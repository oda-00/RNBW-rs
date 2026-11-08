//! Lane-separated ordered-stream codec with an independent 64 KiB LZ window
//! per lane and per-lane raw, rANS, or canonical Huffman storage.

use super::{CodecError, align_up, put_u16, put_u32, rans, read_u16, read_u32};
use rayon::prelude::*;

const MAGIC: &[u8; 4] = b"RNLZ";
const VERSION: u16 = 1;
const HEADER_BYTES: usize = 20;
const DESCRIPTOR_BYTES: usize = 24;
const WINDOW_SIZE: usize = 1 << 16;
const HASH_BITS_COMPACT: usize = 16;
const HASH_BITS_WIDE: usize = 18;
const MAX_CHAIN: usize = 32;
const MAX_MATCH: usize = 130;
const MODE_RAW: u8 = 0;
const MODE_ENTROPY: u8 = 1;
const MODE_LZ_ENTROPY: u8 = 2;
const METHOD_RAW: u8 = 0;
const METHOD_RANS: u8 = 1;
const METHOD_HUFFMAN: u8 = 2;

#[derive(Clone, Debug)]
struct LaneData {
    raw_len: usize,
    token_len: usize,
    model: Vec<u8>,
    payload: Vec<u8>,
    mode: u8,
    entropy_method: u8,
}

#[derive(Clone, Copy, Debug)]
struct LaneDescriptor {
    raw_len: usize,
    token_len: usize,
    payload_offset: usize,
    payload_len: usize,
    model_offset: usize,
    model_len: usize,
    mode: u8,
    entropy_method: u8,
}

/// Lane-split ordered data compressed with independent, multi-window LZ history.
#[derive(Clone, Debug)]
pub struct LaneLz {
    image: Vec<u8>,
}

impl LaneLz {
    /// Encodes independent byte lanes. Each lane has a private 64 KiB match window;
    /// the smallest of raw bytes, entropy-only, and LZ-plus-entropy is stored.
    pub fn encode(lanes: &[&[u8]]) -> Result<Self, CodecError> {
        LaneLzEncoder::default().encode(lanes)
    }

    /// Splits ordered little-endian u32 values into four byte-plane lanes.
    pub fn encode_u32_byte_planes(values: &[u32]) -> Result<Self, CodecError> {
        LaneLzEncoder::default().encode_u32_byte_planes(values)
    }

    /// Encodes a fixed-width interleaved stream as one independent byte lane per
    /// field. For example, RGBA pixels use `lane_width = 4`.
    pub fn encode_interleaved(input: &[u8], lane_width: usize) -> Result<Self, CodecError> {
        LaneLzEncoder::default().encode_interleaved(input, lane_width)
    }

    fn from_lanes(lanes: Vec<LaneData>, raw_total: usize) -> Result<Self, CodecError> {
        let descriptor_end = HEADER_BYTES
            .checked_add(
                lanes
                    .len()
                    .checked_mul(DESCRIPTOR_BYTES)
                    .ok_or(CodecError::LengthOverflow)?,
            )
            .ok_or(CodecError::LengthOverflow)?;
        let header_len = lanes.iter().try_fold(descriptor_end, |position, lane| {
            position
                .checked_add(lane.model.len())
                .ok_or(CodecError::LengthOverflow)
        })?;
        let mut position = align_up(header_len, 8).ok_or(CodecError::LengthOverflow)?;
        let mut descriptors = Vec::with_capacity(lanes.len());
        let mut model_position = descriptor_end;
        for lane in &lanes {
            let model_offset = if lane.model.is_empty() {
                0
            } else {
                let at = model_position;
                model_position = model_position
                    .checked_add(lane.model.len())
                    .ok_or(CodecError::LengthOverflow)?;
                at
            };
            let payload_offset = position;
            position = position
                .checked_add(lane.payload.len())
                .ok_or(CodecError::LengthOverflow)?;
            descriptors.push(LaneDescriptor {
                raw_len: lane.raw_len,
                token_len: lane.token_len,
                payload_offset,
                payload_len: lane.payload.len(),
                model_offset,
                model_len: lane.model.len(),
                mode: lane.mode,
                entropy_method: lane.entropy_method,
            });
            position = align_up(position, 8).ok_or(CodecError::LengthOverflow)?;
        }
        if model_position != header_len
            || header_len > u32::MAX as usize
            || position > u32::MAX as usize
        {
            return Err(CodecError::LengthOverflow);
        }

        let mut image = vec![0u8; position];
        image[..4].copy_from_slice(MAGIC);
        put_u16(&mut image, 4, VERSION)?;
        put_u16(
            &mut image,
            6,
            u16::try_from(lanes.len()).map_err(|_| CodecError::LengthOverflow)?,
        )?;
        put_u32(
            &mut image,
            8,
            u32::try_from(raw_total).map_err(|_| CodecError::LengthOverflow)?,
        )?;
        put_u32(
            &mut image,
            12,
            u32::try_from(header_len).map_err(|_| CodecError::LengthOverflow)?,
        )?;
        put_u32(
            &mut image,
            16,
            u32::try_from(position).map_err(|_| CodecError::LengthOverflow)?,
        )?;
        for (index, (lane, descriptor)) in lanes.iter().zip(&descriptors).enumerate() {
            let at = HEADER_BYTES + index * DESCRIPTOR_BYTES;
            put_u32(
                &mut image,
                at,
                u32::try_from(descriptor.raw_len).map_err(|_| CodecError::LengthOverflow)?,
            )?;
            put_u32(
                &mut image,
                at + 4,
                u32::try_from(descriptor.token_len).map_err(|_| CodecError::LengthOverflow)?,
            )?;
            put_u32(
                &mut image,
                at + 8,
                u32::try_from(descriptor.payload_offset).map_err(|_| CodecError::LengthOverflow)?,
            )?;
            put_u32(
                &mut image,
                at + 12,
                u32::try_from(descriptor.payload_len).map_err(|_| CodecError::LengthOverflow)?,
            )?;
            put_u32(
                &mut image,
                at + 16,
                u32::try_from(descriptor.model_offset).map_err(|_| CodecError::LengthOverflow)?,
            )?;
            put_u16(
                &mut image,
                at + 20,
                u16::try_from(descriptor.model_len).map_err(|_| CodecError::LengthOverflow)?,
            )?;
            image[at + 22] = descriptor.mode;
            image[at + 23] = descriptor.entropy_method;
            if !lane.model.is_empty() {
                let end = descriptor.model_offset + descriptor.model_len;
                image[descriptor.model_offset..end].copy_from_slice(&lane.model);
            }
            let end = descriptor.payload_offset + descriptor.payload_len;
            image[descriptor.payload_offset..end].copy_from_slice(&lane.payload);
        }
        Ok(Self { image })
    }

    /// Parses and validates the serialized lane container.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CodecError> {
        parse_image(bytes)?;
        Ok(Self {
            image: bytes.to_vec(),
        })
    }

    /// Serialized lane container.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.image
    }

    /// Number of lanes stored.
    #[inline]
    pub fn lane_count(&self) -> usize {
        read_u16(&self.image, 6).expect("validated lane header") as usize
    }

    /// Original byte count across all lanes.
    #[inline]
    pub fn raw_bytes(&self) -> usize {
        read_u32(&self.image, 8).expect("validated lane header") as usize
    }

    /// Decodes each lane to its original bytes.
    pub fn decode_lanes(&self) -> Result<Vec<Vec<u8>>, CodecError> {
        let descriptors = parse_image(&self.image)?;
        let mut lanes = Vec::with_capacity(descriptors.len());
        for descriptor in descriptors {
            let payload = &self.image
                [descriptor.payload_offset..descriptor.payload_offset + descriptor.payload_len];
            let model = if descriptor.model_len == 0 {
                &[][..]
            } else {
                &self.image[descriptor.model_offset..descriptor.model_offset + descriptor.model_len]
            };
            let decoded = rans::entropy_decode(
                descriptor.entropy_method,
                descriptor.token_len,
                model,
                payload,
            )?;
            let lane = match descriptor.mode {
                MODE_RAW | MODE_ENTROPY => decoded,
                MODE_LZ_ENTROPY => lz_decode(&decoded, descriptor.raw_len)?,
                _ => return Err(CodecError::InvalidFormat("unknown RNBW lane mode")),
            };
            if lane.len() != descriptor.raw_len {
                return Err(CodecError::InvalidFormat("RNBW lane length mismatch"));
            }
            lanes.push(lane);
        }
        Ok(lanes)
    }

    /// Reassembles four little-endian byte planes into their ordered u32 values.
    pub fn decode_u32_byte_planes(&self) -> Result<Vec<u32>, CodecError> {
        let lanes = self.decode_lanes()?;
        if lanes.len() != 4 || lanes.iter().any(|lane| lane.len() != lanes[0].len()) {
            return Err(CodecError::InvalidFormat(
                "expected four equal byte-plane lanes",
            ));
        }
        let mut values = Vec::with_capacity(lanes[0].len());
        for (index, &low_byte) in lanes[0].iter().enumerate() {
            values.push(
                u32::from(low_byte)
                    | (u32::from(lanes[1][index]) << 8)
                    | (u32::from(lanes[2][index]) << 16)
                    | (u32::from(lanes[3][index]) << 24),
            );
        }
        Ok(values)
    }

    /// Decodes lanes to fixed-width interleaved records.
    pub fn decode_interleaved(&self) -> Result<Vec<u8>, CodecError> {
        let lanes = self.decode_lanes()?;
        if lanes.is_empty() || lanes.iter().any(|lane| lane.len() != lanes[0].len()) {
            return Err(CodecError::InvalidFormat("unequal RNBW interleaved lanes"));
        }
        let output_len = lanes[0]
            .len()
            .checked_mul(lanes.len())
            .ok_or(CodecError::LengthOverflow)?;
        let mut output = Vec::with_capacity(output_len);
        for (record_index, _) in lanes[0].iter().enumerate() {
            for lane in &lanes {
                output.push(lane[record_index]);
            }
        }
        Ok(output)
    }
}

/// Reusable LZ and entropy scratch for repeated encodes.
///
/// The ordinary `LaneLz::encode` API remains convenient for one-shot calls.
/// Callers that encode repeatedly can retain this object to amortize the 64 KiB
/// hash table, 64 KiB window links, and token buffer allocations.
#[derive(Debug, Default)]
pub struct LaneLzEncoder {
    serial: LzWorkspace,
    parallel: Vec<LzWorkspace>,
    interleaved_lanes: Vec<Vec<u8>>,
}

impl LaneLzEncoder {
    /// Encodes independent lanes serially while retaining scratch allocations.
    pub fn encode(&mut self, lanes: &[&[u8]]) -> Result<LaneLz, CodecError> {
        let raw_total = validate_lanes(lanes)?;
        encode_serial_lanes(lanes, raw_total, &mut self.serial)
    }

    /// Encodes independent lanes in parallel, retaining one LZ workspace per
    /// active Rayon worker. Very small lane counts use the serial path.
    pub fn encode_parallel(&mut self, lanes: &[&[u8]]) -> Result<LaneLz, CodecError> {
        let raw_total = validate_lanes(lanes)?;
        encode_parallel_lanes(lanes, raw_total, &mut self.serial, &mut self.parallel)
    }

    /// Splits and encodes interleaved records while retaining the lane buffers.
    ///
    /// This avoids allocating and refilling fresh SoA buffers on repeated image,
    /// shader, and model-data encodes.
    pub fn encode_interleaved(
        &mut self,
        input: &[u8],
        lane_width: usize,
    ) -> Result<LaneLz, CodecError> {
        let LaneLzEncoder {
            serial,
            interleaved_lanes,
            ..
        } = self;
        let raw_total = split_interleaved(input, lane_width, interleaved_lanes)?;
        if lane_width == 4 {
            let lane_refs = [
                interleaved_lanes[0].as_slice(),
                interleaved_lanes[1].as_slice(),
                interleaved_lanes[2].as_slice(),
                interleaved_lanes[3].as_slice(),
            ];
            return encode_serial_lanes(&lane_refs, raw_total, serial);
        }
        let lane_refs: Vec<&[u8]> = interleaved_lanes
            .iter()
            .take(lane_width)
            .map(Vec::as_slice)
            .collect();
        encode_serial_lanes(&lane_refs, raw_total, serial)
    }

    /// Parallel version of [`LaneLzEncoder::encode_interleaved`].
    pub fn encode_interleaved_parallel(
        &mut self,
        input: &[u8],
        lane_width: usize,
    ) -> Result<LaneLz, CodecError> {
        let LaneLzEncoder {
            serial,
            parallel,
            interleaved_lanes,
        } = self;
        let raw_total = split_interleaved(input, lane_width, interleaved_lanes)?;
        if lane_width == 4 {
            let lane_refs = [
                interleaved_lanes[0].as_slice(),
                interleaved_lanes[1].as_slice(),
                interleaved_lanes[2].as_slice(),
                interleaved_lanes[3].as_slice(),
            ];
            return encode_parallel_lanes(&lane_refs, raw_total, serial, parallel);
        }
        let lane_refs: Vec<&[u8]> = interleaved_lanes
            .iter()
            .take(lane_width)
            .map(Vec::as_slice)
            .collect();
        encode_parallel_lanes(&lane_refs, raw_total, serial, parallel)
    }

    /// Splits little-endian u32 values into reusable byte-plane buffers.
    pub fn encode_u32_byte_planes(&mut self, values: &[u32]) -> Result<LaneLz, CodecError> {
        let LaneLzEncoder {
            serial,
            interleaved_lanes,
            ..
        } = self;
        split_u32_byte_planes(values, interleaved_lanes)?;
        let lane_refs = [
            interleaved_lanes[0].as_slice(),
            interleaved_lanes[1].as_slice(),
            interleaved_lanes[2].as_slice(),
            interleaved_lanes[3].as_slice(),
        ];
        let raw_total = values
            .len()
            .checked_mul(4)
            .ok_or(CodecError::LengthOverflow)?;
        encode_serial_lanes(&lane_refs, raw_total, serial)
    }

    /// Parallel version of [`LaneLzEncoder::encode_u32_byte_planes`].
    pub fn encode_u32_byte_planes_parallel(
        &mut self,
        values: &[u32],
    ) -> Result<LaneLz, CodecError> {
        let LaneLzEncoder {
            serial,
            parallel,
            interleaved_lanes,
        } = self;
        split_u32_byte_planes(values, interleaved_lanes)?;
        let lane_refs = [
            interleaved_lanes[0].as_slice(),
            interleaved_lanes[1].as_slice(),
            interleaved_lanes[2].as_slice(),
            interleaved_lanes[3].as_slice(),
        ];
        let raw_total = values
            .len()
            .checked_mul(4)
            .ok_or(CodecError::LengthOverflow)?;
        encode_parallel_lanes(&lane_refs, raw_total, serial, parallel)
    }
}

fn parse_image(image: &[u8]) -> Result<Vec<LaneDescriptor>, CodecError> {
    if image.len() < HEADER_BYTES {
        return Err(CodecError::Truncated);
    }
    if image.get(..4) != Some(MAGIC.as_slice()) || read_u16(image, 4)? != VERSION {
        return Err(CodecError::InvalidFormat("bad RNBW lane signature"));
    }
    let lane_count = read_u16(image, 6)? as usize;
    let raw_total = read_u32(image, 8)? as usize;
    let header_len = read_u32(image, 12)? as usize;
    if read_u32(image, 16)? as usize != image.len() || lane_count == 0 {
        return Err(CodecError::InvalidFormat("invalid RNBW lane header"));
    }
    let descriptor_end = HEADER_BYTES
        .checked_add(
            lane_count
                .checked_mul(DESCRIPTOR_BYTES)
                .ok_or(CodecError::LengthOverflow)?,
        )
        .ok_or(CodecError::LengthOverflow)?;
    if descriptor_end > header_len || header_len > image.len() {
        return Err(CodecError::InvalidFormat("invalid RNBW lane header range"));
    }

    let mut descriptors = Vec::with_capacity(lane_count);
    let mut raw_total_actual = 0usize;
    let mut model_position = descriptor_end;
    let mut payload_position = align_up(header_len, 8).ok_or(CodecError::LengthOverflow)?;
    if payload_position > image.len() {
        return Err(CodecError::InvalidFormat(
            "invalid RNBW lane header padding",
        ));
    }
    if image[header_len..payload_position]
        .iter()
        .any(|&byte| byte != 0)
    {
        return Err(CodecError::InvalidFormat(
            "nonzero RNBW lane header padding",
        ));
    }
    for index in 0..lane_count {
        let at = HEADER_BYTES + index * DESCRIPTOR_BYTES;
        let descriptor = LaneDescriptor {
            raw_len: read_u32(image, at)? as usize,
            token_len: read_u32(image, at + 4)? as usize,
            payload_offset: read_u32(image, at + 8)? as usize,
            payload_len: read_u32(image, at + 12)? as usize,
            model_offset: read_u32(image, at + 16)? as usize,
            model_len: read_u16(image, at + 20)? as usize,
            mode: image[at + 22],
            entropy_method: image[at + 23],
        };
        raw_total_actual = raw_total_actual
            .checked_add(descriptor.raw_len)
            .ok_or(CodecError::LengthOverflow)?;
        let expected_model_len = match descriptor.entropy_method {
            METHOD_RAW => 0,
            METHOD_RANS => 512,
            METHOD_HUFFMAN => 256,
            _ => {
                return Err(CodecError::InvalidFormat(
                    "unknown RNBW lane entropy method",
                ));
            }
        };
        if descriptor.model_len != expected_model_len {
            return Err(CodecError::InvalidFormat("invalid RNBW lane model length"));
        }
        match descriptor.mode {
            MODE_RAW => {
                if descriptor.entropy_method != METHOD_RAW
                    || descriptor.token_len != descriptor.raw_len
                    || descriptor.payload_len != descriptor.raw_len
                {
                    return Err(CodecError::InvalidFormat("invalid raw RNBW lane"));
                }
            }
            MODE_ENTROPY => {
                let stored_len = descriptor
                    .payload_len
                    .checked_add(descriptor.model_len)
                    .ok_or(CodecError::LengthOverflow)?;
                if descriptor.entropy_method == METHOD_RAW
                    || descriptor.token_len != descriptor.raw_len
                    || stored_len >= descriptor.raw_len
                {
                    return Err(CodecError::InvalidFormat("invalid entropy RNBW lane"));
                }
            }
            MODE_LZ_ENTROPY => {
                let stored_len = descriptor
                    .payload_len
                    .checked_add(descriptor.model_len)
                    .ok_or(CodecError::LengthOverflow)?;
                let token_limit = descriptor
                    .raw_len
                    .checked_mul(2)
                    .and_then(|length| length.checked_add(1))
                    .ok_or(CodecError::LengthOverflow)?;
                if stored_len >= descriptor.raw_len || descriptor.token_len > token_limit {
                    return Err(CodecError::InvalidFormat("invalid LZ RNBW lane"));
                }
            }
            _ => return Err(CodecError::InvalidFormat("unknown RNBW lane mode")),
        }
        if descriptor.entropy_method == METHOD_RANS && descriptor.payload_len < 32 {
            return Err(CodecError::InvalidFormat("truncated RNBW lane rANS stream"));
        }
        if descriptor.entropy_method == METHOD_HUFFMAN && descriptor.payload_len == 0 {
            return Err(CodecError::InvalidFormat(
                "truncated RNBW lane Huffman stream",
            ));
        }
        if descriptor.model_len == 0 {
            if descriptor.model_offset != 0 {
                return Err(CodecError::InvalidFormat(
                    "unexpected RNBW lane model offset",
                ));
            }
        } else {
            if descriptor.model_offset != model_position {
                return Err(CodecError::InvalidFormat("noncontiguous RNBW lane models"));
            }
            model_position = model_position
                .checked_add(descriptor.model_len)
                .ok_or(CodecError::LengthOverflow)?;
        }
        if descriptor.payload_offset != payload_position {
            return Err(CodecError::InvalidFormat("noncontiguous RNBW lane payload"));
        }
        let payload_end = descriptor
            .payload_offset
            .checked_add(descriptor.payload_len)
            .ok_or(CodecError::LengthOverflow)?;
        if payload_end > image.len() {
            return Err(CodecError::Truncated);
        }
        payload_position = align_up(payload_end, 8).ok_or(CodecError::LengthOverflow)?;
        if payload_position > image.len()
            || image[payload_end..payload_position]
                .iter()
                .any(|&byte| byte != 0)
        {
            return Err(CodecError::InvalidFormat("bad RNBW lane payload padding"));
        }
        descriptors.push(descriptor);
    }
    if raw_total_actual != raw_total
        || model_position != header_len
        || payload_position != image.len()
    {
        return Err(CodecError::InvalidFormat(
            "RNBW lane lengths disagree with header",
        ));
    }
    Ok(descriptors)
}

fn validate_lanes(lanes: &[&[u8]]) -> Result<usize, CodecError> {
    if lanes.is_empty() || lanes.len() > u16::MAX as usize {
        return Err(CodecError::InvalidFormat("invalid RNBW lane count"));
    }
    let raw_total = lanes.iter().try_fold(0usize, |total, lane| {
        if lane.len() > u32::MAX as usize {
            return Err(CodecError::LengthOverflow);
        }
        total
            .checked_add(lane.len())
            .ok_or(CodecError::LengthOverflow)
    })?;
    if raw_total > u32::MAX as usize {
        return Err(CodecError::LengthOverflow);
    }
    Ok(raw_total)
}

fn encode_serial_lanes(
    lanes: &[&[u8]],
    raw_total: usize,
    workspace: &mut LzWorkspace,
) -> Result<LaneLz, CodecError> {
    let allow_wide_hash = lanes.len() == 1;
    let mut stored = Vec::with_capacity(lanes.len());
    for lane in lanes {
        stored.push(encode_lane(lane, workspace, allow_wide_hash)?);
    }
    LaneLz::from_lanes(stored, raw_total)
}

fn encode_parallel_lanes(
    lanes: &[&[u8]],
    raw_total: usize,
    serial: &mut LzWorkspace,
    parallel: &mut Vec<LzWorkspace>,
) -> Result<LaneLz, CodecError> {
    let workers = rayon::current_num_threads().min(lanes.len());
    if workers < 2 {
        return encode_serial_lanes(lanes, raw_total, serial);
    }
    let allow_wide_hash = lanes.len() == 1;
    let lanes_per_task = lanes.len().div_ceil(workers);
    let tasks = lanes.len().div_ceil(lanes_per_task);
    parallel.resize_with(tasks, LzWorkspace::default);
    parallel.truncate(tasks);

    let mut stored: Vec<Option<LaneData>> =
        std::iter::repeat_with(|| None).take(lanes.len()).collect();
    parallel
        .par_iter_mut()
        .zip(stored.par_chunks_mut(lanes_per_task))
        .zip(lanes.par_chunks(lanes_per_task))
        .try_for_each(|((workspace, output), input)| -> Result<(), CodecError> {
            for (slot, lane) in output.iter_mut().zip(input) {
                *slot = Some(encode_lane(lane, workspace, allow_wide_hash)?);
            }
            Ok(())
        })?;
    let stored = stored
        .into_iter()
        .map(|lane| lane.expect("parallel lane task filled every output slot"))
        .collect();
    LaneLz::from_lanes(stored, raw_total)
}

fn prepare_split_lanes(lanes: &mut Vec<Vec<u8>>, lane_count: usize, records: usize) {
    if lanes.len() < lane_count {
        lanes.resize_with(lane_count, Vec::new);
    }
    for lane in lanes.iter_mut().take(lane_count) {
        lane.clear();
        lane.reserve(records);
    }
}

fn split_interleaved(
    input: &[u8],
    lane_width: usize,
    lanes: &mut Vec<Vec<u8>>,
) -> Result<usize, CodecError> {
    if lane_width == 0 || lane_width > u16::MAX as usize || !input.len().is_multiple_of(lane_width)
    {
        return Err(CodecError::InvalidFormat(
            "invalid interleaved RNBW lane width",
        ));
    }
    if input.len() > u32::MAX as usize {
        return Err(CodecError::LengthOverflow);
    }
    let records = input.len() / lane_width;
    prepare_split_lanes(lanes, lane_width, records);
    for record in input.chunks_exact(lane_width) {
        for (lane, byte) in lanes.iter_mut().zip(record) {
            lane.push(*byte);
        }
    }
    Ok(input.len())
}

fn split_u32_byte_planes(values: &[u32], lanes: &mut Vec<Vec<u8>>) -> Result<(), CodecError> {
    let raw_total = values
        .len()
        .checked_mul(4)
        .ok_or(CodecError::LengthOverflow)?;
    if raw_total > u32::MAX as usize {
        return Err(CodecError::LengthOverflow);
    }
    prepare_split_lanes(lanes, 4, values.len());
    for value in values {
        for (index, lane) in lanes.iter_mut().take(4).enumerate() {
            lane.push((value >> (index * 8)) as u8);
        }
    }
    Ok(())
}

fn encode_lane(
    input: &[u8],
    workspace: &mut LzWorkspace,
    allow_wide_hash: bool,
) -> Result<LaneData, CodecError> {
    let mut best = LaneData {
        raw_len: input.len(),
        token_len: input.len(),
        model: Vec::new(),
        payload: input.to_vec(),
        mode: MODE_RAW,
        entropy_method: METHOD_RAW,
    };

    let (method, model, payload) = rans::entropy_encode(input)?;
    if method != METHOD_RAW && model.len() + payload.len() < best.payload.len() {
        best = LaneData {
            raw_len: input.len(),
            token_len: input.len(),
            model,
            payload,
            mode: MODE_ENTROPY,
            entropy_method: method,
        };
    }

    let hash_bits = if allow_wide_hash && method == METHOD_RAW {
        HASH_BITS_WIDE
    } else {
        HASH_BITS_COMPACT
    };
    let tokens = lz_encode(input, workspace, hash_bits);
    let token_len = tokens.len();
    let (method, model, payload) = rans::entropy_encode(tokens)?;
    if model.len() + payload.len() < best.model.len() + best.payload.len() {
        best = LaneData {
            raw_len: input.len(),
            token_len,
            model,
            payload,
            mode: MODE_LZ_ENTROPY,
            entropy_method: method,
        };
    }
    Ok(best)
}

#[derive(Debug, Default)]
struct LzWorkspace {
    heads: Vec<u32>,
    previous: Vec<u32>,
    touched_heads: Vec<u32>,
    tokens: Vec<u8>,
}

fn lz_encode<'a>(input: &[u8], workspace: &'a mut LzWorkspace, hash_bits: usize) -> &'a [u8] {
    let LzWorkspace {
        heads,
        previous,
        touched_heads,
        tokens: output,
    } = workspace;
    output.clear();
    output.reserve(input.len());
    if input.len() < 3 {
        emit_literals(output, input);
        return output;
    }
    for hash in touched_heads.drain(..) {
        heads[hash as usize] = u32::MAX;
    }
    let hash_size = 1usize << hash_bits;
    if heads.len() != hash_size {
        heads.resize(hash_size, u32::MAX);
    }
    if previous.is_empty() {
        previous.resize(WINDOW_SIZE, u32::MAX);
    }

    let mut index = 0usize;
    let mut literal_start = 0usize;

    while index + 2 < input.len() {
        let hash = hash_at(input, index, hash_bits);
        let mut candidate = heads[hash];
        let mut best_len = 0usize;
        let mut best_distance = 0usize;
        let mut probes = 0usize;
        while candidate != u32::MAX && probes < MAX_CHAIN {
            let candidate_index = candidate as usize;
            let distance = index - candidate_index;
            if distance > WINDOW_SIZE {
                break;
            }
            if input[candidate_index] == input[index]
                && input[candidate_index + 1] == input[index + 1]
                && input[candidate_index + 2] == input[index + 2]
            {
                let mut length = 3usize;
                while index + length < input.len()
                    && input[candidate_index + length] == input[index + length]
                {
                    length += 1;
                }
                if length > best_len {
                    best_len = length;
                    best_distance = distance;
                    if index + length == input.len() {
                        break;
                    }
                }
            }
            candidate = previous[candidate_index % WINDOW_SIZE];
            probes += 1;
        }

        if best_len >= 3 {
            emit_literals(output, &input[literal_start..index]);
            emit_matches(output, best_len, best_distance);
            let end = index + best_len;
            while index < end {
                insert_position(input, index, hash_bits, heads, previous, touched_heads);
                index += 1;
            }
            literal_start = index;
        } else {
            insert_position(input, index, hash_bits, heads, previous, touched_heads);
            index += 1;
        }
    }
    emit_literals(output, &input[literal_start..]);
    output
}

fn hash_at(input: &[u8], index: usize, hash_bits: usize) -> usize {
    let value = u32::from_le_bytes([input[index], input[index + 1], input[index + 2], 0]);
    (value.wrapping_mul(0x1e35_a7bd) >> (32 - hash_bits)) as usize
}

fn insert_position(
    input: &[u8],
    index: usize,
    hash_bits: usize,
    heads: &mut [u32],
    previous: &mut [u32],
    touched_heads: &mut Vec<u32>,
) {
    if index + 2 >= input.len() {
        return;
    }
    let hash = hash_at(input, index, hash_bits);
    let slot = index % WINDOW_SIZE;
    if heads[hash] == u32::MAX {
        touched_heads.push(hash as u32);
    }
    previous[slot] = heads[hash];
    heads[hash] = index as u32;
}

fn emit_literals(output: &mut Vec<u8>, mut literal: &[u8]) {
    while !literal.is_empty() {
        let length = literal.len().min(128);
        output.push((length - 1) as u8);
        output.extend_from_slice(&literal[..length]);
        literal = &literal[length..];
    }
}

fn emit_matches(output: &mut Vec<u8>, mut length: usize, distance: usize) {
    while length != 0 {
        let mut part = length.min(MAX_MATCH);
        if length > MAX_MATCH && length - part < 3 {
            part = length - 3;
        }
        output.push(0x80 | (part - 3) as u8);
        output.extend_from_slice(&((distance - 1) as u16).to_le_bytes());
        length -= part;
    }
}

fn lz_decode(tokens: &[u8], output_len: usize) -> Result<Vec<u8>, CodecError> {
    let mut output = Vec::with_capacity(output_len);
    let mut cursor = 0usize;
    while cursor < tokens.len() {
        let control = tokens[cursor];
        cursor += 1;
        if control & 0x80 == 0 {
            let length = (control as usize) + 1;
            let end = cursor
                .checked_add(length)
                .ok_or(CodecError::LengthOverflow)?;
            if end > tokens.len()
                || output
                    .len()
                    .checked_add(length)
                    .is_none_or(|n| n > output_len)
            {
                return Err(CodecError::InvalidFormat("invalid RNBW lane literal"));
            }
            output.extend_from_slice(&tokens[cursor..end]);
            cursor = end;
        } else {
            let length = (control as usize & 0x7f) + 3;
            let distance_bytes = tokens
                .get(cursor..cursor + 2)
                .ok_or(CodecError::Truncated)?;
            let distance = u16::from_le_bytes([distance_bytes[0], distance_bytes[1]]) as usize + 1;
            cursor += 2;
            if distance > output.len()
                || output
                    .len()
                    .checked_add(length)
                    .is_none_or(|n| n > output_len)
            {
                return Err(CodecError::InvalidFormat("invalid RNBW lane match"));
            }
            for _ in 0..length {
                let value = output[output.len() - distance];
                output.push(value);
            }
        }
    }
    if output.len() != output_len {
        return Err(CodecError::InvalidFormat(
            "RNBW lane output length mismatch",
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_window_lanes_roundtrip_repeated_and_random_data() {
        let mut repeated = Vec::with_capacity(32_000);
        for index in 0..32_000 {
            repeated.push((index % 113) as u8);
        }
        let mut random = Vec::with_capacity(32_000);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..32_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            random.push(state as u8);
        }
        let encoded = LaneLz::encode(&[&repeated, &random]).unwrap();
        assert_eq!(encoded.lane_count(), 2);
        assert_eq!(encoded.decode_lanes().unwrap(), [repeated, random]);
        let parsed = LaneLz::from_bytes(encoded.as_bytes()).unwrap();
        assert_eq!(
            parsed.decode_lanes().unwrap(),
            encoded.decode_lanes().unwrap()
        );
    }

    #[test]
    fn u32_byte_planes_roundtrip_and_corruption_is_rejected() {
        let values: Vec<u32> = (0..50_000)
            .map(|index| (index * 4099) as u32 ^ 0x5a5a_a5a5)
            .collect();
        let encoded = LaneLz::encode_u32_byte_planes(&values).unwrap();
        assert_eq!(encoded.decode_u32_byte_planes().unwrap(), values);
        assert!(encoded.as_bytes().len() < values.len() * 4 + 20);
        let mut corrupt = encoded.as_bytes().to_vec();
        corrupt[0] ^= 1;
        assert!(LaneLz::from_bytes(&corrupt).is_err());
    }

    #[test]
    fn interleaved_lanes_roundtrip_and_reject_bad_widths() {
        let bytes: Vec<u8> = (0..4096)
            .map(|index| ((index / 4) as u8).wrapping_add((index % 4) as u8))
            .collect();
        let encoded = LaneLz::encode_interleaved(&bytes, 4).unwrap();
        assert_eq!(encoded.decode_interleaved().unwrap(), bytes);
        assert!(LaneLz::encode_interleaved(&bytes, 0).is_err());
        assert!(LaneLz::encode_interleaved(&bytes, 3).is_err());
    }

    #[test]
    fn reusable_split_buffers_support_changing_lane_widths() {
        let mut encoder = LaneLzEncoder::default();
        let rgba: Vec<u8> = (0..4096).map(|value| value as u8).collect();
        let rg = rgba[..2048].to_vec();
        let first = encoder.encode_interleaved(&rgba, 4).unwrap();
        assert_eq!(first.decode_interleaved().unwrap(), rgba);
        let second = encoder.encode_interleaved_parallel(&rg, 2).unwrap();
        assert_eq!(second.decode_interleaved().unwrap(), rg);
        let third = encoder.encode_interleaved_parallel(&rgba, 4).unwrap();
        assert_eq!(third.decode_interleaved().unwrap(), rgba);
    }

    #[test]
    fn reusable_serial_and_parallel_encoders_match_one_shot_output() {
        let lanes: Vec<Vec<u8>> = (0..7)
            .map(|lane| {
                (0..20_000)
                    .map(|index| ((index / (lane + 1)) ^ lane) as u8)
                    .collect()
            })
            .collect();
        let refs: Vec<&[u8]> = lanes.iter().map(Vec::as_slice).collect();
        let one_shot = LaneLz::encode(&refs).unwrap();
        let mut reusable = LaneLzEncoder::default();
        let first = reusable.encode(&refs).unwrap();
        let second = reusable.encode(&refs).unwrap();
        assert_eq!(first.as_bytes(), one_shot.as_bytes());
        assert_eq!(second.as_bytes(), one_shot.as_bytes());
        assert_eq!(
            reusable.encode_parallel(&refs).unwrap().as_bytes(),
            one_shot.as_bytes()
        );
        assert_eq!(one_shot.decode_lanes().unwrap(), lanes);
    }
}
