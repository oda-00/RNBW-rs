//! Lane-separated ordered-stream codec with an independent 64 KiB LZ window
//! per lane, followed by one per-lane rANS/Huffman entropy stage.

use super::{CodecError, align_up, put_u16, put_u32, rans, read_u16, read_u32, read_u32_unchecked};
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::OnceLock;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(feature = "wgpu")]
pub mod gpu;

const MAGIC: &[u8; 4] = b"RNLZ";
const VERSION: u16 = 2;
const LEGACY_VERSION: u16 = 1;
const HEADER_BYTES: usize = 20;
const LEGACY_DESCRIPTOR_BYTES: usize = 24;
const DESCRIPTOR_BYTES: usize = 32;
const TOKEN_BYTES: usize = 4;
const WINDOW_SIZE: usize = 1 << 16;
const HASH_BITS_COMPACT: usize = 18;
const HASH_BITS_WIDE: usize = 18;
const MAX_MATCH: usize = 130;
const MATCH_INDEX_TAIL: usize = 16;
const MATCH_INDEX_SKIP_THRESHOLD: usize = 48;
const PARALLEL_MIN_BYTES: usize = 16 * 1024;
const LZ_BLOCK_BYTES: usize = 64 * 1024;
const MODE_RAW: u8 = 0;
const MODE_LZ_ENTROPY: u8 = 2;
const METHOD_RAW: u8 = 0;
const METHOD_RANS: u8 = 1;
const METHOD_HUFFMAN: u8 = 2;

type SplitFourFn = fn(&[u8], &mut [Vec<u8>], usize);
static SPLIT_FOUR_DISPATCH: OnceLock<SplitFourFn> = OnceLock::new();
static RAYON_WORKERS: OnceLock<usize> = OnceLock::new();

#[derive(Clone, Debug)]
struct LaneData {
    raw_len: usize,
    token_len: usize,
    token_count: usize,
    literal_len: usize,
    model: Vec<u8>,
    payload: Vec<u8>,
    mode: u8,
    entropy_method: u8,
}

#[derive(Clone, Copy, Debug)]
struct LaneDescriptor {
    raw_len: usize,
    token_len: usize,
    token_count: usize,
    literal_len: usize,
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
    /// the LZ-plus-entropy result is stored when it shrinks the input; otherwise
    /// the lane is stored raw.
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
                token_count: lane.token_count,
                literal_len: lane.literal_len,
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
            put_u32(
                &mut image,
                at + 24,
                u32::try_from(descriptor.token_count).map_err(|_| CodecError::LengthOverflow)?,
            )?;
            put_u32(
                &mut image,
                at + 28,
                u32::try_from(descriptor.literal_len).map_err(|_| CodecError::LengthOverflow)?,
            )?;
            if !lane.model.is_empty() {
                let end = descriptor.model_offset + descriptor.model_len;
                image[descriptor.model_offset..end].copy_from_slice(&lane.model);
            }
            let end = descriptor.payload_offset + descriptor.payload_len;
            image[descriptor.payload_offset..end].copy_from_slice(&lane.payload);
        }
        Ok(Self { image })
    }

    fn from_raw_single_lane(input: &[u8], raw_total: usize) -> Result<Self, CodecError> {
        let descriptor_end = HEADER_BYTES
            .checked_add(DESCRIPTOR_BYTES)
            .ok_or(CodecError::LengthOverflow)?;
        let payload_offset = align_up(descriptor_end, 8).ok_or(CodecError::LengthOverflow)?;
        let payload_end = payload_offset
            .checked_add(input.len())
            .ok_or(CodecError::LengthOverflow)?;
        let image_len = align_up(payload_end, 8).ok_or(CodecError::LengthOverflow)?;
        if descriptor_end > u32::MAX as usize || image_len > u32::MAX as usize {
            return Err(CodecError::LengthOverflow);
        }

        let mut image = Vec::with_capacity(image_len);
        image.resize(payload_offset, 0);
        image.extend_from_slice(input);
        image.resize(image_len, 0);
        image[..4].copy_from_slice(MAGIC);
        put_u16(&mut image, 4, VERSION)?;
        put_u16(&mut image, 6, 1)?;
        put_u32(
            &mut image,
            8,
            u32::try_from(raw_total).map_err(|_| CodecError::LengthOverflow)?,
        )?;
        put_u32(&mut image, 12, descriptor_end as u32)?;
        put_u32(&mut image, 16, image_len as u32)?;

        let descriptor = HEADER_BYTES;
        let input_len = u32::try_from(input.len()).map_err(|_| CodecError::LengthOverflow)?;
        put_u32(&mut image, descriptor, input_len)?;
        put_u32(&mut image, descriptor + 4, input_len)?;
        put_u32(&mut image, descriptor + 8, payload_offset as u32)?;
        put_u32(&mut image, descriptor + 12, input_len)?;
        image[descriptor + 22] = MODE_RAW;
        image[descriptor + 23] = METHOD_RAW;
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

    /// Decodes each lane to its original bytes, in lane order.
    ///
    /// Multiple lanes are decoded concurrently when the image contains at
    /// least 16 KiB of raw data. A single lane stays on the serial path.
    pub fn decode_lanes(&self) -> Result<Vec<Vec<u8>>, CodecError> {
        // All constructors validate or generate this immutable image. Repeating
        // the full structural validation here would only re-read trusted bytes.
        let count = read_u16(&self.image, 6).expect("validated lane image") as usize;
        let version = read_u16(&self.image, 4).expect("validated lane image");
        let decode = |index| decode_lane(&self.image, index, version);
        if count > 1 && read_u32_unchecked(&self.image, 8) as usize >= PARALLEL_MIN_BYTES {
            (0..count).into_par_iter().map(decode).collect()
        } else {
            (0..count).map(decode).collect()
        }
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
        Ok(self.decode_interleaved_cow()?.into_owned())
    }

    /// Decodes fixed-width interleaved records, borrowing the image payload
    /// when a single raw lane already has the requested byte layout.
    pub fn decode_interleaved_cow(&self) -> Result<Cow<'_, [u8]>, CodecError> {
        // A single raw lane is already the complete interleaved output. Return
        // a view instead of allocating or copying the payload.
        if read_u16(&self.image, 6).expect("validated lane image") == 1 {
            let descriptor = lane_descriptor_unchecked(&self.image, 0);
            if descriptor.mode == MODE_RAW && descriptor.entropy_method == METHOD_RAW {
                if descriptor.payload_len != descriptor.raw_len {
                    return Err(CodecError::InvalidFormat("invalid raw lane length"));
                }
                let end = descriptor
                    .payload_offset
                    .checked_add(descriptor.payload_len)
                    .ok_or(CodecError::LengthOverflow)?;
                return Ok(Cow::Borrowed(&self.image[descriptor.payload_offset..end]));
            }
        }

        let lanes = self.decode_lanes()?;
        if lanes.is_empty() || lanes.iter().any(|lane| lane.len() != lanes[0].len()) {
            return Err(CodecError::InvalidFormat("unequal RNBW interleaved lanes"));
        }
        if lanes.len() == 1 {
            return Ok(Cow::Owned(lanes.into_iter().next().unwrap()));
        }
        let output_len = lanes[0]
            .len()
            .checked_mul(lanes.len())
            .ok_or(CodecError::LengthOverflow)?;
        if lanes.len() == 4 {
            let mut output = Vec::with_capacity(output_len);
            for (((red, green), blue), alpha) in
                lanes[0].iter().zip(&lanes[1]).zip(&lanes[2]).zip(&lanes[3])
            {
                output.push(*red);
                output.push(*green);
                output.push(*blue);
                output.push(*alpha);
            }
            return Ok(Cow::Owned(output));
        }
        let mut output = Vec::with_capacity(output_len);
        for (record_index, _) in lanes[0].iter().enumerate() {
            for lane in &lanes {
                output.push(lane[record_index]);
            }
        }
        Ok(Cow::Owned(output))
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
        if lane_width == 1 {
            let raw_total = validate_lanes(&[input])?;
            return encode_serial_lanes(&[input], raw_total, &mut self.serial);
        }
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
        if lane_width == 1 {
            let raw_total = validate_lanes(&[input])?;
            return encode_parallel_lanes(
                &[input],
                raw_total,
                &mut self.serial,
                &mut self.parallel,
            );
        }
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
    let version = read_u16(image, 4)?;
    if image.get(..4) != Some(MAGIC.as_slice()) || !matches!(version, LEGACY_VERSION | VERSION) {
        return Err(CodecError::InvalidFormat("bad RNBW lane signature"));
    }
    let descriptor_bytes = if version == LEGACY_VERSION {
        LEGACY_DESCRIPTOR_BYTES
    } else {
        DESCRIPTOR_BYTES
    };
    let lane_count = read_u16(image, 6)? as usize;
    let raw_total = read_u32(image, 8)? as usize;
    let header_len = read_u32(image, 12)? as usize;
    if read_u32(image, 16)? as usize != image.len() || lane_count == 0 {
        return Err(CodecError::InvalidFormat("invalid RNBW lane header"));
    }
    let descriptor_end = HEADER_BYTES
        .checked_add(
            lane_count
                .checked_mul(descriptor_bytes)
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
        let at = HEADER_BYTES + index * descriptor_bytes;
        let descriptor = LaneDescriptor {
            raw_len: read_u32(image, at)? as usize,
            token_len: read_u32(image, at + 4)? as usize,
            token_count: if version == LEGACY_VERSION {
                0
            } else {
                read_u32(image, at + 24)? as usize
            },
            literal_len: if version == LEGACY_VERSION {
                0
            } else {
                read_u32(image, at + 28)? as usize
            },
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
                    || descriptor.token_count != 0
                    || descriptor.literal_len != 0
                {
                    return Err(CodecError::InvalidFormat("invalid raw RNBW lane"));
                }
            }
            MODE_LZ_ENTROPY => {
                let stored_len = descriptor
                    .payload_len
                    .checked_add(descriptor.model_len)
                    .ok_or(CodecError::LengthOverflow)?;
                let token_stream_len = descriptor
                    .token_count
                    .checked_mul(TOKEN_BYTES)
                    .and_then(|length| length.checked_add(descriptor.literal_len))
                    .ok_or(CodecError::LengthOverflow)?;
                let malformed_v1 = version == LEGACY_VERSION
                    && descriptor.token_len
                        > descriptor
                            .raw_len
                            .checked_mul(2)
                            .and_then(|length| length.checked_add(1))
                            .ok_or(CodecError::LengthOverflow)?;
                let malformed_v2 = version == VERSION
                    && (descriptor.token_count == 0
                        || descriptor.token_count > descriptor.raw_len
                        || descriptor.literal_len > descriptor.raw_len
                        || token_stream_len != descriptor.token_len);
                if stored_len >= descriptor.raw_len || malformed_v1 || malformed_v2 {
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

#[inline]
fn lane_descriptor_unchecked(image: &[u8], index: usize) -> LaneDescriptor {
    let version = u16::from_le_bytes([image[4], image[5]]);
    lane_descriptor_unchecked_version(image, index, version)
}

#[inline]
fn lane_descriptor_unchecked_version(image: &[u8], index: usize, version: u16) -> LaneDescriptor {
    let descriptor_bytes = if version == LEGACY_VERSION {
        LEGACY_DESCRIPTOR_BYTES
    } else {
        DESCRIPTOR_BYTES
    };
    let at = HEADER_BYTES + index * descriptor_bytes;
    LaneDescriptor {
        raw_len: read_u32_unchecked(image, at) as usize,
        token_len: read_u32_unchecked(image, at + 4) as usize,
        token_count: if version == LEGACY_VERSION {
            0
        } else {
            read_u32_unchecked(image, at + 24) as usize
        },
        literal_len: if version == LEGACY_VERSION {
            0
        } else {
            read_u32_unchecked(image, at + 28) as usize
        },
        payload_offset: read_u32_unchecked(image, at + 8) as usize,
        payload_len: read_u32_unchecked(image, at + 12) as usize,
        model_offset: read_u32_unchecked(image, at + 16) as usize,
        model_len: u16::from_le_bytes([image[at + 20], image[at + 21]]) as usize,
        mode: image[at + 22],
        entropy_method: image[at + 23],
    }
}

fn decode_lane(image: &[u8], index: usize, version: u16) -> Result<Vec<u8>, CodecError> {
    let descriptor = lane_descriptor_unchecked_version(image, index, version);
    let payload =
        &image[descriptor.payload_offset..descriptor.payload_offset + descriptor.payload_len];
    let model = if descriptor.model_len == 0 {
        &[][..]
    } else {
        &image[descriptor.model_offset..descriptor.model_offset + descriptor.model_len]
    };
    let decoded = rans::entropy_decode(
        descriptor.entropy_method,
        descriptor.token_len,
        model,
        payload,
    )?;
    match descriptor.mode {
        MODE_RAW => Ok(decoded),
        MODE_LZ_ENTROPY if version == LEGACY_VERSION => {
            lz_decode_legacy(&decoded, descriptor.raw_len)
        }
        MODE_LZ_ENTROPY => lz_decode(
            &decoded,
            descriptor.token_count,
            descriptor.literal_len,
            descriptor.raw_len,
        ),
        _ => Err(CodecError::InvalidFormat("unknown RNBW lane mode")),
    }
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
    let hash_bits = if lanes.len() == 1 {
        HASH_BITS_WIDE
    } else {
        HASH_BITS_COMPACT
    };
    if lanes.len() == 1 {
        let input = lanes[0];
        if should_skip_lz_sample(input) {
            return LaneLz::from_raw_single_lane(input, raw_total);
        }
        return LaneLz::from_lanes(
            vec![encode_lane_after_sample(input, workspace, hash_bits)?],
            raw_total,
        );
    }
    let mut stored = Vec::with_capacity(lanes.len());
    for lane in lanes {
        stored.push(encode_lane(lane, workspace, hash_bits)?);
    }
    LaneLz::from_lanes(stored, raw_total)
}

fn encode_parallel_lanes(
    lanes: &[&[u8]],
    raw_total: usize,
    serial: &mut LzWorkspace,
    parallel: &mut Vec<LzWorkspace>,
) -> Result<LaneLz, CodecError> {
    if raw_total < PARALLEL_MIN_BYTES {
        return encode_serial_lanes(lanes, raw_total, serial);
    }
    if lanes.len() == 1 {
        let lane = lanes[0];
        if should_skip_lz_sample(lane) {
            return LaneLz::from_raw_single_lane(lane, raw_total);
        }
        if lane.len() <= LZ_BLOCK_BYTES {
            return LaneLz::from_lanes(
                vec![encode_lane_after_sample(lane, serial, HASH_BITS_WIDE)?],
                raw_total,
            );
        }
        let worker_count = rayon_worker_count();
        let block_count = lane.len().div_ceil(LZ_BLOCK_BYTES);
        let tasks = worker_count.min(block_count);
        if tasks < 2 {
            return LaneLz::from_lanes(
                vec![encode_lane_after_sample(lane, serial, HASH_BITS_WIDE)?],
                raw_total,
            );
        }
        return encode_parallel_single_lane(lane, raw_total, worker_count, serial, parallel);
    }
    if lanes.len() < 2 {
        return encode_serial_lanes(lanes, raw_total, serial);
    }
    let worker_count = rayon_worker_count();
    let workers = worker_count.min(lanes.len());
    if workers < 2 {
        return encode_serial_lanes(lanes, raw_total, serial);
    }
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
                *slot = Some(encode_lane(lane, workspace, HASH_BITS_COMPACT)?);
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
    if lane_width == 4 {
        split_four_dispatch(input, &mut lanes[..4], records);
        return Ok(input.len());
    }
    for record in input.chunks_exact(lane_width) {
        for (lane, byte) in lanes.iter_mut().zip(record) {
            lane.push(*byte);
        }
    }
    Ok(input.len())
}

fn split_four_dispatch(input: &[u8], lanes: &mut [Vec<u8>], records: usize) {
    debug_assert_eq!(lanes.len(), 4);
    let split = *SPLIT_FOUR_DISPATCH.get_or_init(|| {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") {
            return split_four_avx2_entry;
        }
        split_four_scalar
    });
    split(input, lanes, records);
}

fn split_four_scalar(input: &[u8], lanes: &mut [Vec<u8>], records: usize) {
    for lane in lanes.iter_mut() {
        lane.resize(records, 0);
    }
    for (record_index, record) in input.as_chunks::<4>().0.iter().enumerate() {
        lanes[0][record_index] = record[0];
        lanes[1][record_index] = record[1];
        lanes[2][record_index] = record[2];
        lanes[3][record_index] = record[3];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn split_four_avx2_entry(input: &[u8], lanes: &mut [Vec<u8>], records: usize) {
    // The dispatch cache calls this entry only after AVX2 feature detection.
    unsafe { split_four_avx2(input, lanes, records) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn split_four_avx2(input: &[u8], lanes: &mut [Vec<u8>], records: usize) {
    const M0: [i8; 32] = [
        0, 4, 8, 12, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 0, 4,
        8, 12, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    ];
    const M1: [i8; 32] = [
        1, 5, 9, 13, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 1, 5,
        9, 13, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    ];
    const M2: [i8; 32] = [
        2, 6, 10, 14, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 2, 6,
        10, 14, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    ];
    const M3: [i8; 32] = [
        3, 7, 11, 15, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 3, 7,
        11, 15, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    ];

    for lane in lanes.iter_mut() {
        lane.resize(records, 0);
    }
    let masks = [M0, M1, M2, M3].map(|mask| unsafe { _mm256_loadu_si256(mask.as_ptr().cast()) });
    let pack = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);
    let vector_records = records & !7;
    let mut record = 0usize;
    while record < vector_records {
        let bytes = unsafe { input.as_ptr().add(record * 4) };
        let interleaved = unsafe { _mm256_loadu_si256(bytes.cast()) };
        for (channel, lane) in lanes.iter_mut().enumerate() {
            let separated = _mm256_shuffle_epi8(interleaved, masks[channel]);
            let packed = _mm256_permutevar8x32_epi32(separated, pack);
            let lane_out = unsafe { lane.as_mut_ptr().add(record).cast::<__m128i>() };
            unsafe { _mm_storel_epi64(lane_out, _mm256_castsi256_si128(packed)) };
        }
        record += 8;
    }
    while record < records {
        let source = unsafe { input.as_ptr().add(record * 4) };
        for (channel, lane) in lanes.iter_mut().enumerate() {
            lane[record] = unsafe { *source.add(channel) };
        }
        record += 1;
    }
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

    // On little-endian targets the in-memory u32 representation is already the
    // byte stream consumed by the four-lane splitter. Reuse its cached AVX2
    // dispatch instead of materializing four scalar `to_le_bytes` pushes per
    // value. The byte view is valid for exactly `raw_total` initialized bytes.
    #[cfg(target_endian = "little")]
    {
        // SAFETY: `u32` has no padding, the checked length is `values.len() * 4`,
        // and `u8` has alignment 1. The view does not outlive `values`.
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), raw_total) };
        split_four_dispatch(bytes, &mut lanes[..4], values.len());
        Ok(())
    }

    #[cfg(target_endian = "big")]
    {
        for &value in values {
            let [x0, x1, x2, x3] = value.to_le_bytes();
            lanes[0].push(x0);
            lanes[1].push(x1);
            lanes[2].push(x2);
            lanes[3].push(x3);
        }
        Ok(())
    }
}

fn encode_lane(
    input: &[u8],
    workspace: &mut LzWorkspace,
    hash_bits: usize,
) -> Result<LaneData, CodecError> {
    if should_skip_lz_sample(input) {
        return Ok(raw_lane_data(input));
    }
    encode_lane_after_sample(input, workspace, hash_bits)
}

fn encode_lane_after_sample(
    input: &[u8],
    workspace: &mut LzWorkspace,
    hash_bits: usize,
) -> Result<LaneData, CodecError> {
    let (token_count, literal_len) = lz_encode(input, workspace, hash_bits);
    choose_lane_data(input, &workspace.tokens, token_count, literal_len)
}

fn encode_parallel_single_lane(
    input: &[u8],
    raw_total: usize,
    worker_count: usize,
    serial: &mut LzWorkspace,
    parallel: &mut Vec<LzWorkspace>,
) -> Result<LaneLz, CodecError> {
    let block_count = input.len().div_ceil(LZ_BLOCK_BYTES);
    let task_count = worker_count.min(block_count);
    let blocks_per_task = block_count.div_ceil(task_count);
    let task_count = block_count.div_ceil(blocks_per_task);
    parallel.resize_with(task_count, LzWorkspace::default);
    parallel.truncate(task_count);

    parallel
        .par_iter_mut()
        .enumerate()
        .for_each(|(task_index, workspace)| {
            let first_block = task_index * blocks_per_task;
            let end_block = (first_block + blocks_per_task).min(block_count);
            let start = first_block * LZ_BLOCK_BYTES;
            let end = (end_block * LZ_BLOCK_BYTES).min(input.len());
            encode_lz_range(input, start, end, workspace, HASH_BITS_WIDE);
        });

    serial.block_tokens.clear();
    serial.block_literals.clear();
    for workspace in parallel.iter() {
        serial
            .block_tokens
            .extend_from_slice(&workspace.block_tokens);
        serial
            .block_literals
            .extend_from_slice(&workspace.block_literals);
    }
    serial.tokens.clear();
    serial
        .tokens
        .reserve(serial.block_tokens.len() + serial.block_literals.len());
    serial.tokens.extend_from_slice(&serial.block_tokens);
    serial.tokens.extend_from_slice(&serial.block_literals);
    let token_count = serial.block_tokens.len() / TOKEN_BYTES;
    let literal_len = serial.block_literals.len();
    let stored = choose_lane_data(input, &serial.tokens, token_count, literal_len)?;
    LaneLz::from_lanes(vec![stored], raw_total)
}

fn encode_lz_range(
    input: &[u8],
    start: usize,
    end: usize,
    workspace: &mut LzWorkspace,
    hash_bits: usize,
) {
    workspace.block_tokens.clear();
    workspace.block_literals.clear();
    let mut block_start = start;
    while block_start < end {
        let block_end = (block_start + LZ_BLOCK_BYTES).min(end);
        let lookahead_end = (block_end + MAX_MATCH).min(input.len());
        let history_start = block_start.saturating_sub(WINDOW_SIZE);
        let context = &input[history_start..lookahead_end];
        let history_len = block_start - history_start;
        let target_len = block_end - block_start;
        let (token_count, literal_len) =
            lz_encode_window(context, history_len, target_len, workspace, hash_bits);
        let token_bytes = token_count * TOKEN_BYTES;
        workspace
            .block_tokens
            .extend_from_slice(&workspace.tokens[..token_bytes]);
        workspace
            .block_literals
            .extend_from_slice(&workspace.tokens[token_bytes..token_bytes + literal_len]);
        block_start = block_end;
    }
}

fn rayon_worker_count() -> usize {
    *RAYON_WORKERS.get_or_init(|| {
        // The process-wide Rayon pool is persistent and uses work stealing.
        // Reuse an embedding application's pool when it initialized Rayon first.
        let _ = rayon::ThreadPoolBuilder::new().build_global();
        rayon::current_num_threads()
    })
}

fn should_skip_lz_sample(input: &[u8]) -> bool {
    const MIN_BYTES: usize = 4096;
    const SAMPLE_BYTES: usize = 32;
    const SAMPLE_COUNT: usize = 64;
    const GRAM_TABLE_SIZE: usize = 4096;
    if input.len() < MIN_BYTES {
        return false;
    }

    let mut byte_counts = [0u16; 256];
    let mut grams = [0u32; GRAM_TABLE_SIZE];
    let mut sample_bytes = 0usize;
    let mut duplicate_grams = 0usize;
    let max_start = input.len() - SAMPLE_BYTES;
    for sample in 0..SAMPLE_COUNT {
        let start = sample * max_start / (SAMPLE_COUNT - 1);
        let bytes = &input[start..start + SAMPLE_BYTES];
        for &byte in bytes {
            byte_counts[byte as usize] += 1;
        }
        sample_bytes += bytes.len();
        for gram in bytes.windows(3) {
            let key = (u32::from(gram[0]) << 16) | (u32::from(gram[1]) << 8) | u32::from(gram[2]);
            let stored_key = key + 1;
            let mut slot = key.wrapping_mul(0x9e37_79b1) as usize & (GRAM_TABLE_SIZE - 1);
            loop {
                match grams[slot] {
                    0 => {
                        grams[slot] = stored_key;
                        break;
                    }
                    existing if existing == stored_key => {
                        duplicate_grams += 1;
                        break;
                    }
                    _ => slot = (slot + 1) & (GRAM_TABLE_SIZE - 1),
                }
            }
        }
    }

    let distinct_bytes = byte_counts.iter().filter(|&&count| count != 0).count();
    let max_frequency = byte_counts.iter().copied().max().unwrap_or(0) as usize;
    distinct_bytes >= 240 && max_frequency <= sample_bytes / 16 && duplicate_grams <= 32
}

fn raw_lane_data(input: &[u8]) -> LaneData {
    LaneData {
        raw_len: input.len(),
        token_len: input.len(),
        token_count: 0,
        literal_len: 0,
        model: Vec::new(),
        payload: input.to_vec(),
        mode: MODE_RAW,
        entropy_method: METHOD_RAW,
    }
}

fn choose_lane_data(
    input: &[u8],
    lz_stream: &[u8],
    token_count: usize,
    literal_len: usize,
) -> Result<LaneData, CodecError> {
    // lz_encode has already counted every literal byte while testing its
    // sliding history window. A very low literal fraction means the 64 KiB
    // window found strong repetition, so give the fixed-capacity Huffman path
    // first refusal and skip rANS when that cheaper code shrinks the tokens.
    let prefer_huffman = input.len() >= 4096 && literal_len <= input.len() / 8;
    let Some((method, model, payload)) =
        rans::entropy_encode_candidate_with_hint(lz_stream, prefer_huffman)?
    else {
        return Ok(raw_lane_data(input));
    };
    if model.len() + payload.len() >= input.len() {
        return Ok(raw_lane_data(input));
    }
    Ok(LaneData {
        raw_len: input.len(),
        token_len: lz_stream.len(),
        token_count,
        literal_len,
        model,
        payload,
        mode: MODE_LZ_ENTROPY,
        entropy_method: method,
    })
}

#[cfg(feature = "wgpu")]
fn encode_lane_from_lz(
    input: &[u8],
    lz_stream: &[u8],
    token_count: usize,
    literal_len: usize,
) -> Result<LaneData, CodecError> {
    choose_lane_data(input, lz_stream, token_count, literal_len)
}

#[derive(Debug, Default)]
struct LzWorkspace {
    // Each hash slot packs its two newest matching positions. Directly
    // testing both candidates avoids the dependent pointer chain.
    heads: Vec<u64>,
    touched_heads: Vec<u32>,
    tokens: Vec<u8>,
    literal_bytes: Vec<u8>,
    block_tokens: Vec<u8>,
    block_literals: Vec<u8>,
}

fn lz_encode(input: &[u8], workspace: &mut LzWorkspace, hash_bits: usize) -> (usize, usize) {
    lz_encode_window(input, 0, input.len(), workspace, hash_bits)
}

fn lz_encode_window(
    input: &[u8],
    history_len: usize,
    target_len: usize,
    workspace: &mut LzWorkspace,
    hash_bits: usize,
) -> (usize, usize) {
    let LzWorkspace {
        heads,
        touched_heads,
        tokens: output,
        literal_bytes,
        ..
    } = workspace;
    let target_end = history_len + target_len;
    debug_assert!(target_end <= input.len());
    output.clear();
    literal_bytes.clear();
    output.reserve(target_len);
    if target_len == 0 {
        return (0, 0);
    }
    if input.len() < 3 {
        emit_literals(output, literal_bytes, &input[history_len..target_end]);
        return finish_token_stream(output, literal_bytes);
    }
    for hash in touched_heads.drain(..) {
        heads[hash as usize] = u64::MAX;
    }
    let hash_size = 1usize << hash_bits;
    if heads.len() != hash_size {
        heads.resize(hash_size, u64::MAX);
    }
    for index in 0..history_len {
        if index + 2 < input.len() {
            let hash = hash_at(input, index, hash_bits);
            insert_position(index, hash, heads, touched_heads);
        }
    }

    let mut index = history_len;
    let mut literal_start = history_len;

    while index < target_end && index + 2 < input.len() {
        let hash = hash_at(input, index, hash_bits);
        let candidates = heads[hash];
        insert_position(index, hash, heads, touched_heads);
        let mut best_len = 0usize;
        let mut best_distance = 0usize;
        let candidate0 = candidates as u32;
        if candidate0 != u32::MAX {
            let candidate_index = candidate0 as usize;
            let distance = index - candidate_index;
            if distance <= WINDOW_SIZE
                && input[candidate_index] == input[index]
                && input[candidate_index + 1] == input[index + 1]
                && input[candidate_index + 2] == input[index + 2]
            {
                let length = 3 + super::extend_match(input, candidate_index + 3, index + 3);
                if length > best_len {
                    best_len = length;
                    best_distance = distance;
                }
            }
        }
        let candidate1 = (candidates >> 32) as u32;
        if candidate1 != u32::MAX {
            let candidate_index = candidate1 as usize;
            let distance = index - candidate_index;
            if distance <= WINDOW_SIZE
                && input[candidate_index] == input[index]
                && input[candidate_index + 1] == input[index + 1]
                && input[candidate_index + 2] == input[index + 2]
            {
                let length = 3 + super::extend_match(input, candidate_index + 3, index + 3);
                if length > best_len {
                    best_len = length;
                    best_distance = distance;
                }
            }
        }
        best_len = best_len.min(target_end - index);
        if best_len >= 3 {
            emit_literals(output, literal_bytes, &input[literal_start..index]);
            emit_matches(output, best_len, best_distance);
            let end = index + best_len;
            // The current position was inserted before candidate matching.
            // Keep the final positions of each match available to nearby
            // lookups, while jumping over older positions hidden by the match.
            index += 1;
            if best_len > MATCH_INDEX_SKIP_THRESHOLD {
                index = end - MATCH_INDEX_TAIL;
            }
            while index < end {
                if index + 2 < input.len() {
                    insert_position(
                        index,
                        hash_at(input, index, hash_bits),
                        heads,
                        touched_heads,
                    );
                }
                index += 1;
            }
            literal_start = index;
        } else {
            index += 1;
        }
    }
    emit_literals(output, literal_bytes, &input[literal_start..target_end]);
    finish_token_stream(output, literal_bytes)
}

fn finish_token_stream(output: &mut Vec<u8>, literal_bytes: &[u8]) -> (usize, usize) {
    let token_count = output.len() / TOKEN_BYTES;
    let literal_len = literal_bytes.len();
    output.extend_from_slice(literal_bytes);
    (token_count, literal_len)
}

fn hash_at(input: &[u8], index: usize, hash_bits: usize) -> usize {
    let value = u32::from_le_bytes([input[index], input[index + 1], input[index + 2], 0]);
    (value.wrapping_mul(0x1e35_a7bd) >> (32 - hash_bits)) as usize
}

fn insert_position(index: usize, hash: usize, heads: &mut [u64], touched_heads: &mut Vec<u32>) {
    let previous = heads[hash];
    if previous == u64::MAX {
        touched_heads.push(hash as u32);
    }
    heads[hash] = (u64::from(previous as u32) << 32) | index as u64;
}

fn emit_literals(output: &mut Vec<u8>, literal_bytes: &mut Vec<u8>, mut literal: &[u8]) {
    while !literal.is_empty() {
        let length = literal.len().min(128);
        let token = ((length - 1) as u32) << 24;
        output.extend_from_slice(&token.to_le_bytes());
        literal_bytes.extend_from_slice(&literal[..length]);
        literal = &literal[length..];
    }
}

fn emit_matches(output: &mut Vec<u8>, mut length: usize, distance: usize) {
    while length != 0 {
        let mut part = length.min(MAX_MATCH);
        if length > MAX_MATCH && length - part < 3 {
            part = length - 3;
        }
        let token = 0x8000_0000 | (((part - 3) as u32) << 24) | (((distance - 1) as u32) << 8);
        output.extend_from_slice(&token.to_le_bytes());
        length -= part;
    }
}

fn lz_decode(
    stream: &[u8],
    token_count: usize,
    literal_len: usize,
    output_len: usize,
) -> Result<Vec<u8>, CodecError> {
    let token_bytes = token_count
        .checked_mul(TOKEN_BYTES)
        .ok_or(CodecError::LengthOverflow)?;
    if token_bytes.checked_add(literal_len) != Some(stream.len()) {
        return Err(CodecError::InvalidFormat(
            "invalid fixed-width RNBW token stream",
        ));
    }
    let mut output = Vec::with_capacity(output_len);
    let mut literal_cursor = token_bytes;
    for token_index in 0..token_count {
        let at = token_index * TOKEN_BYTES;
        let token =
            u32::from_le_bytes([stream[at], stream[at + 1], stream[at + 2], stream[at + 3]]);
        let length = ((token >> 24) & 0x7f) as usize;
        if token & 0x8000_0000 == 0 {
            let length = length + 1;
            let end = literal_cursor
                .checked_add(length)
                .ok_or(CodecError::LengthOverflow)?;
            if token & 0x00ff_ffff != 0
                || end > stream.len()
                || output
                    .len()
                    .checked_add(length)
                    .is_none_or(|n| n > output_len)
            {
                return Err(CodecError::InvalidFormat("invalid RNBW literal token"));
            }
            output.extend_from_slice(&stream[literal_cursor..end]);
            literal_cursor = end;
        } else {
            let length = length + 3;
            let distance = (((token >> 8) & 0xffff) as usize) + 1;
            if token & 0xff != 0
                || distance > output.len()
                || output
                    .len()
                    .checked_add(length)
                    .is_none_or(|n| n > output_len)
            {
                return Err(CodecError::InvalidFormat("invalid RNBW match token"));
            }
            let start = output.len();
            let mut copied = distance.min(length);
            output.extend_from_within(start - distance..start - distance + copied);
            while copied < length {
                let amount = copied.min(length - copied);
                output.extend_from_within(start..start + amount);
                copied += amount;
            }
        }
    }
    if literal_cursor != stream.len() || output.len() != output_len {
        return Err(CodecError::InvalidFormat(
            "RNBW fixed token output length mismatch",
        ));
    }
    Ok(output)
}

fn lz_decode_legacy(tokens: &[u8], output_len: usize) -> Result<Vec<u8>, CodecError> {
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
        let descriptors = parse_image(encoded.as_bytes()).unwrap();
        assert!(descriptors.iter().any(|lane| lane.mode == MODE_LZ_ENTROPY));
        assert!(descriptors.iter().any(|lane| lane.mode == MODE_RAW));
        assert!(
            descriptors
                .iter()
                .filter(|lane| lane.mode == MODE_LZ_ENTROPY)
                .all(|lane| matches!(lane.entropy_method, METHOD_RANS | METHOD_HUFFMAN))
        );
    }

    #[test]
    fn fixed_width_lane_tokens_roundtrip_overlap_and_reject_bad_fields() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&0x0200_0000u32.to_le_bytes()); // 3 literals
        stream.extend_from_slice(&0x8000_0200u32.to_le_bytes()); // length 3, distance 3
        stream.extend_from_slice(b"abc");
        assert_eq!(lz_decode(&stream, 2, 3, 6).unwrap(), b"abcabc");

        let mut invalid_literal = Vec::new();
        invalid_literal.extend_from_slice(&0x0200_0001u32.to_le_bytes());
        invalid_literal.extend_from_slice(b"abc");
        assert!(lz_decode(&invalid_literal, 1, 3, 3).is_err());

        let mut invalid_match = Vec::new();
        invalid_match.extend_from_slice(&0x8000_0100u32.to_le_bytes());
        assert!(lz_decode(&invalid_match, 1, 0, 3).is_err());
    }

    #[test]
    fn encoded_lane_uses_version_two_fixed_width_token_records() {
        let input: Vec<u8> = (0..32_768).map(|index| (index % 61) as u8).collect();
        let encoded = LaneLz::encode_interleaved(&input, 1).unwrap();
        assert_eq!(read_u16(encoded.as_bytes(), 4).unwrap(), VERSION);
        let descriptor = parse_image(encoded.as_bytes()).unwrap()[0];
        assert_eq!(descriptor.mode, MODE_LZ_ENTROPY);
        assert!(descriptor.token_count > 0);
        assert_eq!(
            descriptor.token_len,
            descriptor.token_count * TOKEN_BYTES + descriptor.literal_len
        );
        assert_eq!(encoded.decode_lanes().unwrap()[0], input);
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
    fn single_raw_lane_decode_borrows_the_image_payload() {
        let mut bytes = Vec::with_capacity(32_768);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..32_768 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bytes.push(state as u8);
        }
        let encoded = LaneLz::encode_interleaved(&bytes, 1).unwrap();
        let decoded = encoded.decode_interleaved_cow().unwrap();
        assert!(matches!(decoded, Cow::Borrowed(_)));
        assert_eq!(decoded.as_ref(), bytes);
        assert!(
            encoded
                .as_bytes()
                .windows(decoded.len())
                .any(|window| window.as_ptr() == decoded.as_ptr())
        );
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
    fn four_lane_avx2_split_matches_scalar_for_all_vector_tails() {
        for records in 0..=65 {
            let input = (0..records * 4)
                .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
                .collect::<Vec<_>>();
            let mut expected = std::array::from_fn::<_, 4, _>(|_| Vec::new());
            let mut actual = std::array::from_fn::<_, 4, _>(|_| Vec::new());
            split_four_scalar(&input, &mut expected, records);
            split_four_dispatch(&input, &mut actual, records);
            assert_eq!(actual, expected, "record count {records}");
        }
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

    #[test]
    fn single_lane_parallel_blocks_roundtrip_across_boundaries_without_density_loss() {
        let mut serial = LaneLzEncoder::default();
        let mut parallel = LaneLzEncoder::default();
        for length in [65_537usize, 131_072, 262_147, 786_433] {
            let input = (0..length)
                .map(|index| ((index / 7 + index / 113) % 251) as u8)
                .collect::<Vec<_>>();
            let expected = serial.encode(&[&input]).unwrap();
            let actual = parallel.encode_parallel(&[&input]).unwrap();
            assert_eq!(actual.decode_lanes().unwrap()[0], input, "length={length}");
            assert!(
                actual.as_bytes().len() <= expected.as_bytes().len() + length / 100,
                "parallel block overhead exceeded 1% at {length}: serial={}, parallel={}",
                expected.as_bytes().len(),
                actual.as_bytes().len()
            );
            assert_eq!(
                parallel.encode_parallel(&[&input]).unwrap().as_bytes(),
                actual.as_bytes(),
                "parallel output must be deterministic at {length}"
            );
        }
    }

    #[test]
    fn entropy_sample_skips_random_bytes_but_not_uniform_periodic_sequences() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let random = (0..32_768)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect::<Vec<_>>();
        let periodic = (0..32_768).map(|index| index as u8).collect::<Vec<_>>();
        assert!(should_skip_lz_sample(&random));
        assert!(!should_skip_lz_sample(&periodic));

        let encoded = LaneLz::encode(&[&random]).unwrap();
        let descriptor = parse_image(encoded.as_bytes()).unwrap()[0];
        assert_eq!(descriptor.mode, MODE_RAW);
        let previous_raw_fallback =
            LaneLz::from_lanes(vec![raw_lane_data(&random)], random.len()).unwrap();
        assert_eq!(encoded.as_bytes(), previous_raw_fallback.as_bytes());
        assert_eq!(encoded.decode_lanes().unwrap()[0], random);

        let mut mixed = (0..1_048_576)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect::<Vec<_>>();
        for (index, byte) in mixed[400_000..650_000].iter_mut().enumerate() {
            *byte = index as u8;
        }
        assert!(!should_skip_lz_sample(&mixed));
        let encoded = LaneLz::encode(&[&mixed]).unwrap();
        assert_eq!(
            parse_image(encoded.as_bytes()).unwrap()[0].mode,
            MODE_LZ_ENTROPY
        );
        assert!(encoded.as_bytes().len() < mixed.len());
        assert_eq!(encoded.decode_lanes().unwrap()[0], mixed);
    }

    #[test]
    fn compressed_png_sample_uses_raw_fast_path_and_roundtrips() {
        use std::io::Write;

        let (width, height) = (512u32, 256u32);
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for y in 0..height {
            for x in 0..width {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let noise = (state & 3) as u8;
                pixels.extend_from_slice(&[
                    ((x / 2) as u8).wrapping_add(noise),
                    ((y / 2) as u8).wrapping_add(noise),
                    ((x ^ y) as u8).wrapping_add(noise),
                    255,
                ]);
            }
        }
        let mut scanlines = Vec::with_capacity(131_200);
        for row in pixels.as_chunks::<1024>().0.iter().take(128) {
            scanlines.push(0);
            scanlines.extend_from_slice(row);
        }
        let mut zlib = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        zlib.write_all(&scanlines).unwrap();
        let zlib = zlib.finish().unwrap();
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&256u32.to_be_bytes());
        ihdr.extend_from_slice(&128u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
        for (kind, data) in [
            (b"IHDR", ihdr.as_slice()),
            (b"IDAT", zlib.as_slice()),
            (b"IEND", &[]),
        ] {
            png.extend_from_slice(&(data.len() as u32).to_be_bytes());
            png.extend_from_slice(kind);
            png.extend_from_slice(data);
            let mut crc = !0u32;
            for &byte in kind.iter().chain(data) {
                crc ^= u32::from(byte);
                for _ in 0..8 {
                    crc = (crc >> 1) ^ (0xedb8_8320 & 0u32.wrapping_sub(crc & 1));
                }
            }
            png.extend_from_slice(&(!crc).to_be_bytes());
        }
        assert_eq!(png.len(), 89_883);
        assert!(should_skip_lz_sample(&png));

        let raw_image = LaneLz::from_lanes(vec![raw_lane_data(&png)], png.len()).unwrap();
        let direct_image = LaneLz::from_raw_single_lane(&png, png.len()).unwrap();
        assert_eq!(direct_image.as_bytes(), raw_image.as_bytes());

        let mut full_workspace = LzWorkspace::default();
        let (token_count, literal_len) = lz_encode(&png, &mut full_workspace, HASH_BITS_WIDE);
        let full_lz_result =
            choose_lane_data(&png, &full_workspace.tokens, token_count, literal_len).unwrap();
        assert_eq!(full_lz_result.mode, MODE_RAW);
        let previous_full_encode = LaneLz::from_lanes(vec![full_lz_result], png.len()).unwrap();
        assert_eq!(direct_image.as_bytes(), previous_full_encode.as_bytes());

        let encoded = LaneLz::encode(&[&png]).unwrap();
        assert_eq!(encoded.as_bytes(), direct_image.as_bytes());
        assert_eq!(
            LaneLz::from_bytes(encoded.as_bytes())
                .unwrap()
                .decode_lanes()
                .unwrap()[0],
            png
        );
    }
}
