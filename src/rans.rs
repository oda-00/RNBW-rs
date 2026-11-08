//! Four-state, 12-bit order-0 rANS wrapper for the raw queryable image.

use super::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

const ENTROPY_VERSION: u16 = 4;
const LEGACY_RANS_VERSION: u16 = 3;
const RANS_HEADER: usize = 32;
const STREAM_DESCRIPTOR: usize = 24;
const RANS_SCALE_BITS: u32 = 12;
const RANS_SCALE: u32 = 1 << RANS_SCALE_BITS;
const RANS_LOWER_BOUND: u32 = 1 << 23;
const RANS_MODEL_BYTES: usize = 256 * 2;
const HUFFMAN_MODEL_BYTES: usize = 256;
const METHOD_RAW: u8 = 0;
const METHOD_RANS: u8 = 1;
const METHOD_HUFFMAN: u8 = 2;

#[derive(Clone, Copy)]
struct RawStream {
    kind: u8,
    offset: usize,
    len: usize,
}

#[derive(Clone)]
struct StoredStream {
    kind: u8,
    raw_len: usize,
    method: u8,
    model: Vec<u8>,
    bytes: Vec<u8>,
}

#[derive(Clone, Copy)]
struct StoredDescriptor {
    kind: u8,
    method: u8,
    offset: usize,
    stored_len: usize,
    raw_len: usize,
    model_offset: usize,
    model_len: usize,
}

struct RawMetadata {
    key_count: usize,
    leaf_count: usize,
    node_counts: Vec<usize>,
    offsets: ImageOffsets,
    streams: Vec<RawStream>,
}

pub(super) fn entropy_encode(input: &[u8]) -> Result<(u8, Vec<u8>, Vec<u8>), CodecError> {
    if input.is_empty() {
        return Ok((METHOD_RAW, Vec::new(), Vec::new()));
    }
    let mut selected_method = METHOD_RAW;
    let mut selected_model = Vec::new();
    let mut selected_bytes = input.to_vec();

    let model = build_model(input);
    let coded = rans_encode(input, &model)?;
    if coded.len() + RANS_MODEL_BYTES < selected_bytes.len() {
        selected_method = METHOD_RANS;
        selected_model.reserve_exact(RANS_MODEL_BYTES);
        for frequency in model {
            selected_model.extend_from_slice(&frequency.to_le_bytes());
        }
        selected_bytes = coded;
    }

    if let Some((lengths, coded)) = huffman_encode(input)
        && coded.len() + HUFFMAN_MODEL_BYTES < selected_model.len() + selected_bytes.len()
    {
        selected_method = METHOD_HUFFMAN;
        selected_model = lengths.to_vec();
        selected_bytes = coded;
    }
    Ok((selected_method, selected_model, selected_bytes))
}

pub(super) fn entropy_decode(
    method: u8,
    output_len: usize,
    model_bytes: &[u8],
    payload: &[u8],
) -> Result<Vec<u8>, CodecError> {
    match method {
        METHOD_RAW => {
            if !model_bytes.is_empty() || payload.len() != output_len {
                return Err(CodecError::InvalidFormat("invalid raw entropy stream"));
            }
            Ok(payload.to_vec())
        }
        METHOD_RANS => {
            if model_bytes.len() != RANS_MODEL_BYTES
                || payload.len() + RANS_MODEL_BYTES >= output_len
            {
                return Err(CodecError::InvalidFormat("invalid rANS stream size"));
            }
            let model = read_model(model_bytes)?;
            rans_decode(payload, output_len, &model)
        }
        METHOD_HUFFMAN => {
            if model_bytes.len() != HUFFMAN_MODEL_BYTES
                || payload.len() + HUFFMAN_MODEL_BYTES >= output_len
            {
                return Err(CodecError::InvalidFormat("invalid Huffman stream size"));
            }
            let lengths: &[u8; HUFFMAN_MODEL_BYTES] = model_bytes
                .try_into()
                .map_err(|_| CodecError::InvalidFormat("bad Huffman code lengths"))?;
            huffman_decode(payload, output_len, lengths)
        }
        _ => Err(CodecError::InvalidFormat("unknown entropy method")),
    }
}

pub(super) fn compress_image(image: &[u8]) -> Result<Vec<u8>, CodecError> {
    validate_image(image)?;
    let depth = image[9] as usize;
    let key_width = image[8];
    let raw_header_len = read_u16(image, 6)? as usize;
    let raw_streams = raw_streams(image, depth)?;
    let mut streams = Vec::with_capacity(raw_streams.len());
    for stream in raw_streams {
        let raw = &image[stream.offset..stream.offset + stream.len];
        if raw.is_empty() {
            streams.push(StoredStream {
                kind: stream.kind,
                raw_len: 0,
                method: METHOD_RAW,
                model: Vec::new(),
                bytes: Vec::new(),
            });
            continue;
        }

        let (selected_method, selected_model, selected_bytes) = entropy_encode(raw)?;

        streams.push(StoredStream {
            kind: stream.kind,
            raw_len: raw.len(),
            method: selected_method,
            model: selected_model,
            bytes: selected_bytes,
        });
    }
    let stream_count = streams.len();
    let prefix_offset = RANS_HEADER
        .checked_add(
            stream_count
                .checked_mul(STREAM_DESCRIPTOR)
                .ok_or(CodecError::LengthOverflow)?,
        )
        .ok_or(CodecError::LengthOverflow)?;
    let model_offset_start = prefix_offset
        .checked_add(raw_header_len)
        .ok_or(CodecError::LengthOverflow)?;
    let model_bytes = streams.iter().try_fold(0usize, |total, stream| {
        total
            .checked_add(stream.model.len())
            .ok_or(CodecError::LengthOverflow)
    })?;
    let header_len = model_offset_start
        .checked_add(model_bytes)
        .ok_or(CodecError::LengthOverflow)?;
    let mut position = align_up(header_len, 8).ok_or(CodecError::LengthOverflow)?;
    let mut model_position = model_offset_start;
    let mut descriptors = Vec::with_capacity(stream_count);
    for stream in &streams {
        let method = stream.method;
        let model_len = stream.model.len();
        let model_offset = if method == METHOD_RAW {
            0
        } else {
            model_position
        };
        if method != METHOD_RAW {
            model_position = model_position
                .checked_add(model_len)
                .ok_or(CodecError::LengthOverflow)?;
        }
        let offset = position;
        position = position
            .checked_add(stream.bytes.len())
            .ok_or(CodecError::LengthOverflow)?;
        descriptors.push(StoredDescriptor {
            kind: stream.kind,
            method,
            offset,
            stored_len: stream.bytes.len(),
            raw_len: stream.raw_len,
            model_offset,
            model_len,
        });
        position = align_up(position, 8).ok_or(CodecError::LengthOverflow)?;
    }
    if model_position != header_len
        || header_len > u16::MAX as usize
        || position > u32::MAX as usize
    {
        return Err(CodecError::LengthOverflow);
    }
    let mut output = vec![0u8; position];
    output[0..4].copy_from_slice(MAGIC);
    put_u16(&mut output, 4, ENTROPY_VERSION)?;
    put_u16(&mut output, 6, header_len as u16)?;
    output[8] = key_width;
    output[9] = depth as u8;
    output[10] = stream_count as u8;
    output[11] = 0;
    put_u32(
        &mut output,
        12,
        u32::try_from(image.len()).map_err(|_| CodecError::LengthOverflow)?,
    )?;
    let encoded_len = u32::try_from(output.len()).map_err(|_| CodecError::LengthOverflow)?;
    put_u32(&mut output, 16, encoded_len)?;
    put_u16(
        &mut output,
        20,
        u16::try_from(raw_header_len).map_err(|_| CodecError::LengthOverflow)?,
    )?;
    put_u16(&mut output, 22, 0)?;
    put_u32(&mut output, 24, RANS_HEADER as u32)?;
    put_u32(
        &mut output,
        28,
        u32::try_from(prefix_offset).map_err(|_| CodecError::LengthOverflow)?,
    )?;
    for (index, (stream, descriptor)) in streams.iter().zip(&descriptors).enumerate() {
        let at = RANS_HEADER + index * STREAM_DESCRIPTOR;
        output[at] = descriptor.kind;
        output[at + 1] = descriptor.method;
        put_u16(&mut output, at + 2, 0)?;
        put_u32(
            &mut output,
            at + 4,
            u32::try_from(descriptor.offset).map_err(|_| CodecError::LengthOverflow)?,
        )?;
        put_u32(
            &mut output,
            at + 8,
            u32::try_from(descriptor.stored_len).map_err(|_| CodecError::LengthOverflow)?,
        )?;
        put_u32(
            &mut output,
            at + 12,
            u32::try_from(descriptor.raw_len).map_err(|_| CodecError::LengthOverflow)?,
        )?;
        put_u32(
            &mut output,
            at + 16,
            u32::try_from(descriptor.model_offset).map_err(|_| CodecError::LengthOverflow)?,
        )?;
        put_u32(
            &mut output,
            at + 20,
            u32::try_from(descriptor.model_len).map_err(|_| CodecError::LengthOverflow)?,
        )?;
        if !stream.model.is_empty() {
            let end = descriptor.model_offset + descriptor.model_len;
            output[descriptor.model_offset..end].copy_from_slice(&stream.model);
        }
        output[descriptor.offset..descriptor.offset + descriptor.stored_len]
            .copy_from_slice(&stream.bytes);
    }
    let prefix_end = prefix_offset + raw_header_len;
    output[prefix_offset..prefix_end].copy_from_slice(&image[..raw_header_len]);
    Ok(output)
}

pub(super) fn decompress_image(input: &[u8]) -> Result<Vec<u8>, CodecError> {
    if input.len() < RANS_HEADER {
        return Err(CodecError::Truncated);
    }
    let version = read_u16(input, 4)?;
    if input.get(0..4) != Some(MAGIC.as_slice())
        || ![LEGACY_RANS_VERSION, ENTROPY_VERSION].contains(&version)
    {
        return Err(CodecError::InvalidFormat(
            "bad entropy-coded RNBW signature",
        ));
    }
    let header_len = read_u16(input, 6)? as usize;
    let width = input[8];
    let depth = input[9] as usize;
    let stream_count = input[10] as usize;
    if input[11] != 0
        || !(2..=32).contains(&width)
        || !width.is_multiple_of(2)
        || depth > width as usize / 2
        || stream_count != depth + 2
    {
        return Err(CodecError::InvalidFormat("invalid entropy-coded metadata"));
    }
    let raw_len = read_u32(input, 12)? as usize;
    if read_u32(input, 16)? as usize != input.len() {
        return Err(if read_u32(input, 16)? as usize > input.len() {
            CodecError::Truncated
        } else {
            CodecError::InvalidFormat("trailing entropy-coded bytes")
        });
    }
    let raw_header_len = read_u16(input, 20)? as usize;
    if read_u16(input, 22)? != 0 || read_u32(input, 24)? as usize != RANS_HEADER {
        return Err(CodecError::InvalidFormat("bad entropy descriptor offset"));
    }
    let prefix_offset = read_u32(input, 28)? as usize;
    let expected_prefix_offset = RANS_HEADER
        .checked_add(
            stream_count
                .checked_mul(STREAM_DESCRIPTOR)
                .ok_or(CodecError::LengthOverflow)?,
        )
        .ok_or(CodecError::LengthOverflow)?;
    let prefix_end = prefix_offset
        .checked_add(raw_header_len)
        .ok_or(CodecError::LengthOverflow)?;
    if prefix_offset != expected_prefix_offset
        || prefix_end > input.len()
        || header_len < prefix_end
        || header_len > input.len()
        || raw_header_len != FIXED_HEADER + depth * LEVEL_DESCRIPTOR + TRAILER_LEN
    {
        return Err(CodecError::InvalidFormat("bad raw-header prefix range"));
    }
    let raw_header = &input[prefix_offset..prefix_end];
    let metadata = parse_raw_metadata(raw_header, raw_len, width, depth)?;
    let mut descriptors = Vec::with_capacity(stream_count);
    let mut model_position = prefix_end;
    let mut position = align_up(header_len, 8).ok_or(CodecError::LengthOverflow)?;
    if input[header_len..position].iter().any(|&byte| byte != 0) {
        return Err(CodecError::InvalidFormat("nonzero entropy-header padding"));
    }
    for (index, stream) in metadata.streams.iter().enumerate() {
        let at = RANS_HEADER + index * STREAM_DESCRIPTOR;
        let kind = input[at];
        let method = input[at + 1];
        let max_method = if version == LEGACY_RANS_VERSION {
            METHOD_RANS
        } else {
            METHOD_HUFFMAN
        };
        if kind != stream.kind || method > max_method || read_u16(input, at + 2)? != 0 {
            return Err(CodecError::InvalidFormat(
                "invalid entropy stream descriptor",
            ));
        }
        let descriptor = StoredDescriptor {
            kind,
            method,
            offset: read_u32(input, at + 4)? as usize,
            stored_len: read_u32(input, at + 8)? as usize,
            raw_len: read_u32(input, at + 12)? as usize,
            model_offset: read_u32(input, at + 16)? as usize,
            model_len: read_u32(input, at + 20)? as usize,
        };
        if descriptor.raw_len != stream.len {
            return Err(CodecError::InvalidFormat(
                "entropy stream raw length mismatch",
            ));
        }
        let expected_model_len = match descriptor.method {
            METHOD_RAW => 0,
            METHOD_RANS => RANS_MODEL_BYTES,
            METHOD_HUFFMAN => HUFFMAN_MODEL_BYTES,
            _ => return Err(CodecError::InvalidFormat("unknown entropy method")),
        };
        if descriptor.method == METHOD_RAW {
            if descriptor.model_offset != 0
                || descriptor.model_len != 0
                || descriptor.stored_len != descriptor.raw_len
            {
                return Err(CodecError::InvalidFormat("invalid raw stream descriptor"));
            }
        } else {
            let minimum_payload = if descriptor.method == METHOD_RANS {
                32
            } else {
                1
            };
            if descriptor.model_offset != model_position
                || descriptor.model_len != expected_model_len
                || descriptor.stored_len < minimum_payload
            {
                return Err(CodecError::InvalidFormat("invalid entropy model range"));
            }
            model_position = model_position
                .checked_add(descriptor.model_len)
                .ok_or(CodecError::LengthOverflow)?;
        }
        if descriptor.offset != position {
            return Err(CodecError::InvalidFormat("noncontiguous rANS payload"));
        }
        let data_end = descriptor
            .offset
            .checked_add(descriptor.stored_len)
            .ok_or(CodecError::LengthOverflow)?;
        if data_end > input.len() {
            return Err(CodecError::Truncated);
        }
        position = align_up(data_end, 8).ok_or(CodecError::LengthOverflow)?;
        if position > input.len() || input[data_end..position].iter().any(|&byte| byte != 0) {
            return Err(CodecError::InvalidFormat("bad entropy stream padding"));
        }
        descriptors.push(descriptor);
    }
    if model_position != header_len || position != input.len() {
        return Err(CodecError::InvalidFormat("trailing entropy payload"));
    }

    let mut raw = Vec::new();
    raw.try_reserve_exact(raw_len)
        .map_err(|_| CodecError::LengthOverflow)?;
    raw.resize(raw_len, 0);
    raw[..raw_header_len].copy_from_slice(raw_header);
    for (stream, descriptor) in metadata.streams.iter().zip(&descriptors) {
        let payload = &input[descriptor.offset..descriptor.offset + descriptor.stored_len];
        let decoded = match descriptor.method {
            METHOD_RAW => payload.to_vec(),
            METHOD_RANS => {
                if descriptor.stored_len + RANS_MODEL_BYTES >= descriptor.raw_len {
                    return Err(CodecError::InvalidFormat("rANS stream does not save space"));
                }
                let model = read_model(
                    &input[descriptor.model_offset..descriptor.model_offset + descriptor.model_len],
                )?;
                rans_decode(payload, descriptor.raw_len, &model)?
            }
            METHOD_HUFFMAN => {
                if descriptor.stored_len + HUFFMAN_MODEL_BYTES >= descriptor.raw_len {
                    return Err(CodecError::InvalidFormat(
                        "Huffman stream does not save space",
                    ));
                }
                let lengths: [u8; HUFFMAN_MODEL_BYTES] = input
                    [descriptor.model_offset..descriptor.model_offset + descriptor.model_len]
                    .try_into()
                    .map_err(|_| CodecError::InvalidFormat("bad Huffman code lengths"))?;
                huffman_decode(payload, descriptor.raw_len, &lengths)?
            }
            _ => return Err(CodecError::InvalidFormat("unknown entropy method")),
        };
        if decoded.len() != stream.len {
            return Err(CodecError::InvalidFormat("decoded stream length mismatch"));
        }
        raw[stream.offset..stream.offset + stream.len].copy_from_slice(&decoded);
    }
    rebuild_auxiliary(&mut raw, &metadata)?;
    validate_image(&raw)?;
    Ok(raw)
}

fn raw_streams(image: &[u8], depth: usize) -> Result<Vec<RawStream>, CodecError> {
    let mut streams = Vec::with_capacity(depth + 2);
    for level in 0..depth {
        let at = FIXED_HEADER + level * LEVEL_DESCRIPTOR;
        let nodes = read_u32(image, at)? as usize;
        streams.push(RawStream {
            kind: level as u8,
            offset: read_u32(image, at + 8)? as usize,
            len: nodes.div_ceil(16) * 8,
        });
    }
    let trailer = read_u32(image, 32)? as usize;
    streams.push(RawStream {
        kind: depth as u8,
        offset: read_u32(image, trailer + 4)? as usize,
        len: read_u32(image, trailer + 8)? as usize,
    });
    streams.push(RawStream {
        kind: (depth + 1) as u8,
        offset: read_u32(image, trailer + 20)? as usize,
        len: read_u32(image, trailer + 28)? as usize,
    });
    Ok(streams)
}

fn parse_raw_metadata(
    header: &[u8],
    raw_len: usize,
    width: u8,
    depth: usize,
) -> Result<RawMetadata, CodecError> {
    if header.len() < FIXED_HEADER + TRAILER_LEN
        || header.get(0..4) != Some(MAGIC.as_slice())
        || read_u16(header, 4)? != VERSION
        || header[8] != width
        || header[9] as usize != depth
        || header[11] != 0
        || read_u16(header, 6)? as usize != header.len()
        || read_u32(header, 24)? as usize != raw_len
        || read_u32(header, 28)? as usize != FIXED_HEADER
        || read_u32(header, 32)? as usize != FIXED_HEADER + depth * LEVEL_DESCRIPTOR
        || read_u32(header, 36)? != 0
    {
        return Err(CodecError::InvalidFormat(
            "bad raw header inside entropy stream",
        ));
    }
    let key_count =
        usize::try_from(read_u64(header, 12)?).map_err(|_| CodecError::LengthOverflow)?;
    let leaf_count = read_u32(header, 20)? as usize;
    let mut node_counts = Vec::with_capacity(depth + 1);
    if depth > 0 {
        node_counts.push(read_u32(header, FIXED_HEADER)? as usize);
        if node_counts[0] != 1 {
            return Err(CodecError::InvalidFormat("bad root count in raw header"));
        }
        for level in 0..depth {
            let at = FIXED_HEADER + level * LEVEL_DESCRIPTOR;
            let nodes = read_u32(header, at)? as usize;
            let children = read_u32(header, at + 4)? as usize;
            if nodes != node_counts[level] || children > nodes.saturating_mul(4) {
                return Err(CodecError::InvalidFormat("bad level count in raw header"));
            }
            node_counts.push(children);
        }
        if node_counts[depth] != leaf_count {
            return Err(CodecError::InvalidFormat("bad leaf count in raw header"));
        }
    } else {
        node_counts.push(usize::from(key_count != 0));
        if leaf_count != node_counts[0] {
            return Err(CodecError::InvalidFormat("bad leaf count in raw header"));
        }
    }
    let offsets = image_offsets(width, depth, key_count, leaf_count, &node_counts)?;
    if offsets.total_len != raw_len || offsets.header_len != header.len() {
        return Err(CodecError::InvalidFormat(
            "raw image size disagrees with prefix",
        ));
    }
    for level in 0..depth {
        let at = FIXED_HEADER + level * LEVEL_DESCRIPTOR;
        if read_u32(header, at + 8)? as usize != offsets.levels[level].mask
            || read_u32(header, at + 12)? as usize != offsets.levels[level].rank
        {
            return Err(CodecError::InvalidFormat("bad raw level offsets in prefix"));
        }
    }
    let trailer = FIXED_HEADER + depth * LEVEL_DESCRIPTOR;
    let bucket = offsets.buckets;
    if read_u32(header, trailer)? as usize != bucket.bit_len
        || read_u32(header, trailer + 4)? as usize != bucket.bits
        || read_u32(header, trailer + 8)? as usize != bucket.bytes
        || read_u32(header, trailer + 12)? as usize != bucket.select
        || read_u32(header, trailer + 16)? as usize != bucket.select_count
        || read_u32(header, trailer + 20)? as usize != bucket.low
        || header[trailer + 24] as usize != bucket.low_width
        || read_u32(header, trailer + 28)? as usize != bucket.low_bytes
        || header[trailer + 25..trailer + 28] != [0, 0, 0]
    {
        return Err(CodecError::InvalidFormat(
            "bad raw bucket offsets in prefix",
        ));
    }
    let streams = raw_streams(header, depth)?;
    Ok(RawMetadata {
        key_count,
        leaf_count,
        node_counts,
        offsets,
        streams,
    })
}

fn rebuild_auxiliary(image: &mut [u8], metadata: &RawMetadata) -> Result<(), CodecError> {
    let depth = image[9] as usize;
    for level in 0..depth {
        let at = FIXED_HEADER + level * LEVEL_DESCRIPTOR;
        let node_count = metadata.node_counts[level];
        let mask_offset = read_u32(image, at + 8)? as usize;
        let rank_offset = read_u32(image, at + 12)? as usize;
        let word_count = node_count.div_ceil(16);
        let rank_count = node_count.div_ceil(RANK_STRIDE);
        let mut total = 0u32;
        for block in 0..rank_count {
            put_u32(image, rank_offset + block * 4, total)?;
            let first = block * (RANK_STRIDE / 16);
            let end = (first + RANK_STRIDE / 16).min(word_count);
            for word in first..end {
                total = total
                    .checked_add(read_u64(image, mask_offset + word * 8)?.count_ones())
                    .ok_or(CodecError::LengthOverflow)?;
            }
        }
    }
    let bucket = metadata.offsets.buckets;
    let mut zero = 0usize;
    for position in 0..bucket.bit_len {
        let unary_byte = image[bucket.bits + position / 8];
        if (unary_byte >> (position & 7)) & 1 == 0 {
            if zero.is_multiple_of(SELECT_STRIDE) {
                put_u32(
                    image,
                    bucket.select + (zero / SELECT_STRIDE) * 4,
                    u32::try_from(position).map_err(|_| CodecError::LengthOverflow)?,
                )?;
            }
            zero += 1;
        }
    }
    if zero != metadata.leaf_count || metadata.key_count > bucket.bit_len {
        return Err(CodecError::InvalidFormat("bad decoded bucket counts"));
    }
    Ok(())
}

fn build_model(input: &[u8]) -> [u16; 256] {
    let mut counts = [0u64; 256];
    for &symbol in input {
        counts[symbol as usize] += 1;
    }
    let active = counts.iter().filter(|&&count| count != 0).count();
    let remaining = RANS_SCALE as u64 - active as u64;
    let length = input.len() as u64;
    let mut model = [0u16; 256];
    let mut remainders = Vec::with_capacity(active);
    let mut assigned = 0u32;
    for symbol in 0..256 {
        let count = counts[symbol];
        if count == 0 {
            continue;
        }
        let scaled = count * remaining;
        let frequency = 1 + scaled / length;
        let remainder = scaled % length;
        model[symbol] = frequency as u16;
        assigned += frequency as u32;
        remainders.push((symbol, remainder));
    }
    remainders.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let excess = RANS_SCALE - assigned;
    for &(symbol, _) in remainders.iter().take(excess as usize) {
        model[symbol] += 1;
    }
    model
}

#[derive(Clone)]
struct HuffmanCodebook {
    codes: [u64; 256],
    counts: [u16; 33],
    first_code: [u64; 33],
    first_symbol: [usize; 33],
    symbols: [u8; 256],
    max_len: usize,
}

#[derive(Clone, Copy)]
struct HuffmanNode {
    left: Option<usize>,
    right: Option<usize>,
    symbol: Option<u8>,
}

fn huffman_encode(input: &[u8]) -> Option<([u8; HUFFMAN_MODEL_BYTES], Vec<u8>)> {
    if input.is_empty() {
        return None;
    }
    let mut frequencies = [0u64; 256];
    for &symbol in input {
        frequencies[symbol as usize] += 1;
    }
    let mut nodes = Vec::with_capacity(511);
    let mut queue = BinaryHeap::new();
    for (symbol, &frequency) in frequencies.iter().enumerate() {
        if frequency != 0 {
            let index = nodes.len();
            nodes.push(HuffmanNode {
                left: None,
                right: None,
                symbol: Some(symbol as u8),
            });
            queue.push(Reverse((frequency, symbol as u16, index)));
        }
    }
    while queue.len() > 1 {
        let Reverse((left_frequency, left_minimum, left)) = queue.pop()?;
        let Reverse((right_frequency, right_minimum, right)) = queue.pop()?;
        let frequency = left_frequency.checked_add(right_frequency)?;
        let index = nodes.len();
        nodes.push(HuffmanNode {
            left: Some(left),
            right: Some(right),
            symbol: None,
        });
        queue.push(Reverse((frequency, left_minimum.min(right_minimum), index)));
    }
    let Reverse((_, _, root)) = queue.pop()?;

    let mut lengths = [0u8; 256];
    let mut stack = vec![(root, 0usize)];
    while let Some((node_index, depth)) = stack.pop() {
        let node = nodes[node_index];
        if let Some(symbol) = node.symbol {
            if depth > 32 {
                return None;
            }
            lengths[symbol as usize] = depth.max(1) as u8;
        } else {
            let child_depth = depth.checked_add(1)?;
            if child_depth > 32 {
                return None;
            }
            stack.push((node.right?, child_depth));
            stack.push((node.left?, child_depth));
        }
    }
    let codebook = huffman_codebook(&lengths).ok()?;

    let mut encoded = Vec::with_capacity(input.len());
    let mut bits = 0u64;
    let mut bit_count = 0usize;
    for &symbol in input {
        let index = symbol as usize;
        let length = lengths[index] as usize;
        bits = (bits << length) | codebook.codes[index];
        bit_count += length;
        while bit_count >= 8 {
            bit_count -= 8;
            encoded.push((bits >> bit_count) as u8);
            bits &= (1u64 << bit_count) - 1;
        }
    }
    if bit_count != 0 {
        encoded.push((bits << (8 - bit_count)) as u8);
    }
    Some((lengths, encoded))
}

fn huffman_codebook(lengths: &[u8; HUFFMAN_MODEL_BYTES]) -> Result<HuffmanCodebook, CodecError> {
    let mut counts = [0u16; 33];
    let mut max_len = 0usize;
    for &length in lengths {
        if length > 32 {
            return Err(CodecError::InvalidFormat("Huffman code exceeds 32 bits"));
        }
        if length != 0 {
            counts[length as usize] = counts[length as usize]
                .checked_add(1)
                .ok_or(CodecError::LengthOverflow)?;
            max_len = max_len.max(length as usize);
        }
    }
    if max_len == 0 {
        return Err(CodecError::InvalidFormat("empty Huffman model"));
    }

    let mut first_code = [0u64; 33];
    let mut next_code = [0u64; 33];
    let mut first_symbol = [0usize; 33];
    let mut symbols = [0u8; 256];
    let mut code = 0u64;
    let mut symbol_offset = 0usize;
    for length in 1..=max_len {
        code = (code + counts[length - 1] as u64) << 1;
        let limit = 1u64 << length;
        if code + counts[length] as u64 > limit {
            return Err(CodecError::InvalidFormat("oversubscribed Huffman model"));
        }
        first_code[length] = code;
        next_code[length] = code;
        first_symbol[length] = symbol_offset;
        for (symbol, &symbol_length) in lengths.iter().enumerate() {
            if symbol_length as usize == length {
                symbols[symbol_offset] = symbol as u8;
                symbol_offset += 1;
            }
        }
    }

    let mut codes = [0u64; 256];
    for (symbol, &length) in lengths.iter().enumerate() {
        if length != 0 {
            let slot = length as usize;
            codes[symbol] = next_code[slot];
            next_code[slot] += 1;
        }
    }
    Ok(HuffmanCodebook {
        codes,
        counts,
        first_code,
        first_symbol,
        symbols,
        max_len,
    })
}

fn huffman_decode(
    payload: &[u8],
    output_len: usize,
    lengths: &[u8; HUFFMAN_MODEL_BYTES],
) -> Result<Vec<u8>, CodecError> {
    let codebook = huffman_codebook(lengths)?;
    let bit_len = payload
        .len()
        .checked_mul(8)
        .ok_or(CodecError::LengthOverflow)?;
    let mut bit_position = 0usize;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| CodecError::LengthOverflow)?;
    for _ in 0..output_len {
        let mut code = 0u64;
        let mut found = None;
        for length in 1..=codebook.max_len {
            if bit_position >= bit_len {
                return Err(CodecError::Truncated);
            }
            let byte = payload[bit_position / 8];
            let bit = (byte >> (7 - (bit_position & 7))) & 1;
            bit_position += 1;
            code = (code << 1) | bit as u64;
            let first = codebook.first_code[length];
            let count = codebook.counts[length] as u64;
            if code >= first && code - first < count {
                let index = codebook.first_symbol[length] + (code - first) as usize;
                found = Some(codebook.symbols[index]);
                break;
            }
        }
        output.push(found.ok_or(CodecError::InvalidFormat("invalid Huffman code"))?);
    }

    let expected_bytes = bit_position.div_ceil(8);
    if expected_bytes != payload.len() {
        return Err(CodecError::InvalidFormat("trailing Huffman payload"));
    }
    let padding = expected_bytes * 8 - bit_position;
    if padding != 0 {
        let mask = (1u8 << padding) - 1;
        if payload.last().is_some_and(|byte| byte & mask != 0) {
            return Err(CodecError::InvalidFormat("nonzero Huffman padding"));
        }
    }
    Ok(output)
}

fn rans_encode(input: &[u8], frequencies: &[u16; 256]) -> Result<Vec<u8>, CodecError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let mut cumulative = [0u32; 256];
    let mut total = 0u32;
    for (symbol, frequency) in frequencies.iter().copied().enumerate() {
        cumulative[symbol] = total;
        total += frequency as u32;
    }
    if total != RANS_SCALE {
        return Err(CodecError::InvalidFormat(
            "rANS frequencies do not sum to 4096",
        ));
    }
    let mut states = [RANS_LOWER_BOUND; 4];
    let mut emitted: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::new());
    for index in (0..input.len()).rev() {
        let symbol = input[index] as usize;
        let lane = index & 3;
        let frequency = frequencies[symbol] as u32;
        if frequency == 0 {
            return Err(CodecError::InvalidFormat(
                "rANS model omits an input symbol",
            ));
        }
        let threshold = ((RANS_LOWER_BOUND >> RANS_SCALE_BITS) << 8) * frequency;
        while states[lane] >= threshold {
            emitted[lane].push(states[lane] as u8);
            states[lane] >>= 8;
        }
        states[lane] = ((states[lane] / frequency) << RANS_SCALE_BITS)
            + (states[lane] % frequency)
            + cumulative[symbol];
    }
    for lane in &mut emitted {
        lane.reverse();
    }
    let payload_len = emitted.iter().try_fold(32usize, |sum, lane| {
        sum.checked_add(lane.len())
            .ok_or(CodecError::LengthOverflow)
    })?;
    let mut output = vec![0u8; payload_len];
    for lane in 0..4 {
        put_u32(&mut output, lane * 4, states[lane])?;
        put_u32(
            &mut output,
            16 + lane * 4,
            u32::try_from(emitted[lane].len()).map_err(|_| CodecError::LengthOverflow)?,
        )?;
    }
    let mut cursor = 32usize;
    for lane in emitted {
        output[cursor..cursor + lane.len()].copy_from_slice(&lane);
        cursor += lane.len();
    }
    Ok(output)
}

fn rans_decode(
    payload: &[u8],
    output_len: usize,
    frequencies: &[u16; 256],
) -> Result<Vec<u8>, CodecError> {
    if payload.len() < 32 {
        return Err(CodecError::Truncated);
    }
    let mut cumulative = [0u32; 256];
    let mut decode_table = [0u8; RANS_SCALE as usize];
    let mut total = 0u32;
    for (symbol, frequency) in frequencies.iter().copied().enumerate() {
        cumulative[symbol] = total;
        for slot in total..total + frequency as u32 {
            if slot >= RANS_SCALE {
                return Err(CodecError::InvalidFormat(
                    "rANS model exceeds frequency scale",
                ));
            }
            decode_table[slot as usize] = symbol as u8;
        }
        total = total
            .checked_add(frequency as u32)
            .ok_or(CodecError::LengthOverflow)?;
    }
    if total != RANS_SCALE || frequencies.iter().all(|&frequency| frequency == 0) {
        return Err(CodecError::InvalidFormat("invalid rANS frequency table"));
    }
    let mut states = [0u32; 4];
    let mut lengths = [0usize; 4];
    let mut starts = [0usize; 4];
    for lane in 0..4 {
        states[lane] = read_u32(payload, lane * 4)?;
        lengths[lane] = read_u32(payload, 16 + lane * 4)? as usize;
        if states[lane] < RANS_LOWER_BOUND {
            return Err(CodecError::InvalidFormat("invalid initial rANS state"));
        }
    }
    let mut next_start = 32usize;
    for lane in 0..4 {
        starts[lane] = next_start;
        next_start = next_start
            .checked_add(lengths[lane])
            .ok_or(CodecError::LengthOverflow)?;
        if next_start > payload.len() {
            return Err(CodecError::Truncated);
        }
    }
    if next_start != payload.len() {
        return Err(CodecError::InvalidFormat(
            "rANS lane lengths do not fill payload",
        ));
    }
    let mut cursors = starts;
    let ends = std::array::from_fn::<_, 4, _>(|lane| starts[lane] + lengths[lane]);
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| CodecError::LengthOverflow)?;
    output.resize(output_len, 0);
    for (index, output_byte) in output.iter_mut().enumerate() {
        let lane = index & 3;
        let state = states[lane];
        let slot = state & (RANS_SCALE - 1);
        let symbol = decode_table[slot as usize] as usize;
        let frequency = frequencies[symbol] as u64;
        let offset = (slot - cumulative[symbol]) as u64;
        let mut next = frequency * (state as u64 >> RANS_SCALE_BITS) + offset;
        if next > u32::MAX as u64 {
            return Err(CodecError::InvalidFormat("rANS state overflow"));
        }
        while next < RANS_LOWER_BOUND as u64 {
            if cursors[lane] >= ends[lane] {
                return Err(CodecError::Truncated);
            }
            next = (next << 8) | payload[cursors[lane]] as u64;
            cursors[lane] += 1;
        }
        states[lane] = next as u32;
        *output_byte = symbol as u8;
    }
    for lane in 0..4 {
        if cursors[lane] != ends[lane] || states[lane] != RANS_LOWER_BOUND {
            return Err(CodecError::InvalidFormat(
                "rANS stream has a bad terminal state",
            ));
        }
    }
    Ok(output)
}

fn read_model(bytes: &[u8]) -> Result<[u16; 256], CodecError> {
    if bytes.len() != RANS_MODEL_BYTES {
        return Err(CodecError::InvalidFormat("bad rANS table length"));
    }
    let mut model = [0u16; 256];
    let mut total = 0u32;
    for (index, frequency) in model.iter_mut().enumerate() {
        *frequency = read_u16(bytes, index * 2)?;
        total = total
            .checked_add(*frequency as u32)
            .ok_or(CodecError::LengthOverflow)?;
    }
    if total != RANS_SCALE {
        return Err(CodecError::InvalidFormat("rANS table sum is not 4096"));
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_state_rans_roundtrips_short_and_all_symbol_streams() {
        let mut seed = 0x9e37_79b9u32;
        for length in [1usize, 2, 3, 4, 7, 31, 32, 33, 255, 4096] {
            let mut input = Vec::with_capacity(length);
            for index in 0..length {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                input.push(if index % 3 == 0 {
                    (seed >> 16) as u8
                } else {
                    (index % 5) as u8
                });
            }
            let model = build_model(&input);
            assert_eq!(
                model.iter().map(|&frequency| frequency as u32).sum::<u32>(),
                RANS_SCALE
            );
            let encoded = rans_encode(&input, &model).unwrap();
            assert_eq!(rans_decode(&encoded, input.len(), &model).unwrap(), input);
        }
    }

    #[test]
    fn canonical_huffman_roundtrips_skewed_and_full_alphabets() {
        let mut skewed = vec![0u8; 20_000];
        skewed.extend((0..256).map(|value| value as u8));
        let mut full = Vec::with_capacity(8192);
        for _ in 0..32 {
            full.extend(0..=u8::MAX);
        }
        for input in [&skewed, &full] {
            let (lengths, encoded) = huffman_encode(input).unwrap();
            assert_eq!(
                huffman_decode(&encoded, input.len(), &lengths).unwrap(),
                *input
            );
        }
        let mut invalid = [0u8; HUFFMAN_MODEL_BYTES];
        invalid[..3].fill(1);
        assert!(huffman_codebook(&invalid).is_err());
    }

    #[test]
    fn entropy_image_roundtrips_and_rejects_corruption() {
        let values: Vec<u32> = (0..=u16::MAX as u32).collect();
        let source = Rnbw::encode_width(&values, 16).unwrap();
        let compressed = source.to_compressed_bytes().unwrap();
        assert!(compressed.len() < source.as_bytes().len());
        let header_len = read_u16(&compressed, 6).unwrap() as usize;
        let prefix_start = read_u32(&compressed, 28).unwrap() as usize;
        let prefix_end = prefix_start + read_u16(&compressed, 20).unwrap() as usize;
        let stream_count = compressed[10] as usize;
        assert_eq!(read_u16(&compressed, 4).unwrap(), ENTROPY_VERSION);
        let mut model_count = 0;
        let mut method_count = [0usize; 3];
        for index in 0..stream_count {
            let descriptor = RANS_HEADER + index * STREAM_DESCRIPTOR;
            let method = compressed[descriptor + 1] as usize;
            method_count[method] += 1;
            if method == METHOD_RANS as usize || method == METHOD_HUFFMAN as usize {
                let model_start = read_u32(&compressed, descriptor + 16).unwrap() as usize;
                let model_len = read_u32(&compressed, descriptor + 20).unwrap() as usize;
                assert_eq!(
                    model_len,
                    if method == METHOD_RANS as usize {
                        RANS_MODEL_BYTES
                    } else {
                        HUFFMAN_MODEL_BYTES
                    }
                );
                assert!(model_start >= prefix_end);
                assert!(model_start + model_len <= header_len);
                model_count += 1;
            }
        }
        assert!(model_count > 0, "test image must exercise a stored model");
        assert!(
            method_count[METHOD_HUFFMAN as usize] > 0,
            "test should select Huffman"
        );
        let decoded = Rnbw::from_compressed_bytes(&compressed).unwrap();
        assert_eq!(decoded.as_bytes(), source.as_bytes());
        assert_eq!(decoded.decode(), values);
        for value in [0, 1, 17, 32_768, u16::MAX as u32] {
            assert!(decoded.contains(value));
        }
        let mut corrupt = compressed.clone();
        corrupt[0] ^= 1;
        assert!(Rnbw::from_compressed_bytes(&corrupt).is_err());
        let mut corrupt = compressed;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(Rnbw::from_compressed_bytes(&corrupt).is_err());
    }

    #[test]
    fn version_three_rans_streams_remain_readable() {
        let source = Rnbw::encode(&[3, 0x1234_5678, u32::MAX]).unwrap();
        let mut compressed = source.to_compressed_bytes().unwrap();
        for index in 0..compressed[10] as usize {
            let descriptor = RANS_HEADER + index * STREAM_DESCRIPTOR;
            assert_ne!(compressed[descriptor + 1], METHOD_HUFFMAN);
        }
        put_u16(&mut compressed, 4, LEGACY_RANS_VERSION).unwrap();
        let decoded = Rnbw::from_compressed_bytes(&compressed).unwrap();
        assert_eq!(decoded.as_bytes(), source.as_bytes());
    }
}
