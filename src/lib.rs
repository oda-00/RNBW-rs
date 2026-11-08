//! Lane-implied 4-ary trie codec for sorted sets of unsigned integers.
//!
//! Each trie node is a four-lane mask. A set lane implies the corresponding
//! two key bits, so edges carry no key data. Leaves index unary bucket counts
//! and packed low suffixes. The serialized query form is one contiguous byte
//! image and does not require a heap directory to answer membership queries.

use std::error::Error;
use std::fmt;
use std::sync::OnceLock;

mod rans;
pub mod trnmap;

pub use trnmap::{TriWord, TrnMap, TrnMapError};

const MAGIC: &[u8; 4] = b"RNBW";
const VERSION: u16 = 2;
const FIXED_HEADER: usize = 40;
const LEVEL_DESCRIPTOR: usize = 16;
const TRAILER_LEN: usize = 32;
const SELECT_STRIDE: usize = 64;
const RANK_STRIDE: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    UnsortedInput { index: usize },
    ValueOutOfRange { index: usize, width: u8 },
    Truncated,
    InvalidFormat(&'static str),
    LengthOverflow,
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsortedInput { index } => {
                write!(f, "input is not strictly increasing at index {index}")
            }
            Self::ValueOutOfRange { index, width } => {
                write!(f, "value at index {index} does not fit in {width} bits")
            }
            Self::Truncated => f.write_str("truncated RNBW byte stream"),
            Self::InvalidFormat(reason) => write!(f, "invalid RNBW byte stream: {reason}"),
            Self::LengthOverflow => f.write_str("RNBW stream exceeds the 32-bit format limits"),
        }
    }
}

impl Error for CodecError {}

/// Encoded set in the directly queryable lane-trie representation.
#[derive(Debug, Clone)]
pub struct Rnbw {
    image: Vec<u8>,
}

/// Validated zero-copy view of a directly queryable RNBW image.
#[derive(Debug, Clone, Copy)]
pub struct RnbwView<'a> {
    image: &'a [u8],
}

#[derive(Clone, Copy, Debug, Default)]
struct LevelOffsets {
    mask: usize,
    rank: usize,
    rank_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct BucketOffsets {
    bits: usize,
    bit_len: usize,
    bytes: usize,
    select: usize,
    select_count: usize,
    low: usize,
    low_width: usize,
    low_bytes: usize,
}

#[derive(Debug)]
struct ImageOffsets {
    header_len: usize,
    levels: Vec<LevelOffsets>,
    buckets: BucketOffsets,
    total_len: usize,
}

impl Rnbw {
    /// Encodes a strictly increasing, unique list of 32-bit keys.
    pub fn encode(sorted_values: &[u32]) -> Result<Self, CodecError> {
        Self::encode_width(sorted_values, 32)
    }

    /// Encodes keys using an even key width from 2 through 32 bits.
    ///
    /// This is used by the multi-level manager for a group or chunk whose
    /// directory already implies the higher bits.
    pub fn encode_width(sorted_values: &[u32], width: u8) -> Result<Self, CodecError> {
        validate_input(sorted_values, width)?;
        let max_depth = (width / 2) as usize;
        let mut nodes = Vec::with_capacity(max_depth + 1);
        for depth in 0..=max_depth {
            nodes.push(count_prefixes(sorted_values, width, depth));
        }

        let mut best: Option<(usize, usize)> = None;
        for depth in 0..=max_depth {
            let leaves = nodes[depth];
            let offsets = image_offsets(width, depth, sorted_values.len(), leaves, &nodes)?;
            let size = offsets.total_len;
            if best.is_none_or(|(best_size, best_depth)| {
                size < best_size || (size == best_size && depth > best_depth)
            }) {
                best = Some((size, depth));
            }
        }
        let (total_len, depth) = best.ok_or(CodecError::LengthOverflow)?;
        let leaves = nodes[depth];
        let offsets = image_offsets(width, depth, sorted_values.len(), leaves, &nodes)?;
        if offsets.total_len != total_len {
            return Err(CodecError::InvalidFormat("unstable split cost"));
        }
        let mut image = vec![0u8; total_len];
        write_header(
            &mut image,
            width,
            depth,
            sorted_values.len(),
            leaves,
            &nodes,
            &offsets,
        )?;

        for (level, &node_count) in nodes.iter().take(depth).enumerate() {
            let masks = build_level_masks(sorted_values, width, level, node_count);
            let rank = build_rank_directory(&masks, node_count);
            let offsets_for_level = offsets.levels[level];
            for (word_index, word) in masks.iter().enumerate() {
                let at = offsets_for_level.mask + word_index * 8;
                image[at..at + 8].copy_from_slice(&word.to_le_bytes());
            }
            for (sample_index, count) in rank.iter().enumerate() {
                let at = offsets_for_level.rank + sample_index * 4;
                image[at..at + 4].copy_from_slice(&count.to_le_bytes());
            }
        }

        let bucket_values =
            build_bucket_stream(sorted_values, width, depth, leaves, offsets.buckets)?;
        image[bucket_values.0..bucket_values.0 + bucket_values.1.len()]
            .copy_from_slice(&bucket_values.1);
        image[bucket_values.2..bucket_values.2 + bucket_values.3.len()]
            .copy_from_slice(&bucket_values.3);
        let low = build_low_stream(sorted_values, width - (depth as u8) * 2);
        image[offsets.buckets.low..offsets.buckets.low + low.len()].copy_from_slice(&low);

        let view = RnbwView { image: &image };
        validate_image(view.image)?;
        Ok(Self { image })
    }

    /// Sorts, deduplicates, then encodes arbitrary input.
    pub fn encode_unsorted(values: &[u32]) -> Result<Self, CodecError> {
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        Self::encode(&sorted)
    }

    /// Parses and validates a byte image, copying it into owned storage.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CodecError> {
        validate_image(bytes)?;
        Ok(Self {
            image: bytes.to_vec(),
        })
    }

    /// Parses and validates a byte image while taking ownership of its buffer.
    pub fn from_bytes_owned(image: Vec<u8>) -> Result<Self, CodecError> {
        validate_image(&image)?;
        Ok(Self { image })
    }

    /// Number of keys in the set.
    #[inline]
    pub fn len(&self) -> usize {
        read_u64_unchecked(&self.image, 12) as usize
    }

    /// Whether the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialized and directly queryable image.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.image
    }

    /// Serializes the masks, unary bucket vector, and low bits with four-state
    /// order-0 rANS whenever the model and payload together reduce their size.
    pub fn to_compressed_bytes(&self) -> Result<Vec<u8>, CodecError> {
        rans::compress_image(&self.image)
    }

    /// Restores a validated directly queryable image from the entropy form.
    pub fn from_compressed_bytes(bytes: &[u8]) -> Result<Self, CodecError> {
        Ok(Self {
            image: rans::decompress_image(bytes)?,
        })
    }

    /// Query image has no separately allocated in-memory directory.
    #[inline]
    pub fn runtime_index_bytes(&self) -> usize {
        0
    }

    /// Tests membership without decoding or allocating.
    #[inline]
    pub fn contains(&self, value: u32) -> bool {
        self.view().contains(value)
    }

    /// Decodes all keys in increasing order.
    pub fn decode(&self) -> Vec<u32> {
        self.view().decode()
    }

    /// Borrows the validated queryable image.
    #[inline]
    pub fn view(&self) -> RnbwView<'_> {
        RnbwView { image: &self.image }
    }

    /// Number of trie levels selected for the key set.
    #[inline]
    pub fn depth(&self) -> usize {
        self.image[9] as usize
    }

    /// Selected CPU membership kernel; feature detection is cached process-wide.
    #[inline]
    pub fn simd_backend() -> &'static str {
        dispatch().name
    }
}

impl<'a> RnbwView<'a> {
    /// Parses and validates a borrowed queryable image without copying it.
    pub fn from_bytes(image: &'a [u8]) -> Result<Self, CodecError> {
        validate_image(image)?;
        Ok(Self { image })
    }

    #[inline]
    pub fn len(&self) -> usize {
        read_u64_unchecked(self.image, 12) as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn as_bytes(&self) -> &'a [u8] {
        self.image
    }

    #[inline]
    pub fn runtime_index_bytes(&self) -> usize {
        0
    }

    #[inline]
    pub fn contains(&self, value: u32) -> bool {
        contains_image(self.image, value)
    }

    pub fn decode(&self) -> Vec<u32> {
        let mut output = Vec::with_capacity(self.len());
        let depth = self.image[9] as usize;
        let leaves = read_u32_unchecked(self.image, 20) as usize;
        if leaves == 0 {
            return output;
        }
        decode_node(self.image, 0, 0, 0, depth, &mut output);
        output
    }

    #[inline]
    pub fn depth(&self) -> usize {
        self.image[9] as usize
    }
}

fn validate_input(values: &[u32], width: u8) -> Result<(), CodecError> {
    if !(2..=32).contains(&width) || !width.is_multiple_of(2) {
        return Err(CodecError::InvalidFormat(
            "key width must be even and between 2 and 32",
        ));
    }
    if values.len() > u32::MAX as usize {
        return Err(CodecError::LengthOverflow);
    }
    if width < 32 {
        let limit = 1u32 << width;
        if let Some((index, _)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| **value >= limit)
        {
            return Err(CodecError::ValueOutOfRange { index, width });
        }
    }
    for (index, pair) in values.windows(2).enumerate() {
        if pair[0] >= pair[1] {
            return Err(CodecError::UnsortedInput { index: index + 1 });
        }
    }
    Ok(())
}

fn count_prefixes(values: &[u32], width: u8, depth: usize) -> usize {
    if values.is_empty() {
        return 0;
    }
    if depth == 0 {
        return 1;
    }
    let shift = width as usize - depth * 2;
    let mut previous = None;
    let mut count = 0usize;
    for &value in values {
        let prefix = value >> shift;
        if previous != Some(prefix) {
            count += 1;
            previous = Some(prefix);
        }
    }
    count
}

fn image_offsets(
    width: u8,
    depth: usize,
    key_count: usize,
    leaf_count: usize,
    node_counts: &[usize],
) -> Result<ImageOffsets, CodecError> {
    let header_len = FIXED_HEADER
        .checked_add(
            depth
                .checked_mul(LEVEL_DESCRIPTOR)
                .ok_or(CodecError::LengthOverflow)?,
        )
        .and_then(|value| value.checked_add(TRAILER_LEN))
        .ok_or(CodecError::LengthOverflow)?;
    if header_len > u16::MAX as usize {
        return Err(CodecError::LengthOverflow);
    }
    let bucket_bits = key_count
        .checked_add(leaf_count)
        .ok_or(CodecError::LengthOverflow)?;
    if bucket_bits > u32::MAX as usize || leaf_count > u32::MAX as usize {
        return Err(CodecError::LengthOverflow);
    }
    let mut position = header_len;
    let mut levels = Vec::with_capacity(depth);
    for &node_count in node_counts.iter().take(depth) {
        position = align_up(position, 8).ok_or(CodecError::LengthOverflow)?;
        let mask_len = node_count
            .div_ceil(16)
            .checked_mul(8)
            .ok_or(CodecError::LengthOverflow)?;
        let mask = position;
        position = position
            .checked_add(mask_len)
            .ok_or(CodecError::LengthOverflow)?;
        position = align_up(position, 4).ok_or(CodecError::LengthOverflow)?;
        let rank = position;
        let rank_count = node_count.div_ceil(RANK_STRIDE);
        position = position
            .checked_add(
                rank_count
                    .checked_mul(4)
                    .ok_or(CodecError::LengthOverflow)?,
            )
            .ok_or(CodecError::LengthOverflow)?;
        levels.push(LevelOffsets {
            mask,
            rank,
            rank_count,
        });
        position = align_up(position, 8).ok_or(CodecError::LengthOverflow)?;
    }
    position = align_up(position, 8).ok_or(CodecError::LengthOverflow)?;
    let bucket_offset = position;
    let bucket_bytes = bucket_bits.div_ceil(8);
    position = position
        .checked_add(bucket_bytes)
        .ok_or(CodecError::LengthOverflow)?;
    position = align_up(position, 4).ok_or(CodecError::LengthOverflow)?;
    let select_offset = position;
    let select_count = leaf_count.div_ceil(SELECT_STRIDE);
    position = position
        .checked_add(
            select_count
                .checked_mul(4)
                .ok_or(CodecError::LengthOverflow)?,
        )
        .ok_or(CodecError::LengthOverflow)?;
    position = align_up(position, 8).ok_or(CodecError::LengthOverflow)?;
    let low_offset = position;
    let low_width = width as usize - depth * 2;
    let low_bits = key_count
        .checked_mul(low_width)
        .ok_or(CodecError::LengthOverflow)?;
    let low_bytes = low_bits.div_ceil(8);
    position = position
        .checked_add(low_bytes)
        .ok_or(CodecError::LengthOverflow)?;
    if position > u32::MAX as usize {
        return Err(CodecError::LengthOverflow);
    }
    Ok(ImageOffsets {
        header_len,
        levels,
        buckets: BucketOffsets {
            bits: bucket_offset,
            bit_len: bucket_bits,
            bytes: bucket_bytes,
            select: select_offset,
            select_count,
            low: low_offset,
            low_width,
            low_bytes,
        },
        total_len: position,
    })
}

fn write_header(
    image: &mut [u8],
    width: u8,
    depth: usize,
    key_count: usize,
    leaf_count: usize,
    node_counts: &[usize],
    offsets: &ImageOffsets,
) -> Result<(), CodecError> {
    image[0..4].copy_from_slice(MAGIC);
    put_u16(image, 4, VERSION)?;
    put_u16(
        image,
        6,
        u16::try_from(offsets.header_len).map_err(|_| CodecError::LengthOverflow)?,
    )?;
    image[8] = width;
    image[9] = depth as u8;
    image[10] = complete_levels(node_counts, depth) as u8;
    image[11] = 0;
    put_u64(image, 12, key_count as u64)?;
    put_u32(
        image,
        20,
        u32::try_from(leaf_count).map_err(|_| CodecError::LengthOverflow)?,
    )?;
    put_u32(
        image,
        24,
        u32::try_from(offsets.total_len).map_err(|_| CodecError::LengthOverflow)?,
    )?;
    put_u32(image, 28, FIXED_HEADER as u32)?;
    put_u32(
        image,
        32,
        u32::try_from(FIXED_HEADER + depth * LEVEL_DESCRIPTOR)
            .map_err(|_| CodecError::LengthOverflow)?,
    )?;
    put_u32(image, 36, 0)?;
    for level in 0..depth {
        let at = FIXED_HEADER + level * LEVEL_DESCRIPTOR;
        put_u32(
            image,
            at,
            u32::try_from(node_counts[level]).map_err(|_| CodecError::LengthOverflow)?,
        )?;
        put_u32(
            image,
            at + 4,
            u32::try_from(node_counts[level + 1]).map_err(|_| CodecError::LengthOverflow)?,
        )?;
        put_u32(
            image,
            at + 8,
            u32::try_from(offsets.levels[level].mask).map_err(|_| CodecError::LengthOverflow)?,
        )?;
        put_u32(
            image,
            at + 12,
            u32::try_from(offsets.levels[level].rank).map_err(|_| CodecError::LengthOverflow)?,
        )?;
    }
    let bucket = offsets.buckets;
    let at = FIXED_HEADER + depth * LEVEL_DESCRIPTOR;
    put_u32(
        image,
        at,
        u32::try_from(bucket.bit_len).map_err(|_| CodecError::LengthOverflow)?,
    )?;
    put_u32(
        image,
        at + 4,
        u32::try_from(bucket.bits).map_err(|_| CodecError::LengthOverflow)?,
    )?;
    put_u32(
        image,
        at + 8,
        u32::try_from(bucket.bytes).map_err(|_| CodecError::LengthOverflow)?,
    )?;
    put_u32(
        image,
        at + 12,
        u32::try_from(bucket.select).map_err(|_| CodecError::LengthOverflow)?,
    )?;
    put_u32(
        image,
        at + 16,
        u32::try_from(bucket.select_count).map_err(|_| CodecError::LengthOverflow)?,
    )?;
    put_u32(
        image,
        at + 20,
        u32::try_from(bucket.low).map_err(|_| CodecError::LengthOverflow)?,
    )?;
    image[at + 24] = bucket.low_width as u8;
    image[at + 25..at + 28].fill(0);
    put_u32(
        image,
        at + 28,
        u32::try_from(bucket.low_bytes).map_err(|_| CodecError::LengthOverflow)?,
    )?;
    Ok(())
}

fn complete_levels(node_counts: &[usize], depth: usize) -> usize {
    let mut count = 0usize;
    for level in 0..depth {
        if node_counts[level + 1] != node_counts[level].saturating_mul(4) {
            break;
        }
        count += 1;
    }
    count
}

fn build_level_masks(values: &[u32], width: u8, level: usize, nodes: usize) -> Vec<u64> {
    let mut words = vec![0u64; nodes.div_ceil(16)];
    let parent_shift = width as usize - level * 2;
    let digit_shift = parent_shift - 2;
    let mut previous_parent = None;
    let mut node = 0usize;
    for &value in values {
        let parent = if level == 0 { 0 } else { value >> parent_shift };
        if previous_parent != Some(parent) {
            if previous_parent.is_some() {
                node += 1;
            }
            previous_parent = Some(parent);
        }
        let digit = ((value >> digit_shift) & 3) as usize;
        words[node / 16] |= 1u64 << ((node % 16) * 4 + digit);
    }
    words
}

fn build_rank_directory(masks: &[u64], node_count: usize) -> Vec<u32> {
    let mut directory = Vec::with_capacity(node_count.div_ceil(RANK_STRIDE));
    let mut total = 0u32;
    for block in 0..node_count.div_ceil(RANK_STRIDE) {
        directory.push(total);
        let first_word = block * (RANK_STRIDE / 16);
        let end_word = (first_word + RANK_STRIDE / 16).min(masks.len());
        for word in &masks[first_word..end_word] {
            total += word.count_ones();
        }
    }
    directory
}

fn build_bucket_stream(
    values: &[u32],
    width: u8,
    depth: usize,
    leaves: usize,
    offsets: BucketOffsets,
) -> Result<(usize, Vec<u8>, usize, Vec<u8>), CodecError> {
    let mut bits = vec![0u8; offsets.bytes];
    let mut samples = Vec::with_capacity(offsets.select_count);
    let mut value_start = 0usize;
    let mut bit_position = 0usize;
    let mut leaf = 0usize;
    while value_start < values.len() {
        let prefix = prefix_at_depth(values[value_start], width, depth);
        let mut value_end = value_start + 1;
        while value_end < values.len() && prefix_at_depth(values[value_end], width, depth) == prefix
        {
            value_end += 1;
        }
        for position in bit_position..bit_position + (value_end - value_start) {
            bits[position / 8] |= 1 << (position & 7);
        }
        bit_position += value_end - value_start;
        if leaf.is_multiple_of(SELECT_STRIDE) {
            samples.push(u32::try_from(bit_position).map_err(|_| CodecError::LengthOverflow)?);
        }
        bit_position += 1;
        leaf += 1;
        value_start = value_end;
    }
    if leaf != leaves || bit_position != offsets.bit_len {
        return Err(CodecError::InvalidFormat(
            "bucket sizing disagrees with split",
        ));
    }
    let mut sample_bytes = vec![0u8; samples.len() * 4];
    for (index, sample) in samples.into_iter().enumerate() {
        sample_bytes[index * 4..index * 4 + 4].copy_from_slice(&sample.to_le_bytes());
    }
    Ok((offsets.bits, bits, offsets.select, sample_bytes))
}

fn build_low_stream(values: &[u32], low_width: u8) -> Vec<u8> {
    let bit_len = values.len() * low_width as usize;
    let mut low = vec![0u8; bit_len.div_ceil(8)];
    let mask = low_mask(low_width as usize);
    for (index, value) in values.iter().copied().enumerate() {
        write_packed(
            &mut low,
            index * low_width as usize,
            low_width as usize,
            value & mask,
        );
    }
    low
}

#[inline]
fn prefix_at_depth(value: u32, width: u8, depth: usize) -> u32 {
    if depth == 0 {
        0
    } else {
        value >> (width as usize - 2 * depth)
    }
}

#[inline]
fn low_mask(width: usize) -> u32 {
    match width {
        0 => 0,
        1..=31 => (1u32 << width) - 1,
        _ => u32::MAX,
    }
}

fn validate_image(image: &[u8]) -> Result<(), CodecError> {
    if image.len() < FIXED_HEADER {
        return Err(CodecError::Truncated);
    }
    if image.get(0..4) != Some(MAGIC.as_slice()) {
        return Err(CodecError::InvalidFormat("bad magic"));
    }
    if read_u16(image, 4)? != VERSION {
        return Err(CodecError::InvalidFormat("unsupported version"));
    }
    let header_len = read_u16(image, 6)? as usize;
    let width = image[8];
    let depth = image[9] as usize;
    let complete = image[10] as usize;
    if !(2..=32).contains(&width) || !width.is_multiple_of(2) || depth > width as usize / 2 {
        return Err(CodecError::InvalidFormat(
            "invalid key width or split depth",
        ));
    }
    if image[11] != 0 || read_u32(image, 28)? as usize != FIXED_HEADER {
        return Err(CodecError::InvalidFormat(
            "unsupported flags or descriptor offset",
        ));
    }
    if read_u32(image, 36)? != 0 {
        return Err(CodecError::InvalidFormat("nonzero reserved header bits"));
    }
    let expected_header = FIXED_HEADER + depth * LEVEL_DESCRIPTOR + TRAILER_LEN;
    let trailer_at = FIXED_HEADER + depth * LEVEL_DESCRIPTOR;
    if header_len != expected_header || read_u32(image, 32)? as usize != trailer_at {
        return Err(CodecError::InvalidFormat(
            "invalid header or trailer length",
        ));
    }
    let key_count =
        usize::try_from(read_u64(image, 12)?).map_err(|_| CodecError::LengthOverflow)?;
    let leaf_count = read_u32(image, 20)? as usize;
    let total_len = read_u32(image, 24)? as usize;
    if total_len != image.len() {
        return Err(if total_len > image.len() {
            CodecError::Truncated
        } else {
            CodecError::InvalidFormat("trailing bytes")
        });
    }
    if key_count > u32::MAX as usize || leaf_count > key_count && key_count != 0 {
        return Err(CodecError::InvalidFormat("impossible key or leaf count"));
    }
    if (key_count == 0 && (depth != 0 || leaf_count != 0))
        || (key_count > 0 && leaf_count == 0)
        || (key_count > 0 && depth == 0 && leaf_count != 1)
    {
        return Err(CodecError::InvalidFormat(
            "empty set and trie metadata disagree",
        ));
    }
    let mut nodes = Vec::with_capacity(depth + 1);
    if depth > 0 {
        nodes.push(read_u32(image, FIXED_HEADER)? as usize);
        if nodes[0] != 1 {
            return Err(CodecError::InvalidFormat("trie root count is not one"));
        }
        for level in 0..depth {
            let at = FIXED_HEADER + level * LEVEL_DESCRIPTOR;
            let current_nodes = read_u32(image, at)? as usize;
            let children = read_u32(image, at + 4)? as usize;
            if current_nodes != nodes[level] || children > current_nodes.saturating_mul(4) {
                return Err(CodecError::InvalidFormat("invalid level population"));
            }
            nodes.push(children);
        }
        if nodes[depth] != leaf_count {
            return Err(CodecError::InvalidFormat(
                "leaf count disagrees with trie levels",
            ));
        }
    } else if key_count > 0 {
        nodes.push(1);
    } else {
        nodes.push(0);
    }

    let offsets = image_offsets(width, depth, key_count, leaf_count, &nodes)?;
    if offsets.header_len != header_len || offsets.total_len != total_len {
        return Err(CodecError::InvalidFormat(
            "image size disagrees with its counts",
        ));
    }
    let mut actual_complete = 0usize;
    for level in 0..depth {
        let at = FIXED_HEADER + level * LEVEL_DESCRIPTOR;
        let descriptor = offsets.levels[level];
        if read_u32(image, at + 8)? as usize != descriptor.mask
            || read_u32(image, at + 12)? as usize != descriptor.rank
        {
            return Err(CodecError::InvalidFormat("noncanonical level offsets"));
        }
        let node_count = nodes[level];
        let word_count = node_count.div_ceil(16);
        let mut child_total = 0u32;
        let rank_count = descriptor.rank_count;
        for block in 0..rank_count {
            let expected_rank = child_total;
            if read_u32(image, descriptor.rank + block * 4)? != expected_rank {
                return Err(CodecError::InvalidFormat("bad rank directory entry"));
            }
            let first_word = block * (RANK_STRIDE / 16);
            let end_word = (first_word + RANK_STRIDE / 16).min(word_count);
            for word_index in first_word..end_word {
                let word = read_u64(image, descriptor.mask + word_index * 8)?;
                child_total = child_total
                    .checked_add(word.count_ones())
                    .ok_or(CodecError::LengthOverflow)?;
            }
        }
        if child_total as usize != nodes[level + 1] {
            return Err(CodecError::InvalidFormat(
                "lane population disagrees with level",
            ));
        }
        if !node_count.is_multiple_of(16) {
            let valid_nibbles = node_count % 16;
            let last = read_u64(image, descriptor.mask + (word_count - 1) * 8)?;
            let used = valid_nibbles * 4;
            if last >> used != 0 {
                return Err(CodecError::InvalidFormat("nonzero mask padding"));
            }
        }
        if nodes[level + 1] == nodes[level].saturating_mul(4) && actual_complete == level {
            actual_complete += 1;
        }
    }
    if complete != actual_complete {
        return Err(CodecError::InvalidFormat("incorrect complete-level count"));
    }

    let bucket_at = trailer_at;
    let bucket_bits = read_u32(image, bucket_at)? as usize;
    let bucket_offset = read_u32(image, bucket_at + 4)? as usize;
    let bucket_bytes = read_u32(image, bucket_at + 8)? as usize;
    let select_offset = read_u32(image, bucket_at + 12)? as usize;
    let select_count = read_u32(image, bucket_at + 16)? as usize;
    let low_offset = read_u32(image, bucket_at + 20)? as usize;
    let low_width = image[bucket_at + 24] as usize;
    let low_bytes = read_u32(image, bucket_at + 28)? as usize;
    if image[bucket_at + 25..bucket_at + 28] != [0, 0, 0] {
        return Err(CodecError::InvalidFormat("nonzero reserved bucket bits"));
    }
    let bucket = offsets.buckets;
    if bucket_bits != bucket.bit_len
        || bucket_offset != bucket.bits
        || bucket_bytes != bucket.bytes
        || select_offset != bucket.select
        || select_count != bucket.select_count
        || low_offset != bucket.low
        || low_width != bucket.low_width
        || low_bytes != bucket.low_bytes
    {
        return Err(CodecError::InvalidFormat("noncanonical bucket offsets"));
    }
    validate_buckets(image, key_count, leaf_count, bucket)?;
    Ok(())
}

fn validate_buckets(
    image: &[u8],
    key_count: usize,
    leaf_count: usize,
    bucket: BucketOffsets,
) -> Result<(), CodecError> {
    let bucket_slice = &image[bucket.bits..bucket.bits + bucket.bytes];
    let low_slice = &image[bucket.low..bucket.low + bucket.low_bytes];
    if !bucket.bit_len.is_multiple_of(8) && !bucket_slice.is_empty() {
        let valid = bucket.bit_len % 8;
        if bucket_slice[bucket.bytes - 1] >> valid != 0 {
            return Err(CodecError::InvalidFormat("nonzero unary padding"));
        }
    }
    let low_bit_len = key_count
        .checked_mul(bucket.low_width)
        .ok_or(CodecError::LengthOverflow)?;
    if low_bit_len % 8 != 0 && !low_slice.is_empty() {
        let valid = low_bit_len % 8;
        if low_slice[low_slice.len() - 1] >> valid != 0 {
            return Err(CodecError::InvalidFormat("nonzero low-bit padding"));
        }
    }
    let mut ones = 0usize;
    let mut zeros = 0usize;
    let mut bucket_start = 0usize;
    let mut previous_low = None;
    let mut sample_index = 0usize;
    for position in 0..bucket.bit_len {
        let one = (bucket_slice[position / 8] >> (position & 7)) & 1 != 0;
        if one {
            let low = read_packed(low_slice, ones * bucket.low_width, bucket.low_width);
            if previous_low.is_some_and(|previous| previous >= low) {
                return Err(CodecError::InvalidFormat(
                    "low suffixes are not strictly ordered",
                ));
            }
            previous_low = Some(low);
            ones += 1;
        } else {
            if ones == bucket_start {
                return Err(CodecError::InvalidFormat("empty trie leaf bucket"));
            }
            if zeros.is_multiple_of(SELECT_STRIDE) {
                let sample = read_u32(image, bucket.select + sample_index * 4)? as usize;
                if sample != position {
                    return Err(CodecError::InvalidFormat("bad select-zero sample"));
                }
                sample_index += 1;
            }
            zeros += 1;
            bucket_start = ones;
            previous_low = None;
        }
    }
    if ones != key_count || zeros != leaf_count || sample_index != bucket.select_count {
        return Err(CodecError::InvalidFormat(
            "unary bucket counts do not match header",
        ));
    }
    Ok(())
}

#[inline]
fn mask_at(image: &[u8], level: usize, node: usize) -> u8 {
    let descriptor = FIXED_HEADER + level * LEVEL_DESCRIPTOR;
    let mask_offset = read_u32_unchecked(image, descriptor + 8) as usize;
    let word_offset = mask_offset + (node / 16) * 8;
    let word = read_u64_unchecked(image, word_offset);
    ((word >> ((node % 16) * 4)) & 0x0f) as u8
}

#[inline]
fn rank_before(image: &[u8], level: usize, node: usize) -> usize {
    let descriptor = FIXED_HEADER + level * LEVEL_DESCRIPTOR;
    let node_count = read_u32_unchecked(image, descriptor) as usize;
    debug_assert!(node < node_count);
    let mask_offset = read_u32_unchecked(image, descriptor + 8) as usize;
    let rank_offset = read_u32_unchecked(image, descriptor + 12) as usize;
    let block = node / RANK_STRIDE;
    let mut total = read_u32_unchecked(image, rank_offset + block * 4) as usize;
    let first_word = block * (RANK_STRIDE / 16);
    let word_index = node / 16;
    for word in first_word..word_index {
        total += read_u64_unchecked(image, mask_offset + word * 8).count_ones() as usize;
    }
    let in_word = node % 16;
    if in_word != 0 {
        let word = read_u64_unchecked(image, mask_offset + word_index * 8);
        let lower = word & ((1u64 << (in_word * 4)) - 1);
        total += lower.count_ones() as usize;
    }
    total
}

#[inline]
fn prefix_mask(width: usize, depth: usize) -> u32 {
    low_mask(width - depth * 2)
}

fn contains_image(image: &[u8], value: u32) -> bool {
    let key_count = read_u64_unchecked(image, 12) as usize;
    if key_count == 0 {
        return false;
    }
    let width = image[8] as usize;
    if width < 32 && value >= (1u32 << width) {
        return false;
    }
    let depth = image[9] as usize;
    let complete = image[10] as usize;
    let leaves = read_u32_unchecked(image, 20) as usize;
    let mut node = 0usize;
    let mut level = 0usize;
    while level < complete {
        node = prefix_at_depth(value, width as u8, level + 1) as usize;
        level += 1;
    }
    while level < depth {
        let mask = mask_at(image, level, node);
        let digit = ((value >> (width - (level + 1) * 2)) & 3) as u8;
        let bit = 1u8 << digit;
        if mask & bit == 0 {
            return false;
        }
        node = rank_before(image, level, node) + (mask & (bit - 1)).count_ones() as usize;
        level += 1;
    }
    let leaf = node;
    if leaf >= leaves {
        return false;
    }
    let bucket = bucket_offsets_unchecked(image);
    let (start, end) = bucket_bounds(image, bucket, leaf);
    if start == end {
        return false;
    }
    let target = value & prefix_mask(width, depth);
    if bucket.low_width == 0 {
        return true;
    }
    if let Some(found) = scan_small_bucket(image, bucket, start, end, target) {
        return found;
    }
    let mut low = start;
    let mut high = end;
    while low < high {
        let middle = low + (high - low) / 2;
        let candidate = low_at(image, bucket, middle);
        if candidate < target {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low < end && low_at(image, bucket, low) == target
}

fn scan_small_bucket(
    image: &[u8],
    bucket: BucketView,
    start: usize,
    end: usize,
    target: u32,
) -> Option<bool> {
    let len = end - start;
    if !(8..=256).contains(&len) || !bucket.low_width.is_multiple_of(8) {
        return None;
    }
    let byte_width = bucket.low_width / 8;
    if !matches!(byte_width, 1 | 2 | 4) {
        return None;
    }
    let byte_start = start * byte_width;
    let byte_end = end * byte_width;
    let slice = &image[bucket.low + byte_start..bucket.low + byte_end];
    let dispatch = dispatch();
    let scan = match byte_width {
        1 => dispatch.u8,
        2 => dispatch.u16,
        4 => dispatch.u32,
        _ => None,
    }?;
    // The dispatcher only exposes kernels whose required ISA is present.
    Some(unsafe { scan(slice, len, target) })
}

#[derive(Clone, Copy)]
struct BucketView {
    bits: usize,
    bit_len: usize,
    bytes: usize,
    select: usize,
    low: usize,
    low_width: usize,
    low_bytes: usize,
}

fn bucket_offsets_unchecked(image: &[u8]) -> BucketView {
    let trailer = read_u32_unchecked(image, 32) as usize;
    BucketView {
        bit_len: read_u32_unchecked(image, trailer) as usize,
        bits: read_u32_unchecked(image, trailer + 4) as usize,
        bytes: read_u32_unchecked(image, trailer + 8) as usize,
        select: read_u32_unchecked(image, trailer + 12) as usize,
        low: read_u32_unchecked(image, trailer + 20) as usize,
        low_width: image[trailer + 24] as usize,
        low_bytes: read_u32_unchecked(image, trailer + 28) as usize,
    }
}

fn bucket_bounds(image: &[u8], bucket: BucketView, leaf: usize) -> (usize, usize) {
    let end_zero = select_zero(image, bucket, leaf);
    let end = end_zero - leaf;
    let start = if leaf == 0 {
        0
    } else {
        select_zero(image, bucket, leaf - 1) - (leaf - 1)
    };
    (start, end)
}

fn select_zero(image: &[u8], bucket: BucketView, index: usize) -> usize {
    debug_assert!(index < read_u32_unchecked(image, 20) as usize);
    let sample_index = index / SELECT_STRIDE;
    let sample_zero = sample_index * SELECT_STRIDE;
    let sample_position = read_u32_unchecked(image, bucket.select + sample_index * 4) as usize;
    if index == sample_zero {
        return sample_position;
    }
    let mut remaining = index - sample_zero;
    let mut position = sample_position + 1;
    while position < bucket.bit_len {
        let word_start = position & !63;
        let within = position - word_start;
        let word = read_bits_word(&image[bucket.bits..bucket.bits + bucket.bytes], word_start);
        let valid = (64 - within).min(bucket.bit_len - position);
        let valid_mask = if valid == 64 {
            u64::MAX
        } else {
            (1u64 << valid) - 1
        };
        let zeros = !(word >> within) & valid_mask;
        let count = zeros.count_ones() as usize;
        if remaining <= count {
            return position + select_nth_bit(zeros, remaining - 1);
        }
        remaining -= count;
        position = word_start + 64;
    }
    unreachable!("validated unary bucket vector has the requested zero")
}

fn select_nth_bit(mut word: u64, index: usize) -> usize {
    for _ in 0..index {
        word &= word - 1;
    }
    word.trailing_zeros() as usize
}

fn decode_node(
    image: &[u8],
    level: usize,
    node: usize,
    prefix: u32,
    depth: usize,
    output: &mut Vec<u32>,
) {
    if level == depth {
        let bucket = bucket_offsets_unchecked(image);
        let (start, end) = bucket_bounds(image, bucket, node);
        let low_mask = low_mask(bucket.low_width);
        for index in start..end {
            let low = low_at(image, bucket, index) & low_mask;
            output.push((((prefix as u64) << bucket.low_width) as u32) | low);
        }
        return;
    }
    let mask = mask_at(image, level, node);
    let first_child = rank_before(image, level, node);
    for digit in 0..4u8 {
        if mask & (1 << digit) == 0 {
            continue;
        }
        let child = first_child + (mask & ((1 << digit) - 1)).count_ones() as usize;
        decode_node(
            image,
            level + 1,
            child,
            (prefix << 2) | digit as u32,
            depth,
            output,
        );
    }
}

#[inline]
fn low_at(image: &[u8], bucket: BucketView, index: usize) -> u32 {
    read_packed(
        &image[bucket.low..bucket.low + bucket.low_bytes],
        index * bucket.low_width,
        bucket.low_width,
    )
}

fn write_packed(out: &mut [u8], bit_offset: usize, width: usize, value: u32) {
    if width == 0 {
        return;
    }
    let first = bit_offset / 8;
    let shift = bit_offset & 7;
    let mut word = (value as u64) << shift;
    let count = (width + shift).div_ceil(8);
    for byte in 0..count {
        out[first + byte] |= word as u8;
        word >>= 8;
    }
}

fn read_packed(bytes: &[u8], bit_offset: usize, width: usize) -> u32 {
    if width == 0 {
        return 0;
    }
    let first = bit_offset / 8;
    let shift = bit_offset & 7;
    let count = (width + shift).div_ceil(8);
    let mut word = 0u64;
    for byte in 0..count {
        word |= (bytes[first + byte] as u64) << (byte * 8);
    }
    ((word >> shift) as u32) & low_mask(width)
}

#[inline]
fn read_bits_word(bytes: &[u8], bit_offset: usize) -> u64 {
    let byte_offset = bit_offset / 8;
    let mut buffer = [0u8; 8];
    let available = bytes.len().saturating_sub(byte_offset).min(8);
    buffer[..available].copy_from_slice(&bytes[byte_offset..byte_offset + available]);
    u64::from_le_bytes(buffer)
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
}

fn put_u16(bytes: &mut [u8], at: usize, value: u16) -> Result<(), CodecError> {
    let target = bytes
        .get_mut(at..at.checked_add(2).ok_or(CodecError::LengthOverflow)?)
        .ok_or(CodecError::LengthOverflow)?;
    target.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn put_u32(bytes: &mut [u8], at: usize, value: u32) -> Result<(), CodecError> {
    let target = bytes
        .get_mut(at..at.checked_add(4).ok_or(CodecError::LengthOverflow)?)
        .ok_or(CodecError::LengthOverflow)?;
    target.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn put_u64(bytes: &mut [u8], at: usize, value: u64) -> Result<(), CodecError> {
    let target = bytes
        .get_mut(at..at.checked_add(8).ok_or(CodecError::LengthOverflow)?)
        .ok_or(CodecError::LengthOverflow)?;
    target.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn read_u16(bytes: &[u8], at: usize) -> Result<u16, CodecError> {
    let end = at.checked_add(2).ok_or(CodecError::LengthOverflow)?;
    let source = bytes.get(at..end).ok_or(CodecError::Truncated)?;
    Ok(u16::from_le_bytes([source[0], source[1]]))
}

fn read_u32(bytes: &[u8], at: usize) -> Result<u32, CodecError> {
    let end = at.checked_add(4).ok_or(CodecError::LengthOverflow)?;
    let source = bytes.get(at..end).ok_or(CodecError::Truncated)?;
    Ok(u32::from_le_bytes([
        source[0], source[1], source[2], source[3],
    ]))
}

fn read_u64(bytes: &[u8], at: usize) -> Result<u64, CodecError> {
    let end = at.checked_add(8).ok_or(CodecError::LengthOverflow)?;
    let source = bytes.get(at..end).ok_or(CodecError::Truncated)?;
    Ok(u64::from_le_bytes([
        source[0], source[1], source[2], source[3], source[4], source[5], source[6], source[7],
    ]))
}

#[inline]
fn read_u32_unchecked(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

#[inline]
fn read_u64_unchecked(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes([
        bytes[at],
        bytes[at + 1],
        bytes[at + 2],
        bytes[at + 3],
        bytes[at + 4],
        bytes[at + 5],
        bytes[at + 6],
        bytes[at + 7],
    ])
}

type ScanKernel = unsafe fn(&[u8], usize, u32) -> bool;

#[derive(Clone, Copy)]
struct Dispatch {
    name: &'static str,
    u8: Option<ScanKernel>,
    u16: Option<ScanKernel>,
    u32: Option<ScanKernel>,
}

static DISPATCH: OnceLock<Dispatch> = OnceLock::new();

fn dispatch() -> &'static Dispatch {
    DISPATCH.get_or_init(detect_dispatch)
}

fn detect_dispatch() -> Dispatch {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return Dispatch {
                name: "avx512bw",
                u8: Some(x86_avx512::scan_u8),
                u16: Some(x86_avx512::scan_u16),
                u32: Some(x86_avx512::scan_u32),
            };
        }
        if std::is_x86_feature_detected!("avx2") {
            return Dispatch {
                name: "avx2",
                u8: Some(x86_avx2::scan_u8),
                u16: Some(x86_avx2::scan_u16),
                u32: Some(x86_avx2::scan_u32),
            };
        }
        if std::is_x86_feature_detected!("sse2") {
            return Dispatch {
                name: "sse2",
                u8: Some(x86_sse2::scan_u8),
                u16: Some(x86_sse2::scan_u16),
                u32: Some(x86_sse2::scan_u32),
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return Dispatch {
            name: "neon",
            u8: Some(aarch64_neon::scan_u8),
            u16: Some(aarch64_neon::scan_u16),
            u32: Some(aarch64_neon::scan_u32),
        };
    }
    Dispatch {
        name: "scalar",
        u8: None,
        u16: None,
        u32: None,
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_avx512 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn scan_u8(bytes: &[u8], count: usize, target: u32) -> bool {
        let needle = _mm512_set1_epi8(target as i8);
        let mut index = 0usize;
        while index + 64 <= count {
            let block = unsafe { _mm512_loadu_si512(bytes.as_ptr().add(index).cast()) };
            if _mm512_cmpeq_epi8_mask(block, needle) != 0 {
                return true;
            }
            index += 64;
        }
        bytes[index..count].contains(&(target as u8))
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn scan_u16(bytes: &[u8], count: usize, target: u32) -> bool {
        let needle = _mm512_set1_epi16(target as i16);
        let values = bytes.as_ptr().cast::<i16>();
        let mut index = 0usize;
        while index + 32 <= count {
            let block = unsafe { _mm512_loadu_si512(values.add(index).cast()) };
            if _mm512_cmpeq_epi16_mask(block, needle) != 0 {
                return true;
            }
            index += 32;
        }
        let tail = unsafe {
            std::slice::from_raw_parts(values.add(index).cast::<u8>(), (count - index) * 2)
        };
        tail.chunks_exact(2)
            .any(|pair| u16::from_le_bytes([pair[0], pair[1]]) == target as u16)
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn scan_u32(bytes: &[u8], count: usize, target: u32) -> bool {
        let needle = _mm512_set1_epi32(target as i32);
        let values = bytes.as_ptr().cast::<i32>();
        let mut index = 0usize;
        while index + 16 <= count {
            let block = unsafe { _mm512_loadu_si512(values.add(index).cast()) };
            if _mm512_cmpeq_epi32_mask(block, needle) != 0 {
                return true;
            }
            index += 16;
        }
        let tail = unsafe {
            std::slice::from_raw_parts(values.add(index).cast::<u8>(), (count - index) * 4)
        };
        tail.chunks_exact(4)
            .any(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]) == target)
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_avx2 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn scan_u8(bytes: &[u8], count: usize, target: u32) -> bool {
        let needle = _mm256_set1_epi8(target as i8);
        let mut index = 0usize;
        while index + 32 <= count {
            let block = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index).cast()) };
            if _mm256_movemask_epi8(_mm256_cmpeq_epi8(block, needle)) != 0 {
                return true;
            }
            index += 32;
        }
        bytes[index..count].contains(&(target as u8))
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn scan_u16(bytes: &[u8], count: usize, target: u32) -> bool {
        let needle = _mm256_set1_epi16(target as i16);
        let values = bytes.as_ptr().cast::<i16>();
        let mut index = 0usize;
        while index + 16 <= count {
            let block = unsafe { _mm256_loadu_si256(values.add(index).cast()) };
            if _mm256_movemask_epi8(_mm256_cmpeq_epi16(block, needle)) != 0 {
                return true;
            }
            index += 16;
        }
        let tail = unsafe {
            std::slice::from_raw_parts(values.add(index).cast::<u8>(), (count - index) * 2)
        };
        tail.chunks_exact(2)
            .any(|pair| u16::from_le_bytes([pair[0], pair[1]]) == target as u16)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn scan_u32(bytes: &[u8], count: usize, target: u32) -> bool {
        let needle = _mm256_set1_epi32(target as i32);
        let values = bytes.as_ptr().cast::<i32>();
        let mut index = 0usize;
        while index + 8 <= count {
            let block = unsafe { _mm256_loadu_si256(values.add(index).cast()) };
            if _mm256_movemask_epi8(_mm256_cmpeq_epi32(block, needle)) != 0 {
                return true;
            }
            index += 8;
        }
        let tail = unsafe {
            std::slice::from_raw_parts(values.add(index).cast::<u8>(), (count - index) * 4)
        };
        tail.chunks_exact(4)
            .any(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]) == target)
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_sse2 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub unsafe fn scan_u8(bytes: &[u8], count: usize, target: u32) -> bool {
        let needle = _mm_set1_epi8(target as i8);
        let mut index = 0usize;
        while index + 16 <= count {
            let block = unsafe { _mm_loadu_si128(bytes.as_ptr().add(index).cast()) };
            if _mm_movemask_epi8(_mm_cmpeq_epi8(block, needle)) != 0 {
                return true;
            }
            index += 16;
        }
        bytes[index..count].contains(&(target as u8))
    }

    #[target_feature(enable = "sse2")]
    pub unsafe fn scan_u16(bytes: &[u8], count: usize, target: u32) -> bool {
        let needle = _mm_set1_epi16(target as i16);
        let values = bytes.as_ptr().cast::<i16>();
        let mut index = 0usize;
        while index + 8 <= count {
            let block = unsafe { _mm_loadu_si128(values.add(index).cast()) };
            if _mm_movemask_epi8(_mm_cmpeq_epi16(block, needle)) != 0 {
                return true;
            }
            index += 8;
        }
        let tail = unsafe {
            std::slice::from_raw_parts(values.add(index).cast::<u8>(), (count - index) * 2)
        };
        tail.chunks_exact(2)
            .any(|pair| u16::from_le_bytes([pair[0], pair[1]]) == target as u16)
    }

    #[target_feature(enable = "sse2")]
    pub unsafe fn scan_u32(bytes: &[u8], count: usize, target: u32) -> bool {
        let needle = _mm_set1_epi32(target as i32);
        let values = bytes.as_ptr().cast::<i32>();
        let mut index = 0usize;
        while index + 4 <= count {
            let block = unsafe { _mm_loadu_si128(values.add(index).cast()) };
            if _mm_movemask_epi8(_mm_cmpeq_epi32(block, needle)) != 0 {
                return true;
            }
            index += 4;
        }
        let tail = unsafe {
            std::slice::from_raw_parts(values.add(index).cast::<u8>(), (count - index) * 4)
        };
        tail.chunks_exact(4)
            .any(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]) == target)
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod aarch64_neon {
    use std::arch::aarch64::*;

    pub unsafe fn scan_u8(bytes: &[u8], count: usize, target: u32) -> bool {
        let needle = vdupq_n_u8(target as u8);
        let mut index = 0usize;
        while index + 16 <= count {
            let block = unsafe { vld1q_u8(bytes.as_ptr().add(index)) };
            if vmaxvq_u8(vceqq_u8(block, needle)) != 0 {
                return true;
            }
            index += 16;
        }
        bytes[index..count]
            .iter()
            .any(|&value| value == target as u8)
    }

    pub unsafe fn scan_u16(bytes: &[u8], count: usize, target: u32) -> bool {
        let needle = vdupq_n_u16(target as u16);
        let values = bytes.as_ptr().cast::<u16>();
        let mut index = 0usize;
        while index + 8 <= count {
            let block = unsafe { vld1q_u16(values.add(index)) };
            if vmaxvq_u16(vceqq_u16(block, needle)) != 0 {
                return true;
            }
            index += 8;
        }
        let tail = unsafe {
            std::slice::from_raw_parts(values.add(index).cast::<u8>(), (count - index) * 2)
        };
        tail.chunks_exact(2)
            .any(|pair| u16::from_le_bytes([pair[0], pair[1]]) == target as u16)
    }

    pub unsafe fn scan_u32(bytes: &[u8], count: usize, target: u32) -> bool {
        let needle = vdupq_n_u32(target);
        let values = bytes.as_ptr().cast::<u32>();
        let mut index = 0usize;
        while index + 4 <= count {
            let block = unsafe { vld1q_u32(values.add(index)) };
            if vmaxvq_u32(vceqq_u32(block, needle)) != 0 {
                return true;
            }
            index += 4;
        }
        let tail = unsafe {
            std::slice::from_raw_parts(values.add(index).cast::<u8>(), (count - index) * 4)
        };
        tail.chunks_exact(4)
            .any(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]) == target)
    }
}

/// Plain whole-universe Elias–Fano reference used by the benchmark suite.
#[derive(Debug)]
pub struct PlainEf {
    len: usize,
    low_width: usize,
    high_bit_len: usize,
    high: Vec<u8>,
    low: Vec<u8>,
    zero_samples: Vec<u32>,
}

impl PlainEf {
    pub fn encode(sorted_values: &[u32]) -> Result<Self, CodecError> {
        validate_input(sorted_values, 32)?;
        if sorted_values.is_empty() {
            return Ok(Self {
                len: 0,
                low_width: 0,
                high_bit_len: 0,
                high: Vec::new(),
                low: Vec::new(),
                zero_samples: Vec::new(),
            });
        }
        let ratio = (1u64 << 32) / sorted_values.len() as u64;
        let low_width = (u64::BITS - 1 - ratio.leading_zeros()) as usize;
        let bucket_count = 1usize
            .checked_shl((32 - low_width) as u32)
            .ok_or(CodecError::LengthOverflow)?;
        let high_bit_len = bucket_count
            .checked_add(sorted_values.len())
            .ok_or(CodecError::LengthOverflow)?;
        if high_bit_len > u32::MAX as usize {
            return Err(CodecError::LengthOverflow);
        }
        let mut high = vec![0u8; high_bit_len.div_ceil(8)];
        let low_bit_len = sorted_values
            .len()
            .checked_mul(low_width)
            .ok_or(CodecError::LengthOverflow)?;
        let mut low = vec![0u8; low_bit_len.div_ceil(8)];
        for (index, value) in sorted_values.iter().copied().enumerate() {
            let high_value = (value as u64 >> low_width) as usize;
            let bit = high_value + index;
            high[bit / 8] |= 1 << (bit & 7);
            write_packed(
                &mut low,
                index * low_width,
                low_width,
                value & low_mask(low_width),
            );
        }
        let zero_samples = sample_zero_positions(&high, high_bit_len);
        Ok(Self {
            len: sorted_values.len(),
            low_width,
            high_bit_len,
            high,
            low,
            zero_samples,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn encoded_size(&self) -> usize {
        9 + self.high.len() + self.low.len() + self.zero_samples.len() * 4
    }

    pub fn contains(&self, value: u32) -> bool {
        if self.len == 0 {
            return false;
        }
        let high_value = (value as u64 >> self.low_width) as usize;
        let bucket_count = self.high_bit_len - self.len;
        if high_value >= bucket_count {
            return false;
        }
        let end_zero = select_ef_zero(
            &self.high,
            self.high_bit_len,
            &self.zero_samples,
            high_value,
        );
        let start = if high_value == 0 {
            0
        } else {
            select_ef_zero(
                &self.high,
                self.high_bit_len,
                &self.zero_samples,
                high_value - 1,
            ) - (high_value - 1)
        };
        let end = end_zero - high_value;
        let target = value & low_mask(self.low_width);
        let mut lo = start;
        let mut hi = end;
        while lo < hi {
            let middle = lo + (hi - lo) / 2;
            let candidate = read_packed(&self.low, middle * self.low_width, self.low_width);
            if candidate < target {
                lo = middle + 1;
            } else {
                hi = middle;
            }
        }
        lo < end && read_packed(&self.low, lo * self.low_width, self.low_width) == target
    }

    pub fn decode(&self) -> Vec<u32> {
        let mut output = Vec::with_capacity(self.len);
        let mut index = 0usize;
        for (byte_index, byte) in self.high.iter().copied().enumerate() {
            let mut bits = byte;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let position = byte_index * 8 + bit;
                if position < self.high_bit_len {
                    let high_value = position - index;
                    let low = read_packed(&self.low, index * self.low_width, self.low_width);
                    output.push((((high_value as u64) << self.low_width) as u32) | low);
                    index += 1;
                }
                bits &= bits - 1;
            }
        }
        output
    }
}

fn sample_zero_positions(bits: &[u8], bit_len: usize) -> Vec<u32> {
    let count = bit_len.saturating_sub(
        (0..bit_len)
            .filter(|position| (bits[position / 8] >> (position & 7)) & 1 != 0)
            .count(),
    );
    let mut samples = Vec::with_capacity(count.div_ceil(SELECT_STRIDE));
    let mut zero = 0usize;
    for position in 0..bit_len {
        if (bits[position / 8] >> (position & 7)) & 1 == 0 {
            if zero.is_multiple_of(SELECT_STRIDE) {
                samples.push(position as u32);
            }
            zero += 1;
        }
    }
    samples
}

fn select_ef_zero(bits: &[u8], bit_len: usize, samples: &[u32], index: usize) -> usize {
    let sample_index = index / SELECT_STRIDE;
    let sample_zero = sample_index * SELECT_STRIDE;
    let sample_position = samples[sample_index] as usize;
    if index == sample_zero {
        return sample_position;
    }
    let mut remaining = index - sample_zero;
    let mut position = sample_position + 1;
    while position < bit_len {
        if (bits[position / 8] >> (position & 7)) & 1 == 0 {
            remaining -= 1;
            if remaining == 0 {
                return position;
            }
        }
        position += 1;
    }
    unreachable!("validated EF vector has the requested zero")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(values: &[u32]) {
        let encoded = Rnbw::encode(values).unwrap();
        assert_eq!(encoded.len(), values.len());
        assert_eq!(encoded.decode(), values);
        assert_eq!(
            Rnbw::from_bytes(encoded.as_bytes()).unwrap().decode(),
            values
        );
        let view = RnbwView::from_bytes(encoded.as_bytes()).unwrap();
        assert_eq!(view.as_bytes().as_ptr(), encoded.as_bytes().as_ptr());
        assert_eq!(view.decode(), values);
        for probe in values.iter().copied().chain([0, u32::MAX, 1, 0x1234_5678]) {
            assert_eq!(
                encoded.contains(probe),
                values.binary_search(&probe).is_ok(),
                "{probe}"
            );
        }
    }

    #[test]
    fn empty_singleton_and_unsorted_inputs() {
        roundtrip(&[]);
        roundtrip(&[u32::MAX]);
        let encoded = Rnbw::encode_unsorted(&[4, 1, 4, 0, 2]).unwrap();
        assert_eq!(encoded.decode(), [0, 1, 2, 4]);
        assert!(matches!(
            Rnbw::encode(&[1, 1]),
            Err(CodecError::UnsortedInput { index: 1 })
        ));
    }

    #[test]
    fn deterministic_random_and_clustered_roundtrips() {
        let mut seed = 0x4d59_5df4_d0f3_3173u64;
        let mut values = Vec::new();
        for _ in 0..100_000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            if seed & 3 == 0 {
                values.push((seed >> 16) as u32);
            }
        }
        values.sort_unstable();
        values.dedup();
        roundtrip(&values);

        let clustered: Vec<u32> = (0..1024)
            .flat_map(|run| (0..32).map(move |v| run * 65536 + v))
            .collect();
        roundtrip(&clustered);
    }

    #[test]
    fn width_specific_chunks_roundtrip() {
        let values: Vec<u32> = (0..=u16::MAX as u32).step_by(7).collect();
        let encoded = Rnbw::encode_width(&values, 16).unwrap();
        assert_eq!(encoded.decode(), values);
        for value in 0..=u16::MAX as u32 {
            assert_eq!(
                encoded.contains(value),
                values.binary_search(&value).is_ok()
            );
        }
        assert!(matches!(
            Rnbw::encode_width(&[1 << 16], 16),
            Err(CodecError::ValueOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_corrupt_headers_directories_and_streams() {
        let encoded = Rnbw::encode(&[1, 2, 7, 11, 1000, u32::MAX]).unwrap();
        for end in 0..encoded.as_bytes().len() {
            assert!(
                Rnbw::from_bytes(&encoded.as_bytes()[..end]).is_err(),
                "prefix {end}"
            );
        }
        let mut bad = encoded.as_bytes().to_vec();
        bad[0] ^= 1;
        assert!(Rnbw::from_bytes(&bad).is_err());
        let mut bad = encoded.as_bytes().to_vec();
        bad[24] ^= 1;
        assert!(Rnbw::from_bytes(&bad).is_err());
        let mut bad = encoded.as_bytes().to_vec();
        let depth = bad[9] as usize;
        if depth > 0 {
            let rank_offset = read_u32_unchecked(&bad, FIXED_HEADER + 12) as usize;
            bad[rank_offset] ^= 1;
            assert!(Rnbw::from_bytes(&bad).is_err());
        }
    }

    #[test]
    fn plain_ef_reference_is_correct() {
        let values = [0, 1, 17, 65_535, 1 << 20, 0x7fff_ffff, u32::MAX];
        let ef = PlainEf::encode(&values).unwrap();
        assert_eq!(ef.decode(), values);
        for value in [0, 2, 17, 18, 65_535, 65_536, u32::MAX] {
            assert_eq!(ef.contains(value), values.binary_search(&value).is_ok());
        }
    }

    #[test]
    fn membership_kernel_is_selected_once() {
        assert_eq!(Rnbw::simd_backend(), Rnbw::simd_backend());
        assert!(matches!(
            Rnbw::simd_backend(),
            "avx512bw" | "avx2" | "sse2" | "neon" | "scalar"
        ));
    }

    #[test]
    fn selected_simd_kernels_match_scalar_for_hits_misses_and_tails() {
        let selected = dispatch();
        let cases = [
            (1usize, 97usize, selected.u8, 255u32),
            (2usize, 47usize, selected.u16, 65_535u32),
            (4usize, 27usize, selected.u32, u32::MAX),
        ];
        for (width, count, kernel, miss) in cases {
            let Some(kernel) = kernel else {
                continue;
            };
            let mut bytes = Vec::with_capacity(width * count);
            for index in 0..count {
                let value = match width {
                    1 => (index * 3) as u32,
                    2 => (index * 73) as u32,
                    _ => (index * 65_537) as u32,
                };
                match width {
                    1 => bytes.push(value as u8),
                    2 => bytes.extend_from_slice(&(value as u16).to_le_bytes()),
                    _ => bytes.extend_from_slice(&value.to_le_bytes()),
                }
            }
            let decode = |index: usize| -> u32 {
                let at = index * width;
                match width {
                    1 => bytes[at] as u32,
                    2 => u16::from_le_bytes([bytes[at], bytes[at + 1]]) as u32,
                    _ => {
                        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
                    }
                }
            };
            for target in [decode(0), decode(count / 2), decode(count - 1), miss] {
                let scalar = (0..count).any(|index| decode(index) == target);
                assert_eq!(unsafe { kernel(&bytes, count, target) }, scalar);
            }
        }
    }
}
