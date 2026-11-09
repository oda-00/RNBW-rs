//! 64 KiB independently compressed blocks with a max-key directory.
//! This is the LZ4 lookup baseline used by the RFC benchmark.

use lz4_flex::block::{compress_prepend_size, decompress_size_prepended};

const MAGIC: &[u8; 4] = b"LZ64";
const VERSION: u16 = 1;
const HEADER_BYTES: usize = 16;
const ENTRY_BYTES: usize = 12;
pub const BLOCK_BYTES: usize = 64 * 1024;
const KEYS_PER_BLOCK: usize = BLOCK_BYTES / 4;

#[derive(Clone, Copy, Debug)]
struct Block {
    max_key: u32,
    offset: usize,
    len: usize,
}

/// Sorted u32 keys stored in independent 64 KiB LZ4 blocks.
#[derive(Clone, Debug)]
pub struct Lz4Index {
    key_count: usize,
    directory: Vec<Block>,
    payload: Vec<u8>,
}

impl Lz4Index {
    /// Compresses strictly increasing keys into independently searchable blocks.
    pub fn encode(keys: &[u32]) -> Result<Self, &'static str> {
        if keys.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("LZ4 index input must be strictly increasing");
        }
        if keys.len() > u32::MAX as usize {
            return Err("LZ4 index input is too long");
        }

        let mut directory = Vec::with_capacity(keys.len().div_ceil(KEYS_PER_BLOCK));
        let mut payload = Vec::new();
        for block_keys in keys.chunks(KEYS_PER_BLOCK) {
            let offset = payload.len();
            let mut raw = Vec::with_capacity(block_keys.len() * 4);
            for key in block_keys {
                raw.extend_from_slice(&key.to_le_bytes());
            }
            let compressed = compress_prepend_size(&raw);
            directory.push(Block {
                max_key: *block_keys.last().expect("nonempty key block"),
                offset,
                len: compressed.len(),
            });
            payload.extend_from_slice(&compressed);
        }
        Ok(Self {
            key_count: keys.len(),
            directory,
            payload,
        })
    }

    /// Parses a serialized LZ4 index and validates its canonical directory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < HEADER_BYTES {
            return Err("truncated LZ4 index header");
        }
        if &bytes[..4] != MAGIC || read_u16(bytes, 4)? != VERSION || read_u16(bytes, 6)? != 0 {
            return Err("invalid LZ4 index header");
        }
        let key_count = read_u32(bytes, 8)? as usize;
        let block_count = read_u32(bytes, 12)? as usize;
        if block_count != key_count.div_ceil(KEYS_PER_BLOCK) {
            return Err("invalid LZ4 index block count");
        }
        let directory_end = HEADER_BYTES
            .checked_add(
                block_count
                    .checked_mul(ENTRY_BYTES)
                    .ok_or("LZ4 index overflow")?,
            )
            .ok_or("LZ4 index overflow")?;
        if directory_end > bytes.len() {
            return Err("truncated LZ4 index directory");
        }
        let mut directory = Vec::with_capacity(block_count);
        let mut expected_offset = 0usize;
        for index in 0..block_count {
            let at = HEADER_BYTES + index * ENTRY_BYTES;
            let max_key = read_u32(bytes, at)?;
            let offset = read_u32(bytes, at + 4)? as usize;
            let len = read_u32(bytes, at + 8)? as usize;
            if offset != expected_offset || len < 4 {
                return Err("noncanonical LZ4 index directory");
            }
            expected_offset = expected_offset
                .checked_add(len)
                .ok_or("LZ4 index overflow")?;
            directory.push(Block {
                max_key,
                offset,
                len,
            });
        }
        if directory_end.checked_add(expected_offset) != Some(bytes.len()) {
            return Err("LZ4 index payload length mismatch");
        }
        if directory
            .windows(2)
            .any(|pair| pair[0].max_key >= pair[1].max_key)
        {
            return Err("LZ4 index boundaries are not increasing");
        }
        Ok(Self {
            key_count,
            directory,
            payload: bytes[directory_end..].to_vec(),
        })
    }

    /// Number of keys represented.
    #[inline]
    pub fn len(&self) -> usize {
        self.key_count
    }

    /// Whether the index contains no keys.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.key_count == 0
    }

    /// Exact serialized byte count, including the block index.
    #[inline]
    pub fn encoded_size(&self) -> usize {
        HEADER_BYTES + ENTRY_BYTES * self.directory.len() + self.payload.len()
    }

    /// Serializes the header, max-key directory and compressed blocks.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut output = vec![0u8; self.encoded_size()];
        output[..4].copy_from_slice(MAGIC);
        output[4..6].copy_from_slice(&VERSION.to_le_bytes());
        output[8..12].copy_from_slice(&(self.key_count as u32).to_le_bytes());
        output[12..16].copy_from_slice(&(self.directory.len() as u32).to_le_bytes());
        for (index, block) in self.directory.iter().enumerate() {
            let at = HEADER_BYTES + index * ENTRY_BYTES;
            output[at..at + 4].copy_from_slice(&block.max_key.to_le_bytes());
            output[at + 4..at + 8].copy_from_slice(&(block.offset as u32).to_le_bytes());
            output[at + 8..at + 12].copy_from_slice(&(block.len as u32).to_le_bytes());
        }
        let start = HEADER_BYTES + ENTRY_BYTES * self.directory.len();
        output[start..].copy_from_slice(&self.payload);
        output
    }

    /// Checks membership by binary-searching the block directory, inflating one block,
    /// and binary-searching its sorted keys.
    pub fn contains(&self, key: u32) -> Result<bool, &'static str> {
        let block_index = self.directory.partition_point(|block| block.max_key < key);
        let Some(block) = self.directory.get(block_index) else {
            return Ok(false);
        };
        let compressed = &self.payload[block.offset..block.offset + block.len];
        let raw = decompress_size_prepended(compressed).map_err(|_| "invalid LZ4 block")?;
        if !raw.len().is_multiple_of(4) {
            return Err("LZ4 block has a partial key");
        }
        let mut low = 0usize;
        let mut high = raw.len() / 4;
        while low < high {
            let middle = low + (high - low) / 2;
            let value = read_u32(&raw, middle * 4)?;
            match value.cmp(&key) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Equal => return Ok(true),
                std::cmp::Ordering::Greater => high = middle,
            }
        }
        Ok(false)
    }

    /// Decompresses all blocks into the original sorted keys.
    pub fn decode(&self) -> Result<Vec<u32>, &'static str> {
        let mut output = Vec::with_capacity(self.key_count);
        for block in &self.directory {
            let compressed = &self.payload[block.offset..block.offset + block.len];
            let raw = decompress_size_prepended(compressed).map_err(|_| "invalid LZ4 block")?;
            if !raw.len().is_multiple_of(4) {
                return Err("LZ4 block has a partial key");
            }
            for bytes in raw.as_chunks::<4>().0 {
                output.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
            }
        }
        if output.len() != self.key_count || output.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("LZ4 index decoded keys do not match its header");
        }
        Ok(output)
    }
}

fn read_u16(bytes: &[u8], at: usize) -> Result<u16, &'static str> {
    let slice = bytes.get(at..at + 2).ok_or("truncated LZ4 index field")?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], at: usize) -> Result<u32, &'static str> {
    let slice = bytes.get(at..at + 4).ok_or("truncated LZ4 index field")?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}
