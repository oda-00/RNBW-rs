//! L2 GenTable + VAR Arena storage with explicitly promoted L3 direct pockets.
//! The 32-bit universe is split into 65,536 16-bit chunks and groups of 16.

use crate::trnmap::{TriWord, TrnMapView};
use crate::{CodecError, Rnbw, RnbwView, TrnMap, TrnMapError, scan_u16_candidates};
use std::fmt;
use std::marker::PhantomData;
use std::num::NonZeroU32;

pub const CHUNK_COUNT: usize = 1 << 16;
pub const GROUP_COUNT: usize = 1 << 12;
const CHUNKS_PER_GROUP: usize = 16;
const GROUP_RESERVE_THRESHOLD: usize = 8 * 1024;
const DENSE_TRNMAP_THRESHOLD_BYTES: usize = 8 * 1024;
const DENSE_MAP_HEADER_BYTES: usize = 8;
const GROUP_HEADER: usize = CHUNKS_PER_GROUP * 3;
const ABSENT: u32 = u32::MAX;
const PAGE_BYTES: usize = 1 << 16;
const MAX_PAGE_BYTES: usize = 1 << 24;
const ARENA_MIN_BLOCK_LOG2: usize = 3;
const ARENA_SIZE_CLASSES: usize =
    MAX_PAGE_BYTES.trailing_zeros() as usize - ARENA_MIN_BLOCK_LOG2 + 1;
const VAR_LEN_MASK: u64 = (1 << 24) - 1;
const VAR_BLOCK_LOG_SHIFT: u32 = 24;
const VAR_BLOCK_LOG_MASK: u64 = 0x1f;
const VAR_LIVE_SHIFT: u32 = 43;
const VAR_LIVE_MASK: u64 = 1 << VAR_LIVE_SHIFT;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GroupRecordTag;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DenseTrnMapTag;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Handle<Tag> {
    idx: u32,
    generation: NonZeroU32,
    _tag: PhantomData<fn() -> Tag>,
}
impl<Tag: Copy> Handle<Tag> {
    fn new(idx: u32, generation: NonZeroU32) -> Self {
        Self {
            idx,
            generation,
            _tag: PhantomData,
        }
    }
    fn raw(self) -> u64 {
        (u64::from(self.generation.get()) << 32) | u64::from(self.idx)
    }
    fn from_raw(raw: u64) -> Result<Self, StoreError> {
        let generation = NonZeroU32::new((raw >> 32) as u32).ok_or(StoreError::InvalidRecord)?;
        Ok(Self::new(raw as u32, generation))
    }
}
type GroupRecordHandle = Handle<GroupRecordTag>;
type DenseTrnMapHandle = Handle<DenseTrnMapTag>;

#[derive(Clone, Copy)]
struct VarBlockHeader {
    len: usize,
    block_log2: usize,
    live: bool,
}
impl VarBlockHeader {
    fn decode(raw: u64) -> Result<Self, StoreError> {
        let block_class = ((raw >> VAR_BLOCK_LOG_SHIFT) & VAR_BLOCK_LOG_MASK) as usize;
        if block_class >= ARENA_SIZE_CLASSES {
            return Err(StoreError::InvalidRecord);
        }
        Ok(Self {
            len: (raw & VAR_LEN_MASK) as usize,
            block_log2: block_class + ARENA_MIN_BLOCK_LOG2,
            live: raw & (1 << VAR_LIVE_SHIFT) != 0,
        })
    }
    fn encode(self) -> u64 {
        // The in-block word stores len:24, block_log2:5, and a live bit. Handle
        // generations live in the typed slot table, matching DFE's arena contract.
        (self.len as u64 & VAR_LEN_MASK)
            | (((self.block_log2 - ARENA_MIN_BLOCK_LOG2) as u64) << VAR_BLOCK_LOG_SHIFT)
            | (u64::from(self.live) << VAR_LIVE_SHIFT)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PocketLayout {
    PerChunk32,
    PerPair64,
}

#[derive(Debug)]
pub enum StoreError {
    Unsorted { index: usize },
    ChunkOutOfRange(usize),
    RangeOutOfRange { start: usize, end: usize },
    TooLarge,
    InvalidRecord,
    Codec(CodecError),
    TrnMap(TrnMapError),
}
impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RNBW store error: {self:?}")
    }
}
impl std::error::Error for StoreError {}
impl From<CodecError> for StoreError {
    fn from(e: CodecError) -> Self {
        Self::Codec(e)
    }
}
impl From<TrnMapError> for StoreError {
    fn from(e: TrnMapError) -> Self {
        Self::TrnMap(e)
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    Empty = 0,
    Full = 1,
    Inline1 = 2,
    Quot = 3,
    Runs = 4,
    TrnMap = 5,
}
impl Form {
    fn from(v: u8) -> Result<Self, StoreError> {
        match v {
            0 => Ok(Self::Empty),
            1 => Ok(Self::Full),
            2 => Ok(Self::Inline1),
            3 => Ok(Self::Quot),
            4 => Ok(Self::Runs),
            5 => Ok(Self::TrnMap),
            _ => Err(StoreError::InvalidRecord),
        }
    }
}

#[derive(Default)]
struct VarPage {
    words: Box<[u64]>,
    bump: usize,
}
impl VarPage {
    fn new(bytes: usize) -> Self {
        Self {
            words: vec![0; bytes.div_ceil(8)].into_boxed_slice(),
            bump: 0,
        }
    }
    fn len(&self) -> usize {
        self.words.len() * 8
    }
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast(), self.len()) }
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        let len = self.len();
        unsafe { std::slice::from_raw_parts_mut(self.words.as_mut_ptr().cast(), len) }
    }
    #[inline]
    fn header_raw(&self, offset: usize) -> Result<u64, StoreError> {
        if offset & 7 != 0 {
            return Err(StoreError::InvalidRecord);
        }
        let raw = *self
            .words
            .get(offset >> 3)
            .ok_or(StoreError::InvalidRecord)?;
        Ok(u64::from_le(raw))
    }
    #[inline]
    fn set_header_raw(&mut self, offset: usize, raw: u64) -> Result<(), StoreError> {
        if offset & 7 != 0 {
            return Err(StoreError::InvalidRecord);
        }
        *self
            .words
            .get_mut(offset >> 3)
            .ok_or(StoreError::InvalidRecord)? = raw.to_le();
        Ok(())
    }
    fn header(&self, offset: usize) -> Result<VarBlockHeader, StoreError> {
        VarBlockHeader::decode(self.header_raw(offset)?)
    }
    fn set_header(&mut self, offset: usize, header: VarBlockHeader) -> Result<(), StoreError> {
        self.set_header_raw(offset, header.encode())
    }
}
struct VarSlot {
    page: u32,
    offset: u32,
    generation: NonZeroU32,
}

#[derive(Clone, Copy)]
struct FreeBlock {
    page: u32,
    offset: u32,
    slot: Option<u32>,
}

#[derive(Default)]
struct VarArena {
    pages: Vec<VarPage>,
    slots: Vec<VarSlot>,
    free: Vec<Vec<FreeBlock>>,
    free_mask: u32,
    live_bytes: usize,
    reserved_bytes: usize,
}
impl VarArena {
    fn alloc<Tag: Copy>(&mut self, record: &[u8]) -> Result<Handle<Tag>, StoreError> {
        self.alloc_space(record.len(), Some(record))
    }

    fn alloc_zeroed<Tag: Copy>(&mut self, len: usize) -> Result<Handle<Tag>, StoreError> {
        self.alloc_space(len, None)
    }

    fn alloc_space<Tag: Copy>(
        &mut self,
        len: usize,
        record: Option<&[u8]>,
    ) -> Result<Handle<Tag>, StoreError> {
        if len as u64 > VAR_LEN_MASK {
            return Err(StoreError::TooLarge);
        }
        let need = (len.checked_add(8).ok_or(StoreError::TooLarge)?)
            .max(8)
            .checked_next_power_of_two()
            .ok_or(StoreError::TooLarge)?;
        if need > MAX_PAGE_BYTES {
            return Err(StoreError::TooLarge);
        }
        if self.free.len() != ARENA_SIZE_CLASSES {
            self.free.resize_with(ARENA_SIZE_CLASSES, Vec::new);
        }
        let min_class = need.trailing_zeros() as usize - ARENA_MIN_BLOCK_LOG2;
        let reusable = {
            let candidates = self.free_mask & (u32::MAX << min_class);
            if candidates == 0 {
                None
            } else {
                let class = candidates.trailing_zeros() as usize;
                let free_block = self.free[class].pop().ok_or(StoreError::InvalidRecord)?;
                if self.free[class].is_empty() {
                    self.free_mask &= !(1u32 << class);
                }
                let header =
                    self.pages[free_block.page as usize].header(free_block.offset as usize)?;
                if header.live || header.block_log2 != class + ARENA_MIN_BLOCK_LOG2 {
                    return Err(StoreError::InvalidRecord);
                }
                if let Some(slot) = free_block.slot {
                    let entry = self
                        .slots
                        .get(slot as usize)
                        .ok_or(StoreError::InvalidRecord)?;
                    if entry.page != free_block.page || entry.offset != free_block.offset {
                        return Err(StoreError::InvalidRecord);
                    }
                }
                Some((free_block, class))
            }
        };
        let (page, offset, block, slot, generation) =
            if let Some((free_block, mut class)) = reusable {
                let page = free_block.page;
                let offset = free_block.offset;
                let mut block = 1usize << (class + ARENA_MIN_BLOCK_LOG2);
                while block > need {
                    block >>= 1;
                    class -= 1;
                    let sibling = offset as usize + block;
                    self.pages[page as usize].set_header(
                        sibling,
                        VarBlockHeader {
                            len: 0,
                            block_log2: class + ARENA_MIN_BLOCK_LOG2,
                            live: false,
                        },
                    )?;
                    self.free[class].push(FreeBlock {
                        page,
                        offset: sibling as u32,
                        slot: None,
                    });
                    self.free_mask |= 1u32 << class;
                }
                let slot = if let Some(slot) = free_block.slot {
                    slot
                } else {
                    let slot = u32::try_from(self.slots.len()).map_err(|_| StoreError::TooLarge)?;
                    let generation = NonZeroU32::new(1).expect("one is non-zero");
                    self.slots.push(VarSlot {
                        page,
                        offset,
                        generation,
                    });
                    slot
                };
                let generation = self.slots[slot as usize].generation;
                (page, offset, need, slot, generation)
            } else {
                let p = if self
                    .pages
                    .last()
                    .is_some_and(|page| page.len() - page.bump >= need)
                {
                    self.pages.len() - 1
                } else {
                    if self.pages.len() >= 1 << 20 {
                        return Err(StoreError::TooLarge);
                    }
                    let cap = PAGE_BYTES.max(need);
                    self.pages.push(VarPage::new(cap));
                    self.reserved_bytes += cap;
                    self.pages.len() - 1
                };
                let off = self.pages[p].bump;
                self.pages[p].bump += need;
                let slot = u32::try_from(self.slots.len()).map_err(|_| StoreError::TooLarge)?;
                let generation = NonZeroU32::new(1).expect("one is non-zero");
                self.slots.push(VarSlot {
                    page: p as u32,
                    offset: off as u32,
                    generation,
                });
                (p as u32, off as u32, need, slot, generation)
            };
        let p = &mut self.pages[page as usize];
        let data = &mut p.bytes_mut()[offset as usize + 8..offset as usize + 8 + len];
        if let Some(record) = record {
            data.copy_from_slice(record);
        } else {
            data.fill(0);
        }
        p.set_header(
            offset as usize,
            VarBlockHeader {
                len,
                block_log2: block.trailing_zeros() as usize,
                live: true,
            },
        )?;
        self.live_bytes += len + 8;
        Ok(Handle::new(slot, generation))
    }
    #[inline]
    fn parts<Tag: Copy>(&self, h: Handle<Tag>) -> Result<(usize, usize, u64), StoreError> {
        let slot = self
            .slots
            .get(h.idx as usize)
            .ok_or(StoreError::InvalidRecord)?;
        if slot.generation != h.generation {
            return Err(StoreError::InvalidRecord);
        }
        let page = slot.page as usize;
        let off = slot.offset as usize;
        let p = self.pages.get(page).ok_or(StoreError::InvalidRecord)?;
        let raw = p.header_raw(off)?;
        if raw & VAR_LIVE_MASK == 0 {
            return Err(StoreError::InvalidRecord);
        }
        let block_class = ((raw >> VAR_BLOCK_LOG_SHIFT) & VAR_BLOCK_LOG_MASK) as usize;
        if block_class >= ARENA_SIZE_CLASSES {
            return Err(StoreError::InvalidRecord);
        }
        let block_bytes = 1usize << (block_class + ARENA_MIN_BLOCK_LOG2);
        if off.checked_add(block_bytes).is_none_or(|end| end > p.len()) {
            return Err(StoreError::InvalidRecord);
        }
        let len = (raw & VAR_LEN_MASK) as usize;
        if len > p.len() - off - 8 {
            return Err(StoreError::InvalidRecord);
        }
        Ok((page, off, raw))
    }
    #[inline]
    fn get<Tag: Copy>(&self, h: Handle<Tag>) -> Result<&[u8], StoreError> {
        let (p, o, raw) = self.parts(h)?;
        let len = (raw & VAR_LEN_MASK) as usize;
        Ok(&self.pages[p].bytes()[o + 8..o + 8 + len])
    }
    fn get_mut<Tag: Copy>(&mut self, h: Handle<Tag>) -> Result<&mut [u8], StoreError> {
        let (page, offset, raw) = self.parts(h)?;
        let len = (raw & VAR_LEN_MASK) as usize;
        Ok(&mut self.pages[page].bytes_mut()[offset + 8..offset + 8 + len])
    }
    fn replace_if_fits<Tag: Copy>(
        &mut self,
        h: Handle<Tag>,
        record: &[u8],
    ) -> Result<Option<Handle<Tag>>, StoreError> {
        if record.len() as u64 > VAR_LEN_MASK {
            return Err(StoreError::TooLarge);
        }
        let (page, offset, raw) = self.parts(h)?;
        let slot = &self.slots[h.idx as usize];
        let Some(next_generation) = slot
            .generation
            .get()
            .checked_add(1)
            .and_then(NonZeroU32::new)
        else {
            return Ok(None);
        };
        let block_class = ((raw >> VAR_BLOCK_LOG_SHIFT) & VAR_BLOCK_LOG_MASK) as usize;
        if block_class >= ARENA_SIZE_CLASSES {
            return Err(StoreError::InvalidRecord);
        }
        let block_log2 = block_class + ARENA_MIN_BLOCK_LOG2;
        let block_bytes = 1usize << block_log2;
        if offset
            .checked_add(block_bytes)
            .is_none_or(|end| end > self.pages[page].len())
        {
            return Err(StoreError::InvalidRecord);
        }
        if record.len() + 8 > block_bytes {
            return Ok(None);
        }
        let old_len = (raw & VAR_LEN_MASK) as usize;
        let page_data = &mut self.pages[page];
        page_data.bytes_mut()[offset + 8..offset + 8 + record.len()].copy_from_slice(record);
        page_data.set_header(
            offset,
            VarBlockHeader {
                len: record.len(),
                block_log2,
                live: true,
            },
        )?;
        self.slots[h.idx as usize].generation = next_generation;
        self.live_bytes = self.live_bytes - (old_len + 8) + (record.len() + 8);
        Ok(Some(Handle::new(h.idx, next_generation)))
    }
    fn retire<Tag: Copy>(&mut self, h: Handle<Tag>) -> Result<(), StoreError> {
        let (page, offset, raw) = self.parts(h)?;
        let block_class = ((raw >> VAR_BLOCK_LOG_SHIFT) & VAR_BLOCK_LOG_MASK) as usize;
        if block_class >= ARENA_SIZE_CLASSES {
            return Err(StoreError::InvalidRecord);
        }
        let block_log2 = block_class + ARENA_MIN_BLOCK_LOG2;
        let block = 1usize << block_log2;
        if offset + block > self.pages[page].len() {
            return Err(StoreError::InvalidRecord);
        }
        let len = (raw & VAR_LEN_MASK) as usize;
        self.pages[page].set_header_raw(offset, raw & !VAR_LIVE_MASK)?;
        let class = block_log2 - ARENA_MIN_BLOCK_LOG2;
        if let Some(next_generation) = self.slots[h.idx as usize]
            .generation
            .get()
            .checked_add(1)
            .and_then(NonZeroU32::new)
        {
            self.slots[h.idx as usize].generation = next_generation;
            self.free[class].push(FreeBlock {
                page: page as u32,
                offset: offset as u32,
                slot: Some(h.idx),
            });
            self.free_mask |= 1u32 << class;
        }
        self.live_bytes -= len + 8;
        Ok(())
    }
}

#[inline]
fn read_handle<Tag: Copy>(record: &[u8], offset: u16) -> Result<Handle<Tag>, StoreError> {
    let offset = offset as usize;
    let raw = u64::from_le_bytes(
        record
            .get(offset..offset + 8)
            .ok_or(StoreError::InvalidRecord)?
            .try_into()
            .unwrap(),
    );
    Handle::from_raw(raw)
}

#[inline]
fn append_handle<Tag: Copy>(record: &mut Vec<u8>, handle: Handle<Tag>) -> Result<u16, StoreError> {
    let offset = u16::try_from(record.len()).map_err(|_| StoreError::TooLarge)?;
    record.extend_from_slice(&handle.raw().to_le_bytes());
    Ok(offset)
}

#[inline]
fn quotient_estimate_bytes(count: usize) -> usize {
    let qbits = (count as f64 * (19.0 - (count as f64).log2())).ceil() as usize;
    qbits.div_ceil(8)
}

#[inline]
fn prefers_dense_trnmap(count: usize, run_count: usize) -> bool {
    quotient_estimate_bytes(count) > DENSE_TRNMAP_THRESHOLD_BYTES
        && 2 + run_count * 4 > DENSE_TRNMAP_THRESHOLD_BYTES
}

fn alloc_dense_trnmap(
    arena: &mut VarArena,
    values: &[u16],
) -> Result<DenseTrnMapHandle, StoreError> {
    let bytes = DENSE_MAP_HEADER_BYTES + TrnMap::bytes_for(CHUNK_COUNT)?;
    let handle = arena.alloc_zeroed::<DenseTrnMapTag>(bytes)?;
    let build = (|| {
        let data = arena.get_mut(handle)?;
        data[..4].copy_from_slice(&(values.len() as u32).to_le_bytes());
        data[4..8].copy_from_slice(&(count_runs(values) as u32).to_le_bytes());
        let mut map = TrnMap::in_buffer(&mut data[DENSE_MAP_HEADER_BYTES..], CHUNK_COUNT)?;
        let mut at = 0;
        for word in 0..CHUNK_COUNT.div_ceil(64) {
            let mut present = 0u64;
            while at < values.len() && usize::from(values[at]) / 64 == word {
                present |= 1u64 << (values[at] & 63);
                at += 1;
            }
            map.put(word, TriWord::from_planes(present, present))?;
        }
        debug_assert_eq!(at, values.len());
        Ok::<(), StoreError>(())
    })();
    if let Err(error) = build {
        arena.retire(handle)?;
        return Err(error);
    }
    Ok(handle)
}

pub struct RnbwStore {
    tiers: TrnMap<'static>,
    gen_table: Box<[u64]>,
    group_handles: Box<[Option<GroupRecordHandle>]>,
    arena: VarArena,
    layout: PocketLayout,
    pockets: Option<Box<[u8]>>,
    len: usize,
}

impl RnbwStore {
    pub fn new(layout: PocketLayout) -> Result<Self, StoreError> {
        Ok(Self {
            tiers: TrnMap::new(CHUNK_COUNT)?,
            gen_table: vec![u64::from(ABSENT); GROUP_COUNT].into_boxed_slice(),
            group_handles: vec![None; GROUP_COUNT].into_boxed_slice(),
            arena: VarArena::default(),
            layout,
            pockets: None,
            len: 0,
        })
    }
    pub fn from_sorted(values: &[u32], layout: PocketLayout) -> Result<Self, StoreError> {
        if let Some(i) = values.windows(2).position(|w| w[0] >= w[1]) {
            return Err(StoreError::Unsorted { index: i + 1 });
        }
        let mut s = Self::new(layout)?;
        s.len = values.len();
        let mut at = 0;
        let mut group_values = Vec::new();
        if let Some(&first) = values.first() {
            let first_group = first >> 20;
            let first_group_len = values.partition_point(|value| (value >> 20) == first_group);
            if first_group_len > GROUP_RESERVE_THRESHOLD {
                group_values.reserve_exact(first_group_len);
            }
        }
        while at < values.len() {
            let gid = (values[at] >> 20) as usize;
            group_values.clear();
            let mut boundaries = [0usize; CHUNKS_PER_GROUP + 1];
            let mut current_chunk = 0;
            while at < values.len() && (values[at] >> 20) as usize == gid {
                let v = values[at];
                let chunk = ((v >> 16) & 15) as usize;
                while current_chunk < chunk {
                    boundaries[current_chunk + 1] = group_values.len();
                    current_chunk += 1;
                }
                group_values.push(v as u16);
                at += 1;
            }
            while current_chunk < CHUNKS_PER_GROUP {
                boundaries[current_chunk + 1] = group_values.len();
                current_chunk += 1;
            }
            let chunks: [&[u16]; CHUNKS_PER_GROUP] =
                std::array::from_fn(|i| &group_values[boundaries[i]..boundaries[i + 1]]);
            s.replace_group_slices(gid, &chunks)?;
            for (i, c) in chunks.iter().enumerate() {
                if !c.is_empty() {
                    s.set_tier((gid << 4) + i, 0)?;
                }
            }
        }
        Ok(s)
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn simd_backend() -> &'static str {
        Rnbw::simd_backend()
    }
    pub fn l2_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.arena.reserved_bytes
            + self.arena.pages.capacity() * std::mem::size_of::<VarPage>()
            + self.arena.slots.capacity() * std::mem::size_of::<VarSlot>()
            + self.arena.free.capacity() * std::mem::size_of::<Vec<FreeBlock>>()
            + self
                .arena
                .free
                .iter()
                .map(|blocks| blocks.capacity() * std::mem::size_of::<FreeBlock>())
                .sum::<usize>()
            + self.gen_table.len() * 8
            + self.group_handles.len() * 8
            + self.tiers.storage_bytes()
    }
    pub fn l3_bytes(&self) -> usize {
        self.pockets.as_ref().map_or(0, |p| p.len())
    }
    fn group_handle(&self, g: usize) -> Option<GroupRecordHandle> {
        let cell = self.gen_table[g];
        if cell as u32 == ABSENT {
            None
        } else {
            self.group_handles[g]
        }
    }
    fn set_tier(&mut self, chunk: usize, tier: i8) -> Result<(), StoreError> {
        if self.tiers.get(chunk)? != tier {
            self.tiers.set(chunk, tier)?;
            self.tiers.mark(chunk)?;
        }
        Ok(())
    }
    fn descriptor(record: &[u8], chunk: usize) -> Result<(Form, u16), StoreError> {
        let p = chunk * 3;
        Ok((
            Form::from(*record.get(p).ok_or(StoreError::InvalidRecord)?)?,
            u16::from_le_bytes(
                record
                    .get(p + 1..p + 3)
                    .ok_or(StoreError::InvalidRecord)?
                    .try_into()
                    .unwrap(),
            ),
        ))
    }
    fn contains_l2(&self, chunk: usize, suffix: u16) -> Result<bool, StoreError> {
        let g = chunk >> 4;
        let Some(h) = self.group_handle(g) else {
            return Ok(false);
        };
        let rec = self.arena.get(h)?;
        let (f, o) = Self::descriptor(rec, chunk & 15)?;
        match f {
            Form::Empty => Ok(false),
            Form::Full => Ok(true),
            Form::Inline1 => Ok(o == suffix),
            Form::Quot => {
                let off = o as usize;
                let n = u16::from_le_bytes(
                    rec.get(off..off + 2)
                        .ok_or(StoreError::InvalidRecord)?
                        .try_into()
                        .unwrap(),
                ) as usize;
                let v = RnbwView::from_validated_bytes(
                    rec.get(off + 2..off + 2 + n)
                        .ok_or(StoreError::InvalidRecord)?,
                );
                Ok(v.contains((((chunk & 15) as u32) << 16) | u32::from(suffix)))
            }
            Form::Runs => {
                let off = o as usize;
                let count = u16::from_le_bytes(
                    rec.get(off..off + 2)
                        .ok_or(StoreError::InvalidRecord)?
                        .try_into()
                        .unwrap(),
                ) as usize;
                let mut lo = 0;
                let mut hi = count;
                while lo < hi {
                    let m = (lo + hi) / 2;
                    let p = off + 2 + m * 4;
                    let a = u16::from_le_bytes(
                        rec.get(p..p + 2)
                            .ok_or(StoreError::InvalidRecord)?
                            .try_into()
                            .unwrap(),
                    );
                    if a <= suffix { lo = m + 1 } else { hi = m }
                }
                if lo == 0 {
                    return Ok(false);
                }
                let p = off + 2 + (lo - 1) * 4;
                let a = u16::from_le_bytes(rec[p..p + 2].try_into().unwrap());
                let n = u16::from_le_bytes(rec[p + 2..p + 4].try_into().unwrap());
                Ok(u32::from(suffix) < u32::from(a) + u32::from(n))
            }
            Form::TrnMap => {
                let handle: DenseTrnMapHandle = read_handle(rec, o)?;
                let data = self.arena.get(handle)?;
                let map = TrnMapView::from_dense_chunk(
                    data.get(DENSE_MAP_HEADER_BYTES..)
                        .ok_or(StoreError::InvalidRecord)?,
                )?;
                Ok(map.get_dense(suffix) == 1)
            }
        }
    }
    pub fn contains(&self, key: u32) -> Result<bool, StoreError> {
        let c = (key >> 16) as usize;
        match self.tiers.get(c)? {
            -1 => Ok(false),
            1 => self.pocket_contains(c, key as u16),
            _ => self.contains_l2(c, key as u16),
        }
    }
    fn decode_group(&self, g: usize) -> Result<[Vec<u16>; 16], StoreError> {
        let mut out: [Vec<u16>; 16] = std::array::from_fn(|_| Vec::new());
        let Some(h) = self.group_handle(g) else {
            return Ok(out);
        };
        let rec = self.arena.get(h)?;
        let mut qdone = false;
        for i in 0..16 {
            let (f, o) = Self::descriptor(rec, i)?;
            match f {
                Form::Empty => {}
                Form::Full => out[i].extend(0..=u16::MAX),
                Form::Inline1 => out[i].push(o),
                Form::Runs => {
                    let p = o as usize;
                    let n = u16::from_le_bytes(
                        rec.get(p..p + 2)
                            .ok_or(StoreError::InvalidRecord)?
                            .try_into()
                            .unwrap(),
                    ) as usize;
                    for j in 0..n {
                        let q = p + 2 + j * 4;
                        let a = u16::from_le_bytes(
                            rec.get(q..q + 2)
                                .ok_or(StoreError::InvalidRecord)?
                                .try_into()
                                .unwrap(),
                        );
                        let l = u16::from_le_bytes(
                            rec.get(q + 2..q + 4)
                                .ok_or(StoreError::InvalidRecord)?
                                .try_into()
                                .unwrap(),
                        );
                        out[i]
                            .extend((u32::from(a)..u32::from(a) + u32::from(l)).map(|x| x as u16));
                    }
                }
                Form::TrnMap => {
                    let handle: DenseTrnMapHandle = read_handle(rec, o)?;
                    let data = self.arena.get(handle)?;
                    let map = TrnMapView::from_dense_chunk(
                        data.get(DENSE_MAP_HEADER_BYTES..)
                            .ok_or(StoreError::InvalidRecord)?,
                    )?;
                    for word in 0..CHUNK_COUNT.div_ceil(64) {
                        let (mut present, _) = map.word_dense(word).planes();
                        while present != 0 {
                            let bit = present.trailing_zeros() as usize;
                            out[i].push((word * 64 + bit) as u16);
                            present &= present - 1;
                        }
                    }
                }
                Form::Quot => {
                    if !qdone {
                        qdone = true;
                        let p = o as usize;
                        let n = u16::from_le_bytes(
                            rec.get(p..p + 2)
                                .ok_or(StoreError::InvalidRecord)?
                                .try_into()
                                .unwrap(),
                        ) as usize;
                        let v = RnbwView::from_validated_bytes(
                            rec.get(p + 2..p + 2 + n).ok_or(StoreError::InvalidRecord)?,
                        );
                        for x in v.decode() {
                            let chunk = ((x >> 16) & 15) as usize;
                            out[chunk].push(x as u16);
                        }
                    }
                }
            }
        }
        Ok(out)
    }
    fn replace_group(&mut self, g: usize, chunks: &[Vec<u16>; 16]) -> Result<(), StoreError> {
        let chunk_slices: [&[u16]; CHUNKS_PER_GROUP] =
            std::array::from_fn(|i| chunks[i].as_slice());
        self.replace_group_slices(g, &chunk_slices)
    }
    fn replace_group_slices(
        &mut self,
        g: usize,
        chunks: &[&[u16]; CHUNKS_PER_GROUP],
    ) -> Result<(), StoreError> {
        let old = self.group_handle(g);
        let mut old_dense_maps: [Option<DenseTrnMapHandle>; CHUNKS_PER_GROUP] =
            [None; CHUNKS_PER_GROUP];
        if let Some(handle) = old {
            let previous = self.arena.get(handle)?;
            for (i, slot) in old_dense_maps.iter_mut().enumerate() {
                let (form, offset) = Self::descriptor(previous, i)?;
                if form == Form::TrnMap {
                    *slot = Some(read_handle(previous, offset)?);
                }
            }
        }
        let generation = (self.gen_table[g] >> 32) as u32;
        let next_generation = generation.checked_add(1).ok_or(StoreError::TooLarge)?;
        let mut rec = vec![0u8; GROUP_HEADER];
        let mut forms = [Form::Empty; 16];
        for i in 0..16 {
            let v = chunks[i];
            if v.is_empty() {
                continue;
            }
            if v.len() == 65536 {
                forms[i] = Form::Full;
                put_desc(&mut rec, i, Form::Full, 0)?;
                continue;
            }
            if v.len() == 1 {
                forms[i] = Form::Inline1;
                put_desc(&mut rec, i, Form::Inline1, v[0])?;
                continue;
            }
            let run_count = count_runs(v);
            let run_bytes = 2 + run_count * 4;
            let qbytes = quotient_estimate_bytes(v.len());
            if prefers_dense_trnmap(v.len(), run_count) {
                forms[i] = Form::TrnMap;
            } else if run_bytes < qbytes {
                forms[i] = Form::Runs
            } else {
                forms[i] = Form::Quot
            }
            if forms[i] == Form::Runs {
                let off = append_runs(&mut rec, v, run_count)?;
                put_desc(&mut rec, i, Form::Runs, off)?;
            }
        }
        let mut keys = Vec::new();
        for i in 0..16 {
            if forms[i] == Form::Quot {
                for &x in chunks[i] {
                    keys.push(((i as u32) << 16) | u32::from(x));
                }
            }
        }
        if !keys.is_empty() {
            let encoded = Rnbw::encode_width(&keys, 20)?;
            let image = encoded.as_bytes();
            let start = u16::try_from(rec.len()).map_err(|_| StoreError::TooLarge)?;
            let n = u16::try_from(image.len()).map_err(|_| StoreError::TooLarge)?;
            rec.extend_from_slice(&n.to_le_bytes());
            rec.extend_from_slice(image);
            for (i, form) in forms.iter().enumerate() {
                if *form == Form::Quot {
                    put_desc(&mut rec, i, Form::Quot, start)?;
                }
            }
        }

        let mut new_dense_maps = Vec::new();
        for i in 0..CHUNKS_PER_GROUP {
            if forms[i] == Form::TrnMap {
                let handle = match alloc_dense_trnmap(&mut self.arena, chunks[i]) {
                    Ok(handle) => handle,
                    Err(error) => {
                        for handle in new_dense_maps {
                            self.arena.retire(handle)?;
                        }
                        return Err(error);
                    }
                };
                let offset = match append_handle(&mut rec, handle) {
                    Ok(offset) => offset,
                    Err(error) => {
                        self.arena.retire(handle)?;
                        for handle in new_dense_maps {
                            self.arena.retire(handle)?;
                        }
                        return Err(error);
                    }
                };
                new_dense_maps.push(handle);
                put_desc(&mut rec, i, Form::TrnMap, offset)?;
            }
        }
        let mut replaced_in_place = false;
        let new = if chunks.iter().all(|chunk| chunk.is_empty()) {
            None
        } else if let Some(old_handle) = old {
            match self.arena.replace_if_fits(old_handle, &rec) {
                Ok(Some(handle)) => {
                    replaced_in_place = true;
                    Some(handle)
                }
                Ok(None) => match self.arena.alloc::<GroupRecordTag>(&rec) {
                    Ok(handle) => Some(handle),
                    Err(error) => {
                        for handle in new_dense_maps {
                            self.arena.retire(handle)?;
                        }
                        return Err(error);
                    }
                },
                Err(error) => {
                    for handle in new_dense_maps {
                        self.arena.retire(handle)?;
                    }
                    return Err(error);
                }
            }
        } else {
            match self.arena.alloc::<GroupRecordTag>(&rec) {
                Ok(handle) => Some(handle),
                Err(error) => {
                    for handle in new_dense_maps {
                        self.arena.retire(handle)?;
                    }
                    return Err(error);
                }
            }
        };
        self.gen_table[g] = (u64::from(next_generation) << 32)
            | u64::from(if new.is_some() { g as u32 } else { ABSENT });
        self.group_handles[g] = new;
        for handle in old_dense_maps.into_iter().flatten() {
            self.arena.retire(handle)?;
        }
        if let Some(handle) = old
            && !replaced_in_place
        {
            self.arena.retire(handle)?;
        }
        Ok(())
    }
    fn mutate_dense_chunk(&mut self, key: u32, insert: bool) -> Result<Option<bool>, StoreError> {
        let chunk = (key >> 16) as usize;
        let group = chunk >> 4;
        let suffix = key as u16;
        let tier = self.tiers.get(chunk)?;
        if tier == 1 || (!insert && tier == -1) {
            return Ok(None);
        }
        let Some(group_handle) = self.group_handle(group) else {
            return Ok(None);
        };
        let record = self.arena.get(group_handle)?;
        let (form, offset) = Self::descriptor(record, chunk & 15)?;
        if form != Form::TrnMap {
            return Ok(None);
        }
        let map_handle: DenseTrnMapHandle = read_handle(record, offset)?;
        let (count, run_count, present, left, right) = {
            let data = self.arena.get(map_handle)?;
            let count = u32::from_le_bytes(
                data.get(..4)
                    .ok_or(StoreError::InvalidRecord)?
                    .try_into()
                    .unwrap(),
            );
            let run_count = u32::from_le_bytes(
                data.get(4..8)
                    .ok_or(StoreError::InvalidRecord)?
                    .try_into()
                    .unwrap(),
            );
            if count == 0 || count > CHUNK_COUNT as u32 || run_count > count {
                return Err(StoreError::InvalidRecord);
            }
            let image = data
                .get(DENSE_MAP_HEADER_BYTES..)
                .ok_or(StoreError::InvalidRecord)?;
            let map = TrnMapView::from_dense_chunk(image)?;
            let present = map.get_dense(suffix) == 1;
            if present == insert {
                return Ok(Some(false));
            }
            let left = suffix != 0 && map.get_dense(suffix - 1) == 1;
            let right = suffix != u16::MAX && map.get_dense(suffix + 1) == 1;
            (count, run_count, present, left, right)
        };
        let next_count = if insert { count + 1 } else { count - 1 };
        let next_runs = if insert {
            run_count + u32::from(!left && !right) - u32::from(left && right)
        } else {
            run_count + u32::from(left && right) - u32::from(!left && !right)
        };
        if !prefers_dense_trnmap(next_count as usize, next_runs as usize) {
            return Ok(None);
        }
        let generation = (self.gen_table[group] >> 32) as u32;
        let next_generation = generation.checked_add(1).ok_or(StoreError::TooLarge)?;
        {
            let data = self.arena.get_mut(map_handle)?;
            let mut map = TrnMap::from_existing_buffer(
                data.get_mut(DENSE_MAP_HEADER_BYTES..)
                    .ok_or(StoreError::InvalidRecord)?,
                CHUNK_COUNT,
            )?;
            map.set(usize::from(suffix), if insert { 1 } else { -1 })?;
            map.mark(usize::from(suffix))?;
        }
        {
            let data = self.arena.get_mut(map_handle)?;
            data[..4].copy_from_slice(&next_count.to_le_bytes());
            data[4..8].copy_from_slice(&next_runs.to_le_bytes());
        }
        self.gen_table[group] = (u64::from(next_generation) << 32) | group as u64;
        self.len = if insert { self.len + 1 } else { self.len - 1 };
        if tier == -1 {
            self.set_tier(chunk, 0)?;
        } else {
            self.tiers.mark(chunk)?;
        }
        Ok(Some(present != insert))
    }
    pub fn insert(&mut self, key: u32) -> Result<bool, StoreError> {
        if let Some(changed) = self.mutate_dense_chunk(key, true)? {
            return Ok(changed);
        }
        let c = (key >> 16) as usize;
        let g = c >> 4;
        let mut chunks = self.decode_group(g)?;
        let values = &mut chunks[c & 15];
        match values.binary_search(&(key as u16)) {
            Ok(_) => return Ok(false),
            Err(i) => values.insert(i, key as u16),
        }
        let became_singleton = values.len() == 1;
        let hot = (0..16)
            .filter(|&i| self.tiers.get((g << 4) + i).unwrap_or(-1) == 1)
            .collect::<Vec<_>>();
        self.replace_group(g, &chunks)?;
        self.len += 1;
        if became_singleton {
            self.set_tier(c, 0)?;
        }
        self.tiers.mark(c)?;
        self.refresh_hot_group(g, &chunks, &hot)?;
        if self.tiers.get(c)? == -1 {
            self.set_tier(c, 0)?;
        }
        Ok(true)
    }
    pub fn remove(&mut self, key: u32) -> Result<bool, StoreError> {
        if let Some(changed) = self.mutate_dense_chunk(key, false)? {
            return Ok(changed);
        }
        let c = (key >> 16) as usize;
        let g = c >> 4;
        let mut chunks = self.decode_group(g)?;
        let values = &mut chunks[c & 15];
        let Ok(i) = values.binary_search(&(key as u16)) else {
            return Ok(false);
        };
        values.remove(i);
        let became_empty = values.is_empty();
        let hot = (0..16)
            .filter(|&i| self.tiers.get((g << 4) + i).unwrap_or(-1) == 1)
            .collect::<Vec<_>>();
        self.replace_group(g, &chunks)?;
        self.len -= 1;
        self.tiers.mark(c)?;
        self.set_tier(c, if became_empty { -1 } else { 0 })?;
        self.refresh_hot_group(g, &chunks, &hot)?;
        Ok(true)
    }
    pub fn decode(&self) -> Result<Vec<u32>, StoreError> {
        let mut out = Vec::with_capacity(self.len);
        for g in 0..GROUP_COUNT {
            for (i, v) in self.decode_group(g)?.iter().enumerate() {
                let base = ((g << 20) | (i << 16)) as u32;
                out.extend(v.iter().map(|&x| base | u32::from(x)));
            }
        }
        Ok(out)
    }
    pub fn promote_range(
        &mut self,
        start: usize,
        end: usize,
    ) -> Result<(usize, usize), StoreError> {
        if start > end || end > CHUNK_COUNT {
            return Err(StoreError::RangeOutOfRange { start, end });
        }
        if start == end {
            return Ok((0, 0));
        }
        let mut promoted = 0;
        let mut spills = 0;
        match self.layout {
            PocketLayout::PerChunk32 => {
                let first_group = start / CHUNKS_PER_GROUP;
                let end_group = end.div_ceil(CHUNKS_PER_GROUP);
                for g in first_group..end_group {
                    let base = g * CHUNKS_PER_GROUP;
                    let first = start.saturating_sub(base).min(CHUNKS_PER_GROUP);
                    let last = end.saturating_sub(base).min(CHUNKS_PER_GROUP);
                    let chunks = self.decode_group(g)?;
                    for (i, values) in chunks.iter().enumerate().take(last).skip(first) {
                        if values.is_empty() {
                            continue;
                        }
                        let c = base + i;
                        if values.len() <= 18 {
                            let off = c * 32;
                            let pockets = self
                                .pockets
                                .get_or_insert_with(|| vec![0; 2 * 1024 * 1024].into_boxed_slice());
                            pack32(values, &mut pockets[off..off + 32]);
                            self.set_tier(c, 1)?;
                            promoted += 1;
                        } else {
                            self.set_tier(c, 0)?;
                            spills += 1;
                        }
                    }
                }
            }
            PocketLayout::PerPair64 => {
                let first = start / 2;
                let last = end.div_ceil(2);
                let pairs_per_group = CHUNKS_PER_GROUP / 2;
                let first_group = first / pairs_per_group;
                let end_group = last.div_ceil(pairs_per_group);
                for g in first_group..end_group {
                    let base = g * CHUNKS_PER_GROUP;
                    let first_pair = first.max(g * pairs_per_group);
                    let end_pair = last.min((g + 1) * pairs_per_group);
                    let chunks = self.decode_group(g)?;
                    for p in first_pair..end_pair {
                        let i = p * 2 - base;
                        let a = &chunks[i];
                        let b = &chunks[i + 1];
                        if a.is_empty() && b.is_empty() {
                            continue;
                        }
                        if a.len() + b.len() <= 36 {
                            let off = p * 64;
                            let pockets = self
                                .pockets
                                .get_or_insert_with(|| vec![0; 2 * 1024 * 1024].into_boxed_slice());
                            pack64(a, b, &mut pockets[off..off + 64]);
                            for (c, values) in [(p * 2, a), (p * 2 + 1, b)] {
                                if c >= start && c < end && !values.is_empty() {
                                    self.set_tier(c, 1)?;
                                    promoted += 1;
                                }
                            }
                        } else {
                            for (c, values) in [(p * 2, a), (p * 2 + 1, b)] {
                                if c >= start && c < end && !values.is_empty() {
                                    self.set_tier(c, 0)?;
                                    spills += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok((promoted, spills))
    }
    pub fn demote_range(&mut self, start: usize, end: usize) -> Result<usize, StoreError> {
        if start > end || end > CHUNK_COUNT {
            return Err(StoreError::RangeOutOfRange { start, end });
        }
        let mut n = 0;
        let mut cursor = start;
        while cursor < end {
            let hot = self.tiers.find(crate::trnmap::CLASS_POSITIVE, cursor);
            if hot >= end {
                break;
            }
            let group = hot / CHUNKS_PER_GROUP;
            let base = group * CHUNKS_PER_GROUP;
            let group_end = (base + CHUNKS_PER_GROUP).min(end);
            let chunks = self.decode_group(group)?;
            for c in hot.max(cursor)..group_end {
                if self.tiers.get(c)? == 1 {
                    let local = c - base;
                    self.set_tier(c, if chunks[local].is_empty() { -1 } else { 0 })?;
                    n += 1;
                }
            }
            cursor = group_end;
        }
        Ok(n)
    }
    fn refresh_hot_group(
        &mut self,
        group: usize,
        chunks: &[Vec<u16>; CHUNKS_PER_GROUP],
        hot: &[usize],
    ) -> Result<(), StoreError> {
        match self.layout {
            PocketLayout::PerChunk32 => {
                for &i in hot {
                    let values = &chunks[i];
                    if values.is_empty() {
                        continue;
                    }
                    let c = group * CHUNKS_PER_GROUP + i;
                    if values.len() <= 18 {
                        let off = c * 32;
                        pack32(
                            values,
                            &mut self
                                .pockets
                                .get_or_insert_with(|| vec![0; 2 * 1024 * 1024].into_boxed_slice())
                                [off..off + 32],
                        );
                        self.set_tier(c, 1)?;
                    } else {
                        self.set_tier(c, 0)?;
                    }
                }
            }
            PocketLayout::PerPair64 => {
                let mut previous_pair = usize::MAX;
                for &i in hot {
                    if chunks[i].is_empty() {
                        continue;
                    }
                    let pair = i / 2;
                    if pair == previous_pair {
                        continue;
                    }
                    previous_pair = pair;
                    let left = pair * 2;
                    let a = &chunks[left];
                    let b = &chunks[left + 1];
                    if a.len() + b.len() <= 36 {
                        let pocket = group * (CHUNKS_PER_GROUP / 2) + pair;
                        let off = pocket * 64;
                        pack64(
                            a,
                            b,
                            &mut self
                                .pockets
                                .get_or_insert_with(|| vec![0; 2 * 1024 * 1024].into_boxed_slice())
                                [off..off + 64],
                        );
                    } else {
                        for x in [
                            group * CHUNKS_PER_GROUP + left,
                            group * CHUNKS_PER_GROUP + left + 1,
                        ] {
                            if self.tiers.get(x)? == 1 {
                                self.set_tier(x, 0)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
    fn pocket_contains(&self, c: usize, x: u16) -> Result<bool, StoreError> {
        let p = self.pockets.as_ref().ok_or(StoreError::InvalidRecord)?;
        match self.layout {
            PocketLayout::PerChunk32 => contains32(&p[c * 32..c * 32 + 32], x),
            PocketLayout::PerPair64 => contains64(&p[(c / 2) * 64..(c / 2 + 1) * 64], c & 1, x),
        }
    }
    pub fn dirty_chunks(&mut self) -> Result<Vec<usize>, StoreError> {
        let mut out = Vec::new();
        let mut w = 0;
        loop {
            let next = self.tiers.next_dirty(w);
            if next < 0 {
                break;
            }
            let i = next as usize;
            let bits = self.tiers.take_dirty(i)?;
            for b in 0..64 {
                if bits & (1 << b) != 0 {
                    let c = i * 64 + b;
                    if c < CHUNK_COUNT {
                        out.push(c)
                    }
                }
            }
            w = i + 1;
        }
        Ok(out)
    }
}

fn put_desc(r: &mut [u8], i: usize, f: Form, o: u16) -> Result<(), StoreError> {
    let p = i * 3;
    r[p] = f as u8;
    r[p + 1..p + 3].copy_from_slice(&o.to_le_bytes());
    Ok(())
}
fn count_runs(v: &[u16]) -> usize {
    if v.is_empty() {
        return 0;
    }
    let mut count = 1;
    for i in 1..v.len() {
        count += usize::from(u32::from(v[i]) != u32::from(v[i - 1]) + 1);
    }
    count
}
fn append_runs(r: &mut Vec<u8>, values: &[u16], run_count: usize) -> Result<u16, StoreError> {
    let off = u16::try_from(r.len()).map_err(|_| StoreError::TooLarge)?;
    r.extend_from_slice(&(run_count as u16).to_le_bytes());
    let mut start = 0;
    while start < values.len() {
        let first = values[start];
        let mut end = start + 1;
        while end < values.len() && u32::from(values[end]) == u32::from(values[end - 1]) + 1 {
            end += 1;
        }
        r.extend_from_slice(&first.to_le_bytes());
        r.extend_from_slice(&((end - start) as u16).to_le_bytes());
        start = end;
    }
    Ok(off)
}

fn write_bits(b: &mut [u8], mut at: usize, n: usize, value: u64) {
    for i in 0..n {
        if value & (1 << i) != 0 {
            b[at / 8] |= 1 << (at & 7)
        }
        at += 1
    }
}
fn read_bits(b: &[u8], at: usize, n: usize) -> u64 {
    let mut v = 0;
    for i in 0..n {
        if b[(at + i) / 8] & (1 << ((at + i) & 7)) != 0 {
            v |= 1 << i
        }
    }
    v
}
#[inline]
fn read_bit_window(b: &[u8], at: usize, n: usize) -> u64 {
    debug_assert!((1..64).contains(&n));
    let byte = at / 8;
    let shift = at & 7;
    let word = u64::from_le_bytes(b[byte..byte + 8].try_into().unwrap());
    (word >> shift) & ((1u64 << n) - 1)
}
#[inline]
fn read_u12(b: &[u8], at: usize) -> u16 {
    let byte = at / 8;
    let shift = at & 7;
    let mut word = u32::from(b[byte]);
    if shift + 12 > 8 {
        word |= u32::from(b[byte + 1]) << 8;
    }
    if shift + 12 > 16 {
        word |= u32::from(b[byte + 2]) << 16;
    }
    ((word >> shift) & 0x0fff) as u16
}
#[inline]
fn select_set64(mut bits: u64, mut nth: usize) -> usize {
    let mut base = 0;
    macro_rules! narrow {
        ($shift:literal) => {{
            let lower = bits & ((1u64 << $shift) - 1);
            let count = lower.count_ones() as usize;
            if nth >= count {
                bits >>= $shift;
                base += $shift;
                nth -= count;
            } else {
                bits = lower;
            }
        }};
    }
    narrow!(32);
    narrow!(16);
    narrow!(8);
    narrow!(4);
    narrow!(2);
    let count = (bits & 1).count_ones() as usize;
    base + usize::from(nth >= count)
}
#[inline]
fn ef_range(bits: u64, bit_len: usize, bucket: usize) -> (usize, usize) {
    let zeros = !bits & ((1u64 << bit_len) - 1);
    let end = select_set64(zeros, bucket) - bucket;
    let start = if bucket == 0 {
        0
    } else {
        select_set64(zeros, bucket - 1) - (bucket - 1)
    };
    (start, end)
}
fn pack32(v: &[u16], b: &mut [u8]) {
    b.fill(0);
    let n = v.len();
    write_bits(b, 0, 5, n as u64);
    let hp = 5;
    for (i, &x) in v.iter().enumerate() {
        let bit = hp + (usize::from(x >> 12)) + i;
        b[bit / 8] |= 1 << (bit & 7);
    }
    let lp = hp + 16 + n;
    for (i, &x) in v.iter().enumerate() {
        write_bits(b, lp + i * 12, 12, u64::from(x & 0xfff));
    }
}
fn contains32(b: &[u8], x: u16) -> Result<bool, StoreError> {
    let n = read_bits(b, 0, 5) as usize;
    let hp = 5;
    let (start, end) = ef_range(read_bit_window(b, hp, 16 + n), 16 + n, usize::from(x >> 12));
    if start == end {
        return Ok(false);
    }
    let mut candidates = [0u16; 18];
    let lp = hp + 16 + n;
    for (index, candidate) in candidates[..end - start].iter_mut().enumerate() {
        *candidate = read_u12(b, lp + (start + index) * 12);
    }
    Ok(scan_u16_candidates(&candidates[..end - start], x & 0xfff))
}
fn pack64(a: &[u16], bval: &[u16], out: &mut [u8]) {
    out.fill(0);
    write_bits(out, 0, 6, a.len() as u64);
    write_bits(out, 6, 6, bval.len() as u64);
    let h0 = 12;
    let h1 = h0 + 16 + a.len();
    for (i, &x) in a.iter().enumerate() {
        let p = h0 + usize::from(x >> 12) + i;
        out[p / 8] |= 1 << (p & 7)
    }
    for (i, &x) in bval.iter().enumerate() {
        let p = h1 + usize::from(x >> 12) + i;
        out[p / 8] |= 1 << (p & 7)
    }
    let lp = h1 + 16 + bval.len();
    for (i, &x) in a.iter().chain(bval).enumerate() {
        write_bits(out, lp + i * 12, 12, u64::from(x & 0xfff));
    }
}
fn contains64(b: &[u8], which: usize, x: u16) -> Result<bool, StoreError> {
    let n0 = read_bits(b, 0, 6) as usize;
    let n1 = read_bits(b, 6, 6) as usize;
    let n = if which == 0 { n0 } else { n1 };
    let hs = if which == 0 { 12 } else { 12 + 16 + n0 };
    let lo = if which == 0 { 0 } else { n0 };
    let lp = 12 + 16 + n0 + 16 + n1;
    let (start, end) = ef_range(read_bit_window(b, hs, 16 + n), 16 + n, usize::from(x >> 12));
    if start == end {
        return Ok(false);
    }
    let mut candidates = [0u16; 36];
    let count = end - start;
    for (index, candidate) in candidates[..count].iter_mut().enumerate() {
        *candidate = read_u12(b, lp + (lo + start + index) * 12);
    }
    Ok(scan_u16_candidates(&candidates[..count], x & 0xfff))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_run_records_match_maximal_sorted_runs() {
        let values = [0, 1, 2, 4, 6, 7, u16::MAX - 1, u16::MAX];
        assert_eq!(count_runs(&[]), 0);
        assert_eq!(count_runs(&values), 4);

        let mut record = Vec::new();
        let offset = append_runs(&mut record, &values, count_runs(&values)).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(
            record,
            [
                4, 0, // four runs
                0, 0, 3, 0, // 0..3
                4, 0, 1, 0, // 4..5
                6, 0, 2, 0, // 6..8
                254, 255, 2, 0, // 65534..65536, with no u16 wrap
            ]
        );
    }

    #[test]
    fn var_arena_rejects_retired_handle_after_same_block_reuse() {
        let mut arena = VarArena::default();
        let stale: GroupRecordHandle = arena.alloc::<GroupRecordTag>(b"old record").unwrap();
        arena.retire(stale).unwrap();

        let current: GroupRecordHandle = arena.alloc::<GroupRecordTag>(b"new record").unwrap();
        assert_eq!(stale.idx, current.idx);
        assert_ne!(stale.generation, current.generation);
        assert!(matches!(arena.get(stale), Err(StoreError::InvalidRecord)));
        assert_eq!(arena.get(current).unwrap(), b"new record");
        assert!(matches!(
            arena.retire(stale),
            Err(StoreError::InvalidRecord)
        ));
        assert_eq!(arena.live_bytes, b"new record".len() + 8);
    }

    #[test]
    fn arena_handles_are_tagged_generational_and_option_niche_sized() {
        assert_eq!(std::mem::size_of::<GroupRecordHandle>(), 8);
        assert_eq!(std::mem::size_of::<Option<GroupRecordHandle>>(), 8);
        assert_eq!(std::mem::size_of::<DenseTrnMapHandle>(), 8);
        assert_eq!(std::mem::size_of::<Option<DenseTrnMapHandle>>(), 8);
    }

    #[test]
    fn var_arena_split_blocks_get_independent_live_generations() {
        let mut arena = VarArena::default();
        let stale_large: GroupRecordHandle = arena.alloc::<GroupRecordTag>(&[7; 40]).unwrap();
        arena.retire(stale_large).unwrap();

        let first: GroupRecordHandle = arena.alloc::<GroupRecordTag>(&[1; 0]).unwrap();
        let second: GroupRecordHandle = arena.alloc::<GroupRecordTag>(&[2; 3]).unwrap();
        let third: GroupRecordHandle = arena.alloc::<GroupRecordTag>(&[3; 20]).unwrap();
        assert_eq!(stale_large.idx, first.idx);
        assert_ne!(first.idx, second.idx);
        assert_ne!(second.idx, third.idx);
        assert!(matches!(
            arena.get(stale_large),
            Err(StoreError::InvalidRecord)
        ));
        assert_eq!(arena.get(first).unwrap(), b"");
        assert_eq!(arena.get(second).unwrap(), &[2; 3]);
        assert_eq!(arena.get(third).unwrap(), &[3; 20]);
    }

    #[test]
    fn var_arena_accepts_sequential_power_of_two_blocks_without_padding() {
        let mut arena = VarArena::default();
        let first: GroupRecordHandle = arena.alloc::<GroupRecordTag>(&[1; 24]).unwrap();
        let second: GroupRecordHandle = arena.alloc::<GroupRecordTag>(&[2; 40]).unwrap();
        assert_eq!(arena.slots[first.idx as usize].offset, 0);
        assert_eq!(arena.slots[second.idx as usize].offset, 32);
        assert_eq!(arena.get(first).unwrap(), &[1; 24]);
        assert_eq!(arena.get(second).unwrap(), &[2; 40]);
    }

    #[test]
    fn var_arena_retires_a_block_at_generation_exhaustion() {
        let mut arena = VarArena::default();
        let initial: GroupRecordHandle = arena.alloc::<GroupRecordTag>(b"value").unwrap();
        let generation = NonZeroU32::new(u32::MAX).unwrap();
        arena.slots[initial.idx as usize].generation = generation;
        let exhausted: GroupRecordHandle = Handle::new(initial.idx, generation);
        arena.retire(exhausted).unwrap();

        let replacement: GroupRecordHandle = arena.alloc::<GroupRecordTag>(b"fresh").unwrap();
        assert_ne!(replacement.idx, exhausted.idx);
        assert!(matches!(
            arena.get(exhausted),
            Err(StoreError::InvalidRecord)
        ));
        assert_eq!(arena.get(replacement).unwrap(), b"fresh");
    }

    #[test]
    fn deterministic_var_arena_mutation_fuzz_rejects_all_retired_handles() {
        let mut arena = VarArena::default();
        let mut live: Vec<(GroupRecordHandle, Vec<u8>)> = Vec::new();
        let mut retired = Vec::new();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;

        for step in 0..6000u32 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            if live.is_empty() || (seed & 1 == 0 && live.len() < 64) {
                let len = (seed as usize) & 0x1fff;
                let value = vec![step as u8 ^ (seed >> 32) as u8; len];
                let handle = arena.alloc::<GroupRecordTag>(&value).unwrap();
                live.push((handle, value));
            } else {
                let index = (seed as usize) % live.len();
                let (handle, _) = live.swap_remove(index);
                arena.retire(handle).unwrap();
                retired.push(handle);
            }

            if step % 127 == 0 {
                for (handle, value) in &live {
                    assert_eq!(arena.get(*handle).unwrap(), value);
                }
                for &handle in retired.iter().rev().take(256) {
                    assert!(matches!(arena.get(handle), Err(StoreError::InvalidRecord)));
                }
            }
        }

        for (handle, value) in &live {
            assert_eq!(arena.get(*handle).unwrap(), value);
        }
        for handle in retired {
            assert!(matches!(arena.get(handle), Err(StoreError::InvalidRecord)));
        }
    }

    #[test]
    fn group_generation_exhaustion_leaves_the_group_unchanged() {
        let mut store = RnbwStore::new(PocketLayout::PerChunk32).unwrap();
        let exhausted = (u64::from(u32::MAX) << 32) | u64::from(ABSENT);
        store.gen_table[0] = exhausted;
        let chunks: [Vec<u16>; 16] = std::array::from_fn(|_| Vec::new());

        assert!(matches!(
            store.replace_group(0, &chunks),
            Err(StoreError::TooLarge)
        ));
        assert_eq!(store.gen_table[0], exhausted);
        assert_eq!(store.arena.live_bytes, 0);
        assert_eq!(store.group_handles[0], None);
    }

    #[test]
    fn l2_forms_and_update_roundtrip() {
        let mut v = vec![
            0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38,
        ];
        v.extend(0x10000..0x10000 + 20000);
        v.sort_unstable();
        let mut s = RnbwStore::from_sorted(&v, PocketLayout::PerChunk32).unwrap();
        for x in &v {
            assert!(s.contains(*x).unwrap());
        }
        assert_eq!(s.decode().unwrap(), v);
        assert!(s.insert(0x4000_1234).unwrap());
        assert!(s.contains(0x4000_1234).unwrap());
        assert!(s.remove(0x4000_1234).unwrap());
        assert_eq!(s.decode().unwrap(), v);
    }

    #[test]
    fn dense_chunks_use_arena_backed_trnmap_without_bitmap_storage() {
        let values: Vec<u32> = (0..65_536).step_by(2).collect();
        let mut store = RnbwStore::from_sorted(&values, PocketLayout::PerChunk32).unwrap();
        let group = store.group_handle(0).unwrap();
        let record = store.arena.get(group).unwrap();
        let (form, offset) = RnbwStore::descriptor(record, 0).unwrap();
        assert_eq!(form, Form::TrnMap);
        let old_map: DenseTrnMapHandle = read_handle(record, offset).unwrap();
        let old_image = store.arena.get(old_map).unwrap();
        let old_view = TrnMapView::from_dense_chunk(&old_image[DENSE_MAP_HEADER_BYTES..]).unwrap();
        assert_eq!(old_view.get_dense(0), 1);
        assert_eq!(old_view.get_dense(1), -1);
        assert_eq!(old_view.get_dense(u16::MAX - 1), 1);
        assert_eq!(old_view.get_dense(u16::MAX), -1);
        assert!(store.contains(0).unwrap());
        assert!(!store.contains(1).unwrap());
        assert!(store.contains((u16::MAX - 1) as u32).unwrap());
        assert!(!store.contains(u16::MAX as u32).unwrap());
        assert_eq!(store.decode().unwrap(), values);

        assert!(store.insert(1).unwrap());
        let current_image = store.arena.get(old_map).unwrap();
        assert_eq!(
            u32::from_le_bytes(current_image[..4].try_into().unwrap()),
            32_769
        );
        let current_view =
            TrnMapView::from_dense_chunk(&current_image[DENSE_MAP_HEADER_BYTES..]).unwrap();
        assert_eq!(current_view.get_dense(1), 1);
        assert!(store.contains(1).unwrap());
        assert_eq!(store.len(), values.len() + 1);
    }

    #[test]
    fn deterministic_dense_trnmap_mutations_match_btree_set() {
        use std::collections::BTreeSet;

        let values: Vec<u32> = (0..65_536).step_by(2).collect();
        let mut expected: BTreeSet<u32> = values.iter().copied().collect();
        let mut store = RnbwStore::from_sorted(&values, PocketLayout::PerChunk32).unwrap();
        let mut seed = 0xd1b5_4a32_d192_ed03u64;
        for step in 0..128 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let key = seed as u16 as u32;
            if expected.remove(&key) {
                assert!(store.remove(key).unwrap());
            } else {
                expected.insert(key);
                assert!(store.insert(key).unwrap());
            }
            if step % 16 == 0 {
                assert_eq!(
                    store.decode().unwrap(),
                    expected.iter().copied().collect::<Vec<_>>()
                );
                for probe in [0, 1, key, u16::MAX as u32] {
                    assert_eq!(store.contains(probe).unwrap(), expected.contains(&probe));
                }
            }
        }
        assert_eq!(
            store.decode().unwrap(),
            expected.into_iter().collect::<Vec<_>>()
        );
    }
    #[test]
    fn pockets_dirty_and_spill_are_exact() {
        let v: Vec<u32> = (0..18).map(|x| x * 4096).collect();
        let mut s = RnbwStore::from_sorted(&v, PocketLayout::PerChunk32).unwrap();
        assert_eq!(s.dirty_chunks().unwrap(), vec![0, 1]);
        assert_eq!(s.promote_range(0, 2).unwrap(), (2, 0));
        for x in &v {
            assert!(s.contains(*x).unwrap());
        }
        assert!(!s.contains(19 * 4096).unwrap());
        assert_eq!(s.dirty_chunks().unwrap(), vec![0, 1]);
        assert_eq!(s.dirty_chunks().unwrap(), Vec::<usize>::new());
    }
    #[test]
    fn l3_is_allocated_only_when_a_chunk_promotes() {
        let full: Vec<u32> = (0..65_536).collect();
        let mut store = RnbwStore::from_sorted(&full, PocketLayout::PerChunk32).unwrap();
        assert_eq!(store.promote_range(0, 1).unwrap(), (0, 1));
        assert_eq!(store.l3_bytes(), 0);
        assert!(store.contains(65_535).unwrap());
        assert!(!store.contains(65_536).unwrap());

        let sparse = vec![0, 4096];
        let mut store = RnbwStore::from_sorted(&sparse, PocketLayout::PerChunk32).unwrap();
        assert_eq!(store.promote_range(0, 1).unwrap(), (1, 0));
        assert_eq!(store.l3_bytes(), 2 * 1024 * 1024);
        assert!(store.contains(4096).unwrap());
    }
    #[test]
    fn paired_pocket_matches_l2_across_pair() {
        let v: Vec<u32> = (0..36).map(|x| x * 2048).collect();
        let mut s = RnbwStore::from_sorted(&v, PocketLayout::PerPair64).unwrap();
        assert_eq!(s.promote_range(0, 2).unwrap(), (2, 0));
        for x in &v {
            assert!(s.contains(*x).unwrap());
        }
        assert_eq!(s.decode().unwrap(), v);
    }

    #[test]
    fn range_promotion_batches_groups_and_respects_partial_group_bounds() {
        let values: Vec<u32> = [15u32, 16, 17, 31, 32, 33]
            .into_iter()
            .map(|chunk| (chunk << 16) | 7)
            .collect();

        let mut per_chunk = RnbwStore::from_sorted(&values, PocketLayout::PerChunk32).unwrap();
        per_chunk.dirty_chunks().unwrap();
        assert_eq!(per_chunk.promote_range(16, 32).unwrap(), (3, 0));
        assert_eq!(per_chunk.dirty_chunks().unwrap(), vec![16, 17, 31]);
        assert_eq!(per_chunk.promote_range(16, 16).unwrap(), (0, 0));
        assert!(per_chunk.dirty_chunks().unwrap().is_empty());
        assert_eq!(per_chunk.decode().unwrap(), values);

        let mut per_pair = RnbwStore::from_sorted(&values, PocketLayout::PerPair64).unwrap();
        per_pair.dirty_chunks().unwrap();
        assert_eq!(per_pair.promote_range(17, 33).unwrap(), (3, 0));
        assert_eq!(per_pair.dirty_chunks().unwrap(), vec![17, 31, 32]);
        assert_eq!(per_pair.decode().unwrap(), values);
        for value in &values {
            assert!(per_chunk.contains(*value).unwrap());
            assert!(per_pair.contains(*value).unwrap());
        }
    }

    #[test]
    fn deterministic_store_mutation_fuzz_matches_btree_set() {
        use std::collections::BTreeSet;
        let mut seed = 0xa076_1d64_78bd_642fu64;
        let mut reference = BTreeSet::new();
        for _ in 0..1200 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            reference.insert((seed as u32) & 0x0003_ffff);
        }
        let initial: Vec<_> = reference.iter().copied().collect();
        let mut store = RnbwStore::from_sorted(&initial, PocketLayout::PerChunk32).unwrap();
        store.dirty_chunks().unwrap();
        store.promote_range(0, 4).unwrap();
        for step in 0..5000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let key = (seed as u32) & 0x0003_ffff;
            if step % 3 == 0 {
                assert_eq!(store.insert(key).unwrap(), reference.insert(key));
            } else {
                assert_eq!(store.remove(key).unwrap(), reference.remove(&key));
            }
            assert_eq!(store.contains(key).unwrap(), reference.contains(&key));
            if step % 127 == 0 {
                assert_eq!(
                    store.decode().unwrap(),
                    reference.iter().copied().collect::<Vec<_>>()
                );
            }
        }
        assert_eq!(store.len(), reference.len());
        assert_eq!(
            store.decode().unwrap(),
            reference.iter().copied().collect::<Vec<_>>()
        );
    }
}
