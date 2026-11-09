//! L2 GenTable + VAR Arena storage with explicitly promoted L3 direct pockets.
//! The 32-bit universe is split into 65,536 16-bit chunks and groups of 16.

use crate::{CodecError, Rnbw, RnbwView, TrnMap, TrnMapError, scan_u16_candidates};
use std::fmt;

pub const CHUNK_COUNT: usize = 1 << 16;
pub const GROUP_COUNT: usize = 1 << 12;
const CHUNKS_PER_GROUP: usize = 16;
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
const VAR_GENERATION_SHIFT: u32 = 44;
const VAR_GENERATION_MASK: u32 = (1 << 20) - 1;
const VAR_HANDLE_GENERATION_MASK: u64 = (VAR_GENERATION_MASK as u64) << VAR_GENERATION_SHIFT;
const VAR_LIVE_SHIFT: u32 = 43;
const VAR_LIVE_MASK: u64 = 1 << VAR_LIVE_SHIFT;

#[derive(Clone, Copy)]
struct VarBlockHeader {
    len: usize,
    block_log2: usize,
    generation: u32,
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
            generation: ((raw >> VAR_GENERATION_SHIFT) & u64::from(VAR_GENERATION_MASK)) as u32,
            live: raw & (1 << VAR_LIVE_SHIFT) != 0,
        })
    }
    fn encode(self) -> u64 {
        // The 8-byte in-block header stores len:24, block_log2:5, generation:20, live:1.
        (self.len as u64 & VAR_LEN_MASK)
            | (((self.block_log2 - ARENA_MIN_BLOCK_LOG2) as u64) << VAR_BLOCK_LOG_SHIFT)
            | (u64::from(self.generation) << VAR_GENERATION_SHIFT)
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
    Bitmap = 5,
}
impl Form {
    fn from(v: u8) -> Result<Self, StoreError> {
        match v {
            0 => Ok(Self::Empty),
            1 => Ok(Self::Full),
            2 => Ok(Self::Inline1),
            3 => Ok(Self::Quot),
            4 => Ok(Self::Runs),
            5 => Ok(Self::Bitmap),
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
#[derive(Default)]
struct VarArena {
    pages: Vec<VarPage>,
    free: Vec<Vec<(u32, u32)>>,
    free_mask: u32,
    live_bytes: usize,
    reserved_bytes: usize,
}
impl VarArena {
    fn alloc(&mut self, record: &[u8]) -> Result<u64, StoreError> {
        if record.len() as u64 > VAR_LEN_MASK {
            return Err(StoreError::TooLarge);
        }
        let need = (record.len().checked_add(8).ok_or(StoreError::TooLarge)?)
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
        let reusable = loop {
            let candidates = self.free_mask & (u32::MAX << min_class);
            if candidates == 0 {
                break None;
            }
            let class = candidates.trailing_zeros() as usize;
            let (page, offset) = self.free[class].pop().ok_or(StoreError::InvalidRecord)?;
            if self.free[class].is_empty() {
                self.free_mask &= !(1u32 << class);
            }
            let header = self.pages[page as usize].header(offset as usize)?;
            if header.live || header.block_log2 != class + ARENA_MIN_BLOCK_LOG2 {
                return Err(StoreError::InvalidRecord);
            }
            if header.generation == VAR_GENERATION_MASK {
                continue;
            }
            break Some((page, offset, class, header.generation + 1));
        };
        let (page, offset, block, generation) =
            if let Some((page, offset, mut class, generation)) = reusable {
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
                            generation: 0,
                            live: false,
                        },
                    )?;
                    self.free[class].push((page, sibling as u32));
                    self.free_mask |= 1u32 << class;
                }
                (page, offset, need, generation)
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
                (p as u32, off as u32, need, 1)
            };
        let p = &mut self.pages[page as usize];
        let raw = (u64::from(generation) << 44) | (u64::from(page) << 24) | u64::from(offset);
        let b = p.bytes_mut();
        b[offset as usize + 8..offset as usize + 8 + record.len()].copy_from_slice(record);
        p.set_header(
            offset as usize,
            VarBlockHeader {
                len: record.len(),
                block_log2: block.trailing_zeros() as usize,
                generation,
                live: true,
            },
        )?;
        self.live_bytes += record.len() + 8;
        Ok(raw)
    }
    #[inline]
    fn parts(&self, h: u64) -> Result<(usize, usize, u64), StoreError> {
        let generation = h & VAR_HANDLE_GENERATION_MASK;
        let page = ((h >> 24) & 0xfffff) as usize;
        let off = (h & 0xffffff) as usize;
        let p = self.pages.get(page).ok_or(StoreError::InvalidRecord)?;
        if generation == 0 {
            return Err(StoreError::InvalidRecord);
        }
        let raw = p.header_raw(off)?;
        let expected = generation | VAR_LIVE_MASK;
        if raw & (VAR_HANDLE_GENERATION_MASK | VAR_LIVE_MASK) != expected {
            return Err(StoreError::InvalidRecord);
        }
        let len = (raw & VAR_LEN_MASK) as usize;
        if len > p.len() - off - 8 {
            return Err(StoreError::InvalidRecord);
        }
        Ok((page, off, raw))
    }
    #[inline]
    fn get(&self, h: u64) -> Result<&[u8], StoreError> {
        let (p, o, raw) = self.parts(h)?;
        let len = (raw & VAR_LEN_MASK) as usize;
        Ok(&self.pages[p].bytes()[o + 8..o + 8 + len])
    }
    fn retire(&mut self, h: u64) -> Result<(), StoreError> {
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
        self.free[class].push((page as u32, offset as u32));
        self.free_mask |= 1u32 << class;
        self.live_bytes -= len + 8;
        Ok(())
    }
}

struct BitmapSlot {
    words: Box<[u64; 1024]>,
    generation: u32,
    live: bool,
}
#[derive(Default)]
struct BitmapArena {
    slots: Vec<BitmapSlot>,
    handles: Vec<u64>,
    free: Vec<u32>,
}
impl BitmapArena {
    fn alloc(&mut self, values: &[u16]) -> Result<u16, StoreError> {
        let index = if let Some(i) = self.free.pop() {
            i as usize
        } else {
            if self.slots.len() >= u16::MAX as usize {
                return Err(StoreError::TooLarge);
            }
            self.slots.push(BitmapSlot {
                words: Box::new([0; 1024]),
                generation: 1,
                live: false,
            });
            self.slots.len() - 1
        };
        let s = &mut self.slots[index];
        s.words.fill(0);
        for &v in values {
            s.words[v as usize / 64] |= 1u64 << (v & 63)
        }
        s.live = true;
        let generation = s.generation;
        if self.handles.len() <= index {
            self.handles.resize(index + 1, 0);
        }
        self.handles[index] = (u64::from(generation) << 32) | index as u64;
        Ok(index as u16)
    }
    fn contains(&self, index: u16, value: u16) -> Result<bool, StoreError> {
        let s = self
            .slots
            .get(index as usize)
            .ok_or(StoreError::InvalidRecord)?;
        if !s.live
            || self.handles.get(index as usize).copied()
                != Some((u64::from(s.generation) << 32) | u64::from(index))
        {
            return Err(StoreError::InvalidRecord);
        }
        Ok(s.words[value as usize / 64] & (1u64 << (value & 63)) != 0)
    }
    fn decode(&self, index: u16, out: &mut Vec<u16>) -> Result<(), StoreError> {
        let s = self
            .slots
            .get(index as usize)
            .ok_or(StoreError::InvalidRecord)?;
        if !s.live
            || self.handles.get(index as usize).copied()
                != Some((u64::from(s.generation) << 32) | u64::from(index))
        {
            return Err(StoreError::InvalidRecord);
        }
        for (wi, &mut_word) in s.words.iter().enumerate() {
            let mut w = mut_word;
            while w != 0 {
                let b = w.trailing_zeros() as usize;
                out.push((wi * 64 + b) as u16);
                w &= w - 1;
            }
        }
        Ok(())
    }
    fn retire(&mut self, index: u16) -> Result<(), StoreError> {
        let i = index as usize;
        let s = self.slots.get_mut(i).ok_or(StoreError::InvalidRecord)?;
        if !s.live {
            return Err(StoreError::InvalidRecord);
        }
        s.live = false;
        if let Some(generation) = s.generation.checked_add(1) {
            s.generation = generation;
            self.free.push(index as u32);
        }
        Ok(())
    }
}

pub struct RnbwStore {
    tiers: TrnMap<'static>,
    gen_table: Box<[u64]>,
    group_handles: Box<[u64]>,
    arena: VarArena,
    bitmaps: BitmapArena,
    layout: PocketLayout,
    pockets: Option<Box<[u8]>>,
    len: usize,
}

impl RnbwStore {
    pub fn new(layout: PocketLayout) -> Result<Self, StoreError> {
        Ok(Self {
            tiers: TrnMap::new(CHUNK_COUNT)?,
            gen_table: vec![u64::from(ABSENT); GROUP_COUNT].into_boxed_slice(),
            group_handles: vec![0; GROUP_COUNT].into_boxed_slice(),
            arena: VarArena::default(),
            bitmaps: BitmapArena::default(),
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
        while at < values.len() {
            let gid = (values[at] >> 20) as usize;
            let mut chunks: [Vec<u16>; 16] = std::array::from_fn(|_| Vec::new());
            while at < values.len() && (values[at] >> 20) as usize == gid {
                let v = values[at];
                chunks[((v >> 16) & 15) as usize].push(v as u16);
                at += 1;
            }
            s.replace_group(gid, &chunks)?;
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
            + self.arena.free.capacity() * std::mem::size_of::<Vec<(u32, u32)>>()
            + self
                .arena
                .free
                .iter()
                .map(|blocks| blocks.capacity() * std::mem::size_of::<(u32, u32)>())
                .sum::<usize>()
            + self.bitmaps.slots.len() * 8192
            + self.bitmaps.slots.capacity() * std::mem::size_of::<BitmapSlot>()
            + self.bitmaps.handles.capacity() * std::mem::size_of::<u64>()
            + self.bitmaps.free.capacity() * std::mem::size_of::<u32>()
            + self.gen_table.len() * 8
            + self.group_handles.len() * 8
            + self.tiers.storage_bytes()
    }
    pub fn l3_bytes(&self) -> usize {
        self.pockets.as_ref().map_or(0, |p| p.len())
    }
    fn group_handle(&self, g: usize) -> Option<u64> {
        let cell = self.gen_table[g];
        if cell as u32 == ABSENT {
            None
        } else {
            Some(self.group_handles[g])
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
            Form::Bitmap => self.bitmaps.contains(o, suffix),
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
                Form::Bitmap => self.bitmaps.decode(o, &mut out[i])?,
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
    fn chunk_values(&self, chunk: usize) -> Result<Vec<u16>, StoreError> {
        let mut a = self.decode_group(chunk >> 4)?;
        Ok(std::mem::take(&mut a[chunk & 15]))
    }
    fn replace_group(&mut self, g: usize, chunks: &[Vec<u16>; 16]) -> Result<(), StoreError> {
        let old = self.group_handle(g);
        let generation = (self.gen_table[g] >> 32) as u32;
        let next_generation = generation.checked_add(1).ok_or(StoreError::TooLarge)?;
        let mut rec = vec![0u8; GROUP_HEADER];
        let mut forms = [Form::Empty; 16];
        let mut bitmap_new = Vec::new();
        for i in 0..16 {
            let v = &chunks[i];
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
            let runs = make_runs(v);
            let run_bytes = 2 + runs.len() * 4;
            let qbits = (v.len() as f64 * (18.0 - (v.len() as f64).log2() + 1.0)).ceil() as usize;
            let qbytes = qbits.div_ceil(8);
            if 8192 < qbytes && 8192 < run_bytes {
                forms[i] = Form::Bitmap
            } else if run_bytes < qbytes {
                forms[i] = Form::Runs
            } else {
                forms[i] = Form::Quot
            }
            if forms[i] == Form::Runs {
                let off = append_runs(&mut rec, &runs)?;
                put_desc(&mut rec, i, Form::Runs, off)?;
            }
        }
        let mut keys = Vec::new();
        for i in 0..16 {
            if forms[i] == Form::Quot {
                for &x in &chunks[i] {
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
        for i in 0..16 {
            if forms[i] == Form::Bitmap {
                let slot = match self.bitmaps.alloc(&chunks[i]) {
                    Ok(slot) => slot,
                    Err(error) => {
                        for slot in bitmap_new.drain(..) {
                            self.bitmaps
                                .retire(slot)
                                .expect("new bitmap slot remains live during rollback");
                        }
                        return Err(error);
                    }
                };
                bitmap_new.push(slot);
                put_desc(&mut rec, i, Form::Bitmap, slot)?;
            }
        }
        let new = if chunks.iter().all(Vec::is_empty) {
            None
        } else {
            match self.arena.alloc(&rec) {
                Ok(h) => Some(h),
                Err(e) => {
                    for slot in bitmap_new {
                        self.bitmaps.retire(slot)?;
                    }
                    return Err(e);
                }
            }
        };
        self.gen_table[g] = (u64::from(next_generation) << 32)
            | u64::from(if new.is_some() { g as u32 } else { ABSENT });
        self.group_handles[g] = new.unwrap_or(0);
        if let Some(h) = old {
            let r = self.arena.get(h)?.to_vec();
            for i in 0..16 {
                let (f, o) = Self::descriptor(&r, i)?;
                if f == Form::Bitmap {
                    self.bitmaps.retire(o)?;
                }
            }
            self.arena.retire(h)?;
        }
        Ok(())
    }
    pub fn insert(&mut self, key: u32) -> Result<bool, StoreError> {
        let c = (key >> 16) as usize;
        let mut v = self.chunk_values(c)?;
        match v.binary_search(&(key as u16)) {
            Ok(_) => return Ok(false),
            Err(i) => v.insert(i, key as u16),
        }
        let g = c >> 4;
        let mut chunks = self.decode_group(g)?;
        chunks[c & 15] = v;
        let hot = (0..16)
            .filter(|&i| self.tiers.get((g << 4) + i).unwrap_or(-1) == 1)
            .collect::<Vec<_>>();
        self.replace_group(g, &chunks)?;
        self.len += 1;
        if chunks[c & 15].len() == 1 {
            self.set_tier(c, 0)?;
        }
        self.tiers.mark(c)?;
        for i in hot {
            self.refresh_hot((g << 4) + i)?;
        }
        if self.tiers.get(c)? == -1 {
            self.set_tier(c, 0)?;
        }
        Ok(true)
    }
    pub fn remove(&mut self, key: u32) -> Result<bool, StoreError> {
        let c = (key >> 16) as usize;
        let mut v = self.chunk_values(c)?;
        let Ok(i) = v.binary_search(&(key as u16)) else {
            return Ok(false);
        };
        v.remove(i);
        let g = c >> 4;
        let mut chunks = self.decode_group(g)?;
        chunks[c & 15] = v;
        let hot = (0..16)
            .filter(|&i| self.tiers.get((g << 4) + i).unwrap_or(-1) == 1)
            .collect::<Vec<_>>();
        self.replace_group(g, &chunks)?;
        self.len -= 1;
        self.tiers.mark(c)?;
        self.set_tier(c, if chunks[c & 15].is_empty() { -1 } else { 0 })?;
        for i in hot {
            let ch = (g << 4) + i;
            if !chunks[i].is_empty() {
                self.refresh_hot(ch)?;
            }
        }
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
        let mut promoted = 0;
        let mut spills = 0;
        match self.layout {
            PocketLayout::PerChunk32 => {
                for c in start..end {
                    let v = self.chunk_values(c)?;
                    if v.is_empty() {
                        continue;
                    }
                    if v.len() <= 18 {
                        let off = c * 32;
                        let pockets = self
                            .pockets
                            .get_or_insert_with(|| vec![0; 2 * 1024 * 1024].into_boxed_slice());
                        pack32(&v, &mut pockets[off..off + 32]);
                        self.set_tier(c, 1)?;
                        promoted += 1
                    } else {
                        self.set_tier(c, 0)?;
                        spills += 1
                    }
                }
            }
            PocketLayout::PerPair64 => {
                let first = start / 2;
                let last = end.div_ceil(2);
                for p in first..last {
                    let a = self.chunk_values(p * 2)?;
                    let b = self.chunk_values(p * 2 + 1)?;
                    if a.is_empty() && b.is_empty() {
                        continue;
                    }
                    if a.len() + b.len() <= 36 {
                        let off = p * 64;
                        let pockets = self
                            .pockets
                            .get_or_insert_with(|| vec![0; 2 * 1024 * 1024].into_boxed_slice());
                        pack64(&a, &b, &mut pockets[off..off + 64]);
                        for c in [p * 2, p * 2 + 1] {
                            if c >= start && c < end && !self.chunk_values(c)?.is_empty() {
                                self.set_tier(c, 1)?;
                                promoted += 1;
                            }
                        }
                    } else {
                        for c in p * 2..p * 2 + 2 {
                            if c >= start && c < end && !self.chunk_values(c)?.is_empty() {
                                self.set_tier(c, 0)?;
                                spills += 1;
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
        for c in start..end {
            if self.tiers.get(c)? == 1 {
                self.set_tier(
                    c,
                    if self.chunk_values(c)?.is_empty() {
                        -1
                    } else {
                        0
                    },
                )?;
                n += 1;
            }
        }
        Ok(n)
    }
    fn refresh_hot(&mut self, c: usize) -> Result<(), StoreError> {
        match self.layout {
            PocketLayout::PerChunk32 => {
                let v = self.chunk_values(c)?;
                if v.len() <= 18 && !v.is_empty() {
                    let off = c * 32;
                    pack32(
                        &v,
                        &mut self
                            .pockets
                            .get_or_insert_with(|| vec![0; 2 * 1024 * 1024].into_boxed_slice())
                            [off..off + 32],
                    );
                    self.set_tier(c, 1)?
                } else {
                    self.set_tier(c, 0)?
                }
            }
            PocketLayout::PerPair64 => {
                let p = c / 2;
                let a = self.chunk_values(p * 2)?;
                let b = self.chunk_values(p * 2 + 1)?;
                if a.len() + b.len() <= 36 {
                    let off = p * 64;
                    pack64(
                        &a,
                        &b,
                        &mut self
                            .pockets
                            .get_or_insert_with(|| vec![0; 2 * 1024 * 1024].into_boxed_slice())
                            [off..off + 64],
                    );
                } else {
                    for x in p * 2..p * 2 + 2 {
                        if self.tiers.get(x)? == 1 {
                            self.set_tier(x, 0)?;
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
fn make_runs(v: &[u16]) -> Vec<(u16, u16)> {
    let mut r = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let s = v[i];
        let mut e = i + 1;
        while e < v.len() && u32::from(v[e]) == u32::from(v[e - 1]) + 1 {
            e += 1
        }
        r.push((s, (e - i) as u16));
        i = e;
    }
    r
}
fn append_runs(r: &mut Vec<u8>, runs: &[(u16, u16)]) -> Result<u16, StoreError> {
    let off = u16::try_from(r.len()).map_err(|_| StoreError::TooLarge)?;
    r.extend_from_slice(&(runs.len() as u16).to_le_bytes());
    for &(s, n) in runs {
        r.extend_from_slice(&s.to_le_bytes());
        r.extend_from_slice(&n.to_le_bytes());
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
    fn var_arena_rejects_retired_handle_after_same_block_reuse() {
        let mut arena = VarArena::default();
        let stale = arena.alloc(b"old record").unwrap();
        arena.retire(stale).unwrap();

        let current = arena.alloc(b"new record").unwrap();
        assert_eq!(stale & 0x00ff_ffff, current & 0x00ff_ffff);
        assert_ne!(stale >> 44, current >> 44);
        assert!(matches!(arena.get(stale), Err(StoreError::InvalidRecord)));
        assert_eq!(arena.get(current).unwrap(), b"new record");
        assert!(matches!(
            arena.retire(stale),
            Err(StoreError::InvalidRecord)
        ));
        assert_eq!(arena.live_bytes, b"new record".len() + 8);
    }

    #[test]
    fn var_arena_split_blocks_get_independent_live_generations() {
        let mut arena = VarArena::default();
        let stale_large = arena.alloc(&[7; 40]).unwrap();
        arena.retire(stale_large).unwrap();

        let first = arena.alloc(&[1; 0]).unwrap();
        let second = arena.alloc(&[2; 3]).unwrap();
        let third = arena.alloc(&[3; 20]).unwrap();
        assert_eq!(stale_large & 0x00ff_ffff, first & 0x00ff_ffff);
        assert_ne!(first & 0x00ff_ffff, second & 0x00ff_ffff);
        assert_ne!(second & 0x00ff_ffff, third & 0x00ff_ffff);
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
        let first = arena.alloc(&[1; 24]).unwrap();
        let second = arena.alloc(&[2; 40]).unwrap();
        assert_eq!(first & 0x00ff_ffff, 0);
        assert_eq!(second & 0x00ff_ffff, 32);
        assert_eq!(arena.get(first).unwrap(), &[1; 24]);
        assert_eq!(arena.get(second).unwrap(), &[2; 40]);
    }

    #[test]
    fn var_arena_retires_a_block_at_generation_exhaustion() {
        let mut arena = VarArena::default();
        let initial = arena.alloc(b"value").unwrap();
        let (page, offset, raw) = arena.parts(initial).unwrap();
        let mut header = VarBlockHeader::decode(raw).unwrap();
        header.generation = VAR_GENERATION_MASK;
        arena.pages[page].set_header(offset, header).unwrap();
        let exhausted =
            (initial & !(0x000f_ffff_u64 << 44)) | (u64::from(VAR_GENERATION_MASK) << 44);
        arena.retire(exhausted).unwrap();

        let replacement = arena.alloc(b"fresh").unwrap();
        assert_ne!(replacement & 0x00ff_ffff, exhausted & 0x00ff_ffff);
        assert!(matches!(
            arena.get(exhausted),
            Err(StoreError::InvalidRecord)
        ));
        assert_eq!(arena.get(replacement).unwrap(), b"fresh");
    }

    #[test]
    fn deterministic_var_arena_mutation_fuzz_rejects_all_retired_handles() {
        let mut arena = VarArena::default();
        let mut live: Vec<(u64, Vec<u8>)> = Vec::new();
        let mut retired = Vec::new();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;

        for step in 0..6000u32 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            if live.is_empty() || (seed & 1 == 0 && live.len() < 64) {
                let len = (seed as usize) & 0x1fff;
                let value = vec![step as u8 ^ (seed >> 32) as u8; len];
                let handle = arena.alloc(&value).unwrap();
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
    fn bitmap_arena_does_not_wrap_an_exhausted_generation() {
        let mut arena = BitmapArena::default();
        let first = arena.alloc(&[1, 5, 9]).unwrap();
        let slot = &mut arena.slots[first as usize];
        slot.generation = u32::MAX;
        arena.handles[first as usize] = (u64::from(u32::MAX) << 32) | u64::from(first);
        arena.retire(first).unwrap();
        assert!(!arena.free.contains(&(first as u32)));

        let next = arena.alloc(&[2, 6, 10]).unwrap();
        assert_ne!(next, first);
        assert!(arena.contains(next, 2).unwrap());
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
        assert_eq!(store.group_handles[0], 0);
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
