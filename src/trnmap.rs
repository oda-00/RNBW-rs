//! Hierarchical packed ternary map ported from Symbolic std's `trnx64`/`trnmap`.
//!
//! Each leaf word stores 64 trits in thermometer planes (`p = t >= 1`,
//! `u = t >= 0`). Summary planes record whether each 64-way child group has
//! any positive, negative, or zero values. A parallel hierarchy tracks dirty
//! trits. The map payload is one 8-byte-aligned allocation with the Symbolic
//! 384-byte header and level descriptors.

use std::fmt;

const HEADER_BYTES: usize = 384;
const HEADER_WORDS: usize = HEADER_BYTES / 8;
const MAX_LEVELS: usize = (usize::BITS as usize).div_ceil(6) + 1;
const DESCRIPTOR_WORD: usize = 8;

pub const CLASS_NEGATIVE: u8 = 1;
pub const CLASS_ZERO: u8 = 2;
pub const CLASS_POSITIVE: u8 = 4;
const NEGATIVE: u8 = CLASS_NEGATIVE;
const ZERO: u8 = CLASS_ZERO;
const POSITIVE: u8 = CLASS_POSITIVE;
const ALL_CLASSES: u8 = NEGATIVE | ZERO | POSITIVE;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrnMapError {
    EmptyMap,
    SizeOverflow,
    BufferTooSmall { needed: usize, available: usize },
    IndexOutOfBounds { index: usize, len: usize },
    WordOutOfBounds { word: usize, words: usize },
    InvalidTrit(i8),
    LengthMismatch { left: usize, right: usize },
}

impl fmt::Display for TrnMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMap => f.write_str("trnmap length must be nonzero"),
            Self::SizeOverflow => f.write_str("trnmap allocation size overflows usize"),
            Self::BufferTooSmall { needed, available } => {
                write!(f, "trnmap needs {needed} bytes, buffer has {available}")
            }
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "trit index {index} is outside map length {len}")
            }
            Self::WordOutOfBounds { word, words } => {
                write!(f, "word index {word} is outside {words} words")
            }
            Self::InvalidTrit(value) => write!(f, "trit must be -1, 0, or 1, got {value}"),
            Self::LengthMismatch { left, right } => {
                write!(f, "trnmap lengths differ: {left} and {right}")
            }
        }
    }
}

impl std::error::Error for TrnMapError {}

/// 64 trits packed into the two thermometer planes used by Symbolic std.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TriWord {
    /// Positive plane: set for trit `+1`.
    p: u64,
    /// Nonnegative plane: set for trits `0` and `+1`.
    u: u64,
}

impl TriWord {
    /// Constructs a word while enforcing the invariant `p` is a subset of `u`.
    #[inline]
    pub const fn from_planes(p: u64, u: u64) -> Self {
        Self { p: p & u, u }
    }

    /// Returns the normalized positive and nonnegative planes.
    #[inline]
    pub const fn planes(self) -> (u64, u64) {
        (self.p, self.u)
    }

    /// A word filled with one trit.
    pub const fn splat(trit: i8) -> Result<Self, TrnMapError> {
        match trit {
            -1 => Ok(Self { p: 0, u: 0 }),
            0 => Ok(Self { p: 0, u: u64::MAX }),
            1 => Ok(Self {
                p: u64::MAX,
                u: u64::MAX,
            }),
            value => Err(TrnMapError::InvalidTrit(value)),
        }
    }

    /// Returns the trit at bit `index`.
    #[inline]
    pub const fn get(self, index: u32) -> Option<i8> {
        if index >= 64 {
            return None;
        }
        let positive = ((self.p >> index) & 1) as i8;
        let nonnegative = ((self.u >> index) & 1) as i8;
        Some(positive + nonnegative - 1)
    }

    /// Replaces one trit.
    pub fn set(&mut self, index: u32, trit: i8) -> Result<(), TrnMapError> {
        if index >= 64 {
            return Err(TrnMapError::IndexOutOfBounds {
                index: index as usize,
                len: 64,
            });
        }
        if !(-1..=1).contains(&trit) {
            return Err(TrnMapError::InvalidTrit(trit));
        }
        let bit = 1u64 << index;
        self.p &= !bit;
        self.u &= !bit;
        match trit {
            0 => self.u |= bit,
            1 => {
                self.p |= bit;
                self.u |= bit;
            }
            _ => {}
        }
        Ok(())
    }

    /// Bit mask of trits in the selected classes (1 = -1, 2 = 0, 4 = +1).
    #[inline]
    pub const fn class_mask(self, classes: u8) -> u64 {
        let mut mask = 0;
        if classes & POSITIVE != 0 {
            mask |= self.p;
        }
        if classes & NEGATIVE != 0 {
            mask |= !self.u;
        }
        if classes & ZERO != 0 {
            mask |= self.u & !self.p;
        }
        mask
    }

    #[inline]
    pub const fn count(self, trit: i8) -> Result<u32, TrnMapError> {
        let class = match trit {
            -1 => !self.u,
            0 => self.u & !self.p,
            1 => self.p,
            value => return Err(TrnMapError::InvalidTrit(value)),
        };
        Ok(class.count_ones())
    }

    /// Kleene AND (`min`).
    #[inline]
    pub const fn kleene_and(self, rhs: Self) -> Self {
        Self::from_planes(self.p & rhs.p, self.u & rhs.u)
    }

    /// Kleene OR (`max`).
    #[inline]
    pub const fn kleene_or(self, rhs: Self) -> Self {
        Self::from_planes(self.p | rhs.p, self.u | rhs.u)
    }

    /// Kleene NOT.
    #[inline]
    pub const fn kleene_not(self) -> Self {
        Self::from_planes(!self.u, !self.p)
    }

    /// Ternary dot product against a bit selection.
    #[inline]
    pub const fn dot(self, selected: u64) -> i32 {
        (self.p & selected).count_ones() as i32 - ((!self.u) & selected).count_ones() as i32
    }
}

#[derive(Clone, Copy, Default)]
struct LevelLayout {
    elements: usize,
    presence_offset: usize,
    presence_records: usize,
    dirty_offset: usize,
    dirty_words: usize,
}

#[derive(Clone, Copy)]
struct Layout {
    levels: [LevelLayout; MAX_LEVELS],
    level_count: usize,
    base_words: usize,
    presence_base: usize,
    dirty_base: usize,
    total_words: usize,
    storage_bytes: usize,
    budget_bytes: usize,
}

impl Layout {
    fn for_len(n: usize) -> Result<Self, TrnMapError> {
        if n == 0 {
            return Err(TrnMapError::EmptyMap);
        }
        let mut levels = [LevelLayout::default(); MAX_LEVELS];
        levels[0].elements = n;
        let mut level_count = 1;
        while levels[level_count - 1].elements != 1 {
            if level_count == MAX_LEVELS {
                return Err(TrnMapError::SizeOverflow);
            }
            levels[level_count].elements = div_ceil_64(levels[level_count - 1].elements);
            level_count += 1;
        }

        let mut presence_records = 0usize;
        let mut dirty_words = 0usize;
        for (level, descriptor) in levels[..level_count].iter_mut().enumerate() {
            descriptor.dirty_offset = dirty_words;
            descriptor.dirty_words = div_ceil_64(descriptor.elements);
            dirty_words = dirty_words
                .checked_add(descriptor.dirty_words)
                .ok_or(TrnMapError::SizeOverflow)?;
            if level != 0 {
                descriptor.presence_offset = presence_records;
                descriptor.presence_records = div_ceil_64(descriptor.elements);
                presence_records = presence_records
                    .checked_add(descriptor.presence_records)
                    .ok_or(TrnMapError::SizeOverflow)?;
            }
        }

        let base_words = div_ceil_64(n);
        let presence_words = presence_records
            .checked_mul(3)
            .ok_or(TrnMapError::SizeOverflow)?;
        let dirty_base = HEADER_WORDS
            .checked_add(base_words.checked_mul(2).ok_or(TrnMapError::SizeOverflow)?)
            .and_then(|v| v.checked_add(presence_words))
            .ok_or(TrnMapError::SizeOverflow)?;
        let total_words = dirty_base
            .checked_add(dirty_words)
            .ok_or(TrnMapError::SizeOverflow)?;
        let storage_bytes = total_words
            .checked_mul(8)
            .ok_or(TrnMapError::SizeOverflow)?;
        if storage_bytes > isize::MAX as usize {
            return Err(TrnMapError::SizeOverflow);
        }
        let budget_bytes = storage_bytes
            .checked_add(7)
            .ok_or(TrnMapError::SizeOverflow)?;
        let presence_base = HEADER_WORDS + base_words * 2;
        Ok(Self {
            levels,
            level_count,
            base_words,
            presence_base,
            dirty_base,
            total_words,
            storage_bytes,
            budget_bytes,
        })
    }
}

enum Backing<'a> {
    Owned(Box<[u64]>),
    Borrowed(&'a mut [u64]),
}

impl Backing<'_> {
    #[inline]
    fn words(&self) -> &[u64] {
        match self {
            Self::Owned(words) => words,
            Self::Borrowed(words) => words,
        }
    }

    #[inline]
    fn words_mut(&mut self) -> &mut [u64] {
        match self {
            Self::Owned(words) => words,
            Self::Borrowed(words) => words,
        }
    }
}

/// A 64-way trit map with exact class summaries and hierarchical dirty bits.
///
/// [`TrnMap::new`] owns one contiguous allocation. [`TrnMap::in_buffer`] carves
/// the same layout from caller-owned bytes, requiring `bytes_for(n)` bytes to
/// account for up to seven bytes of alignment slack. New maps contain only -1
/// and have no dirty trits.
pub struct TrnMap<'a> {
    backing: Backing<'a>,
    len: usize,
    level_count: usize,
    base_words: usize,
    trit_base: usize,
    presence_base: usize,
    dirty_base: usize,
}

impl fmt::Debug for TrnMap<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrnMap")
            .field("len", &self.len)
            .field("levels", &self.level_count)
            .field("base_words", &self.base_words)
            .field("storage_bytes", &self.storage_bytes())
            .field("owned", &matches!(&self.backing, Backing::Owned(_)))
            .finish()
    }
}

impl TrnMap<'static> {
    /// Creates an owned map of `len` trits, initialized to -1.
    pub fn new(len: usize) -> Result<Self, TrnMapError> {
        let layout = Layout::for_len(len)?;
        let mut words = vec![0u64; layout.total_words].into_boxed_slice();
        initialize_storage(&mut words, len, layout, true);
        Ok(Self::from_backing(Backing::Owned(words), len, layout))
    }

    /// Exact in-block allocation size, excluding the caller's alignment slack.
    pub fn storage_bytes_for(len: usize) -> Result<usize, TrnMapError> {
        Ok(Layout::for_len(len)?.storage_bytes)
    }

    /// Caller budget: exact in-block size plus seven bytes for 8-byte alignment.
    pub fn bytes_for(len: usize) -> Result<usize, TrnMapError> {
        Ok(Layout::for_len(len)?.budget_bytes)
    }
}

impl<'a> TrnMap<'a> {
    /// Initializes a map in caller-owned bytes, like Symbolic std's window form.
    /// The map borrows the backing storage until it is dropped.
    pub fn in_buffer(buffer: &'a mut [u8], len: usize) -> Result<Self, TrnMapError> {
        let layout = Layout::for_len(len)?;
        if buffer.len() < layout.budget_bytes {
            return Err(TrnMapError::BufferTooSmall {
                needed: layout.budget_bytes,
                available: buffer.len(),
            });
        }
        let available = buffer.len();
        // SAFETY: u64 accepts every bit pattern; only the aligned middle is used,
        // and the exclusive borrow of `buffer` is retained by the returned map.
        let (_, aligned, _) = unsafe { buffer.align_to_mut::<u64>() };
        if aligned.len() < layout.total_words {
            return Err(TrnMapError::BufferTooSmall {
                needed: layout.budget_bytes,
                available,
            });
        }
        let words = &mut aligned[..layout.total_words];
        initialize_storage(words, len, layout, false);
        Ok(Self::from_backing(Backing::Borrowed(words), len, layout))
    }

    fn from_backing(backing: Backing<'a>, len: usize, layout: Layout) -> Self {
        Self {
            backing,
            len,
            level_count: layout.level_count,
            base_words: layout.base_words,
            trit_base: HEADER_WORDS,
            presence_base: layout.presence_base,
            dirty_base: layout.dirty_base,
        }
    }

    /// Number of trits.
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Number of 64-trit leaf words.
    #[inline]
    pub const fn word_len(&self) -> usize {
        self.base_words
    }

    /// Number of 64-way hierarchy levels, including the leaf level.
    #[inline]
    pub const fn levels(&self) -> usize {
        self.level_count
    }

    /// Exact size of the contiguous map block.
    #[inline]
    pub fn storage_bytes(&self) -> usize {
        self.backing.words()[2] as usize
    }

    /// Reads trit `index`.
    pub fn get(&self, index: usize) -> Result<i8, TrnMapError> {
        if index >= self.len {
            return Err(TrnMapError::IndexOutOfBounds {
                index,
                len: self.len,
            });
        }
        let word = self.read_leaf(index / 64);
        Ok(word.get((index & 63) as u32).expect("masked trit index"))
    }

    /// Writes trit `index`, updating only summary levels whose state changes.
    pub fn set(&mut self, index: usize, trit: i8) -> Result<(), TrnMapError> {
        if index >= self.len {
            return Err(TrnMapError::IndexOutOfBounds {
                index,
                len: self.len,
            });
        }
        if !(-1..=1).contains(&trit) {
            return Err(TrnMapError::InvalidTrit(trit));
        }
        let word_index = index / 64;
        let mut word = self.read_leaf(word_index);
        let old = word;
        word.set((index & 63) as u32, trit)?;
        if word != old {
            self.write_leaf(word_index, word);
            self.recompute_summaries(word_index);
        }
        Ok(())
    }

    /// Reads one packed group of 64 trits.
    pub fn word(&self, word: usize) -> Result<TriWord, TrnMapError> {
        if word >= self.base_words {
            return Err(TrnMapError::WordOutOfBounds {
                word,
                words: self.base_words,
            });
        }
        Ok(self.read_leaf(word))
    }

    /// Replaces one packed group and updates its summaries.
    pub fn put(&mut self, word: usize, value: TriWord) -> Result<(), TrnMapError> {
        if word >= self.base_words {
            return Err(TrnMapError::WordOutOfBounds {
                word,
                words: self.base_words,
            });
        }
        let valid = valid_bits(self.len.saturating_sub(word * 64).min(64));
        let value = TriWord::from_planes(value.p & valid, value.u & valid);
        if self.read_leaf(word) != value {
            self.write_leaf(word, value);
            self.recompute_summaries(word);
        }
        Ok(())
    }

    /// The map's trit if all entries are +1 or all are -1; otherwise zero.
    pub fn top(&self) -> i8 {
        if self.level_count == 1 {
            let word = self.read_leaf(0);
            let valid = valid_bits(self.len);
            if word.p & valid == valid {
                1
            } else if word.u & valid == 0 {
                -1
            } else {
                0
            }
        } else {
            let level = self.level_count - 1;
            let planes = self.read_presence(level, 0);
            let positive = planes[0] & 1 != 0;
            let negative = planes[1] & 1 != 0;
            let zero = planes[2] & 1 != 0;
            if !zero && !negative {
                1
            } else if !zero && !positive {
                -1
            } else {
                0
            }
        }
    }

    /// First index at or after `start` belonging to `classes`, or `len`.
    /// Class bits are 1 = -1, 2 = 0, 4 = +1. Runs in O(log_64(n)) word loads.
    pub fn find(&self, classes: u8, start: usize) -> usize {
        let classes = classes & ALL_CLASSES;
        if start >= self.len || classes == 0 {
            return self.len;
        }
        let leaf_index = start / 64;
        let leaf_mask = self.leaf_class_mask(leaf_index, classes) & (u64::MAX << (start & 63));
        if leaf_mask != 0 {
            return leaf_index * 64 + leaf_mask.trailing_zeros() as usize;
        }
        if self.level_count == 1 {
            return self.len;
        }

        // Climb one level per miss. Each summary word covers 64 child groups.
        let mut level = 1usize;
        let mut target = leaf_index + 1;
        while level < self.level_count {
            let record = target / 64;
            let record_count = self.level_elements(level).div_ceil(64);
            if record >= record_count {
                return self.len;
            }
            let mask =
                self.presence_class_mask(level, record, classes) & (u64::MAX << (target & 63));
            if mask != 0 {
                let mut child = record * 64 + mask.trailing_zeros() as usize;
                for lower in (1..level).rev() {
                    let lower_mask = self.presence_class_mask(lower, child, classes);
                    debug_assert_ne!(lower_mask, 0);
                    child = child * 64 + lower_mask.trailing_zeros() as usize;
                }
                let leaf = child;
                let leaf_mask = self.leaf_class_mask(leaf, classes);
                debug_assert_ne!(leaf_mask, 0);
                return leaf * 64 + leaf_mask.trailing_zeros() as usize;
            }
            target = record + 1;
            level += 1;
        }
        self.len
    }

    /// Length of the equal-trit run beginning at `index`.
    pub fn run(&self, index: usize) -> Result<usize, TrnMapError> {
        let trit = self.get(index)?;
        let different = ALL_CLASSES ^ trit_class(trit);
        Ok(self.find(different, index) - index)
    }

    /// Count occurrences of one trit, using summary levels to skip empty groups.
    pub fn count(&self, trit: i8) -> Result<usize, TrnMapError> {
        let classes = trit_class_checked(trit)?;
        let mut count = 0usize;
        let mut start = 0usize;
        while start < self.len {
            let index = self.find(classes, start);
            if index == self.len {
                break;
            }
            let word_index = index / 64;
            count += self.leaf_class_mask(word_index, classes).count_ones() as usize;
            start = ((word_index + 1) * 64).min(self.len);
        }
        Ok(count)
    }

    /// Whether any trit is dirty. O(1).
    #[inline]
    pub fn dirty(&self) -> bool {
        let top = self.level_count - 1;
        self.read_dirty(top, 0) != 0
    }

    /// Marks one trit dirty and propagates only newly set summary bits.
    pub fn mark(&mut self, index: usize) -> Result<(), TrnMapError> {
        if index >= self.len {
            return Err(TrnMapError::IndexOutOfBounds {
                index,
                len: self.len,
            });
        }
        let dirty_word = index / 64;
        let leaf_bit = 1u64 << (index & 63);
        let old = self.read_dirty(0, dirty_word);
        if old & leaf_bit != 0 {
            return Ok(());
        }
        self.write_dirty(0, dirty_word, old | leaf_bit);
        let mut child = dirty_word;
        for level in 1..self.level_count {
            let record = child / 64;
            let bit = 1u64 << (child & 63);
            let old = self.read_dirty(level, record);
            if old & bit != 0 {
                break;
            }
            self.write_dirty(level, record, old | bit);
            child = record;
        }
        Ok(())
    }

    /// First leaf word at or after `word` containing dirty trits, or -1.
    pub fn next_dirty(&self, word: usize) -> isize {
        if word >= self.base_words {
            return -1;
        }
        if self.level_count == 1 {
            return if self.read_dirty(0, 0) != 0 { 0 } else { -1 };
        }
        let mut level = 1usize;
        let mut target = word;
        while level < self.level_count {
            let record = target / 64;
            let record_count = self.level_elements(level).div_ceil(64);
            if record >= record_count {
                return -1;
            }
            let mask = self.read_dirty(level, record) & (u64::MAX << (target & 63));
            if mask != 0 {
                let mut child = record * 64 + mask.trailing_zeros() as usize;
                for lower in (1..level).rev() {
                    let lower_mask = self.read_dirty(lower, child);
                    debug_assert_ne!(lower_mask, 0);
                    child = child * 64 + lower_mask.trailing_zeros() as usize;
                }
                return child as isize;
            }
            target = record + 1;
            level += 1;
        }
        -1
    }

    /// Returns and clears dirty bits for leaf word `word`, pruning empty ancestors.
    pub fn take_dirty(&mut self, word: usize) -> Result<u64, TrnMapError> {
        if word >= self.base_words {
            return Err(TrnMapError::WordOutOfBounds {
                word,
                words: self.base_words,
            });
        }
        let bits = self.read_dirty(0, word);
        if bits == 0 {
            return Ok(0);
        }
        self.write_dirty(0, word, 0);
        let mut child = word;
        for level in 1..self.level_count {
            let record = child / 64;
            let bit = 1u64 << (child & 63);
            let remaining = self.read_dirty(level, record) & !bit;
            self.write_dirty(level, record, remaining);
            if remaining != 0 {
                break;
            }
            child = record;
        }
        Ok(bits)
    }

    /// Kleene OR (`max`) in place, visiting only right-hand words with 0 or +1.
    pub fn or_assign(&mut self, rhs: &Self) -> Result<(), TrnMapError> {
        self.ensure_same_len(rhs)?;
        self.merge(rhs, true);
        Ok(())
    }

    /// Kleene AND (`min`) in place, visiting only right-hand words with -1 or 0.
    pub fn and_assign(&mut self, rhs: &Self) -> Result<(), TrnMapError> {
        self.ensure_same_len(rhs)?;
        self.merge(rhs, false);
        Ok(())
    }

    /// Kleene NOT in place; swaps positive/negative summaries and preserves dirty bits.
    pub fn not_assign(&mut self) {
        for word_index in 0..self.base_words {
            let old = self.read_leaf(word_index);
            let valid = valid_bits(self.len.saturating_sub(word_index * 64).min(64));
            let inverted = TriWord::from_planes(!old.u & valid, !old.p & valid);
            self.write_leaf(word_index, inverted);
        }
        for level in 1..self.level_count {
            let records = self.level_elements(level).div_ceil(64);
            for record in 0..records {
                let at = self.presence_index(level, record);
                let words = self.backing.words_mut();
                words.swap(at, at + 1);
            }
        }
    }

    fn ensure_same_len(&self, rhs: &Self) -> Result<(), TrnMapError> {
        if self.len == rhs.len {
            Ok(())
        } else {
            Err(TrnMapError::LengthMismatch {
                left: self.len,
                right: rhs.len,
            })
        }
    }

    fn merge(&mut self, rhs: &Self, is_or: bool) {
        let classes = if is_or {
            ZERO | POSITIVE
        } else {
            NEGATIVE | ZERO
        };
        let mut start = 0usize;
        while start < self.len {
            let index = rhs.find(classes, start);
            if index == rhs.len {
                break;
            }
            let word_index = index / 64;
            let lhs_word = self.read_leaf(word_index);
            let rhs_word = rhs.read_leaf(word_index);
            let merged = if is_or {
                lhs_word.kleene_or(rhs_word)
            } else {
                lhs_word.kleene_and(rhs_word)
            };
            if merged != lhs_word {
                self.write_leaf(word_index, merged);
                self.recompute_summaries(word_index);
            }
            start = ((word_index + 1) * 64).min(self.len);
        }
    }

    fn level_elements(&self, level: usize) -> usize {
        self.backing.words()[DESCRIPTOR_WORD + 3 * level] as usize
    }

    #[inline]
    fn presence_index(&self, level: usize, record: usize) -> usize {
        let offset = self.backing.words()[DESCRIPTOR_WORD + 3 * level + 1] as usize;
        self.presence_base + (offset + record) * 3
    }

    #[inline]
    fn dirty_index(&self, level: usize, word: usize) -> usize {
        let offset = self.backing.words()[DESCRIPTOR_WORD + 3 * level + 2] as usize;
        self.dirty_base + offset + word
    }

    #[inline]
    fn read_leaf(&self, index: usize) -> TriWord {
        let words = self.backing.words();
        let at = self.trit_base + index * 2;
        TriWord::from_planes(words[at], words[at + 1])
    }

    #[inline]
    fn write_leaf(&mut self, index: usize, value: TriWord) {
        let at = self.trit_base + index * 2;
        let words = self.backing.words_mut();
        words[at] = value.p;
        words[at + 1] = value.u;
    }

    #[inline]
    fn read_presence(&self, level: usize, record: usize) -> [u64; 3] {
        let at = self.presence_index(level, record);
        let words = self.backing.words();
        [words[at], words[at + 1], words[at + 2]]
    }

    #[inline]
    fn presence_class_mask(&self, level: usize, record: usize, classes: u8) -> u64 {
        let planes = self.read_presence(level, record);
        let mut mask = 0;
        if classes & POSITIVE != 0 {
            mask |= planes[0];
        }
        if classes & NEGATIVE != 0 {
            mask |= planes[1];
        }
        if classes & ZERO != 0 {
            mask |= planes[2];
        }
        mask
    }

    #[inline]
    fn leaf_class_mask(&self, word: usize, classes: u8) -> u64 {
        let valid = valid_bits(self.len.saturating_sub(word * 64).min(64));
        self.read_leaf(word).class_mask(classes) & valid
    }

    fn recompute_summaries(&mut self, leaf_word: usize) {
        let mut child = leaf_word;
        for level in 1..self.level_count {
            let record = child / 64;
            let bit = 1u64 << (child & 63);
            let present = if level == 1 {
                let valid = valid_bits(self.len.saturating_sub(child * 64).min(64));
                let word = self.read_leaf(child);
                [
                    word.p & valid != 0,
                    !word.u & valid != 0,
                    word.u & !word.p & valid != 0,
                ]
            } else {
                let lower = self.read_presence(level - 1, child);
                [lower[0] != 0, lower[1] != 0, lower[2] != 0]
            };
            let at = self.presence_index(level, record);
            let words = self.backing.words_mut();
            let old = [
                words[at] & bit != 0,
                words[at + 1] & bit != 0,
                words[at + 2] & bit != 0,
            ];
            if old == present {
                break;
            }
            for plane in 0..3 {
                if present[plane] {
                    words[at + plane] |= bit;
                } else {
                    words[at + plane] &= !bit;
                }
            }
            child = record;
        }
    }

    #[inline]
    fn read_dirty(&self, level: usize, word: usize) -> u64 {
        self.backing.words()[self.dirty_index(level, word)]
    }

    #[inline]
    fn write_dirty(&mut self, level: usize, word: usize, value: u64) {
        let at = self.dirty_index(level, word);
        self.backing.words_mut()[at] = value;
    }
}

fn initialize_storage(words: &mut [u64], len: usize, layout: Layout, owned: bool) {
    words.fill(0);
    let trit_base = HEADER_WORDS;
    let presence_base = layout.presence_base;
    let dirty_base = layout.dirty_base;
    words[0] = len as u64;
    words[1] = layout.level_count as u64;
    words[2] = layout.storage_bytes as u64;
    words[3] = u64::from(owned);
    words[4] = words.as_mut_ptr().wrapping_add(trit_base) as usize as u64;
    words[5] = words.as_mut_ptr().wrapping_add(presence_base) as usize as u64;
    words[6] = words.as_mut_ptr().wrapping_add(dirty_base) as usize as u64;

    for (level, descriptor) in layout.levels[..layout.level_count].iter().enumerate() {
        let at = DESCRIPTOR_WORD + 3 * level;
        words[at] = descriptor.elements as u64;
        words[at + 1] = descriptor.presence_offset as u64;
        words[at + 2] = descriptor.dirty_offset as u64;
    }

    // Maps start all -1. For each level, one bit marks each valid child group.
    for level in 1..layout.level_count {
        let descriptor = layout.levels[level];
        let child_count = descriptor.elements;
        for record in 0..descriptor.presence_records {
            let count = child_count.saturating_sub(record * 64).min(64);
            let valid = valid_bits(count);
            let at = presence_base + (descriptor.presence_offset + record) * 3;
            words[at + 1] = valid;
        }
    }
}

#[inline]
const fn div_ceil_64(value: usize) -> usize {
    (value >> 6) + if value & 63 != 0 { 1 } else { 0 }
}

#[inline]
const fn valid_bits(count: usize) -> u64 {
    if count >= 64 {
        u64::MAX
    } else if count == 0 {
        0
    } else {
        (1u64 << count) - 1
    }
}

#[inline]
const fn trit_class(trit: i8) -> u8 {
    match trit {
        -1 => NEGATIVE,
        0 => ZERO,
        1 => POSITIVE,
        _ => 0,
    }
}

#[inline]
fn trit_class_checked(trit: i8) -> Result<u8, TrnMapError> {
    let class = trit_class(trit);
    if class == 0 {
        Err(TrnMapError::InvalidTrit(trit))
    } else {
        Ok(class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trit_word_matches_scalar_kleene_algebra() {
        for a in -1..=1 {
            for b in -1..=1 {
                let wa = TriWord::splat(a).unwrap();
                let wb = TriWord::splat(b).unwrap();
                assert_eq!(wa.kleene_and(wb).get(0), Some(a.min(b)));
                assert_eq!(wa.kleene_or(wb).get(0), Some(a.max(b)));
                assert_eq!(wa.kleene_not().get(0), Some(-a));
                assert_eq!(wa.count(a).unwrap(), 64);
            }
        }
        let mut word = TriWord::splat(-1).unwrap();
        for i in 0..64 {
            let trit = (i % 3) as i8 - 1;
            word.set(i, trit).unwrap();
            assert_eq!(word.get(i), Some(trit));
        }
        assert_eq!(word.dot(u64::MAX), (0..64).map(|i| i % 3 - 1).sum());
    }

    #[test]
    fn exact_layout_matches_symbolic_byte_budget() {
        let expected = [
            (1, 408, 415),
            (63, 440, 447),
            (64, 440, 447),
            (65, 496, 503),
            (70, 496, 503),
            (4095, 1984, 1991),
            (4096, 1984, 1991),
            (4097, 2072, 2079),
        ];
        for (len, storage, budget) in expected {
            assert_eq!(
                TrnMap::storage_bytes_for(len).unwrap(),
                storage,
                "len={len}"
            );
            assert_eq!(TrnMap::bytes_for(len).unwrap(), budget, "len={len}");
        }
        assert_eq!(TrnMap::new(0).unwrap_err(), TrnMapError::EmptyMap);
    }

    #[test]
    fn invalid_inputs_and_tail_padding_are_checked() {
        let mut map = TrnMap::new(65).unwrap();
        assert_eq!(
            map.get(65),
            Err(TrnMapError::IndexOutOfBounds { index: 65, len: 65 })
        );
        assert_eq!(
            map.set(65, 0),
            Err(TrnMapError::IndexOutOfBounds { index: 65, len: 65 })
        );
        assert_eq!(map.set(0, 2), Err(TrnMapError::InvalidTrit(2)));
        assert_eq!(
            map.mark(65),
            Err(TrnMapError::IndexOutOfBounds { index: 65, len: 65 })
        );
        assert!(matches!(map.count(2), Err(TrnMapError::InvalidTrit(2))));
        assert!(matches!(
            map.word(2),
            Err(TrnMapError::WordOutOfBounds { .. })
        ));

        map.put(1, TriWord::splat(1).unwrap()).unwrap();
        assert_eq!(map.get(64), Ok(1));
        assert_eq!(map.count(1), Ok(1));
        assert_eq!(map.top(), 0);

        let mut other = TrnMap::new(64).unwrap();
        assert!(matches!(
            map.or_assign(&other),
            Err(TrnMapError::LengthMismatch { .. })
        ));
        other.not_assign();

        let required = TrnMap::bytes_for(65).unwrap();
        let mut short_buffer = vec![0; required - 1];
        assert!(matches!(
            TrnMap::in_buffer(&mut short_buffer, 65),
            Err(TrnMapError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn hierarchy_boundaries_search_counts_and_dirty_are_exact() {
        for len in [1, 63, 64, 65, 4095, 4096, 4097, 262_145] {
            let mut map = TrnMap::new(len).unwrap();
            assert_eq!(map.top(), -1, "len={len}");
            assert_eq!(map.count(-1).unwrap(), len);
            assert_eq!(map.count(0).unwrap(), 0);
            assert_eq!(map.count(1).unwrap(), 0);
            assert_eq!(map.find(NEGATIVE, 0), 0);
            assert_eq!(map.find(ZERO | POSITIVE, 0), len);

            let points = [0, len / 2, len - 1];
            let mut expected = vec![-1; len];
            for (slot, &index) in points.iter().enumerate() {
                let trit = [0, 1, -1][slot];
                map.set(index, trit).unwrap();
                expected[index] = trit;
                assert_eq!(map.get(index), Ok(trit));
                assert_eq!(map.find(trit_class(trit), index), index);
                assert_eq!(map.find(trit_class(trit), index + 1), len);
            }
            let expected_top = if expected.iter().all(|&v| v == 1) {
                1
            } else if expected.iter().all(|&v| v == -1) {
                -1
            } else {
                0
            };
            assert_eq!(map.top(), expected_top, "len={len}");
            map.mark(points[0]).unwrap();
            map.mark(points[2]).unwrap();
            assert!(map.dirty());
            assert_eq!(map.next_dirty(0), (points[0] / 64) as isize);
            let first_word = points[0] / 64;
            assert_ne!(map.take_dirty(first_word).unwrap(), 0);
            let expected_next = if points[2] / 64 == first_word {
                -1
            } else {
                (points[2] / 64) as isize
            };
            assert_eq!(map.next_dirty(first_word + 1), expected_next);
            while map.next_dirty(0) >= 0 {
                let word = map.next_dirty(0) as usize;
                map.take_dirty(word).unwrap();
            }
            assert!(!map.dirty());
        }
    }

    #[test]
    fn randomized_operations_match_naive_map() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for len in [1, 63, 64, 65, 4095, 4096, 4097, 262_145] {
            let mut map = TrnMap::new(len).unwrap();
            let mut naive = vec![-1i8; len];
            let mut dirty = vec![false; len];
            for step in 0..1800 {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let index = (seed as usize) % len;
                let trit = ((seed >> 32) % 3) as i8 - 1;
                match (seed >> 60) as u8 & 7 {
                    0 | 1 => {
                        map.set(index, trit).unwrap();
                        naive[index] = trit;
                    }
                    2 => {
                        let classes = ((seed >> 48) as u8 & 7).max(1);
                        let start = (seed.rotate_left(17) as usize) % len;
                        let expected = naive
                            .iter()
                            .enumerate()
                            .skip(start)
                            .find(|(_, value)| classes & trit_class(**value) != 0)
                            .map_or(len, |(i, _)| i);
                        assert_eq!(map.find(classes, start), expected, "len={len}, step={step}");
                    }
                    3 => {
                        let expected = naive[index..]
                            .iter()
                            .take_while(|&&v| v == naive[index])
                            .count();
                        assert_eq!(map.run(index), Ok(expected), "len={len}, step={step}");
                    }
                    4 => {
                        map.mark(index).unwrap();
                        dirty[index] = true;
                    }
                    5 => {
                        let word = (seed.rotate_left(11) as usize) % map.word_len();
                        let expected = dirty
                            .iter()
                            .enumerate()
                            .skip(word * 64)
                            .take(64)
                            .enumerate()
                            .fold(0u64, |bits, (bit, (_, flag))| {
                                bits | (u64::from(*flag) << bit)
                            });
                        assert_eq!(
                            map.take_dirty(word).unwrap(),
                            expected,
                            "len={len}, step={step}"
                        );
                        for value in dirty.iter_mut().skip(word * 64).take(64) {
                            *value = false;
                        }
                    }
                    6 => {
                        let t = [-1, 0, 1][((seed >> 40) % 3) as usize];
                        assert_eq!(
                            map.count(t).unwrap(),
                            naive.iter().filter(|&&v| v == t).count()
                        );
                    }
                    _ => assert_eq!(map.get(index), Ok(naive[index])),
                }
                if step % 97 == 0 {
                    assert_eq!(map.dirty(), dirty.iter().any(|&v| v));
                    let start_word = (seed as usize) % map.word_len();
                    let expected = dirty
                        .chunks(64)
                        .enumerate()
                        .skip(start_word)
                        .find(|(_, chunk)| chunk.iter().any(|&v| v))
                        .map_or(-1, |(i, _)| i as isize);
                    assert_eq!(
                        map.next_dirty(start_word),
                        expected,
                        "len={len}, step={step}"
                    );
                    if len < 10_000 {
                        for (i, &value) in naive.iter().enumerate() {
                            assert_eq!(map.get(i), Ok(value), "len={len}, i={i}");
                        }
                    } else {
                        for i in (0..len).step_by(997) {
                            assert_eq!(map.get(i), Ok(naive[i]), "len={len}, i={i}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn merges_and_not_match_scalar_kleene_results() {
        for len in [1, 63, 64, 65, 4097] {
            let mut lhs = TrnMap::new(len).unwrap();
            let mut rhs = TrnMap::new(len).unwrap();
            let mut a = vec![-1i8; len];
            let mut b = vec![-1i8; len];
            for i in 0..len {
                let av = (i % 3) as i8 - 1;
                let bv = ((i * 7 + 1) % 3) as i8 - 1;
                lhs.set(i, av).unwrap();
                rhs.set(i, bv).unwrap();
                a[i] = av;
                b[i] = bv;
            }
            lhs.or_assign(&rhs).unwrap();
            for (i, value) in a.iter_mut().enumerate() {
                *value = (*value).max(b[i]);
                assert_eq!(lhs.get(i), Ok(*value), "OR len={len}, i={i}");
            }
            lhs.and_assign(&rhs).unwrap();
            for (i, value) in a.iter_mut().enumerate() {
                *value = (*value).min(b[i]);
                assert_eq!(lhs.get(i), Ok(*value), "AND len={len}, i={i}");
            }
            lhs.not_assign();
            for (i, value) in a.iter_mut().enumerate() {
                *value = -*value;
                assert_eq!(lhs.get(i), Ok(*value), "NOT len={len}, i={i}");
            }
            assert_eq!(
                lhs.count(-1).unwrap() + lhs.count(0).unwrap() + lhs.count(1).unwrap(),
                len
            );
        }
    }

    #[test]
    fn caller_buffer_alignment_and_canaries_are_preserved() {
        for offset in 0..8 {
            let len = 4097;
            let budget = TrnMap::bytes_for(len).unwrap();
            let mut backing = vec![0xa5; offset + budget + 16];
            {
                let mut map =
                    TrnMap::in_buffer(&mut backing[offset..offset + budget], len).unwrap();
                map.set(len - 1, 1).unwrap();
                map.mark(len - 1).unwrap();
                assert_eq!(map.get(len - 1), Ok(1));
                assert_eq!(map.next_dirty(0), (len / 64) as isize);
            }
            assert!(backing[..offset].iter().all(|&b| b == 0xa5));
            assert!(backing[offset + budget..].iter().all(|&b| b == 0xa5));
        }
    }
}
