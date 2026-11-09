use rnbw::{TriWord, TrnMap, TrnMapError};

/// Build one TRNMAP trit per little-endian 32-bit word in an opaque byte stream.
/// A trailing partial word is zero-padded. Trit classes are raw-word zero,
/// high-bit-set nonzero, and other nonzero; no file or tensor decoding occurs.
pub fn build_raw_word_trnmap(bytes: &[u8]) -> Result<TrnMap<'static>, TrnMapError> {
    let words = bytes.len().div_ceil(4);
    let mut map = TrnMap::new(words)?;
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime dispatch guarantees AVX2, and each vector load is
        // bounded to eight complete source words in the current leaf.
        unsafe { fill_avx2(&mut map, bytes)? };
        return Ok(map);
    }
    fill_scalar(&mut map, bytes)?;
    Ok(map)
}

#[inline]
pub fn raw_word_class(bytes: &[u8], position: usize) -> i8 {
    let at = position * 4;
    let mut raw = [0u8; 4];
    let count = bytes.len().saturating_sub(at).min(4);
    raw[..count].copy_from_slice(&bytes[at..at + count]);
    let word = u32::from_le_bytes(raw);
    if word & 0x7fff_ffff == 0 {
        0
    } else if word & 0x8000_0000 != 0 {
        -1
    } else {
        1
    }
}

pub fn scalar_find_class(bytes: &[u8], start: usize, trit: i8) -> usize {
    let words = bytes.len().div_ceil(4);
    (start..words)
        .find(|&position| raw_word_class(bytes, position) == trit)
        .unwrap_or(words)
}

pub fn scalar_count_class(bytes: &[u8], trit: i8) -> usize {
    (0..bytes.len().div_ceil(4))
        .filter(|&position| raw_word_class(bytes, position) == trit)
        .count()
}

fn fill_scalar(map: &mut TrnMap<'_>, bytes: &[u8]) -> Result<(), TrnMapError> {
    let words = bytes.len().div_ceil(4);
    for leaf in 0..words.div_ceil(64) {
        let start = leaf * 64;
        let count = (words - start).min(64);
        let mut p = 0u64;
        let mut u = 0u64;
        for bit in 0..count {
            let class = raw_word_class(bytes, start + bit);
            if class >= 0 {
                u |= 1u64 << bit;
            }
            if class > 0 {
                p |= 1u64 << bit;
            }
        }
        map.put(leaf, TriWord::from_planes(p, u))?;
    }
    Ok(())
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2")]
unsafe fn fill_avx2(map: &mut TrnMap<'_>, bytes: &[u8]) -> Result<(), TrnMapError> {
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_castsi256_ps, _mm256_cmpeq_epi32, _mm256_loadu_si256,
        _mm256_movemask_ps, _mm256_set1_epi32,
    };

    let words = bytes.len().div_ceil(4);
    let full_words = bytes.len() / 4;
    let zero = _mm256_set1_epi32(0);
    let magnitude_mask = _mm256_set1_epi32(0x7fff_ffff);
    for leaf in 0..words.div_ceil(64) {
        let start = leaf * 64;
        let count = (words - start).min(64);
        let mut p = 0u64;
        let mut u = 0u64;
        let mut bit = 0usize;
        while bit + 8 <= count && start + bit + 8 <= full_words {
            // SAFETY: `full_words` counts only complete four-byte words.
            let packed = unsafe {
                _mm256_loadu_si256(bytes.as_ptr().add((start + bit) * 4).cast::<__m256i>())
            };
            let signs = _mm256_movemask_ps(_mm256_castsi256_ps(packed)) as u32;
            let magnitudes = _mm256_and_si256(packed, magnitude_mask);
            let zeroes = _mm256_cmpeq_epi32(magnitudes, zero);
            let zeroes = _mm256_movemask_ps(_mm256_castsi256_ps(zeroes)) as u32;
            let positive = (!(signs | zeroes) & 0xff) as u64;
            let nonnegative = (positive as u32 | zeroes) as u64;
            p |= positive << bit;
            u |= nonnegative << bit;
            bit += 8;
        }
        while bit < count {
            let class = raw_word_class(bytes, start + bit);
            if class >= 0 {
                u |= 1u64 << bit;
            }
            if class > 0 {
                p |= 1u64 << bit;
            }
            bit += 1;
        }
        map.put(leaf, TriWord::from_planes(p, u))?;
    }
    Ok(())
}
