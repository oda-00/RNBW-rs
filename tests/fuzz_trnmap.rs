use rnbw::trnmap::{CLASS_NEGATIVE, CLASS_POSITIVE, CLASS_ZERO, TrnMap};

#[test]
fn deterministic_operation_fuzz_matches_naive_map() {
    let lengths = [
        1usize, 2, 63, 64, 65, 127, 128, 129, 4095, 4096, 4097, 8193, 65_535,
    ];
    let mut state = 0x6a09_e667_f3bc_c909u64;
    let mut operations = 0usize;

    for len in lengths {
        let mut map = TrnMap::new(len).expect("fuzz map allocation");
        let mut reference = vec![-1i8; len];
        let mut dirty = vec![false; len];

        for step in 0..2_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let index = state as usize % len;
            let trit = ((state >> 32) % 3) as i8 - 1;
            match (state >> 60) as u8 & 7 {
                0 | 1 => {
                    map.set(index, trit).expect("valid fuzz set");
                    reference[index] = trit;
                }
                2 => {
                    let classes = ((state >> 48) as u8) & 7;
                    let start = state.rotate_left(19) as usize % len;
                    let expected = reference
                        .iter()
                        .enumerate()
                        .skip(start)
                        .find(|(_, value)| classes & class(**value) != 0)
                        .map_or(len, |(found, _)| found);
                    assert_eq!(map.find(classes, start), expected, "len={len}, step={step}");
                }
                3 => {
                    map.mark(index).expect("valid fuzz mark");
                    dirty[index] = true;
                }
                4 => {
                    let from = state.rotate_left(7) as usize % map.word_len();
                    let expected = dirty
                        .chunks(64)
                        .enumerate()
                        .skip(from)
                        .find(|(_, chunk)| chunk.iter().any(|&is_dirty| is_dirty))
                        .map_or(-1, |(word, _)| word as isize);
                    assert_eq!(map.next_dirty(from), expected, "len={len}, step={step}");
                }
                5 => {
                    let word = state.rotate_left(29) as usize % map.word_len();
                    let expected = dirty
                        .iter()
                        .skip(word * 64)
                        .take(64)
                        .enumerate()
                        .fold(0u64, |bits, (bit, &is_dirty)| {
                            bits | (u64::from(is_dirty) << bit)
                        });
                    assert_eq!(map.take_dirty(word), Ok(expected), "len={len}, step={step}");
                    dirty
                        .iter_mut()
                        .skip(word * 64)
                        .take(64)
                        .for_each(|v| *v = false);
                }
                6 => {
                    let actual = reference[index];
                    let expected = reference[index..]
                        .iter()
                        .take_while(|&&v| v == actual)
                        .count();
                    assert_eq!(map.run(index), Ok(expected), "len={len}, step={step}");
                }
                _ => {
                    let trit = [-1, 0, 1][((state >> 40) % 3) as usize];
                    let expected = reference.iter().filter(|&&value| value == trit).count();
                    assert_eq!(map.count(trit), Ok(expected), "len={len}, step={step}");
                    assert_eq!(map.dirty(), dirty.iter().any(|&v| v));
                }
            }
            operations += 1;
        }

        for (index, &expected) in reference.iter().enumerate() {
            assert_eq!(
                map.get(index),
                Ok(expected),
                "final len={len}, index={index}"
            );
        }
        assert_eq!(map.dirty(), dirty.iter().any(|&v| v));
    }

    assert_eq!(operations, lengths.len() * 2_000);
}

#[test]
fn every_class_code_and_out_of_range_start_are_total() {
    let mut map = TrnMap::new(257).unwrap();
    for i in 0..257 {
        map.set(i, (i % 3) as i8 - 1).unwrap();
    }
    for classes in 0..=u8::MAX {
        let expected = (0..257)
            .find(|&i| classes & class(map.get(i).unwrap()) != 0)
            .unwrap_or(257);
        assert_eq!(map.find(classes, 0), expected, "classes={classes:#04x}");
        assert_eq!(map.find(classes, 257), 257);
        assert_eq!(map.find(classes, usize::MAX), 257);
    }
}

fn class(trit: i8) -> u8 {
    match trit {
        -1 => CLASS_NEGATIVE,
        0 => CLASS_ZERO,
        1 => CLASS_POSITIVE,
        _ => unreachable!(),
    }
}
