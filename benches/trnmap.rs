use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rnbw::trnmap::{CLASS_NEGATIVE, CLASS_POSITIVE, CLASS_ZERO, TriWord, TrnMap};
use std::hint::black_box;

const SIZES: [usize; 5] = [64, 4096, 65_536, 1_000_000, 16_000_000];

#[derive(Clone, Copy)]
enum Shape {
    Negative,
    Zero,
    Uniform,
    Alternating,
    Clustered,
    Sparse,
}

impl Shape {
    const ALL: [Self; 6] = [
        Self::Negative,
        Self::Zero,
        Self::Uniform,
        Self::Alternating,
        Self::Clustered,
        Self::Sparse,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Negative => "all-negative",
            Self::Zero => "all-zero",
            Self::Uniform => "uniform-ternary",
            Self::Alternating => "alternating-negative-positive",
            Self::Clustered => "4096-trit-clusters",
            Self::Sparse => "positive-1-in-4096",
        }
    }

    fn value(self, index: usize) -> i8 {
        match self {
            Self::Negative => -1,
            Self::Zero => 0,
            Self::Uniform => {
                let mut value = (index as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
                value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                (((value ^ (value >> 31)) % 3) as i8) - 1
            }
            Self::Alternating => {
                if index & 1 == 0 {
                    -1
                } else {
                    1
                }
            }
            Self::Clustered => match (index / 4096) % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            },
            Self::Sparse => {
                if index & 4095 == 0 {
                    1
                } else {
                    -1
                }
            }
        }
    }

    fn query(self) -> (i8, usize) {
        match self {
            Self::Negative => (-1, 0),
            Self::Zero => (0, 0),
            Self::Uniform => (1, 17),
            Self::Alternating => (1, 17),
            Self::Clustered => (1, 1),
            Self::Sparse => (1, 1),
        }
    }

    fn absent_query(self) -> Option<(i8, usize)> {
        match self {
            Self::Negative | Self::Zero => Some((1, 1)),
            Self::Alternating | Self::Sparse => Some((0, 1)),
            Self::Uniform | Self::Clustered => None,
        }
    }
}

fn make_map(len: usize, shape: Shape) -> (TrnMap<'static>, Vec<i8>) {
    let mut map = TrnMap::new(len).expect("allocate map");
    let values: Vec<_> = (0..len).map(|index| shape.value(index)).collect();
    for word_index in 0..map.word_len() {
        let start = word_index * 64;
        let count = (len - start).min(64);
        let mut p = 0u64;
        let mut u = 0u64;
        for (bit, &value) in values[start..start + count].iter().enumerate() {
            if value >= 0 {
                u |= 1u64 << bit;
            }
            if value > 0 {
                p |= 1u64 << bit;
            }
        }
        map.put(word_index, TriWord::from_planes(p, u))
            .expect("fill map word");
    }
    (map, values)
}

fn random_starts(len: usize) -> Vec<usize> {
    let mut state = 0x243f_6a88_85a3_08d3u64;
    (0..1024)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state as usize % len
        })
        .collect()
}

fn trnmap_benchmarks(c: &mut Criterion) {
    eprintln!("trnmap cache/workload matrix; timings are from this runner only");
    for len in SIZES {
        let bytes = TrnMap::storage_bytes_for(len).expect("exact bytes");
        eprintln!(
            "trnmap len={len}: {bytes} B ({:.3} MiB), alignment budget={} B, levels={}, leaf words={}",
            bytes as f64 / (1024.0 * 1024.0),
            TrnMap::bytes_for(len).expect("caller bytes"),
            TrnMap::new(len).expect("level count").levels(),
            len.div_ceil(64),
        );
    }

    // Get and mutation cost across maps sized from one leaf to beyond LLC.
    for len in SIZES {
        let (map, values) = make_map(len, Shape::Uniform);
        let probe = len * 3 / 5;
        assert_eq!(map.get(probe).unwrap(), values[probe]);
        let expected = values
            .iter()
            .enumerate()
            .skip(probe)
            .find(|&(_, value)| *value == 1)
            .map_or(len, |(index, _)| index);
        assert_eq!(map.find(CLASS_POSITIVE, probe), expected);
        let mut group = c.benchmark_group(format!("trnmap/point/{len}"));
        group.throughput(Throughput::Elements(1));
        group.bench_function("get-packed", |b| {
            b.iter(|| black_box(map.get(black_box(probe)).expect("valid index")))
        });
        group.bench_function("get-unchecked", |b| {
            b.iter(|| {
                // SAFETY: the fixed benchmark probe is less than `len`.
                black_box(unsafe { map.get_unchecked(black_box(probe)) })
            })
        });
        group.bench_function("get-scalar-byte", |b| {
            b.iter(|| black_box(values[black_box(probe)]))
        });
        group.bench_function("find-one-value", |b| {
            b.iter(|| black_box(map.find(CLASS_POSITIVE, black_box(probe))))
        });
        group.bench_function("find-unchecked-one-value", |b| {
            b.iter(|| {
                // SAFETY: the fixed benchmark probe is less than `len`.
                black_box(unsafe { map.find_unchecked(CLASS_POSITIVE, black_box(probe)) })
            })
        });
        group.bench_function("find-scalar-scan", |b| {
            b.iter(|| {
                black_box(
                    values
                        .iter()
                        .enumerate()
                        .skip(black_box(probe))
                        .find(|&(_, value)| *value == 1)
                        .map_or(len, |(index, _)| index),
                )
            })
        });
        group.finish();

        if len == 1_000_000 {
            let indices = random_starts(len);
            let mut batch = c.benchmark_group("trnmap/batch-get-1k-random/1m-map");
            batch.throughput(Throughput::Elements(indices.len() as u64));
            batch.bench_function("packed-trnmap", |b| {
                b.iter(|| {
                    let mut sum = 0i64;
                    for &index in black_box(&indices) {
                        sum += i64::from(map.get(index).expect("valid random index"));
                    }
                    black_box(sum)
                })
            });
            batch.bench_function("packed-trnmap-unchecked", |b| {
                b.iter(|| {
                    let mut sum = 0i64;
                    for &index in black_box(&indices) {
                        // SAFETY: random_starts returns indexes in `0..len`.
                        sum += i64::from(unsafe { map.get_unchecked(index) });
                    }
                    black_box(sum)
                })
            });
            batch.bench_function("byte-array", |b| {
                b.iter(|| {
                    let mut sum = 0i64;
                    for &index in black_box(&indices) {
                        sum += i64::from(values[index]);
                    }
                    black_box(sum)
                })
            });
            batch.finish();

            let mut searches = c.benchmark_group("trnmap/batch-find-1k-random/1m-map");
            searches.throughput(Throughput::Elements(indices.len() as u64));
            searches.bench_function("hierarchical-positive", |b| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for &start in black_box(&indices) {
                        sum = sum.wrapping_add(map.find(CLASS_POSITIVE, start));
                    }
                    black_box(sum)
                })
            });
            searches.bench_function("hierarchical-positive-unchecked", |b| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for &start in black_box(&indices) {
                        // SAFETY: random_starts returns every start in `0..len`.
                        sum =
                            sum.wrapping_add(unsafe { map.find_unchecked(CLASS_POSITIVE, start) });
                    }
                    black_box(sum)
                })
            });
            searches.bench_function("hierarchical-positive-unroll4", |b| {
                b.iter(|| {
                    let mut sums = [0usize; 4];
                    for starts in black_box(&indices).chunks_exact(4) {
                        for lane in 0..4 {
                            sums[lane] =
                                sums[lane].wrapping_add(map.find(CLASS_POSITIVE, starts[lane]));
                        }
                    }
                    black_box(sums.into_iter().fold(0usize, usize::wrapping_add))
                })
            });
            searches.bench_function("hierarchical-positive-unroll4-explicit", |b| {
                b.iter(|| {
                    let mut sums = [0usize; 4];
                    for starts in black_box(&indices).chunks_exact(4) {
                        sums[0] = sums[0].wrapping_add(map.find(CLASS_POSITIVE, starts[0]));
                        sums[1] = sums[1].wrapping_add(map.find(CLASS_POSITIVE, starts[1]));
                        sums[2] = sums[2].wrapping_add(map.find(CLASS_POSITIVE, starts[2]));
                        sums[3] = sums[3].wrapping_add(map.find(CLASS_POSITIVE, starts[3]));
                    }
                    black_box(sums.into_iter().fold(0usize, usize::wrapping_add))
                })
            });
            searches.bench_function("hierarchical-positive-unroll8-explicit", |b| {
                b.iter(|| {
                    let mut sums = [0usize; 8];
                    for starts in black_box(&indices).chunks_exact(8) {
                        sums[0] = sums[0].wrapping_add(map.find(CLASS_POSITIVE, starts[0]));
                        sums[1] = sums[1].wrapping_add(map.find(CLASS_POSITIVE, starts[1]));
                        sums[2] = sums[2].wrapping_add(map.find(CLASS_POSITIVE, starts[2]));
                        sums[3] = sums[3].wrapping_add(map.find(CLASS_POSITIVE, starts[3]));
                        sums[4] = sums[4].wrapping_add(map.find(CLASS_POSITIVE, starts[4]));
                        sums[5] = sums[5].wrapping_add(map.find(CLASS_POSITIVE, starts[5]));
                        sums[6] = sums[6].wrapping_add(map.find(CLASS_POSITIVE, starts[6]));
                        sums[7] = sums[7].wrapping_add(map.find(CLASS_POSITIVE, starts[7]));
                    }
                    black_box(sums.into_iter().fold(0usize, usize::wrapping_add))
                })
            });
            searches.bench_function("hierarchical-positive-unroll16-explicit", |b| {
                b.iter(|| {
                    let mut sums = [0usize; 16];
                    for starts in black_box(&indices).chunks_exact(16) {
                        sums[0] = sums[0].wrapping_add(map.find(CLASS_POSITIVE, starts[0]));
                        sums[1] = sums[1].wrapping_add(map.find(CLASS_POSITIVE, starts[1]));
                        sums[2] = sums[2].wrapping_add(map.find(CLASS_POSITIVE, starts[2]));
                        sums[3] = sums[3].wrapping_add(map.find(CLASS_POSITIVE, starts[3]));
                        sums[4] = sums[4].wrapping_add(map.find(CLASS_POSITIVE, starts[4]));
                        sums[5] = sums[5].wrapping_add(map.find(CLASS_POSITIVE, starts[5]));
                        sums[6] = sums[6].wrapping_add(map.find(CLASS_POSITIVE, starts[6]));
                        sums[7] = sums[7].wrapping_add(map.find(CLASS_POSITIVE, starts[7]));
                        sums[8] = sums[8].wrapping_add(map.find(CLASS_POSITIVE, starts[8]));
                        sums[9] = sums[9].wrapping_add(map.find(CLASS_POSITIVE, starts[9]));
                        sums[10] = sums[10].wrapping_add(map.find(CLASS_POSITIVE, starts[10]));
                        sums[11] = sums[11].wrapping_add(map.find(CLASS_POSITIVE, starts[11]));
                        sums[12] = sums[12].wrapping_add(map.find(CLASS_POSITIVE, starts[12]));
                        sums[13] = sums[13].wrapping_add(map.find(CLASS_POSITIVE, starts[13]));
                        sums[14] = sums[14].wrapping_add(map.find(CLASS_POSITIVE, starts[14]));
                        sums[15] = sums[15].wrapping_add(map.find(CLASS_POSITIVE, starts[15]));
                    }
                    black_box(sums.into_iter().fold(0usize, usize::wrapping_add))
                })
            });
            searches.bench_function("hierarchical-positive-unchecked-unroll16", |b| {
                b.iter(|| {
                    let mut sums = [0usize; 16];
                    for starts in black_box(&indices).chunks_exact(16) {
                        // SAFETY: random_starts returns every start in `0..len`.
                        sums[0] = sums[0]
                            .wrapping_add(unsafe { map.find_unchecked(CLASS_POSITIVE, starts[0]) });
                        sums[1] = sums[1]
                            .wrapping_add(unsafe { map.find_unchecked(CLASS_POSITIVE, starts[1]) });
                        sums[2] = sums[2]
                            .wrapping_add(unsafe { map.find_unchecked(CLASS_POSITIVE, starts[2]) });
                        sums[3] = sums[3]
                            .wrapping_add(unsafe { map.find_unchecked(CLASS_POSITIVE, starts[3]) });
                        sums[4] = sums[4]
                            .wrapping_add(unsafe { map.find_unchecked(CLASS_POSITIVE, starts[4]) });
                        sums[5] = sums[5]
                            .wrapping_add(unsafe { map.find_unchecked(CLASS_POSITIVE, starts[5]) });
                        sums[6] = sums[6]
                            .wrapping_add(unsafe { map.find_unchecked(CLASS_POSITIVE, starts[6]) });
                        sums[7] = sums[7]
                            .wrapping_add(unsafe { map.find_unchecked(CLASS_POSITIVE, starts[7]) });
                        sums[8] = sums[8]
                            .wrapping_add(unsafe { map.find_unchecked(CLASS_POSITIVE, starts[8]) });
                        sums[9] = sums[9]
                            .wrapping_add(unsafe { map.find_unchecked(CLASS_POSITIVE, starts[9]) });
                        sums[10] = sums[10].wrapping_add(unsafe {
                            map.find_unchecked(CLASS_POSITIVE, starts[10])
                        });
                        sums[11] = sums[11].wrapping_add(unsafe {
                            map.find_unchecked(CLASS_POSITIVE, starts[11])
                        });
                        sums[12] = sums[12].wrapping_add(unsafe {
                            map.find_unchecked(CLASS_POSITIVE, starts[12])
                        });
                        sums[13] = sums[13].wrapping_add(unsafe {
                            map.find_unchecked(CLASS_POSITIVE, starts[13])
                        });
                        sums[14] = sums[14].wrapping_add(unsafe {
                            map.find_unchecked(CLASS_POSITIVE, starts[14])
                        });
                        sums[15] = sums[15].wrapping_add(unsafe {
                            map.find_unchecked(CLASS_POSITIVE, starts[15])
                        });
                    }
                    black_box(sums.into_iter().fold(0usize, usize::wrapping_add))
                })
            });
            searches.bench_function("hierarchical-positive-unroll8", |b| {
                b.iter(|| {
                    let mut sums = [0usize; 8];
                    for starts in black_box(&indices).chunks_exact(8) {
                        for lane in 0..8 {
                            sums[lane] =
                                sums[lane].wrapping_add(map.find(CLASS_POSITIVE, starts[lane]));
                        }
                    }
                    black_box(sums.into_iter().fold(0usize, usize::wrapping_add))
                })
            });
            searches.bench_function("scalar-scan-positive", |b| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for &start in black_box(&indices) {
                        let found = values
                            .iter()
                            .enumerate()
                            .skip(start)
                            .find(|&(_, &value)| value == 1)
                            .map_or(len, |(index, _)| index);
                        sum = sum.wrapping_add(found);
                    }
                    black_box(sum)
                })
            });
            searches.finish();
        }

        let mut changing = TrnMap::new(len).expect("allocate mutable map");
        let mut flip = false;
        let mut set_group = c.benchmark_group("trnmap/set-one");
        set_group.bench_function(BenchmarkId::from_parameter(len), |b| {
            b.iter(|| {
                flip = !flip;
                black_box(changing.set(probe, if flip { 1 } else { -1 })).unwrap();
            })
        });
        set_group.finish();

        let mut dirty_map = TrnMap::new(len).expect("allocate dirty map");
        let index = len - 1;
        let mut dirty_group = c.benchmark_group("trnmap/dirty-mark-next-take");
        dirty_group.bench_function(BenchmarkId::from_parameter(len), |b| {
            b.iter(|| {
                dirty_map.mark(black_box(index)).unwrap();
                black_box(dirty_map.next_dirty(0));
                black_box(dirty_map.take_dirty(index / 64).unwrap());
            })
        });
        dirty_group.finish();
    }

    // Search and count through six occupancy shapes at each hierarchy/cache size.
    for len in SIZES {
        for shape in Shape::ALL {
            let (map, values) = make_map(len, shape);
            let (target, start) = shape.query();
            let class = match target {
                -1 => CLASS_NEGATIVE,
                0 => CLASS_ZERO,
                _ => CLASS_POSITIVE,
            };
            let expected_find = values
                .iter()
                .enumerate()
                .skip(start)
                .find(|&(_, value)| *value == target)
                .map_or(len, |(index, _)| index);
            let expected_count = values.iter().filter(|&&value| value == target).count();
            let run_start = start.min(len - 1);
            let run_expected = values[run_start..]
                .iter()
                .take_while(|&&value| value == values[run_start])
                .count();
            assert_eq!(
                map.find(class, start),
                expected_find,
                "find setup: {} n={len}",
                shape.name()
            );
            assert_eq!(
                map.count(target).unwrap(),
                expected_count,
                "count setup: {} n={len}",
                shape.name()
            );
            assert_eq!(map.run(run_start).unwrap(), run_expected);
            let mut group = c.benchmark_group(format!("trnmap/scenario/{}/{len}", shape.name()));
            group.throughput(Throughput::Elements(1));
            group.bench_function("find-hierarchical", |b| {
                b.iter(|| black_box(map.find(class, black_box(start))))
            });
            group.bench_function("find-scalar-scan", |b| {
                b.iter(|| {
                    black_box(
                        values
                            .iter()
                            .enumerate()
                            .skip(black_box(start))
                            .find(|&(_, value)| *value == target)
                            .map_or(len, |(index, _)| index),
                    )
                })
            });
            group.bench_function("run-hierarchical", |b| {
                b.iter(|| black_box(map.run(black_box(run_start)).unwrap()))
            });
            group.bench_function("run-scalar-scan", |b| {
                b.iter(|| {
                    black_box(
                        values[run_start..]
                            .iter()
                            .take_while(|&&value| value == values[run_start])
                            .count(),
                    )
                })
            });
            group.bench_function("count-hierarchical", |b| {
                b.iter(|| black_box(map.count(black_box(target)).unwrap()))
            });
            group.bench_function("count-scalar-scan", |b| {
                b.iter(|| black_box(values.iter().filter(|&&value| value == target).count()))
            });
            group.finish();

            let starts = random_starts(len);
            let mut group =
                c.benchmark_group(format!("trnmap/random-starts/{}/{len}", shape.name()));
            group.throughput(Throughput::Elements(starts.len() as u64));
            group.bench_function("find-hierarchical-1k", |b| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for &query_start in black_box(&starts) {
                        sum = sum.wrapping_add(map.find(class, query_start));
                    }
                    black_box(sum)
                })
            });
            group.bench_function("find-scalar-scan-1k", |b| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for &query_start in black_box(&starts) {
                        let found = values
                            .iter()
                            .enumerate()
                            .skip(query_start)
                            .find(|&(_, value)| *value == target)
                            .map_or(len, |(index, _)| index);
                        sum = sum.wrapping_add(found);
                    }
                    black_box(sum)
                })
            });
            group.finish();

            if let Some((absent, absent_start)) = shape.absent_query() {
                let absent_class = match absent {
                    -1 => CLASS_NEGATIVE,
                    0 => CLASS_ZERO,
                    _ => CLASS_POSITIVE,
                };
                assert_eq!(map.find(absent_class, absent_start), len);
                let mut group = c.benchmark_group(format!("trnmap/absent/{}/{len}", shape.name()));
                group.bench_function("find-hierarchical", |b| {
                    b.iter(|| black_box(map.find(absent_class, black_box(absent_start))))
                });
                group.bench_function("find-scalar-scan", |b| {
                    b.iter(|| {
                        black_box(
                            values
                                .iter()
                                .enumerate()
                                .skip(black_box(absent_start))
                                .find(|&(_, value)| *value == absent)
                                .map_or(len, |(index, _)| index),
                        )
                    })
                });
                group.finish();
            }
        }
    }

    // Dense/vector-style whole-map operations, with setup excluded from timings.
    let (rhs, _) = make_map(1_000_000, Shape::Clustered);
    c.bench_function("trnmap/merge-or-1m-clustered", |b| {
        b.iter_batched(
            || TrnMap::new(1_000_000).expect("allocate left map"),
            |mut lhs| {
                black_box(lhs.or_assign(black_box(&rhs))).unwrap();
                black_box(lhs.top());
            },
            BatchSize::SmallInput,
        )
    });
    c.bench_function("trnmap/merge-and-1m-clustered", |b| {
        b.iter_batched(
            || TrnMap::new(1_000_000).expect("allocate left map"),
            |mut lhs| {
                black_box(lhs.and_assign(black_box(&rhs))).unwrap();
                black_box(lhs.top());
            },
            BatchSize::SmallInput,
        )
    });
    let (mut inverted, _) = make_map(1_000_000, Shape::Uniform);
    c.bench_function("trnmap/not-assign-1m", |b| {
        b.iter(|| {
            inverted.not_assign();
            black_box(inverted.top());
        })
    });
    c.bench_function("trnmap/build-1m", |b| {
        b.iter(|| black_box(TrnMap::new(black_box(1_000_000)).expect("allocate map")))
    });
}

criterion_group!(benches, trnmap_benchmarks);
criterion_main!(benches);
