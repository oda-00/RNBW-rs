use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rnbw::{
    Rnbw,
    store::{CHUNK_COUNT, PocketLayout, RnbwStore},
};
use roaring::RoaringBitmap;
use std::hint::black_box;
use std::time::{Duration, Instant};

#[path = "../src/datasets.rs"]
mod datasets;

fn benchmarks(c: &mut Criterion) {
    eprintln!(
        "RNBW store benchmark; SIMD={}; 1,000,000 requested keys; Criterion optimized build",
        RnbwStore::simd_backend()
    );
    for name in ["uniform", "clustered", "dense", "morton-shell", "ecs"] {
        let values = datasets::make_dataset(name, 1_000_000).unwrap();
        let rnbw = Rnbw::encode(&values).unwrap();
        let mut roaring = RoaringBitmap::new();
        roaring.extend(values.iter().copied());
        roaring.optimize();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let hits: Vec<u32> = (0..1024)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                values[seed as usize % values.len()]
            })
            .collect();
        let mut misses = Vec::with_capacity(1024);
        while misses.len() < 1024 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let x = seed as u32;
            if values.binary_search(&x).is_err() {
                misses.push(x);
            }
        }
        let mut stores = Vec::new();
        for (label, layout) in [
            ("p32", PocketLayout::PerChunk32),
            ("p64", PocketLayout::PerPair64),
        ] {
            let store = RnbwStore::from_sorted(&values, layout).unwrap();
            let mut hot = RnbwStore::from_sorted(&values, layout).unwrap();
            let (promoted, spills) = hot.promote_range(0, CHUNK_COUNT).unwrap();
            let bytes = hot.l2_bytes() + hot.l3_bytes();
            eprintln!(
                "store dataset={name} keys={} layout={label} l2={}B l3={}B total={}B bits/key={:.3} promoted={} spills={}",
                values.len(),
                hot.l2_bytes(),
                hot.l3_bytes(),
                bytes,
                bytes as f64 * 8.0 / values.len() as f64,
                promoted,
                spills
            );
            assert_eq!(hot.decode().unwrap(), values, "store decode {name}/{label}");
            assert!(hits.iter().all(|&x| hot.contains(x).unwrap()));
            assert!(misses.iter().all(|&x| !hot.contains(x).unwrap()));
            stores.push((label, layout, store, hot));
        }
        let mut build = c.benchmark_group(format!("store-build/{name}"));
        build.throughput(Throughput::Elements(values.len() as u64));
        build.bench_function("rnbw-l2-p32", |b| {
            b.iter(|| RnbwStore::from_sorted(black_box(&values), PocketLayout::PerChunk32).unwrap())
        });
        build.bench_function("rnbw-l2-p64", |b| {
            b.iter(|| RnbwStore::from_sorted(black_box(&values), PocketLayout::PerPair64).unwrap())
        });
        build.bench_function("rnbw-query-image", |b| {
            b.iter(|| Rnbw::encode(black_box(&values)).unwrap())
        });
        build.finish();
        let mut access = c.benchmark_group(format!("store-contains/{name}"));
        access.throughput(Throughput::Elements(hits.len() as u64));
        access.bench_function("rnbw-image/hits", |b| {
            b.iter(|| {
                black_box(&hits)
                    .iter()
                    .filter(|&&x| rnbw.contains(x))
                    .count()
            })
        });
        access.bench_function("roaring/hits", |b| {
            b.iter(|| {
                black_box(&hits)
                    .iter()
                    .filter(|&&x| roaring.contains(x))
                    .count()
            })
        });
        access.bench_function("rnbw-image/misses", |b| {
            b.iter(|| {
                black_box(&misses)
                    .iter()
                    .filter(|&&x| rnbw.contains(x))
                    .count()
            })
        });
        access.bench_function("roaring/misses", |b| {
            b.iter(|| {
                black_box(&misses)
                    .iter()
                    .filter(|&&x| roaring.contains(x))
                    .count()
            })
        });
        for (label, _, l2, hot) in &stores {
            access.bench_function(BenchmarkId::new(format!("l2-{label}/hits"), "1024"), |b| {
                b.iter(|| {
                    black_box(&hits)
                        .iter()
                        .filter(|&&x| l2.contains(x).unwrap())
                        .count()
                })
            });
            access.bench_function(
                BenchmarkId::new(format!("l2-{label}/misses"), "1024"),
                |b| {
                    b.iter(|| {
                        black_box(&misses)
                            .iter()
                            .filter(|&&x| l2.contains(x).unwrap())
                            .count()
                    })
                },
            );
            access.bench_function(
                BenchmarkId::new(format!("l2-l3-{label}/hits"), "1024"),
                |b| {
                    b.iter(|| {
                        black_box(&hits)
                            .iter()
                            .filter(|&&x| hot.contains(x).unwrap())
                            .count()
                    })
                },
            );
            access.bench_function(
                BenchmarkId::new(format!("l2-l3-{label}/misses"), "1024"),
                |b| {
                    b.iter(|| {
                        black_box(&misses)
                            .iter()
                            .filter(|&&x| hot.contains(x).unwrap())
                            .count()
                    })
                },
            );
        }
        access.finish();
        let mut decode = c.benchmark_group(format!("store-decode/{name}"));
        decode.throughput(Throughput::Elements(values.len() as u64));
        for (label, _, l2, hot) in &stores {
            decode.bench_function(format!("l2-{label}"), |b| {
                b.iter(|| black_box(l2).decode().unwrap())
            });
            decode.bench_function(format!("l2-l3-{label}"), |b| {
                b.iter(|| black_box(hot).decode().unwrap())
            });
        }
        decode.finish();

        let insert_key = values
            .windows(2)
            .find_map(|pair| (pair[1] - pair[0] > 1).then_some(pair[0] + 1))
            .unwrap_or_else(|| values.last().copied().unwrap().checked_add(1).unwrap());
        let remove_key = values[values.len() / 2];
        for (label, layout, l2, hot) in &mut stores {
            benchmark_mutation_paths(
                c, name, label, *layout, &values, l2, hot, insert_key, remove_key,
            );
        }

        let mut promotion = c.benchmark_group(format!("store-promote/{name}"));
        promotion.throughput(Throughput::Elements(CHUNK_COUNT as u64));
        for (label, layout) in [
            ("p32", PocketLayout::PerChunk32),
            ("p64", PocketLayout::PerPair64),
        ] {
            let mut store = RnbwStore::from_sorted(&values, layout).unwrap();
            let expected = store.promote_range(0, CHUNK_COUNT).unwrap();
            assert_eq!(
                store.decode().unwrap(),
                values,
                "promoted decode {name}/{label}"
            );
            assert!(hits.iter().all(|&x| store.contains(x).unwrap()));
            assert!(misses.iter().all(|&x| !store.contains(x).unwrap()));
            black_box(store.dirty_chunks().unwrap());
            promotion.bench_function(format!("{label}/full-range-refresh"), |b| {
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for _ in 0..iterations {
                        let start = Instant::now();
                        let actual = store.promote_range(0, CHUNK_COUNT).unwrap();
                        elapsed += start.elapsed();
                        black_box(actual);
                    }
                    elapsed
                });
            });
            black_box(expected);
        }
        promotion.finish();
    }
    dense_trnmap_benchmarks(c);
}

fn dense_trnmap_benchmarks(c: &mut Criterion) {
    let values: Vec<u32> = (0..65_536).step_by(2).collect();
    let store = RnbwStore::from_sorted(&values, PocketLayout::PerChunk32).unwrap();
    let encoded = Rnbw::encode_width(&values, 16).unwrap();
    let mut roaring = RoaringBitmap::new();
    roaring.extend(values.iter().copied());
    roaring.optimize();
    let hits: Vec<u32> = values.iter().step_by(32).take(1024).copied().collect();
    let misses: Vec<u32> = (1..65_536).step_by(32).take(1024).collect();
    let total_bytes = store.l2_bytes() + store.l3_bytes();
    eprintln!(
        "store dense-trnmap keys={} l2={}B l3={}B total={}B bits/key={:.3} rnbw-image={}B roaring={}B",
        values.len(),
        store.l2_bytes(),
        store.l3_bytes(),
        total_bytes,
        total_bytes as f64 * 8.0 / values.len() as f64,
        encoded.as_bytes().len(),
        roaring.serialized_size()
    );

    let mut build = c.benchmark_group("store-dense-trnmap-build");
    build.throughput(Throughput::Elements(values.len() as u64));
    build.bench_function("rnbw-store", |b| {
        b.iter(|| RnbwStore::from_sorted(black_box(&values), PocketLayout::PerChunk32).unwrap())
    });
    build.bench_function("rnbw-image", |b| {
        b.iter(|| Rnbw::encode_width(black_box(&values), 16).unwrap())
    });
    build.bench_function("roaring", |b| {
        b.iter(|| {
            let mut bitmap = RoaringBitmap::new();
            bitmap.extend(black_box(&values).iter().copied());
            bitmap.optimize();
            bitmap
        })
    });
    build.finish();

    let mut access = c.benchmark_group("store-dense-trnmap-contains");
    access.throughput(Throughput::Elements(hits.len() as u64));
    access.bench_function("rnbw-store/hits-1024", |b| {
        b.iter(|| {
            black_box(&hits)
                .iter()
                .filter(|&&x| store.contains(x).unwrap())
                .count()
        })
    });
    access.bench_function("rnbw-image/hits-1024", |b| {
        b.iter(|| {
            black_box(&hits)
                .iter()
                .filter(|&&x| encoded.contains(x))
                .count()
        })
    });
    access.bench_function("roaring/hits-1024", |b| {
        b.iter(|| {
            black_box(&hits)
                .iter()
                .filter(|&&x| roaring.contains(x))
                .count()
        })
    });
    access.bench_function("rnbw-store/misses-1024", |b| {
        b.iter(|| {
            black_box(&misses)
                .iter()
                .filter(|&&x| store.contains(x).unwrap())
                .count()
        })
    });
    access.bench_function("rnbw-image/misses-1024", |b| {
        b.iter(|| {
            black_box(&misses)
                .iter()
                .filter(|&&x| encoded.contains(x))
                .count()
        })
    });
    access.bench_function("roaring/misses-1024", |b| {
        b.iter(|| {
            black_box(&misses)
                .iter()
                .filter(|&&x| roaring.contains(x))
                .count()
        })
    });
    access.finish();

    let mut update_store = RnbwStore::from_sorted(&values, PocketLayout::PerChunk32).unwrap();
    let mut update = c.benchmark_group("store-dense-trnmap-update");
    update.throughput(Throughput::Elements(2));
    update.bench_function("insert-remove-pair", |b| {
        b.iter(|| {
            black_box(update_store.insert(1).unwrap());
            black_box(update_store.remove(1).unwrap());
        })
    });
    update.finish();

    let mut decode = c.benchmark_group("store-dense-trnmap-decode");
    decode.throughput(Throughput::Elements(values.len() as u64));
    decode.bench_function("rnbw-store", |b| {
        b.iter(|| black_box(&store).decode().unwrap())
    });
    decode.bench_function("rnbw-image", |b| b.iter(|| black_box(&encoded).decode()));
    decode.bench_function("roaring", |b| {
        b.iter(|| black_box(&roaring).iter().collect::<Vec<_>>())
    });
    decode.finish();
}

#[allow(clippy::too_many_arguments)]
fn benchmark_mutation_paths(
    c: &mut Criterion,
    dataset: &str,
    layout_name: &str,
    layout: PocketLayout,
    values: &[u32],
    l2: &mut RnbwStore,
    l3: &mut RnbwStore,
    insert_key: u32,
    remove_key: u32,
) {
    for (tier, store) in [("l2", l2), ("l2-l3", l3)] {
        store.dirty_chunks().unwrap();
        let name = format!("store-update/{dataset}/{tier}-{layout_name}/insert-hot-reset");
        c.bench_function(&name, |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let start = Instant::now();
                    let inserted = store.insert(black_box(insert_key)).unwrap();
                    elapsed += start.elapsed();
                    black_box(inserted);
                    black_box(store.remove(insert_key).unwrap());
                    black_box(store.dirty_chunks().unwrap());
                }
                elapsed
            });
        });

        let name = format!("store-update/{dataset}/{tier}-{layout_name}/remove-hot-reset");
        c.bench_function(&name, |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let start = Instant::now();
                    let removed = store.remove(black_box(remove_key)).unwrap();
                    elapsed += start.elapsed();
                    black_box(removed);
                    black_box(store.insert(remove_key).unwrap());
                    black_box(store.dirty_chunks().unwrap());
                }
                elapsed
            });
        });

        let name = format!("store-update/{dataset}/{tier}-{layout_name}/dirty-drain-hot-reset");
        c.bench_function(&name, |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    black_box(store.insert(insert_key).unwrap());
                    let start = Instant::now();
                    let dirty = store.dirty_chunks().unwrap();
                    elapsed += start.elapsed();
                    black_box(dirty);
                    black_box(store.remove(insert_key).unwrap());
                    black_box(store.dirty_chunks().unwrap());
                }
                elapsed
            });
        });
    }

    let mut demote = RnbwStore::from_sorted(values, layout).unwrap();
    let (promoted, _) = demote.promote_range(0, CHUNK_COUNT).unwrap();
    demote.dirty_chunks().unwrap();
    let name = format!("store-demote/{dataset}/{layout_name}/full-range-hot-reset");
    if promoted == 0 {
        let mut group = c.benchmark_group(format!("store-demote-noop/{dataset}/{layout_name}"));
        const BATCH: usize = 256;
        group.throughput(Throughput::Elements(BATCH as u64));
        group.bench_function("full-range", |b| {
            b.iter(|| {
                for _ in 0..BATCH {
                    black_box(demote.demote_range(0, CHUNK_COUNT).unwrap());
                }
            });
        });
        group.finish();
    } else {
        c.bench_function(&name, |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let start = Instant::now();
                    let demoted = demote.demote_range(0, CHUNK_COUNT).unwrap();
                    elapsed += start.elapsed();
                    black_box(demoted);
                    black_box(demote.promote_range(0, CHUNK_COUNT).unwrap());
                    black_box(demote.dirty_chunks().unwrap());
                }
                elapsed
            });
        });
    }
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
