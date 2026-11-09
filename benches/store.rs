use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rnbw::{
    Rnbw,
    store::{CHUNK_COUNT, PocketLayout, RnbwStore},
};
use roaring::RoaringBitmap;
use std::hint::black_box;

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
    }
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
