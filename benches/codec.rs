use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lz4_flex::block::{compress_prepend_size, decompress_size_prepended};
use rnbw::{PlainEf, Rnbw, RnbwView};
use roaring::RoaringBitmap;
use std::hint::black_box;

#[path = "../src/datasets.rs"]
mod datasets;

fn codec_benchmarks(criterion: &mut Criterion) {
    for workload in ["uniform", "clustered", "dense", "morton-shell", "ecs"] {
        let values = datasets::make_dataset(workload, 1_000_000).expect("valid benchmark workload");
        let rnbw = Rnbw::encode(&values).expect("RNBW encode");
        let serialized = rnbw.as_bytes();
        let ef = PlainEf::encode(&values).expect("plain EF encode");
        let mut roaring = RoaringBitmap::new();
        roaring.extend(values.iter().copied());
        let raw_bytes = to_le_bytes(&values);
        let lz4 = compress_prepend_size(&raw_bytes);
        let raw_size = values.len() * std::mem::size_of::<u32>();
        eprintln!(
            "dataset={workload} keys={} bytes/key: rnbw={:.4} (+{} runtime index bytes) plain-ef={:.4} raw={:.4} roaring-serialized={:.4} lz4={:.4}",
            values.len(),
            rnbw.as_bytes().len() as f64 / values.len() as f64,
            rnbw.runtime_index_bytes(),
            ef.encoded_size() as f64 / values.len() as f64,
            raw_size as f64 / values.len() as f64,
            roaring.serialized_size() as f64 / values.len() as f64,
            lz4.len() as f64 / values.len() as f64,
        );

        let hits = sampled_hits(&values, 1024);
        let misses = sampled_misses(&values, 1024);
        let mut build_group = criterion.benchmark_group(format!("build/{workload}"));
        build_group.throughput(Throughput::Elements(values.len() as u64));
        build_group.bench_with_input(
            BenchmarkId::new("rnbw", values.len()),
            &values,
            |b, values| {
                b.iter(|| Rnbw::encode(black_box(values)).expect("RNBW build"));
            },
        );
        build_group.bench_with_input(
            BenchmarkId::new("plain-ef", values.len()),
            &values,
            |b, values| {
                b.iter(|| PlainEf::encode(black_box(values)).expect("EF build"));
            },
        );
        build_group.bench_with_input(
            BenchmarkId::new("raw-vec", values.len()),
            &values,
            |b, values| {
                b.iter(|| black_box(values).to_vec());
            },
        );
        build_group.bench_with_input(
            BenchmarkId::new("roaring", values.len()),
            &values,
            |b, values| {
                b.iter(|| {
                    let mut bitmap = RoaringBitmap::new();
                    bitmap.extend(black_box(values).iter().copied());
                    bitmap
                });
            },
        );
        build_group.bench_with_input(
            BenchmarkId::new("lz4-block", values.len()),
            &raw_bytes,
            |b, bytes| {
                b.iter(|| compress_prepend_size(black_box(bytes)));
            },
        );
        build_group.finish();

        let mut contains_group = criterion.benchmark_group(format!("contains/{workload}"));
        contains_group.throughput(Throughput::Elements(hits.len() as u64));
        contains_group.bench_function("rnbw/hits", |b| {
            b.iter(|| count_hits(&hits, |v| rnbw.contains(v)))
        });
        contains_group.bench_function("rnbw/misses", |b| {
            b.iter(|| count_hits(&misses, |v| rnbw.contains(v)))
        });
        contains_group.bench_function("plain-ef/hits", |b| {
            b.iter(|| count_hits(&hits, |v| ef.contains(v)))
        });
        contains_group.bench_function("plain-ef/misses", |b| {
            b.iter(|| count_hits(&misses, |v| ef.contains(v)))
        });
        contains_group.bench_function("raw-binary-search/hits", |b| {
            b.iter(|| count_hits(&hits, |v| values.binary_search(&v).is_ok()))
        });
        contains_group.bench_function("raw-binary-search/misses", |b| {
            b.iter(|| count_hits(&misses, |v| values.binary_search(&v).is_ok()))
        });
        contains_group.bench_function("roaring/hits", |b| {
            b.iter(|| count_hits(&hits, |v| roaring.contains(v)))
        });
        contains_group.bench_function("roaring/misses", |b| {
            b.iter(|| count_hits(&misses, |v| roaring.contains(v)))
        });
        contains_group.bench_function("lz4-decode-plus-search/hits", |b| {
            b.iter(|| {
                let decoded = decompress_size_prepended(&lz4).expect("LZ4 decode");
                let decoded = from_le_bytes(&decoded).expect("u32 bytes");
                count_hits(&hits, |v| decoded.binary_search(&v).is_ok())
            })
        });
        contains_group.bench_function("lz4-decode-plus-search/misses", |b| {
            b.iter(|| {
                let decoded = decompress_size_prepended(&lz4).expect("LZ4 decode");
                let decoded = from_le_bytes(&decoded).expect("u32 bytes");
                count_hits(&misses, |v| decoded.binary_search(&v).is_ok())
            })
        });
        contains_group.finish();

        let mut decode_group = criterion.benchmark_group(format!("decode/{workload}"));
        decode_group.throughput(Throughput::Elements(values.len() as u64));
        decode_group.bench_function("rnbw", |b| b.iter(|| black_box(&rnbw).decode()));
        decode_group.bench_function("rnbw-parse", |b| {
            b.iter(|| RnbwView::from_bytes(black_box(serialized)).expect("RNBW parse"))
        });
        decode_group.bench_function("rnbw-parse-plus-decode", |b| {
            b.iter(|| {
                RnbwView::from_bytes(black_box(serialized))
                    .expect("RNBW parse")
                    .decode()
            })
        });
        decode_group.bench_function("plain-ef", |b| b.iter(|| black_box(&ef).decode()));
        decode_group.bench_function("raw-copy", |b| b.iter(|| black_box(&values).clone()));
        decode_group.bench_function("roaring-iterate", |b| {
            b.iter(|| black_box(&roaring).iter().collect::<Vec<_>>())
        });
        decode_group.bench_function("lz4-block", |b| {
            b.iter(|| {
                let bytes = decompress_size_prepended(black_box(&lz4)).expect("LZ4 decode");
                from_le_bytes(&bytes).expect("u32 bytes")
            })
        });
        decode_group.finish();
    }
}

fn count_hits(probes: &[u32], mut contains: impl FnMut(u32) -> bool) -> usize {
    black_box(
        probes
            .iter()
            .filter(|&&probe| contains(black_box(probe)))
            .count(),
    )
}

fn sampled_hits(values: &[u32], count: usize) -> Vec<u32> {
    let stride = values.len().div_ceil(count);
    values.iter().step_by(stride).take(count).copied().collect()
}

fn sampled_misses(values: &[u32], count: usize) -> Vec<u32> {
    let mut misses = Vec::with_capacity(count);
    let mut candidate = 0u32;
    while misses.len() < count {
        if values.binary_search(&candidate).is_err() {
            misses.push(candidate);
        }
        candidate = candidate.wrapping_add(1);
    }
    misses
}

fn to_le_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn from_le_bytes(bytes: &[u8]) -> Result<Vec<u32>, &'static str> {
    if !bytes.len().is_multiple_of(4) {
        return Err("byte count is not divisible by four");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes([v[0], v[1], v[2], v[3]]))
        .collect())
}

criterion_group!(benches, codec_benchmarks);
criterion_main!(benches);
