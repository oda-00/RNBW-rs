use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lz4_flex::block::{compress_prepend_size, decompress_size_prepended};
#[path = "support/lz4_index.rs"]
mod lz4_index;
use rnbw::{LaneLz, LaneLzEncoder, PlainEf, Rnbw, RnbwView};
use roaring::RoaringBitmap;
use std::hint::black_box;

use lz4_index::Lz4Index;

#[path = "../src/datasets.rs"]
mod datasets;
fn codec_benchmarks(criterion: &mut Criterion) {
    eprintln!(
        "benchmark environment: arch={} RNBW-SIMD={} rayon-workers={}",
        std::env::consts::ARCH,
        Rnbw::simd_backend(),
        rayon::current_num_threads()
    );
    for workload in ["uniform", "clustered", "dense", "morton-shell", "ecs"] {
        let values = datasets::make_dataset(workload, 1_000_000).expect("valid benchmark workload");
        let rnbw = Rnbw::encode(&values).expect("RNBW encode");
        let serialized = rnbw.as_bytes();
        let entropy = rnbw
            .to_compressed_bytes()
            .expect("RNBW hybrid entropy encode");
        let lane_lz = LaneLz::encode_u32_byte_planes(&values).expect("RNBW lane LZ encode");
        let ef = PlainEf::encode(&values).expect("plain EF encode");
        let delta_varint = delta_varint_encode(&values).expect("delta-varint encode");
        let mut roaring = RoaringBitmap::new();
        roaring.extend(values.iter().copied());
        roaring.optimize();
        let lz4 = Lz4Index::encode(&values).expect("LZ4 indexed blocks");
        let lz4_bytes = lz4.to_bytes();
        let lz4_restored = Lz4Index::from_bytes(&lz4_bytes).expect("LZ4 index parse");
        let raw_size = values.len() * std::mem::size_of::<u32>();
        let restored = Rnbw::from_compressed_bytes(&entropy).expect("RNBW hybrid entropy decode");
        assert_eq!(rnbw.decode(), values, "RNBW decode: {workload}");
        assert_eq!(
            restored.decode(),
            values,
            "RNBW entropy roundtrip: {workload}"
        );
        assert_eq!(
            lane_lz
                .decode_u32_byte_planes()
                .expect("RNBW lane LZ decode"),
            values,
            "RNBW lane LZ roundtrip: {workload}"
        );
        assert_eq!(ef.decode(), values, "Plain EF decode: {workload}");
        assert_eq!(
            delta_varint_decode(&delta_varint).expect("delta-varint decode"),
            values,
            "delta-varint roundtrip: {workload}"
        );
        let delta_values = delta_varint_decode(&delta_varint).expect("delta-varint indexed values");
        assert_eq!(
            roaring.iter().collect::<Vec<_>>(),
            values,
            "Roaring decode: {workload}"
        );
        assert_eq!(
            lz4.decode().expect("LZ4 decode"),
            values,
            "LZ4 decode: {workload}"
        );
        assert_eq!(lz4_restored.len(), values.len());
        assert!(!lz4_restored.is_empty());
        assert_eq!(lz4_restored.decode().expect("LZ4 parsed decode"), values);
        eprintln!(
            "dataset={workload} keys={} bytes/key: rnbw-query={:.4} rnbw-hybrid={:.4} lane-lz-hybrid={:.4} plain-ef={:.4} delta-varint={:.4} raw={:.4} roaring-run-optimized={:.4} lz4-64k-indexed={:.4} (+{} RNBW runtime index bytes)",
            values.len(),
            rnbw.as_bytes().len() as f64 / values.len() as f64,
            entropy.len() as f64 / values.len() as f64,
            lane_lz.as_bytes().len() as f64 / values.len() as f64,
            ef.encoded_size() as f64 / values.len() as f64,
            delta_varint.len() as f64 / values.len() as f64,
            raw_size as f64 / values.len() as f64,
            roaring.serialized_size() as f64 / values.len() as f64,
            lz4.encoded_size() as f64 / values.len() as f64,
            rnbw.runtime_index_bytes(),
        );

        let hits = sampled_hits(&values, 1024, workload_seed(workload));
        let misses = sampled_misses(
            &values,
            1024,
            workload_seed(workload) ^ 0xa076_1d64_78bd_642f,
        );
        assert_eq!(count_hits(&hits, |v| rnbw.contains(v)), hits.len());
        assert_eq!(count_hits(&misses, |v| rnbw.contains(v)), 0);
        assert_eq!(count_hits(&hits, |v| ef.contains(v)), hits.len());
        assert_eq!(count_hits(&misses, |v| ef.contains(v)), 0);
        assert_eq!(count_hits(&hits, |v| roaring.contains(v)), hits.len());
        assert_eq!(count_hits(&misses, |v| roaring.contains(v)), 0);
        assert_eq!(
            count_hits(&hits, |v| lz4.contains(v).expect("LZ4 hit search")),
            hits.len()
        );
        assert_eq!(
            count_hits(&misses, |v| lz4.contains(v).expect("LZ4 miss search")),
            0
        );
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
            BenchmarkId::new("delta-varint", values.len()),
            &values,
            |b, values| {
                b.iter(|| delta_varint_encode(black_box(values)).expect("delta-varint encode"));
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
                    bitmap.optimize();
                    bitmap
                });
            },
        );
        build_group.bench_with_input(
            BenchmarkId::new("lz4-64k-indexed", values.len()),
            &values,
            |b, values| {
                b.iter(|| Lz4Index::encode(black_box(values)).expect("LZ4 indexed blocks"));
            },
        );
        build_group.bench_with_input(
            BenchmarkId::new("rnbw-hybrid-entropy", values.len()),
            &values,
            |b, values| {
                b.iter(|| {
                    Rnbw::encode(black_box(values))
                        .expect("RNBW encode")
                        .to_compressed_bytes()
                        .expect("RNBW hybrid entropy encode")
                });
            },
        );
        build_group.bench_with_input(
            BenchmarkId::new("rnbw-lane-lz-hybrid", values.len()),
            &values,
            |b, values| {
                b.iter(|| LaneLz::encode_u32_byte_planes(black_box(values)).expect("RNBW lane LZ"));
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
        contains_group.bench_function("delta-varint-decoded-binary-search/hits", |b| {
            b.iter(|| count_hits(&hits, |v| delta_values.binary_search(&v).is_ok()))
        });
        contains_group.bench_function("delta-varint-decoded-binary-search/misses", |b| {
            b.iter(|| count_hits(&misses, |v| delta_values.binary_search(&v).is_ok()))
        });
        contains_group.bench_function("delta-varint-decode-plus-search/hits", |b| {
            b.iter(|| {
                let decoded = delta_varint_decode(black_box(&delta_varint)).unwrap();
                count_hits(&hits, |v| decoded.binary_search(&v).is_ok())
            })
        });
        contains_group.bench_function("delta-varint-decode-plus-search/misses", |b| {
            b.iter(|| {
                let decoded = delta_varint_decode(black_box(&delta_varint)).unwrap();
                count_hits(&misses, |v| decoded.binary_search(&v).is_ok())
            })
        });
        contains_group.bench_function("roaring/hits", |b| {
            b.iter(|| count_hits(&hits, |v| roaring.contains(v)))
        });
        contains_group.bench_function("roaring/misses", |b| {
            b.iter(|| count_hits(&misses, |v| roaring.contains(v)))
        });
        contains_group.bench_function("lz4-64k-block-plus-search/hits", |b| {
            b.iter(|| {
                count_hits(&hits, |v| {
                    black_box(&lz4).contains(v).expect("LZ4 hit search")
                })
            })
        });
        contains_group.bench_function("lz4-64k-block-plus-search/misses", |b| {
            b.iter(|| {
                count_hits(&misses, |v| {
                    black_box(&lz4).contains(v).expect("LZ4 miss search")
                })
            })
        });
        contains_group.bench_function("rnbw-hybrid-restore-plus-search/hits", |b| {
            b.iter(|| {
                let restored = Rnbw::from_compressed_bytes(black_box(&entropy))
                    .expect("RNBW hybrid entropy decode");
                count_hits(&hits, |v| restored.contains(v))
            })
        });
        contains_group.bench_function("rnbw-hybrid-restore-plus-search/misses", |b| {
            b.iter(|| {
                let restored = Rnbw::from_compressed_bytes(black_box(&entropy))
                    .expect("RNBW hybrid entropy decode");
                count_hits(&misses, |v| restored.contains(v))
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
        decode_group.bench_function("delta-varint", |b| {
            b.iter(|| delta_varint_decode(black_box(&delta_varint)).expect("delta-varint decode"))
        });
        decode_group.bench_function("raw-copy", |b| b.iter(|| black_box(&values).clone()));
        decode_group.bench_function("roaring-iterate", |b| {
            b.iter(|| black_box(&roaring).iter().collect::<Vec<_>>())
        });
        decode_group.bench_function("lz4-64k-indexed", |b| {
            b.iter(|| black_box(&lz4).decode().expect("LZ4 decode"))
        });
        decode_group.bench_function("rnbw-hybrid-entropy", |b| {
            b.iter(|| {
                Rnbw::from_compressed_bytes(black_box(&entropy))
                    .expect("RNBW hybrid entropy decode")
                    .decode()
            })
        });
        decode_group.bench_function("rnbw-lane-lz-hybrid", |b| {
            b.iter(|| {
                black_box(&lane_lz)
                    .decode_u32_byte_planes()
                    .expect("RNBW lane LZ decode")
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

fn sampled_hits(values: &[u32], count: usize, mut state: u64) -> Vec<u32> {
    (0..count)
        .map(|_| values[(next_random(&mut state) as usize) % values.len()])
        .collect()
}

fn sampled_misses(values: &[u32], count: usize, mut state: u64) -> Vec<u32> {
    let mut misses = Vec::with_capacity(count);
    while misses.len() < count {
        let candidate = next_random(&mut state) as u32;
        if values.binary_search(&candidate).is_err() {
            misses.push(candidate);
        }
    }
    misses
}

fn workload_seed(workload: &str) -> u64 {
    workload.bytes().fold(0x9e37_79b9_7f4a_7c15, |state, byte| {
        state.rotate_left(7) ^ byte as u64
    })
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn delta_varint_encode(values: &[u32]) -> Result<Vec<u8>, &'static str> {
    if values.len() > u32::MAX as usize || values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("delta-varint expects strictly increasing u32 values");
    }
    let mut encoded = Vec::with_capacity(8 + values.len() * 2);
    encoded.extend_from_slice(&(values.len() as u32).to_le_bytes());
    if let Some((&first, rest)) = values.split_first() {
        encoded.extend_from_slice(&first.to_le_bytes());
        let mut previous = first;
        for &value in rest {
            let mut delta = value - previous;
            while delta >= 0x80 {
                encoded.push((delta as u8) | 0x80);
                delta >>= 7;
            }
            encoded.push(delta as u8);
            previous = value;
        }
    }
    Ok(encoded)
}

fn delta_varint_decode(bytes: &[u8]) -> Result<Vec<u32>, &'static str> {
    let count_bytes = bytes.get(..4).ok_or("truncated delta-varint count")?;
    let count = u32::from_le_bytes(count_bytes.try_into().unwrap()) as usize;
    if count == 0 {
        return if bytes.len() == 4 {
            Ok(Vec::new())
        } else {
            Err("trailing bytes after empty delta-varint")
        };
    }
    if count > bytes.len().saturating_sub(8) + 1 {
        return Err("delta-varint count exceeds available payload");
    }
    let first_bytes = bytes.get(4..8).ok_or("truncated delta-varint first key")?;
    let mut values = Vec::with_capacity(count);
    let mut value = u32::from_le_bytes(first_bytes.try_into().unwrap());
    values.push(value);
    let mut cursor = 8usize;
    while values.len() < count {
        let start = cursor;
        let mut shift = 0u32;
        let mut delta = 0u32;
        loop {
            let byte = *bytes.get(cursor).ok_or("truncated delta-varint value")?;
            cursor += 1;
            if shift == 28 && (byte & 0x70) != 0 {
                return Err("delta-varint value overflows u32");
            }
            delta |= u32::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                if cursor - start > 1 && byte == 0 {
                    return Err("noncanonical delta-varint value");
                }
                break;
            }
            shift += 7;
            if shift > 28 {
                return Err("delta-varint value is too long");
            }
        }
        if delta == 0 {
            return Err("delta-varint keys are not strictly increasing");
        }
        value = value
            .checked_add(delta)
            .ok_or("delta-varint key overflows u32")?;
        values.push(value);
    }
    if cursor != bytes.len() {
        return Err("trailing delta-varint bytes");
    }
    Ok(values)
}

fn lane_type_benchmarks(criterion: &mut Criterion) {
    for (name, bytes, lane_width) in synthetic_lane_datasets() {
        let lane = LaneLz::encode_interleaved(&bytes, lane_width).expect("RNBW lane encode");
        let lz4_interleaved = compress_prepend_size(&bytes);
        let split = split_interleaved(&bytes, lane_width);
        let records = bytes.len() / lane_width;
        let lane_refs: Vec<&[u8]> = split.chunks_exact(records).collect();
        let lz4_split = compress_prepend_size(&split);
        assert_eq!(lane.decode_interleaved().expect("RNBW lane decode"), bytes);
        assert_eq!(decompress_size_prepended(&lz4_interleaved).unwrap(), bytes);
        assert_eq!(
            interleave_lanes(&decompress_size_prepended(&lz4_split).unwrap(), lane_width),
            bytes
        );
        let mut serial_encoder = LaneLzEncoder::default();
        assert_eq!(
            serial_encoder.encode(&lane_refs).unwrap().as_bytes(),
            lane.as_bytes(),
            "reused serial encoder output differs for {name}"
        );
        let mut parallel_encoder = LaneLzEncoder::default();
        assert_eq!(
            parallel_encoder
                .encode_parallel(&lane_refs)
                .unwrap()
                .as_bytes(),
            lane.as_bytes(),
            "reused parallel encoder output differs for {name}"
        );
        let mut interleaved_serial_encoder = LaneLzEncoder::default();
        assert_eq!(
            interleaved_serial_encoder
                .encode_interleaved(&bytes, lane_width)
                .unwrap()
                .as_bytes(),
            lane.as_bytes(),
            "reused interleaved serial output differs for {name}"
        );
        let mut interleaved_parallel_encoder = LaneLzEncoder::default();
        assert_eq!(
            interleaved_parallel_encoder
                .encode_interleaved_parallel(&bytes, lane_width)
                .unwrap()
                .as_bytes(),
            lane.as_bytes(),
            "reused interleaved parallel output differs for {name}"
        );
        eprintln!(
            "lane-dataset={name} raw={} RNBW-multilane={} ({:.3}x) LZ4-interleaved={} ({:.3}x) LZ4-SoA={} ({:.3}x)",
            bytes.len(),
            lane.as_bytes().len(),
            bytes.len() as f64 / lane.as_bytes().len() as f64,
            lz4_interleaved.len(),
            bytes.len() as f64 / lz4_interleaved.len() as f64,
            lz4_split.len(),
            bytes.len() as f64 / lz4_split.len() as f64,
        );

        let mut build = criterion.benchmark_group(format!("lane-build/{name}"));
        build.throughput(Throughput::Bytes(bytes.len() as u64));
        build.bench_function("rnbw-multilane", |b| {
            b.iter(|| LaneLz::encode_interleaved(black_box(&bytes), lane_width).unwrap())
        });
        build.bench_function("rnbw-multilane-reuse-serial", |b| {
            b.iter(|| serial_encoder.encode(black_box(&lane_refs)).unwrap())
        });
        build.bench_function("rnbw-multilane-reuse-parallel", |b| {
            b.iter(|| {
                parallel_encoder
                    .encode_parallel(black_box(&lane_refs))
                    .unwrap()
            })
        });
        build.bench_function("rnbw-interleaved-reuse-serial", |b| {
            b.iter(|| {
                interleaved_serial_encoder
                    .encode_interleaved(black_box(&bytes), lane_width)
                    .unwrap()
            })
        });
        build.bench_function("rnbw-interleaved-reuse-parallel", |b| {
            b.iter(|| {
                interleaved_parallel_encoder
                    .encode_interleaved_parallel(black_box(&bytes), lane_width)
                    .unwrap()
            })
        });
        build.bench_function("lz4-interleaved", |b| {
            b.iter(|| compress_prepend_size(black_box(&bytes)))
        });
        build.bench_function("lz4-SoA-split-plus-compress", |b| {
            b.iter(|| {
                let split = split_interleaved(black_box(&bytes), lane_width);
                compress_prepend_size(&split)
            })
        });
        build.bench_function("raw-copy", |b| b.iter(|| black_box(&bytes).clone()));
        build.finish();

        let mut decode = criterion.benchmark_group(format!("lane-decode/{name}"));
        decode.throughput(Throughput::Bytes(bytes.len() as u64));
        decode.bench_function("rnbw-multilane", |b| {
            b.iter(|| black_box(&lane).decode_interleaved().unwrap())
        });
        decode.bench_function("lz4-interleaved", |b| {
            b.iter(|| decompress_size_prepended(black_box(&lz4_interleaved)).unwrap())
        });
        decode.bench_function("lz4-SoA-decompress-plus-interleave", |b| {
            b.iter(|| {
                let split = decompress_size_prepended(black_box(&lz4_split)).unwrap();
                interleave_lanes(&split, lane_width)
            })
        });
        decode.bench_function("raw-copy", |b| b.iter(|| black_box(&bytes).clone()));
        decode.finish();
    }
}

fn synthetic_lane_datasets() -> Vec<(&'static str, Vec<u8>, usize)> {
    let mut datasets = Vec::new();

    // Deterministic 512x512 RGBA frame: smooth gradients, low-amplitude sensor noise,
    // and an opaque alpha channel. This is synthetic image data, not a captured asset.
    let mut rgba = Vec::with_capacity(512 * 512 * 4);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for y in 0..512u32 {
        for x in 0..512u32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state & 7) as u8;
            rgba.extend_from_slice(&[
                ((x >> 1) as u8).wrapping_add(noise),
                ((y >> 1) as u8).wrapping_add(noise),
                ((x ^ y) as u8).wrapping_add(noise),
                255,
            ]);
        }
    }
    datasets.push(("graphics-rgba8-synthetic", rgba, 4));

    // SPIR-V-like shader word fields: repeated opcodes, ids, and immediate operands.
    let mut shader = Vec::with_capacity(65_536 * 4);
    for index in 0..65_536u32 {
        let opcode: u32 =
            [0x0001, 0x0013, 0x0021, 0x0033, 0x0041, 0x006f][(index as usize * 17) % 6];
        let word = opcode | ((index & 0x3ff) << 16) | (((index * 13) & 0x7f) << 8);
        shader.extend_from_slice(&word.to_le_bytes());
    }
    datasets.push(("graphics-shader-words-synthetic", shader, 4));

    // Quantized model tensor blocks: 16 independently correlated output channels,
    // sparse near-zero weights, and repeated block patterns. Not a real model dump.
    let mut weights = Vec::with_capacity(1 << 20);
    for index in 0..(1 << 20usize) {
        let lane = index & 15;
        let position = index >> 4;
        let phase = (position / 256 + lane * 7) & 31;
        let value = if (position + lane * 11) % 8 < 5 {
            0i8
        } else {
            ((phase * 7 + lane * 3) % 63) as i8 - 31
        };
        weights.push(value as u8);
    }
    datasets.push(("ai-model-int8-weights-synthetic", weights, 16));

    // Transformer kernel instruction stream with repeated opcode/control fields.
    let mut kernel = Vec::with_capacity(65_536 * 4);
    for index in 0..65_536u32 {
        let opcode: u32 = [0x13, 0x33, 0x37, 0x63, 0x6f][(index as usize * 11) % 5];
        let register_fields = ((index & 31) << 7) | (((index * 3) & 31) << 15);
        let immediate = ((index / 64) & 0xfff) << 20;
        kernel.extend_from_slice(&(opcode | register_fields | immediate).to_le_bytes());
    }
    datasets.push(("ai-kernel-words-synthetic", kernel, 4));

    // Incompressible payload control case.
    let mut random = Vec::with_capacity(256 * 1024);
    let mut random_state = 0x726e_6277_6461_7461u64;
    for _ in 0..256 * 1024 {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 7;
        random_state ^= random_state << 17;
        random.push(random_state as u8);
    }
    datasets.push(("random-bytes-control", random, 1));
    datasets
}

fn split_interleaved(bytes: &[u8], lane_width: usize) -> Vec<u8> {
    assert!(lane_width > 0 && bytes.len().is_multiple_of(lane_width));
    let records = bytes.len() / lane_width;
    let mut output = vec![0u8; bytes.len()];
    for (record_index, record) in bytes.chunks_exact(lane_width).enumerate() {
        for (lane, byte) in record.iter().enumerate() {
            output[lane * records + record_index] = *byte;
        }
    }
    output
}

fn interleave_lanes(bytes: &[u8], lane_width: usize) -> Vec<u8> {
    assert!(lane_width > 0 && bytes.len().is_multiple_of(lane_width));
    let records = bytes.len() / lane_width;
    let mut output = vec![0u8; bytes.len()];
    for (record_index, record) in output.chunks_exact_mut(lane_width).enumerate() {
        for (lane, byte) in record.iter_mut().enumerate() {
            *byte = bytes[lane * records + record_index];
        }
    }
    output
}

criterion_group!(benches, codec_benchmarks, lane_type_benchmarks);
criterion_main!(benches);
