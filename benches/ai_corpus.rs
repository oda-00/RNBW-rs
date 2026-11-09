use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rnbw::{LaneLz, LaneLzEncoder, trnmap::CLASS_ZERO};
use std::{collections::BTreeSet, hint::black_box, time::Instant};

#[path = "support/trnmap_index.rs"]
mod trnmap_index;

fn benchmarks(c: &mut Criterion) {
    let path = std::env::var_os("RNBW_AI_CORPUS")
        .expect("set RNBW_AI_CORPUS to a real model artifact path");
    let path = std::path::PathBuf::from(path);
    let label = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model-artifact");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("read AI corpus {}: {error}", path.display()));
    assert!(!bytes.is_empty(), "AI corpus must not be empty");
    let (tensor_payload, dtypes) = safetensors_payload(&bytes);

    eprintln!(
        "real-ai-corpus={} bytes={} tensor-payload={} dtypes={dtypes:?}; model artifact remains opaque and is not tensor-decoded",
        path.display(),
        bytes.len(),
        tensor_payload.len()
    );
    match std::env::var("RNBW_AI_BENCH").as_deref().unwrap_or("all") {
        "codec" => benchmark_codec(c, label, &bytes),
        "index" => benchmark_undecoded_index(c, &format!("{label}-tensor-payload"), tensor_payload),
        "all" => {
            benchmark_codec(c, label, &bytes);
            benchmark_undecoded_index(c, &format!("{label}-tensor-payload"), tensor_payload);
        }
        other => panic!("RNBW_AI_BENCH must be codec, index, or all; got {other}"),
    }
}

fn safetensors_payload(bytes: &[u8]) -> (&[u8], Vec<String>) {
    assert!(bytes.len() >= 8, "truncated safetensors length prefix");
    let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap()))
        .expect("safetensors header length fits usize");
    let payload_start = 8usize
        .checked_add(header_len)
        .filter(|&end| end <= bytes.len())
        .expect("safetensors header range is valid");
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..payload_start]).expect("valid safetensors JSON header");
    let object = header.as_object().expect("safetensors header object");
    let mut spans = Vec::with_capacity(object.len());
    let mut dtypes = BTreeSet::new();
    for (name, tensor) in object {
        if name == "__metadata__" {
            continue;
        }
        dtypes.insert(tensor["dtype"].as_str().expect("tensor dtype").to_owned());
        let offsets = tensor["data_offsets"]
            .as_array()
            .filter(|offsets| offsets.len() == 2)
            .expect("tensor data offsets");
        let start = usize::try_from(offsets[0].as_u64().expect("offset start"))
            .expect("offset start fits usize");
        let end = usize::try_from(offsets[1].as_u64().expect("offset end"))
            .expect("offset end fits usize");
        assert!(start <= end, "tensor offset order");
        spans.push((start, end));
    }
    spans.sort_unstable();
    let mut covered = 0usize;
    for (start, end) in spans {
        assert_eq!(start, covered, "tensor payload has a gap or overlap");
        covered = end;
    }
    let payload = &bytes[payload_start..];
    assert_eq!(covered, payload.len(), "tensor offsets cover full payload");
    assert!(!dtypes.is_empty(), "safetensors file contains tensors");
    assert!(
        payload.len().is_multiple_of(4),
        "tensor payload length is word-aligned"
    );
    (payload, dtypes.into_iter().collect())
}

fn benchmark_codec(c: &mut Criterion, label: &str, bytes: &[u8]) {
    assert!(
        bytes.len().is_multiple_of(4),
        "four-byte lane corpus has a tail"
    );
    let raw_len = bytes.len();
    let codec = LaneLz::encode_interleaved(bytes, 4).expect("RNBW encode");
    let decoded = codec.decode_interleaved().expect("RNBW decode");
    assert_eq!(decoded, bytes, "RNBW real model roundtrip");
    drop(decoded);

    let mut serial = LaneLzEncoder::default();
    let serial_image = serial
        .encode_interleaved(bytes, 4)
        .expect("RNBW serial reuse encode");
    assert_eq!(serial_image.as_bytes(), codec.as_bytes());
    drop(serial_image);

    let mut parallel = LaneLzEncoder::default();
    let parallel_image = parallel
        .encode_interleaved_parallel(bytes, 4)
        .expect("RNBW parallel reuse encode");
    assert_eq!(
        parallel_image
            .decode_interleaved()
            .expect("parallel decode"),
        bytes
    );
    let parallel_size = parallel_image.as_bytes().len();
    drop(parallel_image);

    let lz4_size = lz4_flex::block::compress_prepend_size(bytes).len();
    let snappy_size = snap::raw::Encoder::new().compress_vec(bytes).unwrap().len();
    let zstd_size = zstd::bulk::compress(bytes, -3).unwrap().len();
    eprintln!(
        "real-ai-codec={label} raw={raw_len}B rnbw={}B ({:.4}) rnbw-parallel={}B lz4={lz4_size}B ({:.4}) snappy={snappy_size}B ({:.4}) zstd-fast={zstd_size}B ({:.4})",
        codec.as_bytes().len(),
        codec.as_bytes().len() as f64 / raw_len as f64,
        parallel_size,
        lz4_size as f64 / raw_len as f64,
        snappy_size as f64 / raw_len as f64,
        zstd_size as f64 / raw_len as f64,
    );

    let mut encode = c.benchmark_group(format!("ai-real-encode/{label}"));
    encode.throughput(Throughput::Bytes(raw_len as u64));
    encode.bench_function("rnbw-interleaved-reuse-serial", |b| {
        b.iter(|| serial.encode_interleaved(black_box(bytes), 4).unwrap())
    });
    encode.bench_function("rnbw-interleaved-reuse-parallel", |b| {
        b.iter(|| {
            parallel
                .encode_interleaved_parallel(black_box(bytes), 4)
                .unwrap()
        })
    });
    encode.bench_function("lz4", |b| {
        b.iter(|| lz4_flex::block::compress_prepend_size(black_box(bytes)))
    });
    encode.bench_function("snappy", |b| {
        b.iter(|| {
            snap::raw::Encoder::new()
                .compress_vec(black_box(bytes))
                .unwrap()
        })
    });
    encode.bench_function("zstd-fast", |b| {
        b.iter(|| zstd::bulk::compress(black_box(bytes), -3).unwrap())
    });
    encode.finish();

    let mut decode = c.benchmark_group(format!("ai-real-decode/{label}"));
    decode.throughput(Throughput::Bytes(raw_len as u64));
    decode.bench_function("rnbw", |b| {
        b.iter(|| black_box(&codec).decode_interleaved().unwrap())
    });
    let lz4 = lz4_flex::block::compress_prepend_size(bytes);
    assert_eq!(
        lz4_flex::block::decompress_size_prepended(&lz4).unwrap(),
        bytes
    );
    decode.bench_function("lz4", |b| {
        b.iter(|| lz4_flex::block::decompress_size_prepended(black_box(&lz4)).unwrap())
    });
    drop(lz4);

    let snappy = snap::raw::Encoder::new().compress_vec(bytes).unwrap();
    assert_eq!(
        snap::raw::Decoder::new().decompress_vec(&snappy).unwrap(),
        bytes
    );
    decode.bench_function("snappy", |b| {
        b.iter(|| {
            snap::raw::Decoder::new()
                .decompress_vec(black_box(&snappy))
                .unwrap()
        })
    });
    drop(snappy);

    let zstd = zstd::bulk::compress(bytes, -3).unwrap();
    assert_eq!(zstd::bulk::decompress(&zstd, raw_len).unwrap(), bytes);
    decode.bench_function("zstd-fast", |b| {
        b.iter(|| zstd::bulk::decompress(black_box(&zstd), raw_len).unwrap())
    });
    decode.finish();
}

fn benchmark_undecoded_index(c: &mut Criterion, label: &str, bytes: &[u8]) {
    let started = Instant::now();
    let word_count = bytes.len().div_ceil(4);
    let index = trnmap_index::build_raw_word_trnmap(bytes).expect("full model TRNMAP allocation");
    let build_time = started.elapsed();
    let index_bytes = index.storage_bytes();
    let zero_count = index.count(0).expect("zero-class count");
    let negative_count = index.count(-1).expect("negative-class count");
    let positive_count = index.count(1).expect("positive-class count");
    assert_eq!(zero_count + negative_count + positive_count, word_count);

    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let positions: Vec<usize> = (0..1024)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as usize % word_count
        })
        .collect();
    let zero_starts: Vec<usize> = (0..32)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as usize % word_count
        })
        .collect();
    for &position in &positions {
        assert_eq!(
            index.get(position).unwrap(),
            trnmap_index::raw_word_class(bytes, position)
        );
    }
    let first_zero = index.find(CLASS_ZERO, 0);
    assert_eq!(first_zero, trnmap_index::scalar_find_class(bytes, 0, 0));
    assert_eq!(
        index.count(0).unwrap(),
        trnmap_index::scalar_count_class(bytes, 0)
    );
    for &start in &zero_starts {
        assert_eq!(
            index.find(CLASS_ZERO, start),
            trnmap_index::scalar_find_class(bytes, start, 0)
        );
    }

    eprintln!(
        "real-ai-trnmap-index={label} raw-u32-words={word_count} bytes={index_bytes} build={build_time:?} positive={positive_count} zero={zero_count} negative={negative_count} first-zero={first_zero} queried-without-tensor-decode=true",
    );

    let mut get = c.benchmark_group(format!("ai-undecoded-index-get/{label}"));
    get.throughput(Throughput::Elements(positions.len() as u64));
    get.bench_function("trnmap-random-1024", |b| {
        b.iter(|| {
            black_box(&positions)
                .iter()
                .map(|&position| index.get(position).unwrap())
                .fold(0i64, |sum, value| sum + i64::from(value))
        })
    });
    get.bench_function("raw-word-classify-random-1024", |b| {
        b.iter(|| {
            black_box(&positions)
                .iter()
                .map(|&position| trnmap_index::raw_word_class(bytes, position))
                .fold(0i64, |sum, value| sum + i64::from(value))
        })
    });
    get.finish();

    let mut find = c.benchmark_group(format!("ai-undecoded-index-find-zero/{label}"));
    find.bench_function("trnmap-hierarchical", |b| {
        b.iter(|| black_box(&index).find(CLASS_ZERO, 0))
    });
    find.bench_function("raw-linear-scan", |b| {
        b.iter(|| trnmap_index::scalar_find_class(black_box(bytes), 0, 0))
    });
    find.finish();

    let mut multi_find = c.benchmark_group(format!("ai-undecoded-index-find-zero-random/{label}"));
    multi_find.throughput(Throughput::Elements(zero_starts.len() as u64));
    multi_find.bench_function("trnmap-hierarchical-32-starts", |b| {
        b.iter(|| {
            black_box(&zero_starts)
                .iter()
                .map(|&start| index.find(CLASS_ZERO, start))
                .sum::<usize>()
        })
    });
    multi_find.bench_function("raw-linear-scan-32-starts", |b| {
        b.iter(|| {
            black_box(&zero_starts)
                .iter()
                .map(|&start| trnmap_index::scalar_find_class(black_box(bytes), start, 0))
                .sum::<usize>()
        })
    });
    multi_find.finish();

    let mut count = c.benchmark_group(format!("ai-undecoded-index-count-zero/{label}"));
    count.throughput(Throughput::Elements(word_count as u64));
    count.bench_function("trnmap-summary-count", |b| {
        b.iter(|| black_box(&index).count(0).unwrap())
    });
    count.bench_function("raw-linear-count", |b| {
        b.iter(|| trnmap_index::scalar_count_class(black_box(bytes), 0))
    });
    count.finish();
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
