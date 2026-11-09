use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use flate2::{Compression, read::ZlibDecoder, write::ZlibEncoder};
use rnbw::{LaneLz, LaneLzEncoder, Rnbw, trnmap::CLASS_ZERO};
use std::{
    hint::black_box,
    io::{Read, Write},
};

#[path = "support/trnmap_index.rs"]
mod trnmap_index;

#[derive(Clone, Copy)]
enum ExternalCodec {
    Lz4,
    Snappy,
    ZstdFast,
    ZstdBalanced,
    BrotliFast,
    BrotliBalanced,
    ZlibFast,
    ZlibDefault,
}

const EXTERNAL_CODECS: [ExternalCodec; 8] = [
    ExternalCodec::Lz4,
    ExternalCodec::Snappy,
    ExternalCodec::ZstdFast,
    ExternalCodec::ZstdBalanced,
    ExternalCodec::BrotliFast,
    ExternalCodec::BrotliBalanced,
    ExternalCodec::ZlibFast,
    ExternalCodec::ZlibDefault,
];

#[derive(Clone, Copy)]
enum LaneLayout {
    Interleaved(usize),
    U32RecordFields,
}

impl LaneLayout {
    fn label(self) -> &'static str {
        match self {
            Self::Interleaved(1) => "1-byte-lane",
            Self::Interleaved(4) => "4-byte-lanes",
            Self::Interleaved(16) => "16-byte-lanes",
            Self::Interleaved(_) => "interleaved-lanes",
            Self::U32RecordFields => "4-u32-field-lanes",
        }
    }
}

impl ExternalCodec {
    fn name(self) -> &'static str {
        match self {
            Self::Lz4 => "lz4-fast",
            Self::Snappy => "snappy",
            Self::ZstdFast => "zstd-fast-3",
            Self::ZstdBalanced => "zstd-level-1",
            Self::BrotliFast => "brotli-quality-0",
            Self::BrotliBalanced => "brotli-quality-5",
            Self::ZlibFast => "zlib-fast",
            Self::ZlibDefault => "zlib-default",
        }
    }
}

fn file_codecs(c: &mut Criterion) {
    eprintln!(
        "RNBW file codec matrix; host SIMD query backend={}; every RNBW lane uses multi-window LZ + entropy with raw fallback; comparators: LZ4, Snappy, Zstd, Brotli, Zlib",
        Rnbw::simd_backend()
    );
    for (name, mut bytes, layout) in file_corpus() {
        if let LaneLayout::Interleaved(lane_width) = layout {
            pad_to(&mut bytes, lane_width);
        }
        let raw_len = bytes.len();
        benchmark_raw_word_index(c, name, &bytes);
        let external: Vec<_> = EXTERNAL_CODECS
            .into_iter()
            .map(|codec| {
                let compressed = external_encode(codec, &bytes);
                assert_eq!(
                    external_decode(codec, &compressed, raw_len),
                    bytes,
                    "{} roundtrip: {name}",
                    codec.name()
                );
                (codec, compressed)
            })
            .collect();
        let mut serial_encoder = LaneLzEncoder::default();
        let mut parallel_encoder = LaneLzEncoder::default();
        let codec = encode_rnbw(&bytes, layout)
            .unwrap_or_else(|error| panic!("RNBW encode {name}: {error}"));
        let mut serial_field_workspace = FieldLaneWorkspace::default();
        let mut parallel_field_workspace = FieldLaneWorkspace::default();
        let parallel_codec = encode_rnbw_parallel(
            &mut parallel_encoder,
            &mut parallel_field_workspace,
            &bytes,
            layout,
        )
        .unwrap_or_else(|error| panic!("RNBW parallel encode {name}: {error}"));
        assert_eq!(
            decode_rnbw(&parallel_codec, layout),
            bytes,
            "RNBW parallel roundtrip: {name}"
        );
        assert_eq!(decode_rnbw(&codec, layout), bytes, "RNBW roundtrip: {name}");
        let decoded_cow = match layout {
            LaneLayout::Interleaved(_) => Some(codec.decode_interleaved_cow().unwrap()),
            LaneLayout::U32RecordFields => None,
        };
        if let Some(decoded_cow) = &decoded_cow {
            assert_eq!(
                decoded_cow.as_ref(),
                bytes,
                "RNBW borrowed roundtrip: {name}"
            );
        }
        let ratios = external
            .iter()
            .map(|(codec, compressed)| {
                format!(
                    "{}={:.4}",
                    codec.name(),
                    compressed.len() as f64 / raw_len as f64
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "file-dataset={name} bytes={raw_len} structural-lanes={} rnbw={:.4} rnbw-parallel={:.4} parallel-size-delta={:+.2}% raw=1.0000 {ratios} cow-borrowed={}",
            layout.label(),
            codec.as_bytes().len() as f64 / raw_len as f64,
            parallel_codec.as_bytes().len() as f64 / raw_len as f64,
            (parallel_codec.as_bytes().len() as f64 / codec.as_bytes().len() as f64 - 1.0) * 100.0,
            matches!(decoded_cow, Some(std::borrow::Cow::Borrowed(_))),
        );

        let mut build = c.benchmark_group(format!("file-encode/{name}"));
        build.throughput(Throughput::Bytes(raw_len as u64));
        build.bench_function("rnbw-hybrid-lz-entropy-one-shot", |b| {
            b.iter(|| encode_rnbw(black_box(&bytes), layout).unwrap())
        });
        build.bench_function("rnbw-hybrid-lz-entropy-reuse-serial", |b| {
            b.iter(|| {
                encode_rnbw_serial(
                    &mut serial_encoder,
                    &mut serial_field_workspace,
                    black_box(&bytes),
                    layout,
                )
                .unwrap()
            })
        });
        build.bench_function("rnbw-hybrid-lz-entropy-reuse-parallel", |b| {
            b.iter(|| {
                encode_rnbw_parallel(
                    &mut parallel_encoder,
                    &mut parallel_field_workspace,
                    black_box(&bytes),
                    layout,
                )
                .unwrap()
            })
        });
        for &(codec, _) in &external {
            build.bench_function(format!("{}-encode", codec.name()), |b| {
                b.iter(|| external_encode(codec, black_box(&bytes)))
            });
        }
        build.bench_function("raw-copy-control", |b| b.iter(|| black_box(&bytes).clone()));
        build.finish();

        let mut decode = c.benchmark_group(format!("file-decode/{name}"));
        decode.throughput(Throughput::Bytes(raw_len as u64));
        if matches!(layout, LaneLayout::Interleaved(_)) {
            decode.bench_function("rnbw-cow-borrow-when-possible", |b| {
                b.iter(|| black_box(&codec).decode_interleaved_cow().unwrap())
            });
        }
        decode.bench_function("rnbw-hybrid-lz-entropy", |b| {
            b.iter(|| decode_rnbw(black_box(&codec), layout))
        });
        for &(external_codec, ref compressed) in &external {
            decode.bench_function(format!("{}-decode", external_codec.name()), |b| {
                b.iter(|| external_decode(external_codec, black_box(compressed), raw_len))
            });
        }
        decode.bench_function("raw-copy-control", |b| b.iter(|| black_box(&bytes).clone()));
        decode.finish();

        let mut roundtrip = c.benchmark_group(format!("file-roundtrip/{name}"));
        roundtrip.throughput(Throughput::Bytes(raw_len as u64));
        roundtrip.bench_function("rnbw-hybrid-lz-entropy-reuse-serial", |b| {
            b.iter(|| {
                let encoded = encode_rnbw_serial(
                    &mut serial_encoder,
                    &mut serial_field_workspace,
                    black_box(&bytes),
                    layout,
                )
                .unwrap();
                black_box(decode_rnbw(&encoded, layout))
            })
        });
        roundtrip.bench_function("rnbw-hybrid-lz-entropy-reuse-parallel", |b| {
            b.iter(|| {
                let encoded = encode_rnbw_parallel(
                    &mut parallel_encoder,
                    &mut parallel_field_workspace,
                    black_box(&bytes),
                    layout,
                )
                .unwrap();
                black_box(decode_rnbw(&encoded, layout))
            })
        });
        for codec in EXTERNAL_CODECS {
            roundtrip.bench_function(format!("{}-encode-decode", codec.name()), |b| {
                b.iter(|| {
                    let compressed = external_encode(codec, black_box(&bytes));
                    black_box(external_decode(codec, &compressed, raw_len))
                })
            });
        }
        roundtrip.finish();
    }
}

fn benchmark_raw_word_index(c: &mut Criterion, name: &str, bytes: &[u8]) {
    let word_count = bytes.len().div_ceil(4);
    let started = std::time::Instant::now();
    let index = trnmap_index::build_raw_word_trnmap(bytes).expect("general file TRNMAP");
    let build_time = started.elapsed();
    let index_bytes = index.storage_bytes();
    let zeros = index.count(0).expect("zero word count");
    assert_eq!(
        zeros,
        trnmap_index::scalar_count_class(bytes, 0),
        "file index count {name}"
    );
    let first_zero = index.find(CLASS_ZERO, 0);
    assert_eq!(
        first_zero,
        trnmap_index::scalar_find_class(bytes, 0, 0),
        "file index find {name}"
    );

    let positions: Vec<usize> = (0..1024).map(|i| i * word_count / 1024).collect();
    assert!(positions.iter().all(|&position| {
        index.get(position).unwrap() == trnmap_index::raw_word_class(bytes, position)
    }));
    let starts: Vec<usize> = (1..=32).map(|i| i * word_count / 33).collect();
    assert!(starts.iter().all(|&start| {
        index.find(CLASS_ZERO, start) == trnmap_index::scalar_find_class(bytes, start, 0)
    }));

    eprintln!(
        "general-trnmap-index={name} bytes={} words={word_count} index-bytes={index_bytes} build={build_time:?} zero-words={zeros} first-zero={first_zero}",
        bytes.len()
    );

    let mut get = c.benchmark_group(format!("file-undecoded-index-get/{name}"));
    get.throughput(Throughput::Elements(positions.len() as u64));
    get.bench_function("trnmap-1024-even-positions", |b| {
        b.iter(|| {
            black_box(&positions)
                .iter()
                .map(|&position| index.get(position).unwrap())
                .fold(0i64, |sum, value| sum + i64::from(value))
        })
    });
    get.bench_function("raw-word-classify-1024", |b| {
        b.iter(|| {
            black_box(&positions)
                .iter()
                .map(|&position| trnmap_index::raw_word_class(bytes, position))
                .fold(0i64, |sum, value| sum + i64::from(value))
        })
    });
    get.finish();

    let mut find = c.benchmark_group(format!("file-undecoded-index-find-zero/{name}"));
    find.throughput(Throughput::Elements(starts.len() as u64));
    find.bench_function("trnmap-hierarchical-32-starts", |b| {
        b.iter(|| {
            black_box(&starts)
                .iter()
                .map(|&start| index.find(CLASS_ZERO, start))
                .sum::<usize>()
        })
    });
    find.bench_function("raw-linear-32-starts", |b| {
        b.iter(|| {
            black_box(&starts)
                .iter()
                .map(|&start| trnmap_index::scalar_find_class(bytes, start, 0))
                .sum::<usize>()
        })
    });
    find.finish();

    let mut count = c.benchmark_group(format!("file-undecoded-index-count-zero/{name}"));
    count.throughput(Throughput::Elements(word_count as u64));
    count.bench_function("trnmap-summary-count", |b| {
        b.iter(|| black_box(&index).count(0).unwrap())
    });
    count.bench_function("raw-linear-count", |b| {
        b.iter(|| trnmap_index::scalar_count_class(black_box(bytes), 0))
    });
    count.finish();
}

fn encode_rnbw(input: &[u8], layout: LaneLayout) -> Result<LaneLz, String> {
    match layout {
        LaneLayout::Interleaved(width) => {
            LaneLz::encode_interleaved(input, width).map_err(|error| error.to_string())
        }
        LaneLayout::U32RecordFields => {
            let mut workspace = FieldLaneWorkspace::default();
            let lanes = workspace.split(input)?;
            LaneLz::encode(&lanes).map_err(|error| error.to_string())
        }
    }
}

fn encode_rnbw_serial(
    encoder: &mut LaneLzEncoder,
    fields: &mut FieldLaneWorkspace,
    input: &[u8],
    layout: LaneLayout,
) -> Result<LaneLz, String> {
    match layout {
        LaneLayout::Interleaved(width) => encoder
            .encode_interleaved(input, width)
            .map_err(|error| error.to_string()),
        LaneLayout::U32RecordFields => {
            let lanes = fields.split(input)?;
            encoder.encode(&lanes).map_err(|error| error.to_string())
        }
    }
}

fn encode_rnbw_parallel(
    encoder: &mut LaneLzEncoder,
    fields: &mut FieldLaneWorkspace,
    input: &[u8],
    layout: LaneLayout,
) -> Result<LaneLz, String> {
    match layout {
        LaneLayout::Interleaved(width) => encoder
            .encode_interleaved_parallel(input, width)
            .map_err(|error| error.to_string()),
        LaneLayout::U32RecordFields => {
            let lanes = fields.split(input)?;
            encoder
                .encode_parallel(&lanes)
                .map_err(|error| error.to_string())
        }
    }
}

fn decode_rnbw(codec: &LaneLz, layout: LaneLayout) -> Vec<u8> {
    match layout {
        LaneLayout::Interleaved(_) => codec.decode_interleaved().unwrap(),
        LaneLayout::U32RecordFields => {
            let lanes = codec.decode_lanes().unwrap();
            assert_eq!(lanes.len(), 4);
            assert!(lanes.iter().all(|lane| lane.len() == lanes[0].len()));
            assert_eq!(lanes[0].len() % 4, 0);
            let mut output = Vec::with_capacity(codec.raw_bytes());
            for record in 0..lanes[0].len() / 4 {
                let at = record * 4;
                for lane in &lanes {
                    output.extend_from_slice(&lane[at..at + 4]);
                }
            }
            output
        }
    }
}

#[derive(Default)]
struct FieldLaneWorkspace {
    lanes: [Vec<u8>; 4],
}

impl FieldLaneWorkspace {
    fn split(&mut self, input: &[u8]) -> Result<[&[u8]; 4], String> {
        if !input.len().is_multiple_of(16) {
            return Err("u32-field input is not a whole 16-byte record".into());
        }
        let records = input.len() / 16;
        for lane in &mut self.lanes {
            lane.clear();
            lane.reserve(records * 4);
        }
        for record in input.chunks_exact(16) {
            for (field, lane) in self.lanes.iter_mut().enumerate() {
                lane.extend_from_slice(&record[field * 4..field * 4 + 4]);
            }
        }
        Ok([
            self.lanes[0].as_slice(),
            self.lanes[1].as_slice(),
            self.lanes[2].as_slice(),
            self.lanes[3].as_slice(),
        ])
    }
}

fn external_encode(codec: ExternalCodec, bytes: &[u8]) -> Vec<u8> {
    match codec {
        ExternalCodec::Lz4 => lz4_flex::block::compress_prepend_size(bytes),
        ExternalCodec::Snappy => snap::raw::Encoder::new().compress_vec(bytes).unwrap(),
        ExternalCodec::ZstdFast => zstd::bulk::compress(bytes, -3).unwrap(),
        ExternalCodec::ZstdBalanced => zstd::bulk::compress(bytes, 1).unwrap(),
        ExternalCodec::BrotliFast => brotli_compress(bytes, 0),
        ExternalCodec::BrotliBalanced => brotli_compress(bytes, 5),
        ExternalCodec::ZlibFast => zlib_compress(bytes, Compression::fast()),
        ExternalCodec::ZlibDefault => zlib_compress(bytes, Compression::default()),
    }
}

fn external_decode(codec: ExternalCodec, bytes: &[u8], raw_len: usize) -> Vec<u8> {
    match codec {
        ExternalCodec::Lz4 => lz4_flex::block::decompress_size_prepended(bytes).unwrap(),
        ExternalCodec::Snappy => snap::raw::Decoder::new().decompress_vec(bytes).unwrap(),
        ExternalCodec::ZstdFast | ExternalCodec::ZstdBalanced => {
            zstd::bulk::decompress(bytes, raw_len).unwrap()
        }
        ExternalCodec::BrotliFast | ExternalCodec::BrotliBalanced => {
            let mut decoder = brotli::Decompressor::new(bytes, 1 << 15);
            let mut output = Vec::with_capacity(raw_len);
            decoder.read_to_end(&mut output).unwrap();
            output
        }
        ExternalCodec::ZlibFast | ExternalCodec::ZlibDefault => {
            let mut decoder = ZlibDecoder::new(bytes);
            let mut output = Vec::with_capacity(raw_len);
            decoder.read_to_end(&mut output).unwrap();
            output
        }
    }
}

fn brotli_compress(bytes: &[u8], quality: u32) -> Vec<u8> {
    let mut output = Vec::with_capacity(bytes.len());
    {
        let mut encoder = brotli::CompressorWriter::new(&mut output, 1 << 15, quality, 22);
        encoder.write_all(bytes).unwrap();
    }
    output
}

fn zlib_compress(bytes: &[u8], level: Compression) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::with_capacity(bytes.len()), level);
    encoder.write_all(bytes).unwrap();
    encoder.finish().unwrap()
}

fn file_corpus() -> Vec<(&'static str, Vec<u8>, LaneLayout)> {
    vec![
        (
            "image-rgba8-raster",
            rgba_raster(),
            LaneLayout::Interleaved(4),
        ),
        (
            "image-png-stored-deflate",
            png_raster(),
            LaneLayout::Interleaved(1),
        ),
        (
            "image-png-compressed-deflate",
            png_raster_compressed(),
            LaneLayout::Interleaved(1),
        ),
        (
            "graphics-shader-machine-words-synthetic",
            shader_words(),
            LaneLayout::Interleaved(4),
        ),
        (
            "graphics-wgsl-source",
            shader_source(),
            LaneLayout::Interleaved(1),
        ),
        (
            "ai-model-int8-tensor-synthetic",
            model_weights(),
            LaneLayout::Interleaved(16),
        ),
        (
            "native-executable-binary",
            executable_bytes(),
            LaneLayout::Interleaved(1),
        ),
        (
            "structured-binary-records",
            binary_records(),
            LaneLayout::U32RecordFields,
        ),
        ("sparse-binary", sparse_binary(), LaneLayout::Interleaved(1)),
        (
            "random-binary-incompressible",
            random_bytes(1 << 20, 0xd1b5_4a32_d192_ed03),
            LaneLayout::Interleaved(1),
        ),
        (
            "repository-rust-source",
            repository_source(),
            LaneLayout::Interleaved(1),
        ),
        (
            "repository-technical-prose",
            repository_prose(),
            LaneLayout::Interleaved(1),
        ),
        (
            "natural-style-prose-generated",
            natural_prose(),
            LaneLayout::Interleaved(1),
        ),
        ("json-records", json_records(), LaneLayout::Interleaved(1)),
        ("csv-table", csv_table(), LaneLayout::Interleaved(1)),
        (
            "server-log-records",
            log_records(),
            LaneLayout::Interleaved(1),
        ),
    ]
}

fn rgba_raster() -> Vec<u8> {
    let (width, height) = (512u32, 256u32);
    let mut bytes = Vec::with_capacity((width * height * 4) as usize);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for y in 0..height {
        for x in 0..width {
            state = xorshift(state);
            let noise = (state & 3) as u8;
            bytes.extend_from_slice(&[
                ((x / 2) as u8).wrapping_add(noise),
                ((y / 2) as u8).wrapping_add(noise),
                ((x ^ y) as u8).wrapping_add(noise),
                255,
            ]);
        }
    }
    bytes
}

fn png_raster() -> Vec<u8> {
    png_raster_with_deflate(false)
}

fn png_raster_compressed() -> Vec<u8> {
    png_raster_with_deflate(true)
}

fn png_raster_with_deflate(compressed: bool) -> Vec<u8> {
    let (width, height) = (256u32, 128u32);
    let pixels = rgba_raster();
    let row_bytes = width as usize * 4;
    let mut scanlines = Vec::with_capacity((row_bytes + 1) * height as usize);
    for row in pixels.chunks_exact(row_bytes).take(height as usize) {
        scanlines.push(0); // PNG filter: None
        scanlines.extend_from_slice(row);
    }
    let zlib = if compressed {
        zlib_compress(&scanlines, Compression::default())
    } else {
        stored_zlib(&scanlines)
    };

    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA
    png_chunk(&mut png, b"IHDR", &ihdr);
    png_chunk(&mut png, b"IDAT", &zlib);
    png_chunk(&mut png, b"IEND", &[]);
    png
}

fn stored_zlib(scanlines: &[u8]) -> Vec<u8> {
    let mut zlib = vec![0x78, 0x01];
    let mut remaining = scanlines;
    while !remaining.is_empty() {
        let length = remaining.len().min(u16::MAX as usize);
        let final_block = length == remaining.len();
        zlib.push(u8::from(final_block)); // DEFLATE stored block, byte aligned
        let len = length as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(&remaining[..length]);
        remaining = &remaining[length..];
    }
    zlib.extend_from_slice(&adler32(scanlines).to_be_bytes());
    zlib
}

fn png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & 0u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in bytes {
        a = (a + u32::from(byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    (b << 16) | a
}

fn shader_words() -> Vec<u8> {
    let mut out = Vec::with_capacity(256 * 1024);
    for i in 0..65_536u32 {
        let op = [1u32, 0x13, 0x21, 0x33, 0x41, 0x6f][(i as usize * 17) % 6];
        let word = op | ((i & 0x3ff) << 16) | (((i * 13) & 0x7f) << 8);
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

fn shader_source() -> Vec<u8> {
    include_bytes!("data/graphics-wgsl-source.txt").to_vec()
}

fn model_weights() -> Vec<u8> {
    (0..(1 << 20usize))
        .map(|i| {
            let lane = i & 15;
            let position = i >> 4;
            let phase = (position / 256 + lane * 7) & 31;
            if (position + lane * 11) % 8 < 5 {
                0
            } else {
                ((phase * 7 + lane * 3) % 63) as u8
            }
        })
        .collect()
}

fn executable_bytes() -> Vec<u8> {
    include_bytes!("data/native-executable-binary.exe").to_vec()
}

fn binary_records() -> Vec<u8> {
    let mut out = Vec::with_capacity(16 * 32_768);
    let mut state = 0x243f_6a88_85a3_08d3u64;
    for id in 0..32_768u32 {
        state = xorshift(state);
        out.extend_from_slice(&id.to_le_bytes());
        out.extend_from_slice(&(id / 32).to_le_bytes());
        out.extend_from_slice(&(state as u32).to_le_bytes());
        out.extend_from_slice(&((state >> 32) as u32).to_le_bytes());
    }
    out
}

fn sparse_binary() -> Vec<u8> {
    let mut out = vec![0u8; 1 << 20];
    let mut state = 0xa076_1d64_78bd_642fu64;
    for block in 0..512usize {
        state = xorshift(state);
        let start = block * 2048 + (state as usize & 1023);
        for (offset, byte) in out[start..start + 64].iter_mut().enumerate() {
            *byte = (state.rotate_left(offset as u32) as u8).wrapping_add(offset as u8);
        }
    }
    out
}

fn random_bytes(length: usize, mut state: u64) -> Vec<u8> {
    (0..length)
        .map(|_| {
            state = xorshift(state);
            (state >> 24) as u8
        })
        .collect()
}

fn repository_source() -> Vec<u8> {
    include_bytes!("data/repository-rust-source.txt").to_vec()
}

fn repository_prose() -> Vec<u8> {
    include_bytes!("data/repository-technical-prose.txt").to_vec()
}

fn natural_prose() -> Vec<u8> {
    const SUBJECTS: [&str; 16] = [
        "the river",
        "a small workshop",
        "the evening train",
        "an old orchard",
        "the mountain road",
        "a quiet library",
        "the coastal village",
        "a garden gate",
        "the winter market",
        "a distant lighthouse",
        "the wooden bridge",
        "a narrow lane",
        "the morning fog",
        "a paper map",
        "the open field",
        "a stone house",
    ];
    const VERBS: [&str; 16] = [
        "followed the changing light",
        "carried news from the valley",
        "waited beyond the station",
        "remembered the rain",
        "turned toward the harbor",
        "kept its windows open",
        "moved slowly through the day",
        "held the last warmth of summer",
        "crossed the ridge",
        "gathered beneath the trees",
        "marked the edge of town",
        "returned before sunset",
        "left a clear path in the snow",
        "stood beside the road",
        "changed with the season",
        "made room for another visitor",
    ];
    const ENDINGS: [&str; 16] = [
        "while the lamps came on.",
        "and no one hurried home.",
        "as the clouds lifted.",
        "before the first frost arrived.",
        "with a note pinned to the door.",
        "and the whole street grew quiet.",
        "until the last bus passed.",
        "where the map showed no road.",
        "after the market had closed.",
        "as the first birds returned.",
        "without disturbing the sleepers.",
        "and the conversation continued.",
        "when the weather finally cleared.",
        "under a sky full of stars.",
        "though the forecast promised rain.",
        "because someone had left a light on.",
    ];
    let mut out = Vec::with_capacity(512 * 1024);
    let mut state = 0x6a09_e667_f3bc_c909u64;
    for _ in 0..8192 {
        state = xorshift(state);
        let a = state as usize & 15;
        let b = (state >> 8) as usize & 15;
        let c = (state >> 16) as usize & 15;
        out.extend_from_slice(SUBJECTS[a].as_bytes());
        out.push(b' ');
        out.extend_from_slice(VERBS[b].as_bytes());
        out.push(b' ');
        out.extend_from_slice(ENDINGS[c].as_bytes());
        out.push(b'\n');
    }
    out
}

fn json_records() -> Vec<u8> {
    let mut out = Vec::with_capacity(1 << 20);
    out.extend_from_slice(b"[\n");
    for i in 0..16_384u32 {
        let state = xorshift(u64::from(i) ^ 0x510e_527f_ade6_82d1);
        let _ = std::fmt::Write::write_fmt(
            &mut StringWriter(&mut out),
            format_args!(
                "{{\"id\":{i},\"group\":{},\"active\":{},\"weight\":{},\"label\":\"node-{:04}\"}}{}\n",
                i / 64,
                i % 10 != 0,
                (state >> 20) as u16,
                i % 997,
                if i == 16_383 { "" } else { "," }
            ),
        );
    }
    out.extend_from_slice(b"]\n");
    out
}

struct StringWriter<'a>(&'a mut Vec<u8>);

impl std::fmt::Write for StringWriter<'_> {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        self.0.extend_from_slice(value.as_bytes());
        Ok(())
    }
}

fn csv_table() -> Vec<u8> {
    let mut out = Vec::with_capacity(1 << 20);
    out.extend_from_slice(b"id,group,active,weight,label\n");
    for i in 0..32_768u32 {
        let state = xorshift(u64::from(i) ^ 0x1f83_d9ab_fb41_bd6b);
        let _ = std::fmt::Write::write_fmt(
            &mut StringWriter(&mut out),
            format_args!(
                "{i},{},{},{},node-{:04}\n",
                i / 64,
                i % 10 != 0,
                (state >> 20) as u16,
                i % 997
            ),
        );
    }
    out
}

fn log_records() -> Vec<u8> {
    const LEVELS: [&str; 5] = ["INFO", "DEBUG", "WARN", "INFO", "TRACE"];
    const EVENTS: [&str; 8] = [
        "cache lookup completed",
        "worker entered batch",
        "request was accepted",
        "page flush finished",
        "record checksum matched",
        "connection returned to pool",
        "background scan advanced",
        "index generation changed",
    ];
    let mut out = Vec::with_capacity(1 << 20);
    for i in 0..16_384u32 {
        let _ = std::fmt::Write::write_fmt(
            &mut StringWriter(&mut out),
            format_args!(
                "2026-11-08T{:02}:{:02}:{:02}.{:06}Z {} worker={} request={} {}\n",
                (i / 3600) % 24,
                (i / 60) % 60,
                i % 60,
                (i.wrapping_mul(7919)) % 1_000_000,
                LEVELS[i as usize % LEVELS.len()],
                i % 64,
                i.wrapping_mul(2654435761),
                EVENTS[i as usize % EVENTS.len()],
            ),
        );
    }
    out
}

fn pad_to(bytes: &mut Vec<u8>, alignment: usize) {
    let padding = (alignment - bytes.len() % alignment) % alignment;
    bytes.resize(bytes.len() + padding, 0);
}

fn xorshift(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^ (state << 17)
}

criterion_group!(benches, file_codecs);
criterion_main!(benches);
