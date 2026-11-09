//! Timed functions are production kernels; fixture construction and checks are untimed.
use super::{FieldLaneWorkspace, LaneLayout, file_corpus, pad_to};
use criterion::{Criterion, Throughput};
use rnbw::{entropy_stage_bench as entropy, lane_lz::stage_bench as lz};
use std::hint::black_box;

struct EntropyFixture {
    input: Vec<u8>,
    counts: [u64; 256],
    model: [u16; 256],
    rans: Vec<u8>,
    huffman: Option<([u8; 256], Vec<u8>)>,
}

pub fn benchmarks(c: &mut Criterion) {
    histogram_cutoff(c);
    for (name, mut bytes, layout) in file_corpus() {
        let mut split = Vec::new();
        let mut fields = FieldLaneWorkspace::default();
        let mut group = c.benchmark_group(format!("stage/{name}"));
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        let inputs = match layout {
            LaneLayout::Interleaved(width) => {
                pad_to(&mut bytes, width);
                lz::split(&bytes, width, &mut split).unwrap();
                group.bench_function("split-reuse", |b| {
                    b.iter(|| {
                        lz::split(black_box(&bytes), width, black_box(&mut split)).unwrap();
                        black_box(&split);
                    })
                });
                split
            }
            LaneLayout::U32RecordFields => {
                fields.split(&bytes).unwrap();
                group.bench_function("split-reuse", |b| {
                    b.iter(|| {
                        fields.split(black_box(&bytes)).unwrap();
                        black_box(&fields.lanes);
                    })
                });
                fields.lanes.into_iter().collect()
            }
        };
        if matches!(layout, LaneLayout::Interleaved(16)) {
            assert_eq!(lz::join_sixteen(&inputs).unwrap(), bytes);
            group.bench_function("interleave", |b| {
                b.iter(|| black_box(lz::join_sixteen(black_box(&inputs)).unwrap()))
            });
        }
        let mut lanes: Vec<_> = inputs
            .into_iter()
            .map(|input| {
                let lane = lz::Lane::new(input.clone());
                assert_eq!(lane.expand().unwrap(), input, "LZ stage roundtrip {name}");
                lane
            })
            .collect();
        let fixtures: Vec<_> = lanes
            .iter()
            .map(|lane| {
                let input = lane.stream().to_vec();
                let counts = entropy::histogram(&input);
                let model = entropy::normalize(&counts, input.len());
                let rans = entropy::encode_rans(&input, &model).unwrap();
                assert_eq!(
                    entropy::decode_rans(&rans, input.len(), &model).unwrap(),
                    input,
                    "rANS stage roundtrip {name}"
                );
                let huffman = entropy::encode_huffman(&input, &counts);
                if let Some((lengths, payload)) = &huffman {
                    assert_eq!(
                        entropy::decode_huffman(payload, input.len(), lengths).unwrap(),
                        input,
                        "Huffman stage roundtrip {name}"
                    );
                }
                EntropyFixture {
                    input,
                    counts,
                    model,
                    rans,
                    huffman,
                }
            })
            .collect();
        let stream_bytes: usize = fixtures.iter().map(|f| f.input.len()).sum();
        eprintln!(
            "stage dataset={name} raw_bytes={} lanes={} stream_bytes={stream_bytes} sampled_raw_lanes={}",
            bytes.len(),
            lanes.len(),
            lanes.iter().filter(|l| l.sample()).count()
        );
        group.throughput(Throughput::Elements(lanes.len() as u64));
        group.bench_function("sample", |b| {
            b.iter(|| {
                for lane in black_box(&lanes) {
                    black_box(lane.sample());
                }
            })
        });
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_function("lz-parse-reuse", |b| {
            b.iter(|| {
                for lane in black_box(&mut lanes) {
                    black_box(lane.parse());
                    black_box(lane.stream());
                }
            })
        });
        group.bench_function("lz-expand", |b| {
            b.iter(|| {
                for lane in black_box(&lanes) {
                    black_box(lane.expand().unwrap());
                }
            })
        });
        group.throughput(Throughput::Bytes(stream_bytes as u64));
        group.bench_function("hybrid-entropy-selection", |b| {
            b.iter(|| {
                for lane in black_box(&lanes) {
                    black_box(lane.hybrid_select().unwrap());
                }
            })
        });
        group.bench_function("histogram", |b| {
            b.iter(|| {
                for f in black_box(&fixtures) {
                    black_box(entropy::histogram(&f.input));
                }
            })
        });
        // Model construction consumes a fixed 256-bin histogram per lane.
        group.throughput(Throughput::Elements(lanes.len() as u64));
        group.bench_function("normalize-model", |b| {
            b.iter(|| {
                for f in black_box(&fixtures) {
                    black_box(entropy::normalize(&f.counts, f.input.len()));
                }
            })
        });
        group.throughput(Throughput::Bytes(stream_bytes as u64));
        group.bench_function("rans-encode", |b| {
            b.iter(|| {
                for f in black_box(&fixtures) {
                    black_box(entropy::encode_rans(&f.input, &f.model).unwrap());
                }
            })
        });
        group.bench_function("rans-decode", |b| {
            b.iter(|| {
                for f in black_box(&fixtures) {
                    black_box(entropy::decode_rans(&f.rans, f.input.len(), &f.model).unwrap());
                }
            })
        });
        if fixtures.iter().all(|f| f.huffman.is_some()) {
            group.bench_function("huffman-encode-model-and-payload", |b| {
                b.iter(|| {
                    for f in black_box(&fixtures) {
                        black_box(entropy::encode_huffman(&f.input, &f.counts).unwrap());
                    }
                })
            });
            group.bench_function("huffman-decode-table-and-payload", |b| {
                b.iter(|| {
                    for f in black_box(&fixtures) {
                        let (lengths, payload) = f.huffman.as_ref().unwrap();
                        black_box(
                            entropy::decode_huffman(payload, f.input.len(), lengths).unwrap(),
                        );
                    }
                })
            });
        } else {
            eprintln!(
                "stage dataset={name} Huffman cannot represent a forced stream; forced Huffman benchmarks omitted"
            );
        }
        group.finish();
    }
}

fn histogram_cutoff(c: &mut Criterion) {
    for shape in ["constant", "four-symbol", "uniform"] {
        for length in [64usize, 256, 512, 1023, 1024, 2048, 4096, 16384] {
            let mut seed = 0x9e37_79b9u32;
            let input: Vec<_> = (0..length)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    match shape {
                        "constant" => 0,
                        "four-symbol" => (seed & 3) as u8,
                        _ => seed as u8,
                    }
                })
                .collect();
            assert_eq!(
                entropy::histogram(&input),
                entropy::histogram_serial_reference(&input)
            );
            let mut group = c.benchmark_group(format!("histogram-cutoff/{shape}/{length}"));
            group.throughput(Throughput::Bytes(length as u64));
            group.bench_function("production", |b| {
                b.iter(|| black_box(entropy::histogram(black_box(&input))))
            });
            group.bench_function("serial-reference", |b| {
                b.iter(|| black_box(entropy::histogram_serial_reference(black_box(&input))))
            });
            group.finish();
        }
    }
}
