//! Process profiler for actual input files. Setup/validation precede the inner timer;
//! Hyperfine/Callgrind also include that setup, process startup, and output.
use rnbw::{LaneLz, LaneLzEncoder, lane_lz::stage_bench};
use std::{error::Error, hint::black_box, time::Instant};

const USAGE: &str = "usage: lane_profile split|interleave|encode|encode-parallel|decode|copy INPUT LANE_WIDTH ITERATIONS";

fn main() -> Result<(), Box<dyn Error>> {
    let [operation, path, width, iterations]: [String; 4] = std::env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| USAGE)?;
    let width: usize = width.parse()?;
    let iterations: usize = iterations.parse()?;
    if width == 0 || iterations == 0 {
        return Err("lane width and iteration count must be positive".into());
    }
    let input = std::fs::read(path)?;
    if !input.len().is_multiple_of(width) {
        return Err("input length must be divisible by the lane width; no implicit padding".into());
    }
    let elapsed = match operation.as_str() {
        "split" => {
            let mut lanes = Vec::new();
            stage_bench::split(&input, width, &mut lanes)?;
            for (record, bytes) in input.chunks_exact(width).enumerate() {
                for (lane, &byte) in bytes.iter().enumerate() {
                    assert_eq!(lanes[lane][record], byte);
                }
            }
            measure(iterations, || {
                stage_bench::split(black_box(&input), width, black_box(&mut lanes)).unwrap();
                black_box(&lanes);
            })
        }
        "interleave" => {
            if width != 16 {
                return Err("the isolated interleave kernel takes exactly sixteen lanes".into());
            }
            let mut lanes = Vec::new();
            stage_bench::split(&input, width, &mut lanes)?;
            assert_eq!(stage_bench::join_sixteen(&lanes)?, input);
            measure(iterations, || {
                stage_bench::join_sixteen(black_box(&lanes)).unwrap()
            })
        }
        "encode" | "encode-parallel" => {
            let mut encoder = LaneLzEncoder::default();
            if operation == "encode" {
                let image = encoder.encode_interleaved(&input, width)?;
                assert_eq!(image.decode_interleaved()?, input);
                measure(iterations, || {
                    encoder
                        .encode_interleaved(black_box(&input), width)
                        .unwrap()
                })
            } else {
                let image = encoder.encode_interleaved_parallel(&input, width)?;
                assert_eq!(image.decode_interleaved()?, input);
                measure(iterations, || {
                    encoder
                        .encode_interleaved_parallel(black_box(&input), width)
                        .unwrap()
                })
            }
        }
        "decode" => {
            let image = LaneLz::encode_interleaved(&input, width)?;
            assert_eq!(image.decode_interleaved()?, input);
            measure(iterations, || {
                black_box(&image).decode_interleaved().unwrap()
            })
        }
        "copy" => measure(iterations, || black_box(&input).to_vec()),
        _ => return Err(USAGE.into()),
    };
    println!(
        "operation={operation} raw_bytes={} lane_width={width} iterations={iterations} total_ns={elapsed} ns_per_call={:.3}",
        input.len(),
        elapsed as f64 / iterations as f64
    );
    Ok(())
}

fn measure<T>(iterations: usize, mut operation: impl FnMut() -> T) -> u128 {
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(operation());
    }
    start.elapsed().as_nanos()
}
