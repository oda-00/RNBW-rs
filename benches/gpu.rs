use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rnbw::{LaneLzGpu, LaneLzRestarted};
use std::future::Future;
use std::hint::black_box;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread;
use std::time::Duration;

struct ThreadWake(thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park(),
        }
    }
}

fn gpu_benchmarks(c: &mut Criterion) {
    let gpu = block_on(LaneLzGpu::new_vulkan()).expect("initialize Vulkan compute device");
    eprintln!("RNBW GPU adapter: {:?}", gpu.adapter_info());
    let size = 16 * 1024 * 1024;
    let all_workloads = ["repeat", "json_records", "random"];
    let selected_workload = std::env::var("RNBW_GPU_CASE").ok();
    let workloads = all_workloads.into_iter().filter(|workload| {
        selected_workload
            .as_deref()
            .is_none_or(|selected| selected == *workload)
    });
    let all_block_sizes = [64 * 1024, 256 * 1024, 1024 * 1024];
    let selected_blocks = std::env::var("RNBW_GPU_BLOCKS").ok();
    let block_sizes: Vec<usize> = all_block_sizes
        .into_iter()
        .filter(|block| {
            selected_blocks.as_deref().is_none_or(|selected| {
                selected
                    .split(',')
                    .filter_map(|value| value.parse::<usize>().ok())
                    .any(|value| value == *block)
            })
        })
        .collect();

    for workload in workloads {
        let input = make_workload(workload, size);
        for &block_size in &block_sizes {
            let cpu_encoded = LaneLzRestarted::encode_cpu(&[&input], block_size)
                .expect("CPU restart-block encode");
            let gpu_encoded = gpu
                .encode_restart_blocks(&[&input], block_size)
                .expect("GPU restart-block encode");
            assert_eq!(cpu_encoded.decode_lanes().unwrap(), [input.as_slice()]);
            assert_eq!(
                gpu.decode_restart_blocks(gpu_encoded.as_bytes()).unwrap(),
                [input.as_slice()]
            );

            eprintln!(
                "workload={workload} bytes={size} block={block_size} cpu_image={} gpu_image={} ratio_cpu={:.5} ratio_gpu={:.5}",
                cpu_encoded.as_bytes().len(),
                gpu_encoded.as_bytes().len(),
                cpu_encoded.as_bytes().len() as f64 / size as f64,
                gpu_encoded.as_bytes().len() as f64 / size as f64,
            );

            let mut group = c.benchmark_group(format!("gpu-restart/{workload}/block-{block_size}"));
            group.throughput(Throughput::Bytes(size as u64));
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(200));
            group.measurement_time(Duration::from_millis(800));
            group.bench_function(BenchmarkId::new("encode", "cpu"), |b| {
                b.iter(|| {
                    black_box(
                        LaneLzRestarted::encode_cpu(&[black_box(&input)], block_size).unwrap(),
                    )
                })
            });
            group.bench_function(BenchmarkId::new("encode", "gpu"), |b| {
                b.iter(|| {
                    black_box(
                        gpu.encode_restart_blocks(&[black_box(&input)], block_size)
                            .unwrap(),
                    )
                })
            });
            group.bench_function(BenchmarkId::new("decode", "cpu"), |b| {
                b.iter(|| {
                    black_box(
                        LaneLzRestarted::from_bytes(gpu_encoded.as_bytes())
                            .unwrap()
                            .decode_lanes()
                            .unwrap(),
                    )
                })
            });
            group.bench_function(BenchmarkId::new("decode", "gpu"), |b| {
                b.iter(|| {
                    black_box(
                        gpu.decode_restart_blocks(black_box(gpu_encoded.as_bytes()))
                            .unwrap(),
                    )
                })
            });
            group.finish();
        }
    }
}

fn make_workload(kind: &str, len: usize) -> Vec<u8> {
    match kind {
        "repeat" => {
            let pattern =
                b"RNBW lane restart compression: field=position; value=000000; state=stable\n";
            (0..len)
                .map(|index| pattern[index % pattern.len()])
                .collect()
        }
        "json_records" => {
            let mut out = Vec::with_capacity(len);
            let mut row = 0u32;
            while out.len() < len {
                let line = format!(
                    "{{\"entity\":{row},\"component\":\"transform\",\"position\":[{},{},{}],\"active\":true}}\n",
                    row.wrapping_mul(17) % 100_003,
                    row.wrapping_mul(31) % 70_001,
                    row.wrapping_mul(47) % 30_007,
                );
                out.extend_from_slice(line.as_bytes());
                row = row.wrapping_add(1);
            }
            out.truncate(len);
            out
        }
        "random" => {
            let mut state = 0x9e3779b97f4a7c15u64;
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state as u8
                })
                .collect()
        }
        _ => unreachable!("benchmark workload is a fixed list"),
    }
}

criterion_group!(benches, gpu_benchmarks);
criterion_main!(benches);
