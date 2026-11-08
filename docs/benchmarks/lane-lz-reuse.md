# Lane LZ workspace and parallel measurements

## What changed

`LaneLzEncoder` retains the hand-written LZ hash heads, 64 KiB predecessor window,
and token buffer across calls. It clears only hash heads touched by the previous
lane. `encode_parallel` processes independent lanes in bounded groups, retaining
one workspace per active task. `LaneLz::encode` remains the one-shot serial API.
The reusable encoder also retains split buffers for interleaved input.

The workspace uses 524,288 bytes for the 65,536-entry `u32` hash and predecessor
arrays, plus up to 131,072 bytes for the touched-head list and a retained token
buffer. The fixed lookup tables are allocated only when a lane has at least three
bytes. For `P` active tasks, retained scratch is at most `P × (655,360 bytes +
token capacity)`, plus the raw input size for retained interleaved SoA buffers.
`P ≤ min(lane_count, Rayon worker count)`. Encoding is O(n × 32) worst case for
the bounded match-chain search, with O(n) tokenization and entropy passes;
parallel wall time is bounded by the largest task plus container assembly.

This applies the reusable scratch-buffer and dense-index patterns used by the
public [DFE resource manager](https://github.com/oda-00/dumpster_fire_engine/blob/115726ef89b6394d6383b4d1f00c78a575d07366/src/resource_manager/manager.rs)
and [stage storage](https://github.com/oda-00/dumpster_fire_engine/blob/115726ef89b6394d6383b4d1f00c78a575d07366/src/resource_manager/world_manager/stage.rs).
Rayon supplies parallelism only across independent codec lanes; the codec and its
LZ implementation remain hand-written.

## Measurements

Quick repeated-workload run on Windows x86_64, RNBW AVX2, 16 logical processors,
with Criterion 0.8.2, 20 samples, 300 ms warmup, and 1 s requested measurement.
Criterion extended slow cases to collect the requested samples. Values below are
the Criterion mean and 95% confidence interval. A few groups still had 10–15%
outliers, so use these to choose the path, not as portable throughput promises.

| Synthetic workload | Raw bytes | Serial reuse | Parallel reuse | Parallel speedup |
|---|---:|---:|---:|---:|
| RGBA8 image | 1,048,576 | 24.309 ms [24.231, 24.409] | 10.168 ms [10.088, 10.257] | 2.39× |
| Shader words | 262,144 | 2.588 ms [2.548, 2.638] | 1.444 ms [1.425, 1.463] | 1.79× |
| AI int8 weights | 1,048,576 | 8.083 ms [8.060, 8.108] | 1.609 ms [1.526, 1.713] | 5.02× |
| AI kernel words | 262,144 | 1.903 ms [1.882, 1.921] | 0.808 ms [0.750, 0.867] | 2.35× |
| Random-byte control | 262,144 | 6.833 ms [6.802, 6.881] | 7.059 ms [7.005, 7.131] | 0.97× |

This table passes already-split lane slices and measures the codec path. The
random-byte case uses one lane and falls back to serial; its 3.3% difference
between separate encoder instances is bench variance, not parallel work. For
structured lanes, explicit parallel encoding won 1.79–5.02× in this comparison.

The retained interleaved API was also measured end to end, including the repeated
interleave-to-SoA pass. It holds onto lane buffers between iterations:

| Synthetic workload | Serial interleaved reuse | Parallel interleaved reuse | Speedup |
|---|---:|---:|---:|
| RGBA8 image | 26.011 ms [25.918, 26.114] | 13.765 ms [13.723, 13.809] | 1.89× |
| Shader words | 3.231 ms [3.205, 3.257] | 2.170 ms [2.105, 2.226] | 1.49× |
| AI int8 weights | 10.873 ms [10.691, 11.085] | 4.205 ms [4.165, 4.242] | 2.59× |
| AI kernel words | 2.408 ms [2.398, 2.423] | 1.370 ms [1.359, 1.380] | 1.76× |
| Random-byte control | 7.186 ms [7.115, 7.281] | 7.323 ms [7.202, 7.463] | 0.98× |

The interleaved results show 1.49–2.59× wins on structured inputs; the random
control remains serial. The two tables include different amounts of work and
should not be compared as if their split costs were identical.

The quick-run command was:

```sh
cargo bench --locked --bench codec -- rnbw-multilane-reuse \
  --sample-size 20 --warm-up-time 0.3 --measurement-time 1.0
cargo bench --locked --bench codec -- rnbw-interleaved-reuse \
  --sample-size 20 --warm-up-time 0.3 --measurement-time 1.0
```

The synthetic image, shader, and AI inputs are deterministic test data. They are
not captured production assets or a substitute for a real application corpus.
The lane benchmark compares against LZ4 and raw copy; it does not yet include the
full common-codec matrix.
