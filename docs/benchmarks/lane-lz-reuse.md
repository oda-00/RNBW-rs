# Lane LZ workspace and parallel measurements

> The measurements below are a historical snapshot from the predecessor-chain
> matcher. The current source uses two stored candidates per hash and has been
> remeasured in `general-file-codec-and-trnmap-summary-2026-11-09.txt`.

## What changed

`LaneLzEncoder` retains its LZ table, touched-hash list, tokens, literals, and
interleaved split buffers between calls. The current matcher stores the two
newest positions for each 3-byte hash in one `u64`; it tests at most those two
candidates rather than following a predecessor chain. The table has `2^18`
entries, so its exact allocated size is 2,097,152 bytes per initialized
workspace. The touched-hash list can retain up to 1,048,576 bytes, in addition
to token/literal capacity.

`encode_parallel` processes independent lanes or 64 KiB blocks in bounded groups
and retains one workspace per task. With 16 initialized tasks, the hash tables
alone occupy 32 MiB; the touched lists and token/literal buffers add workload-
dependent memory. The old 64 KiB predecessor ring is no longer present. The
current LZ matcher still uses a hand-written hash-head table; the separate
TrnMap benchmarks do not mean the codec matcher is TrnMap-backed.

This applies the reusable scratch-buffer and dense-index patterns used by the
public [DFE resource manager](https://github.com/oda-00/dumpster_fire_engine/blob/115726ef89b6394d6383b4d1f00c78a575d07366/src/resource_manager/manager.rs)
and [stage storage](https://github.com/oda-00/dumpster_fire_engine/blob/115726ef89b6394d6383b4d1f00c78a575d07366/src/resource_manager/world_manager/stage.rs).
Rayon supplies parallelism only across independent codec lanes; the codec and its
LZ implementation remain hand-written.

## Historical measurements (predecessor-chain matcher)

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
