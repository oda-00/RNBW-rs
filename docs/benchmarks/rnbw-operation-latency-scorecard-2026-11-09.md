# RNBW operation latency scorecard

Date: 2026-11-09. This report puts the current measurements on one unit basis:
whole-call latency plus amortized cost per input item. A lookup uses ns/lookup;
a key-store batch uses ns/probe; byte codecs use ns/input-byte; bulk index builds,
decodes, and transforms use ns/key or ns/trit. Whole-call latency is always shown
alongside the normalized value so amortization cannot hide a slow request.

## Meaning of the 1 ns target

For a stream operation, 1 ns/input-byte means 1 GB/s. A 1 MiB encode at that rate
still takes about 1.05 ms. Requiring the entire 1 MiB call to finish in 1 ns would
require about 1.1 PB/s and is not a meaningful CPU codec target. For a scalar
lookup, 1 ns means one complete lookup per nanosecond under the stated cache and
workload conditions. This report marks a path as meeting 1 ns only when the full
reported confidence interval is at or below 1 ns.

Criterion point-operation runs used 30 samples; the multi-codec corpus summary
used its documented 10-sample runs and 95% intervals. The latter timings below
are its center estimates divided by the exact raw-input byte counts. See the
source summaries for every confidence interval and comparator. Measurements are
local to this Windows AVX2 host and are not portable guarantees.

## Point operations and query batches

| Path and scope | Whole operation | Normalized cost | 1 ns status |
|---|---:|---:|---|
| TRNMAP safe packed `get`, one hot index in 1M-trit map | 0.853 ns [0.814–0.918] | 0.853 ns/lookup | Meets, hot-only |
| TRNMAP unchecked packed `get`, same index | 0.823 ns [0.816–0.833] | 0.823 ns/lookup | Meets, unsafe API; hot-only |
| Scalar byte-array `get`, same index | 0.682 ns [0.679–0.684] | 0.682 ns/lookup | Meets; 1.25× faster than packed safe `get` |
| TRNMAP safe positive `find`, same hot start and immediate result | 0.737 ns [0.735–0.741] | 0.737 ns/search | Meets, favorable path only |
| TRNMAP unchecked positive `find`, same case | 0.723 ns [0.716–0.735] | 0.723 ns/search | Meets, unsafe API; favorable path only |
| Scalar byte-array positive scan, same hot case | 1.186 ns [1.153–1.215] | 1.186 ns/search | Misses |
| 1,024 random packed `get`s, safe | 1.0167 µs [1.0080–1.0235] | 0.993 ns/get [0.9844–0.9995] | Meets narrowly; upper bound 0.9995 ns |
| 1,024 random packed `get`s, unchecked | 0.9038 µs [0.8992–0.9092] | 0.883 ns/get [0.878–0.888] | Meets |
| 1,024 random positive hierarchical `find`s, safe | 1.1626 µs [1.1601–1.1649] | 1.135 ns/search | Misses |
| Same batch, unchecked | 1.1500 µs [1.1327–1.1687] | 1.123 ns/search | Misses |
| Same batch, explicit 16-way unroll | 1.0659 µs [1.0203–1.1337] | 1.041 ns/search [0.996–1.107] | Not established; interval crosses 1 ns |
| 16M sparse / absent hierarchical search | roughly 11–12 ns | roughly 11–12 ns/search | Misses; long-range skip path |

The random batch means include the loop and accumulation over 1,024 queries; the
reported per-query value is total batch time divided by 1,024. The explicit
unroll result is not accepted as a win because its interval crosses the target
and repeat runs were variable. The source experiment logs are linked from the
performance ledger.

## Store operations

The full 186-row store matrix is in
[`store-trnmap-current-summary-2026-11-09.txt`](store-trnmap-current-summary-2026-11-09.txt).
Every path in that matrix is normalized in
[`store-ns-op-normalized-2026-11-09.txt`](store-ns-op-normalized-2026-11-09.txt),
with timed unit counts checked against the benchmark loops.
Build/decode consume about one million keys and are reported as both total time
and ns/key here. The `contains` figures are 1,024-query batch averages, not
isolated dependent-load latency. Update rows time one mutation (with reset work
outside the timer).

| Path and workload | Whole operation | Normalized cost |
|---|---:|---:|
| L2 p32 build, uniform, 999,882 keys | 23.338 ms | 23.34 ns/key |
| L2 p32 build, clustered, 1M keys | 2.929 ms | 2.93 ns/key |
| L2 p32 build, dense, 1M keys | 1.756 ms | 1.76 ns/key |
| L2 p32 + L3 p32 hit, uniform, 1,024 probes | 50.707 µs | 49.52 ns/probe |
| L2 p32 hit, clustered, 1,024 probes | 12.029 µs | 11.75 ns/probe |
| L2 p32 miss, clustered, 1,024 probes | 5.605 µs | 5.47 ns/probe |
| L2 p32 miss, ECS, 1,024 probes | 3.060 µs | 2.99 ns/probe |
| L2 p32 decode, uniform, 999,882 keys | 34.204 ms | 34.21 ns/key |
| L2 + L3 p32 decode, ECS, 1M keys | 2.067 ms | 2.07 ns/key |
| Hot insert, uniform, L2 p32 | 10.990 µs | 10.990 µs/insert |
| Hot insert, ECS, L2 p32 | 145.89 ns | 145.89 ns/insert |
| Dirty drain, uniform, L2 p32 | 106.05 ns | 106.05 ns/drain |
| Uniform p32 full-range promotion/refresh | 56.425 ms | 56.425 ms/call |

Dense-build and all other shapes/layouts remain directly available as individual
rows in the linked matrix; this scorecard does not infer a per-key denominator
for operations whose benchmark processes a different unit. No store lookup or
mutation currently meets 1 ns.

## Byte-codec totals and ns/input-byte

The following table covers all 16 current corpus types. Times are total function
latency; `ns/B` is that total divided by raw input bytes. Ratios are compressed
bytes/raw bytes. The detailed source matrix has the 95% intervals and LZ4,
Snappy, Zstd, Brotli, zlib, and raw-copy comparators.

| Dataset | Raw B | Ratio | Encode one-shot total / ns/B | Encode reuse serial total / ns/B | Encode reuse parallel total / ns/B | Decode one-shot total / ns/B | Decode parallel-form total / ns/B |
|---|---:|---:|---:|---:|---:|---:|---:|
| ai-model-int8-tensor-synthetic | 1048576 | 0.0206 | 4.3238 ms / 4.123 ns/B | 3.9503 ms / 3.767 ns/B | 2.6075 ms / 2.487 ns/B | 1.618 ms / 1.543 ns/B | 1.6346 ms / 1.559 ns/B |
| csv-table | 995409 | 0.4314 | 6.1315 ms / 6.160 ns/B | 5.6834 ms / 5.710 ns/B | 3.3395 ms / 3.355 ns/B | 7.9763 ms / 8.013 ns/B | 7.6711 ms / 7.706 ns/B |
| graphics-shader-machine-words-synthetic | 262144 | 0.0140 | 969.44 µs / 3.698 ns/B | 518.6 µs / 1.978 ns/B | 204.04 µs / 0.778 ns/B | 365.65 µs / 1.395 ns/B | 370.76 µs / 1.414 ns/B |
| graphics-wgsl-source | 8267 | 0.3610 | 519.83 µs / 62.880 ns/B | 54.184 µs / 6.554 ns/B | 55.819 µs / 6.752 ns/B | 50.971 µs / 6.166 ns/B | 51.937 µs / 6.282 ns/B |
| image-png-compressed-deflate | 89883 | 1.0007 | 1.7639 ms / 19.624 ns/B | 1.3203 ms / 14.689 ns/B | 1.182 ms / 13.150 ns/B | 1.9995 µs / 0.022 ns/B | 2.0935 µs / 0.023 ns/B |
| image-png-stored-deflate | 131278 | 0.9283 | 3.038 ms / 23.142 ns/B | 2.6199 ms / 19.957 ns/B | 2.5077 ms / 19.102 ns/B | 1.4232 ms / 10.841 ns/B | 1.3652 ms / 10.399 ns/B |
| image-rgba8-raster | 524288 | 0.5630 | 6.9212 ms / 13.201 ns/B | 5.9896 ms / 11.424 ns/B | 2.8617 ms / 5.458 ns/B | 2.6423 ms / 5.040 ns/B | 2.6746 ms / 5.101 ns/B |
| json-records | 1209516 | 0.1966 | 4.9247 ms / 4.072 ns/B | 4.6317 ms / 3.829 ns/B | 2.7627 ms / 2.284 ns/B | 4.2417 ms / 3.507 ns/B | 4.2092 ms / 3.480 ns/B |
| native-executable-binary | 5437440 | 0.5560 | 66.245 ms / 12.183 ns/B | 60.53 ms / 11.132 ns/B | 45.892 ms / 8.440 ns/B | 35.266 ms / 6.486 ns/B | 33.71 ms / 6.200 ns/B |
| natural-style-prose-generated | 586022 | 0.1535 | 2.3067 ms / 3.936 ns/B | 1.9321 ms / 3.297 ns/B | 1.1312 ms / 1.930 ns/B | 1.6922 ms / 2.888 ns/B | 1.7833 ms / 3.043 ns/B |
| random-binary-incompressible | 1048576 | 1.0001 | 436.18 µs / 0.416 ns/B | 384.82 µs / 0.367 ns/B | 523.19 µs / 0.499 ns/B | 197.81 µs / 0.189 ns/B | 256.35 µs / 0.244 ns/B |
| repository-rust-source | 275729 | 0.3161 | 2.0096 ms / 7.288 ns/B | 1.5249 ms / 5.530 ns/B | 1.0363 ms / 3.758 ns/B | 1.8116 ms / 6.570 ns/B | 1.7574 ms / 6.374 ns/B |
| repository-technical-prose | 26666 | 0.5937 | 869.73 µs / 32.616 ns/B | 295.27 µs / 11.073 ns/B | 278.19 µs / 10.432 ns/B | 310.82 µs / 11.656 ns/B | 310.09 µs / 11.629 ns/B |
| server-log-records | 1398527 | 0.2819 | 6.699 ms / 4.790 ns/B | 6.6722 ms / 4.771 ns/B | 3.6296 ms / 2.595 ns/B | 7.5238 ms / 5.380 ns/B | 7.3049 ms / 5.223 ns/B |
| sparse-binary | 1048576 | 0.0462 | 2.7828 ms / 2.654 ns/B | 2.1145 ms / 2.017 ns/B | 760 µs / 0.725 ns/B | 977.52 µs / 0.932 ns/B | 1.0109 ms / 0.964 ns/B |
| structured-binary-records | 524288 | 0.6725 | 2.8339 ms / 5.405 ns/B | 2.4205 ms / 4.617 ns/B | 2.1831 ms / 4.164 ns/B | 1.0629 ms / 2.027 ns/B | 1.0656 ms / 2.032 ns/B |

All codec rows are full calls; none is a sub-nanosecond call. Some amortized
encode/decode values fall below 1 ns/B because the call handles a large buffer.
That is a throughput result and does not make the whole codec operation finish
within one nanosecond. For example, incompressible 1 MiB RNBW reuse-parallel
encode is 523.19 µs (0.499 ns/B), while AI-int8 1 MiB encode is 2.608 ms
(2.487 ns/B). The measured high-ratio shader-word corpus encodes in 204.04 µs
(0.778 ns/B), but parallel-form decode is 370.76 µs (1.414 ns/B).

## Full-call size and time still matter

| Operation family | Complexity / work | Current scale | Target status |
|---|---|---:|---|
| TRNMAP `get` | O(1) packed leaf access | 0.85 ns hot; 0.99 ns/random batch | Meets only the tested hot/random get cases |
| TRNMAP `find` | O(log base 64 N) summary traversal plus leaf select | 0.74 ns immediate hot hit; ~11–12 ns long sparse/absent search | Mixed; not generally under 1 ns |
| TRNMAP `set` and store updates | O(log base 64 N) affected summary path | 22 ns map set historically; ~146 ns ECS hot store insert | Misses |
| Store membership | O(1) expected/direct chunk path; multiple dependent loads | ~3–181 ns/probe in shown shapes | Misses |
| L1 index build / store build | O(N) bulk scan and layout construction | ~2.9–29.7 ms per 1M keys depending shape/layout | Misses as whole-call 1 ns; report ns/key |
| Lane LZ + entropy codec | O(N) parse/model/emit; decode O(N) | Corpus: ~0.2 µs to 45.9 ms encode, ~2 µs to 33.7 ms decode | Misses as whole-call 1 ns; report ns/B |
| `count`, bitwise map combine, whole-range promotion | O(N/word) or O(number of affected chunks) | µs to tens of ms by range and density | Misses as whole-call 1 ns; normalize by trit/chunk only for throughput |

These are the measured gaps, not a declaration that the 1 ns goal is met
globally. The only paths currently demonstrated below the per-item target are
hot point reads/searches and a 1,024-item random packed-get batch. The next
optimization pass should target the high-cost store query path and codec
encode/decode stages using paired, workload-matched before/after runs; it must
retain total latency, normalized latency, output bytes, and exact roundtrip
checks together.
