# Codec stage audit and sixteen-lane SIMD result

Date: 2026-11-09. Parent source: `ff4341d`. Retained change: AVX2 forward and inverse 16-lane transpose inside the existing RNLZ codec. No wire-format, entropy model, match policy, TRNMAP, Arena, or GenTable changes. The LZ and entropy implementations remain hand-written; external codecs are benchmark controls only.

## Verified full-operation gain

Windows x86-64 AVX2, Rust 1.95.0 / LLVM 22.1.2. The saved original binary and retained SIMD binary use the same frozen corpus. Criterion: 30 samples, 100 ms warmup, requested 300 ms per case for repeated baseline and 200 ms for final matrix (Criterion extends collection where needed). No compiler, tests, or profiler ran concurrently with these timings. These short captures support a large local gain, not a portable worst-case bound.

Input: **1,048,576 synthetic int8 bytes, 16 lanes**. All full-codec forms retain **21,568 encoded bytes**, a **48.6187:1 raw/encoded ratio**. This is synthetic data, not a production model-weight claim. All 16 corpus metadata rows, including raw sizes and RNBW/comparator encoded sizes, are identical between the saved baseline and final capture.

| Operation | Original center (µs) | Retained center [interval] (µs) | Speedup | Retained ns/raw byte |
|---|---:|---:|---:|---:|
| Split, reused outputs | 2412.900 | 48.113 [47.661–48.546] | 50.15× | 0.045884 |
| Complete one-shot encode | 2920.500 | 598.550 [593.580–604.550] | 4.88× | 0.570822 |
| Complete reuse-serial encode | 2773.200 | 609.520 [606.160–613.670] | 4.55× | 0.581284 |
| Complete reuse-parallel encode | 2302.400 | 199.580 [195.110–205.120] | 11.54× | 0.190334 |
| Complete decode, serial-produced image | 1649.100 | 330.640 [327.640–333.870] | 4.99× | 0.315323 |
| Complete decode, parallel-produced image | 1683.000 | 373.630 [359.860–385.590] | 4.50× | 0.356321 |

Decode uses the existing internal lane-parallel policy in both image forms; “serial-produced” describes the encoder, not a serial decode implementation. The isolated inverse transpose is 253.33 µs [243.71–262.98], including output allocation and destruction. The transpose helper is shared, inlined, and allocation-free itself; the owned codec still allocates its outputs. The 0.190334 ns/byte parallel-encode figure is throughput amortization: the complete call takes 199.58 µs. It does not satisfy a ≤1 ns whole-operation limit.

On this same capture LZ4 encode/decode is 123.81/203.30 µs, Snappy encode 142.30 µs, and Zstd level 1 encode 89.415 µs. RNBW remains slower. Their encoded/raw fractions are 0.0190 / 0.0664 / 0.0015 respectively, versus RNBW 0.0206. RNBW does not win density on this synthetic tensor fixture either.

## Complete retained-candidate file matrix

All times below are whole calls in **µs**, not ns/byte. RNBW encode columns are one-shot / reused serial / reused parallel. Decode restores the serial-produced image. External encoders are independent full one-shot calls. No pipeline stages are added together. Intervals and all 177 rows are in the linked normalized capture.

| Dataset | Raw bytes | RNBW bytes | RNBW encode one / serial / parallel (µs) | RNBW decode (µs) | LZ4 encode / decode (µs) | Snappy encode (µs) | Zstd-1 encode (µs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| image-rgba8-raster | 524288 | 295200 | 5957.600 / 5358.000 / 2685.500 | 1600.500 | 168.460 / 21.597 | 29.620 | 867.620 |
| image-png-stored-deflate | 131278 | 121864 | 3334.700 / 2670.100 / 2567.800 | 1416.700 | 66.632 / 6.530 | 6.393 | 120.130 |
| image-png-compressed-deflate | 89883 | 89944 | 5.600 / 6.070 / 5.730 | 1.810 | 7.033 / 3.277 | 5.201 | 78.841 |
| graphics-shader-machine-words-synthetic | 262144 | 3672 | 102.520 / 92.229 / 93.061 | 367.210 | 17.057 / 9.309 | 20.890 | 33.235 |
| graphics-wgsl-source | 8267 | 2992 | 55.400 / 52.627 / 51.042 | 27.015 | 6.965 / 2.863 | 6.641 | 21.320 |
| ai-model-int8-tensor-synthetic | 1048576 | 21568 | 598.550 / 609.520 / 199.580 | 330.640 | 123.810 / 203.300 | 142.300 | 89.415 |
| native-executable-binary | 5437440 | 3023424 | 62903.000 / 61926.000 / 47342.000 | 34968.000 | 8287.500 / 2449.800 | 7866.700 | 12307.000 |
| structured-binary-records | 524288 | 352584 | 2417.800 / 2224.200 / 2164.200 | 1080.100 | 701.200 / 201.000 | 481.510 | 1415.300 |
| sparse-binary | 1048576 | 48424 | 1010.200 / 613.220 / 537.090 | 659.930 | 132.290 / 483.540 | 141.170 | 288.700 |
| random-binary-incompressible | 1048576 | 1048632 | 246.730 / 211.740 / 234.610 | 248.870 | 428.580 / 215.900 | 342.210 | 544.500 |
| repository-rust-source | 275729 | 87216 | 2370.600 / 1649.900 / 1034.900 | 893.940 | 432.130 / 108.390 | 375.540 | 671.280 |
| repository-technical-prose | 26666 | 15840 | 294.980 / 290.510 / 285.050 | 163.860 | 33.695 / 11.360 | 27.260 | 73.498 |
| natural-style-prose-generated | 586022 | 89832 | 2453.700 / 1988.600 / 1080.700 | 743.040 | 291.940 / 126.120 | 322.390 | 529.040 |
| json-records | 1209516 | 237792 | 5780.700 / 4907.300 / 2716.800 | 2303.600 | 1160.900 / 477.210 | 909.140 | 1769.200 |
| csv-table | 995409 | 429432 | 7389.800 / 6741.200 / 3576.400 | 4521.100 | 1830.100 / 367.270 | 1622.900 | 2386.600 |
| server-log-records | 1398527 | 394256 | 7573.000 / 7083.500 / 3722.600 | 3643.300 | 1466.600 / 485.480 | 1335.900 | 2329.200 |

## Density across all captured codec controls

Encoded bytes / raw bytes (lower is smaller). These size observations include Zstd, Brotli, and zlib settings whose timings were not rerun in the final speed filter. Earlier full timing captures remain historical rather than being relabeled as current.

| Dataset | RNBW | LZ4 | Snappy | Zstd-fast-3 | Zstd-1 | Brotli-0 | Brotli-5 | Zlib-fast | Zlib-default |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| image-rgba8-raster | 0.5630 | 1.0020 | 0.9999 | 0.9746 | 0.7928 | 0.7870 | 0.6079 | 0.7459 | 0.6805 |
| image-png-stored-deflate | 0.9283 | 0.9962 | 1.0001 | 0.9545 | 0.8180 | 0.8028 | 0.6091 | 0.7439 | 0.6848 |
| image-png-compressed-deflate | 1.0007 | 1.0040 | 1.0001 | 1.0001 | 0.9701 | 1.0001 | 0.9666 | 0.9646 | 0.9525 |
| graphics-shader-machine-words-synthetic | 0.0140 | 0.0509 | 0.4090 | 0.0471 | 0.0376 | 0.0386 | 0.0367 | 0.7748 | 0.0434 |
| graphics-wgsl-source | 0.3619 | 0.3698 | 0.3272 | 0.3197 | 0.2585 | 0.2731 | 0.2112 | 0.2804 | 0.2175 |
| ai-model-int8-tensor-synthetic | 0.0206 | 0.0190 | 0.0664 | 0.0019 | 0.0015 | 0.0021 | 0.0025 | 0.0343 | 0.0157 |
| native-executable-binary | 0.5560 | 0.5544 | 0.5636 | 0.5503 | 0.4342 | 0.4538 | 0.3469 | 0.4514 | 0.3823 |
| structured-binary-records | 0.6725 | 0.7595 | 0.7577 | 0.6997 | 0.5790 | 0.7177 | 0.5815 | 0.6347 | 0.6349 |
| sparse-binary | 0.0462 | 0.0382 | 0.0789 | 0.0336 | 0.0333 | 0.0392 | 0.0311 | 0.0399 | 0.0351 |
| random-binary-incompressible | 1.0001 | 1.0039 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0009 | 1.0002 |
| repository-rust-source | 0.3163 | 0.3332 | 0.3288 | 0.2975 | 0.2350 | 0.2542 | 0.1894 | 0.2902 | 0.2021 |
| repository-technical-prose | 0.5940 | 0.5840 | 0.5652 | 0.5777 | 0.4187 | 0.4447 | 0.3537 | 0.4761 | 0.3759 |
| natural-style-prose-generated | 0.1533 | 0.1558 | 0.1446 | 0.0952 | 0.0908 | 0.1123 | 0.0678 | 0.1227 | 0.0598 |
| json-records | 0.1966 | 0.2395 | 0.2464 | 0.2037 | 0.1317 | 0.1963 | 0.1100 | 0.1571 | 0.1306 |
| csv-table | 0.4314 | 0.5051 | 0.5024 | 0.5033 | 0.2681 | 0.3897 | 0.2292 | 0.3400 | 0.2874 |
| server-log-records | 0.2819 | 0.3256 | 0.3240 | 0.3104 | 0.1804 | 0.2517 | 0.1717 | 0.2142 | 0.1857 |

## Cost and correctness model

- Forward split reads and writes each input byte once, in 512-byte tiles (32 records), followed by 0–31 tail records. Inverse join does the same. Work remains O(N); unavoidable input/output traversal is not relabeled O(1).
- Added serialized bytes: **0**. Added per-record heap scratch: **0**. Fixed SIMD arrays are 16 × 32 B each; compiler register pressure spills some of them to the stack. Existing lane/output allocations remain. Reuse split preserves capacities; join allocates the returned output.
- Hardware detection is cached once per transform. Non-AVX2 targets retain correct scalar fallbacks. No alignment promise is imposed on caller bytes: all SIMD loads/stores are unaligned-capable.
- Each tile load is bounded by a complete 32-record tile. Each store initializes a disjoint output range within reserved capacity. Vector lengths are published only after the tile and tail writes finish. Equal lane lengths and total-size overflow are checked before join.
- The boundary test covers 100 record lengths × five offsets, every tail length, growing buffers, scalar parity, original byte reconstruction, and exact serialized-image equality. The codec fuzzer now includes a 16-lane seed, interleaved decode on accepted equal-lane mutations, and 96 changing-width/reuse/random-input cases.

## Experiments rejected instead of shipped

| Experiment | Measured result | Decision |
|---|---|---|
| Floor reciprocal rANS division with correction | rANS encode geomean 1.2651× baseline across 16 streams; unchanged decode 0.9674× | Reverted arithmetic change |
| Bounded-state rounded-up rANS reciprocal | rANS encode 1.1743× baseline; hybrid selection 1.0654× | Reverted; saved patch |
| Four histogram counter banks | Histogram 0.5400× baseline, but constant 1 KiB 1.70× and uniform 2 KiB 1.58× slower than serial; full reused encode mixed | Reverted; saved patch and all controls |
| Fixed `[Vec<u8>;16]` SIMD boundaries | Split 48.113 → 37.291 µs, but interleave 253.33 → 292.64 µs; full decode and unchanged LZ4 control both shifted slower | Reverted; no full-pipeline win established |

The reciprocal trials followed runtime constant-division research ([libdivide](https://libdivide.com/)) and the bounded-state principle in [ryg_rans](https://github.com/rygorous/ryg_rans/blob/master/rans_byte.h). No dependency or wire-format change was introduced. Their setup cost and code generation matter on these short per-lane streams; arithmetic instruction count alone did not predict a win.

## Stage coverage and profiling

The opt-in `bench-internals` feature exposes actual production kernels to Criterion. The original stage run contains **176 cases** across all 16 file datasets: split, sample, LZ parse/emit, LZ expansion, hybrid entropy selection, histogram, model normalization, rANS encode/decode, and Huffman encode/decode. It validates roundtrips outside timing. The final run adds the inverse 16-lane transpose. Raw-byte, token-stream-byte, lane-sample, and model denominators are separated. Some forced kernels are skipped by normal hybrid selection; their times must not be summed into total codec time.

Current stage gaps: individual match probe/extension, token emit, container assembly, independent Huffman table build, rank/select substeps, and Arena growth. These are not claimed measured merely because their parent operation is timed.

New `examples/lane_profile.rs` profiles real files using production APIs: split, interleave, reused serial encode, reused parallel encode, complete decode, and raw copy. It validates before the inner timer and prints batch total plus per-call time. File reads, setup, validation, startup, and reporting are included by process profilers. Example:

```sh
cargo build --locked --release --features bench-internals --example lane_profile
hyperfine --shell=none --warmup 2 --runs 10 "target/release/examples/lane_profile encode INPUT 16 100"
valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes target/release/examples/lane_profile decode INPUT 16 10
```

WSL uses Rust 1.99.1, Hyperfine 1.19.0, Valgrind 3.26.0; it is separate from the Windows Criterion comparison. The six Hyperfine cases below each execute 100 calls over 1 MiB input, with 8 Rayon workers and 2 warmups/10 process runs. Do not divide by 100 and call that isolated codec latency: the process also performs setup and byte-equality validation.

The WSL input uses the exact `model_weights()` generator in `benches/files.rs`; SHA-256 `047c56b1f45871edec4009ab6bdca9a5233412532c03e8bb05544bff5ebc45e5`. Raw Callgrind files are retained as `lane-encode-callgrind-wsl-2026-11-09.out` and `lane-decode-callgrind-wsl-2026-11-09.out` beside the readable annotations.

| WSL whole process, 100 calls | Mean ± σ (ms) | Median (ms) | Min–max (ms) |
|---|---:|---:|---:|
| split | 12.773 ± 0.692 | 12.508 | 12.081–13.927 |
| interleave | 12.767 ± 0.270 | 12.883 | 12.275–13.175 |
| encode | 70.717 ± 1.718 | 70.163 | 68.740–74.122 |
| encode-parallel | 68.823 ± 18.600 | 57.343 | 52.455–96.431 |
| decode | 38.164 ± 2.747 | 38.436 | 34.229–42.529 |
| copy | 9.852 ± 0.500 | 9.868 | 9.179–10.439 |

Callgrind whole-process totals, one worker and 10 timed calls plus setup: encode **115,086,973 instructions**, decode **64,415,015 instructions**. Encode attribution places LZ parsing at 60.96%, entropy choice at 16.80%, and the sample gate at 12.56% of whole-process instructions; split is now 4.70%. Decode attribution places LZ expansion at 37.65% and Huffman decode at 35.03%. These inclusive function counts are not wall-time percentages and overlapping rows must not be added. Cache and branch figures are simulations, not hardware counters. This points the next work toward LZ and Huffman rather than another format change.

Validation: 57 library tests, 3 binary tests, 2 codec integration tests, and 3 TRNMAP integration tests passed; 20,523 malformed-stream candidates included 224 directed header mutations. Clippy all targets with the stage feature and rustfmt passed. Standalone split/join Memcheck: **zero errors, zero definitely/indirectly/possibly lost bytes**, 548 B reachable at exit. The full Rust-test executable reported three possible-leak contexts (2,480 B) rooted in test-thread/Rayon TLS allocation; this report does not call that run a clean leak check. No invalid-access errors were reported. Raw outputs are retained.

GitHub CI now runs production stage measurements, builds the profiler, and runs byte-codec Hyperfine/Callgrind and standalone transform Memcheck on the pinned executable fixture. Local checks are complete; the new remote workflow is only verified when its run finishes.

## Evidence and corrections

- [Final 177-row normalized matrix](avx2-sixteen-lane-final-comparison-2026-11-09.txt) and [raw capture](avx2-sixteen-lane-final-matrix-2026-11-09.txt).
- [Repeated original-binary comparison](avx2-sixteen-lane-repeat-comparison-2026-11-09.txt) and [raw baseline](avx2-sixteen-lane-repeat-baseline-2026-11-09.txt).
- [All 176 stage intervals](codec-stages-baseline-summary-2026-11-09.txt), [histogram cutoff](histogram-cutoff-summary-2026-11-09.txt), and [rejected fixed-array trial](avx2-sixteen-lane-fixed-array-comparison-2026-11-09.txt).
- [Emitted SIMD assembly](avx2-sixteen-lane-slice-assembly-2026-11-09.txt): unpacks and vector stores are present; transpose is inlined; stack spills remain.
- [Hyperfine raw](lane-simd-hyperfine-wsl-2026-11-09.txt), [Hyperfine JSON](lane-simd-hyperfine-wsl-2026-11-09.json), [encode Callgrind](lane-encode-callgrind-wsl-2026-11-09.txt), [decode Callgrind](lane-decode-callgrind-wsl-2026-11-09.txt).
- [Split Memcheck](lane-split-memcheck-wsl-2026-11-09.txt), [join Memcheck](lane-join-memcheck-wsl-2026-11-09.txt), [test-executable Memcheck with TLS findings](avx2-sixteen-lane-memcheck-2026-11-09.txt), [tests/fuzz log](codec-stage-tests-2026-11-09.txt).
- Corrected six dense-TRNMAP store build/decode normalization rows: the fixture has **32,768 keys**, not 1,000,000. Original totals were unchanged; the earlier ns/key values understated cost by 30.517578125×. See [corrected table](store-ns-op-normalized-2026-11-09.txt).

Retained Windows benchmark executable SHA-256: `523522bf2663e471919b342bb41e6b61b4479b4f14c1b4d4e83ab49dc8dc6383`. Original stage-baseline executable SHA-256: `287b49fb919071c74938439b96af225f1ac0ae04ee8f9453b07e672e28be845e`. Local binaries are ignored; raw captures and normalized comparisons are versioned.
