# RNBW performance path ledger

Audit date: 2026-11-09
Branch observed: `main`
Scope: source-level path and growth audit of the integer index, L2/L3 store, byte codec, entropy paths, TRNMAP, CPU dispatch, test/fuzz harnesses, and current benchmark coverage.

This ledger records measured end-to-end times where a benchmark exists and labels the remaining operations **unmeasured**. It does not assign internal-stage timings from whole-operation measurements. The current worktree replaces the dense `BitmapArena` form with arena-backed TRNMAP images and adds an in-place dense-TRNMAP mutation path. Any benchmark capture made before that change is historical and is not evidence for the current dense form. The experimental unaligned hash-load edit was reverted because its paired run was inconclusive; the captures remain in `docs/benchmarks/general-lz-word-load-*`.

## Naming used in this ledger

- **L1 index** is `Rnbw` / `RnbwView` in [`src/lib.rs`](../../src/lib.rs): a sorted `u32` set represented by one directly queryable byte image. Each trie node is a four-bit mask for four two-bit branches; set branches imply the key bits. Per-level rank samples locate nodes. Leaves index a unary bucket vector; low suffixes are bit-packed. `contains` does not decode the set and does not allocate a runtime directory.
- **L2** is `RnbwStore` in [`src/store.rs`](../../src/store.rs): a 64-bit generational group table, typed group-record handles, a variable-record paged arena, and a TRNMAP tier map. Each group describes 16 chunks, with empty/full/inline/run/quotient/TRNMAP forms. Dense chunk TRNMAP images are allocated from the paged arena; there is no bitmap payload form in the current source.
- **L3** is the optional, explicit hot-pocket array attached to L2. `PerChunk32` stores up to 18 suffixes in 32 B per chunk; `PerPair64` stores up to 36 across a chunk pair in 64 B. The 2 MiB L3 array is allocated only when a pocket is promoted.
- **Byte stream codec** is `LaneLz` / `LaneLzEncoder` in [`src/lane_lz.rs`](../../src/lane_lz.rs), with entropy implementation in [`src/rans.rs`](../../src/rans.rs). It is separate from the L1 integer index and the L2/L3 set store.

## Measurement conditions and evidence strength

The current Windows captures report x86-64, AVX2 dispatch, Rust/Cargo 1.95.0, and Rayon configured for 16 workers. The exact CPU SKU and cache sizes were not exposed. Criterion runs are short exploratory runs (typically 10 samples, 50–200 ms requested warm-up/measurement); the raw logs contain intervals and warnings where Criterion extended collection. Timings below are medians/point estimates from the named summary, not portable guarantees. No CPU counters were captured.

WSL has been used earlier in this session. A fresh launch from the current Codex shell now fails (`wsl --list` returns `E_ACCESSDENIED`; explicit distro launch returns `E_ACCESSDENIED` or `ERROR_NO_SYSTEM_RESOURCES`), so that attempt produced no new local profile. This does not erase the earlier WSL use or establish that the host lacks Hyperfine or Valgrind/Callgrind. The repository's Ubuntu `profiling-tools` workflow also installs Hyperfine and Valgrind, runs the process matrix and Callgrind/Massif/Memcheck, and uploads `reports/`; its artifact is required before treating that workflow run as profiling evidence.

## Public operation and growth ledger

`N` is number of values/input bytes as appropriate; `D` is selected L1 trie depth (at most 16 for `u32`); `B` is a chunk's key count; `G` is the key count decoded from one L2 group; `H ≤ 16` is the number of hot chunks in that group; `W = ceil(N/64)` is the TRNMAP leaf-word count. Fixed-width limits make several inner multipliers bounded, but the bound is still recorded.

| Surface / path | Growth and worst case | Current timing evidence |
|---|---|---|
| `Rnbw::encode` / `encode_width` (L1 build) | Input validation O(N); prefix counting O(N); exact split evaluates at most 17 depths; level masks rescan values up to D times. O(N·D), hence O(N) for fixed 32-bit keys, with a ≤16-level rescan factor. Output and scratch O(image + N) bytes. | 1M-key RNBW query-image build is 23.076–27.063 ms across five distributions in the current store summary; full data in [store summary](../benchmarks/general-encode-decode-store-current-summary-2026-11-09.txt). The 16-file byte-codec table is a different path and is below. |
| `Rnbw::encode_unsorted` | Copy + sort + deduplicate: O(N log N), O(N) extra values; then L1 build. | No separate timing in current matrix. |
| `Rnbw::from_bytes`, `RnbwView::from_bytes` | Full format validation O(image bytes + represented values/bucket metadata); owned form adds an O(image) copy; view borrows. | Integer codec harness has parse and parse-plus-decode cases; full five-distribution results in [codec Criterion capture](../benchmarks/codec-current-2026-11-09.txt). |
| `Rnbw::contains` / `RnbwView::contains` | O(D + bucket search). Mask/rank descent is bounded by D and at most seven packed-word popcounts per rank. SIMD scans selected small byte-aligned buckets; other bucket widths binary-search packed suffixes. The unary zero-select helper advances from 64-leaf samples; worst-case work is proportional to the encoded gap/bucket span (up to O(N)), not proven constant by current source. | L1 hit/miss batches of 1,024 are in [store summary](../benchmarks/store-summary-2026-11-09.txt); per-layout details below. The integer codec matrix also times L1 hits/misses. |
| `Rnbw::decode` / `RnbwView::decode` | O(N + image bytes visited), plus O(N) output allocation. | Integer decode and parse-plus-decode are in [codec Criterion capture](../benchmarks/codec-current-2026-11-09.txt). |
| L1 entropy wrapper `to_compressed_bytes` / `from_compressed_bytes` | O(image bytes + 256-symbol model work); histogram and coding are linear in image size. | Current integer codec matrix includes compressed forms where applicable; individual rANS substage is not separately timed. |
| `PlainEf::encode` / `contains` / `decode` | Build O(N); membership O(select-sample gap + packed low-bit search), worst gap may scale with encoded high-bit span; decode O(N). Reference baseline only. | Build/query/decode paths in [codec Criterion capture](../benchmarks/codec-current-2026-11-09.txt). |
| `LaneLz::encode*` one-shot/reuse serial | O(input bytes + output bytes) for current bounded two-candidate matching; each consumed position is advanced once, with at most two candidate extensions. Fixed 2^18-entry `u64` hash heads are 2 MiB per initialized workspace; touched-head list can add up to about 1 MiB/workspace. Multi-lane split/pack adds linear passes. | All 16 file workloads × one-shot and reuse-serial in [current file summary](../benchmarks/general-encode-decode-store-current-summary-2026-11-09.txt) and [complete 367-case report](../benchmarks/rnbw-codec-results-2026-11-09.txt). |
| `LaneLz::encode*` parallel, semantic lanes | O(N) useful work + task dispatch/synchronization; dispatch gate is 16 KiB aggregate input. Tiny cases can lose to serial. | Full-size file matrix and cutoff sweep in [lane codec report](../benchmarks/codec-lane-current-2026-11-09.txt). |
| `LaneLz::encode_interleaved*` / `encode_u32_byte_planes*` | O(N); width-4 split has AVX2 plus scalar fallback; byte-plane transform is additional linear read/write. Single-lane horizontal mode uses 64 KiB blocks and rehashes up to 64 KiB prior history per block, bounded O(N) overhead. | File and lane reports above; the 16 KiB cutoff is not universally profitable for single-lane data. |
| `should_skip_lz_sample` | For lanes ≥4 KiB, fixed 64 × 32-byte samples; fixed 256-byte histogram and 4,096-slot 3-gram table. O(1) in N for fixed constants, 16,896 B scratch. | Included only in end-to-end encode timings; no isolated sample-gate timer. |
| `LaneLz::from_bytes` / image parser | O(serialized bytes + lane descriptors), with checked ranges and validation. | Parser case appears in integer/lane codec benches only where the public `LaneLz` API is exercised; no dedicated parse-only table is captured in the current file summary. |
| `decode_lanes`, `decode_interleaved`, `decode_u32_byte_planes` | O(compressed bytes + output bytes); independent lanes may run in parallel above the gate. Interleave restoration is another O(N) pass. | Serial/parallel decode rows across all 16 file datasets in the timing table below. |
| `decode_interleaved_cow` | O(1) and zero-copy for an already-raw single lane; otherwise the owned decode path is O(N) and allocates. | The 16-corpus report records borrowed eligibility; decode benchmark row is explicitly a borrow-when-possible path, not full decompression. |
| LZ match loop / hash / emit | O(N) with two fixed candidate checks per selected position, 64 KiB history limit, SIMD match extension where available. Token/literal output O(N). | Only included in LaneLz whole-encode time; no internal hash, extension, token-emission, or allocation-stage timings. |
| LZ decode overlap copy | O(output bytes + tokens); long matches are copied by repeated doubling with `extend_from_within`, bounded by bytes emitted. | Included in decode matrix; an alternate overlap-copy experiment was rejected and recorded in [rejected experiment](../benchmarks/lz-decode-overlap-copy-rejected-2026-11-09.txt). |
| `symbol_histogram`, `build_model`, Huffman path | Histogram O(N); alphabet/model work is bounded by 256 symbols; Huffman symbol coding/decoding O(N), with fixed-width code/table work. | No standalone entropy timer; only full codec totals. |
| rANS encode/decode | O(N) symbols, fixed 4-state/12-bit model. Encode has per-symbol integer division and remainder; decode has table/state work per symbol. | No standalone rANS timer; only full codec totals. |
| `RnbwStore::from_sorted` | O(N) sorted validation + one traversal of keys/groups + per-group form encoding; O(encoded store + transient group values). | 1M keys: L2 p32/p64 and L1 build by distribution in [current summary](../benchmarks/general-encode-decode-store-current-summary-2026-11-09.txt); complete medians in [store summary](../benchmarks/store-summary-2026-11-09.txt). |
| `RnbwStore::contains` | O(1) form dispatch for empty/full/inline/TRNMAP/pocket; run forms O(log run_count); quotient form delegates to L1 suffix search and inherits bucket-span worst case. Dense TRNMAP membership is O(1) after chunk, GenTable, typed-handle, and arena lookup. | Broad 1M-key hit/miss batches across L1, L2 p32/p64, and L2+L3 p32/p64 are in [store summary](../benchmarks/store-summary-2026-11-09.txt). A fresh isolated 32,768-key dense-TRNMAP comparison is in [dense TRNMAP capture](../benchmarks/store-dense-trnmap-typed-arena-2026-11-09.txt). |
| `RnbwStore::insert` / `remove` | Dense TRNMAP chunks use an in-place update path: O(log_64 CHUNK_COUNT) TRNMAP summary/dirty propagation plus constant metadata work. Other forms rebuild the owning 16-chunk group; sorted `Vec::insert/remove` adds O(B) movement, and hot-peer refresh can repeat group work up to H≤16. | A 1M-key update/dirty matrix is in [`store-update-arena-optimized`](store-update-arena-optimized-2026-11-09.txt); the focused dense insert/remove pair is in [dense TRNMAP capture](../benchmarks/store-dense-trnmap-typed-arena-2026-11-09.txt). These measure distinct update cases; do not treat one as a universal mutation latency. |
| `RnbwStore::decode` | O(number of groups + N) and O(N) output allocation; currently checks all group slots. | L2 and L2+L3 p32/p64 decode by distribution in store summary. |
| `promote_range` | O(groups touched × (group decode + in-range pocket packing)); each group is decoded once per range refresh. Pocket packing is at most 36 suffixes/chunk pair. | Full-range p32/p64 timings and the verified uniform-p32 A/B are in [current summary](../benchmarks/general-encode-decode-store-current-summary-2026-11-09.txt). |
| `demote_range` | O(range chunks + touched groups × group decode + promoted pockets packed); current indexed path avoids revisiting untouched chunks and batches peers by group. | Full-range per-shape timings are in [`store-demote-indexed`](store-demote-indexed-2026-11-09.txt). Dense and clustered no-promotion cases are near no-op timings; uniform and Morton-shell perform actual demotions and take milliseconds. |
| `dirty_chunks` | O(dirty summary words + dirty leaf words + returned chunks); TRNMAP summaries skip clean ranges. Output allocation O(dirty chunk count). | Insert/remove plus hot-reset dirty-drain timings are in [`store-update-arena-optimized`](store-update-arena-optimized-2026-11-09.txt); TRNMAP dirty mark/next/take has a separate component benchmark below. |
| `VarArena::alloc/get/retire` | Handle decode/check O(1); free-size class reuse is O(1) when a free block exists. Page acquisition/free-block search can grow with pages/pages-of-class; exact worst-case allocator scan should be benchmarked as arena population grows. | Unit/fuzz correctness coverage exists; no isolated allocator timings. |
| Dense arena-backed `TrnMap` chunk | Build scans the fixed 1,024 leaf words plus input values; membership O(1) after arena handle validation; update O(log_64 CHUNK_COUNT) summary repair. The live image stores two trit planes and TRNMAP metadata, not a one-bit bitmap. | Current isolated comparison: 32,768 even keys use 157,512 B total store bytes; build estimate 99.806 µs; 1,024 hits 13.879 µs; 1,024 misses 14.233 µs; insert/remove pair 227.13 ns; decode 145.23 µs. See [capture](../benchmarks/store-dense-trnmap-typed-arena-2026-11-09.txt). The store is ~10× slower than Roaring for hot membership and uses ~19.2× its bytes on this narrow workload; no across-the-board win is claimed. |
| `TrnMap::get` / `top` | O(1). | 1M hot `get` 2.10 ns vs scalar byte 0.68 ns; exact matrix in [TRNMAP report](../benchmarks/trnmap.md). |
| `TrnMap::set` / `mark` / `take_dirty` | O(log_64 N) summary propagation; stops when ancestors already match. | 1M changing `set` 22.31 ns; dirty mark+next+take composite 16.03 ns. |
| `TrnMap::next_dirty` | O(log_64 N) to locate next dirty summary and leaf word; return is a word index, not per-trit enumeration. | Only timed in the composite dirty benchmark above. |
| `TrnMap::find` | O(log_64 N) hierarchy search; ≤5 levels at 16M trits. | 16M absent find 11.33–12.45 ns; 1M uniform hot find 4.57 ns. Full shape/size matrix in TRNMAP report. |
| `TrnMap::run` | O(log_64 N) via find-next-different, even for a long homogeneous run. | 16M all-negative run 14.69 ns; full matrix in TRNMAP report. |
| `TrnMap::count` | O(ceil(W/64) + matching leaf words), with a dense fallback O(W); asymptotically O(N/64) worst case. Current docs claim O((matching words + 1)·log_64 N), which omits the unconditional level-one summary scan. | 1M uniform count 101.97 µs; 16M sparse count 75.12 µs; dense/all-class rows in report. **Complexity documentation should be corrected after source review.** |
| `TrnMap::or_assign` / `and_assign` | O(W) worst case; implementation skips RHS leaf words that cannot alter output and repairs only changed summaries. | 1M clustered OR 165.56 µs / AND 66.50 µs. |
| `TrnMap::not_assign` | O(W) leaf word pass plus summary-plane swap/repair. | 1M 19.41 µs. |
| `TrnMap::new` / `in_buffer` | O(total storage words) initialization; memory O(N/64 + summary records + dirty records). | 1M allocate/init 10.45 µs; exact storage matrix in [TRNMAP report](../benchmarks/trnmap.md). |
| GPU restart framing / wgpu encode/decode | Host packing, buffer allocation/upload, dispatch, synchronous readback/wait and unpack all contribute. Shader LZ work is serial inside one workgroup per lane; fixed 65,536-entry history/head arrays per lane. | **No GPU timing artifact in this checkout.** CPU-only evidence cannot substantiate a GPU advantage. |
| Real AI artifact path | Safetensors parse O(header + tensor count); payload codecs O(payload bytes). | **Unmeasured:** `RNBW_AI_CORPUS` is unset and no `real-ai-codec=` output is present. Synthetic AI corpus results are not a real model benchmark. |

### Growth and explosion audit

| Path | Worst growth found | Finding |
|---|---|---|
| L1 sorted build | O(N·D), D≤16 | Bounded linear scan multiplier for fixed `u32`; not quadratic. |
| L1 unsorted build / benchmark dataset canonicalization | O(N log N) | Sorting is expected and documented; no quadratic algorithm identified. |
| L1/PlainEF sampled select | O(encoded sample gap / bucket span), worst may scale with N or universe span | Not proved O(1); add adversarial skewed-bucket tests and latency cases before claiming bounded constant lookup. |
| LaneLz encode/decode | O(N) under fixed 2 candidates and fixed window; output/temporary memory O(N + workspace) | No unbounded backtracking or quadratic loop found. Parallel single-lane history rehash adds a bounded linear pass. |
| Entropy | O(N) plus fixed 256-symbol alphabet | rANS division/remainder are likely throughput costs, not asymptotic explosions. |
| Store insert/remove | Dense TRNMAP chunk fast path O(log_64 CHUNK_COUNT); other forms O(G·(H+1)) with H≤16 and O(B) sorted-vector movement | Dense map membership updates mutate in place; non-TRNMAP forms still rebuild the 16-chunk group and may refresh hot peers. Full current-path timing is in the 186-row Criterion report. |
| Store demote | O(range chunks + touched groups·G + pocket packing) | Current indexed demotion visits touched groups and batches peer work. Full-range performance is measured; partial-range/adversarial hot-neighbor costs remain open. |
| TRNMAP count / boolean operations | O(N/64) worst | Linear in packed words by design. No superlinear path identified. |
| Corpus generation | `make_dataset` sort O(N log N); Morton-shell uses HashSet and bounded attempts, then sort | Not timed codec work. No observed unbounded quadratic core path; HashSet adversarial collision worst-case is theoretical. |

No unbounded quadratic-or-worse path was identified in the production codec/index implementation during this source pass. This is not a proof of absence: sampled-select worst-case span, non-TRNMAP group rewrite amplification, partial-range demotion, and allocator growth/search under long mutation sequences still need dedicated adversarial measurements.

## Whole-file byte codec path timings

All values are medians in microseconds from the current summary. Encode columns are **one-shot / workspace-reuse serial / workspace-reuse parallel**. Decode columns are serial image / parallel-encoded image. Comparator columns show the fastest measured external encode and decode on that dataset; the comparator is not required to have the same compression ratio. This is a 16-corpus, 10-sample quick run, not a stable cross-machine grade.

| Dataset (raw B) | RNBW ratio | Encode one-shot / serial / parallel (µs) | Decode serial / parallel (µs) | Fastest comparator encode / decode |
|---|---:|---:|---:|---:|
| RGBA raster (524,288) | 0.5630 | 6,714 / 5,484 / 2,695 | 1,632 / 1,627 | Snappy 29.8 / Snappy 19.3 |
| PNG stored-deflate (131,278) | 0.9283 | 3,227 / 2,673 / 2,573 | 1,440 / 1,372 | Snappy 6.7 / Snappy 5.2 |
| PNG compressed-deflate (89,883) | 1.0007 | 6.3 / 6.1 / 5.9 | 1.9 / 2.0 | Snappy 5.1 / Zstd-fast 2.3 |
| Synthetic shader words (262,144) | 0.0140 | 482 / 93.6 / 98.3 | 376.5 / 366.7 | Snappy 19.1 / Zstd-fast 6.7 |
| WGSL source (8,267) | 0.3610 | 458 / 52.8 / 51.8 | 26.9 / 27.3 | Snappy 6.7 / LZ4 2.7 |
| Synthetic AI int8 (1,048,576) | 0.0206 | 3,136 / 2,725 / 2,480 | 1,625 / 1,626 | Zstd-fast 76.7 / LZ4 237.4 |
| Native executable (5,437,440) | 0.5560 | 63,512 / 63,487 / 46,982 | 35,063 / 34,084 | Snappy 7,733 / LZ4 2,427 |
| Structured records (524,288) | 0.6725 | 2,723 / 2,271 / 2,220 | 1,074 / 1,079 | Snappy 496.8 / Snappy 180.2 |
| Sparse binary (1,048,576) | 0.0462 | 1,104 / 588 / 584 | 672 / 685 | Snappy 126.3 / Zstd-level-1 230.4 |
| Random binary (1,048,576) | 1.0001 | 241 / 212 / 214 | 209 / 215 | Snappy 401.4 / Snappy 210.9 |
| Rust source (275,729) | 0.3163 | 2,312 / 1,586 / 1,007 | 888 / 871 | Snappy 364.7 / LZ4 96.3 |
| Technical prose (26,666) | 0.5937 | 723 / 285 / 281 | 164 / 165 | Snappy 28.1 / Snappy 13.4 |
| Generated natural prose (586,022) | 0.1533 | 2,203 / 1,774 / 1,044 | 743 / 733 | LZ4 291.4 / LZ4 129.7 |
| JSON records (1,209,516) | 0.1966 | 5,291 / 5,142 / 2,758 | 2,282 / 2,290 | Snappy 884.6 / LZ4 492.2 |
| CSV (995,409) | 0.4314 | 6,525 / 6,428 / 3,257 | 4,532 / 4,518 | Zstd-fast 1,615 / LZ4 373.0 |
| Server logs (1,398,527) | 0.2819 | 7,548 / 6,482 / 3,890 | 3,777 / 3,700 | Snappy 1,272 / LZ4 505.2 |

Full per-codec ratios, confidence intervals, all eight external baselines, and raw output are in [current summary](../benchmarks/general-encode-decode-store-current-summary-2026-11-09.txt), [full codec report](../benchmarks/rnbw-codec-results-2026-11-09.txt), and [raw Criterion log](../benchmarks/general-encode-decode-current-2026-11-09.txt).

## L1 index vs L2 vs promoted L3

This current-worktree matrix uses 1M-key shaped corpora and 1,024 hit/miss probes. Values are microseconds per batch; divide by 1,024 for the measured batch average per probe. `L1` is the RNBW directly queryable image. L2/L3 columns are store APIs; p32/p64 pocket layouts are kept separate. Current source uses arena-backed TRNMAP dense chunks, not the bitmap form in the earlier store captures.

| Shape | L1 hit / miss | L2 p32 hit / miss | L2+L3 p32 hit / miss | L2 p64 hit / miss | L2+L3 p64 hit / miss | L3 bytes |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 133.33 / 94.063 | 185.78 / 110.21 | 50.707 / 41.572 | 110.76 / 110.69 | 51.441 / 39.809 | 2,097,152 |
| Clustered | 474.36 / 56.263 | 12.029 / 5.6052 | 12.251 / 5.7175 | 12.238 / 5.4715 | 11.601 / 5.4064 | 0 |
| Dense | 366.6 / 16.596 | 7.0225 / 2.9269 | 7.4976 / 3.1275 | 7.0112 / 3.0509 | 7.2945 / 3.2577 | 0 |
| Morton-shell | 382.62 / 42.107 | 194.23 / 9.4022 | 195.63 / 9.4254 | 225.51 / 9.0117 | 199.26 / 8.9383 | 2,097,152 |
| ECS | 372.36 / 16.723 | 13.596 / 3.060 | 15.010 / 3.3492 | 13.734 / 3.0359 | 14.370 / 3.2777 | 0 |

The matrix has 186 timed Criterion rows, including builds, hits/misses, update/dirty-drain, promotion/demotion, and decode. Full intervals and exact commands are in the [current store summary](../benchmarks/store-trnmap-current-summary-2026-11-09.txt) and [raw log](../benchmarks/store-trnmap-current-2026-11-09.txt). L3 clearly improves uniform membership in this run; Morton-shell is noisy and mixed. Clustered, dense, and ECS have no L3 allocation, so their L2+L3 comparison is a separate L2 repeat. The old [store summary](../benchmarks/store-summary-2026-11-09.txt) and the earlier L2/L3 table are historical bitmap-form data and must not be used to grade the current TRNMAP representation.

Only the uniform p32 full-range promotion has a captured source-level A/B pair: 454.070 ms → 57.161 ms (7.94× faster). Other promotion rows are not accepted as wins without a paired baseline.

## TRNMAP measured paths

TRNMAP results are a separate ternary-index benchmark, not a byte-codec result. Exact storage grows from 440 B at 64 trits to 6,127,456 B at 16M trits. The source-level complexity and representative current measurements are:

| Operation | Complexity | Representative timing |
|---|---|---:|
| get, 1M hot index | O(1) | 2.10 ns |
| find, 1M uniform hot start | O(log_64 N) | 4.57 ns |
| absent find, 16M all-zero | O(log_64 N) | 11.33 ns |
| run, 16M all-negative | O(log_64 N) | 14.69 ns |
| count, 1M uniform | O(N/64) worst | 101.97 µs |
| count, 16M sparse | O(N/64) worst | 75.12 µs |
| changing set, 1M map | O(log_64 N) | 22.31 ns |
| dirty mark + next + take, 1M | O(log_64 N) composite | 16.03 ns |
| clustered OR / AND, 1M | O(N/64) worst | 165.56 / 66.50 µs |
| NOT, 1M | O(N/64) | 19.41 µs |
| allocate/init, 1M | O(N/64) | 10.45 µs |

Full 64 →16M shapes and scalar comparison timings are in [`docs/benchmarks/trnmap.md`](../benchmarks/trnmap.md). That report's `count` complexity paragraph needs correction to match the current unconditional summary-record scan described above.

## Why the byte codec is still slower

Current full-operation medians show substantial gaps: for 1 MiB synthetic AI bytes, RNBW reuse-serial encode is 2.725 ms vs the fastest measured Zstd encode at 76.7 µs; RNBW decode is 1.625 ms vs LZ4 at 237.4 µs. For 5.44 MiB native executable bytes, RNBW decode is 34.084 ms vs LZ4 at 2.427 ms. Strong density on selected synthetic shapes (shader ratio 0.0140) does not imply fast execution on other data.

The code makes several full passes: lane split/interleave, LZ hash/extend/token production, byte histogram/model, entropy coding, container assembly; decode parses streams, entropy-decodes, LZ-expands, then re-interleaves. rANS encode performs integer divide/remainder per symbol. These mechanisms explain likely cost centers but have **not** been isolated with cycle counters. The current 2 MiB hash-head workspace and per-lane scratch can also pressure cache; 16 concurrently initialized workspaces can reserve at least 32 MiB for hash heads before touched-head lists and token buffers. This is a cost hypothesis grounded in source layout, not a measured cache-miss result.

The latest unaligned-u32 hash-load candidate is not retained: its paired Criterion runs show broad timing movement in the unchanged decode control (including double-digit-percent shifts), so the encode differences cannot be confidently attributed to the hash edit. Logs are preserved. No source-format change is approved or made in this audit.

## Latest research directions and what they suggest for experiments

Research checked on 2026-11-09. These results are directions, not predicted RNBW gains. No technique below is applied. The user has reserved approval for changes to the storage/codec structure.

| Direction | Reported result / core idea | RNBW-specific experiment and cost to measure |
|---|---|---|
| **Format-aware reversible transform graphs (OpenZL, ICDE 2026)** | The graph model composes reversible transforms and codecs into a self-describing plan; OpenZL emphasizes structured inputs and a universal decoder. Its 2026 LZ report uses a changeable wire format and backend entropy choices, reporting >2× Zstd decode in its own Silesia run. | Test field-wise transforms only on corpus with known schema (records, ECS, graphics, tensors), while keeping one hybrid format. For each transform, count bytes read/written (typically an extra 2N traffic per forward/inverse pass) and metadata bytes; benchmark transform+encode+decode end to end. No transform is universally useful on byte soup. |
| **SIMD-friendly Huffman / PivCo-Huffman (2026 preprint)** | A wavelet-tree-inspired pivot-coded Huffman is proposed for SIMD encode/decode; paper also combines ANS selectively at skewed nodes. | Compare against current canonical Huffman and rANS on identical per-lane symbols. Cost: new code representation, pivot/tree metadata, SIMD tables; overhead is format-dependent and must be recorded exactly. Measure bytes/symbol and CPU instructions/symbol, not just entropy stage. |
| **Interleaved ANS / FSE** | Zstd standardizes Huffman for literals and FSE for sequence symbols; interleaving independent coder states hides serial state dependency. RNBW already uses four rANS states. | Sweep state count and block size without changing outer transform semantics. Additional serialized state cost is `(new_states − current_states) × serialized_state_bytes` per entropy stream; table setup is amortized over block symbols. Measure short files separately because setup may dominate. |
| **Encode-time decode-parallel LZ (ACEAPEX, 2026 preprint)** | Resolves references to absolute offsets to remove back-reference dependency chains; authors report CPU multithread and GPU decode gains on FASTQ/enwik9 with byte-perfect checks. | Determine whether a lane token layout can resolve absolute references while preserving compact 32-bit token + literal stream. Cost is extra offset bits/bytes per affected match and index/metadata; measure density and decode time together. |
| **Match-granularity tuning** | A follow-up 2026 preprint reports that increasing minimum match length by distance class improved its GPU decode throughput and ratio across its tested corpora by removing short far matches. | Sweep minimum match thresholds in the existing matcher, with exact roundtrip and same corpus. Current 32-bit token width means no fixed extra metadata, but skipped matches increase literal bytes; record encoded bytes, encode time, decode time, and match-length histogram. Treat the published GPU result as a hypothesis for CPU, not a prediction. |
| **Approximate/long-range LZ precompression (SEA 2026)** | Shared-memory parallel approximate parsing captures long-range repeats that a short sliding window misses, then delegates local redundancy to a standard backend. | Test only large repetitive source/log/model corpora. Cost is an additional input pass plus an index/phrase structure whose bytes per input byte must be measured; expected work remains linear or near-linear for the engineered algorithm, but small-file latency will likely lose. |
| **GPU-oriented string dictionary compression (GSST, 2025)** | FSST-derived format is designed around SIMT-friendly independent strings; TU Delft reports 191 GB/s decompression on A100 and better ratio/throughput tradeoffs than tested nvCOMP LZ formats for its string data. | Consider only string-heavy lanes, with CPU and GPU parity and transfer-inclusive benchmarks. Current wgpu shader has serial one-workgroup-per-lane work and synchronous readback; split/independent jobs, device-resident buffers, and batch size must be measured. GPU launch + PCIe transfer can dominate small streams. |
| **Domain transforms / numeric codecs (ASPLOS 2025; IoTDB SIMD 2025)** | Scientific float codecs combine predictor/bit-plane/union-find transforms with lossless entropy; IoTDB work decodes packed delta data in SIMD and fuses query operators to avoid materializing intermediates. | Add real typed corpora (f32/f64, integer IDs, ECS columns) and evaluate reversible XOR/delta/bitshuffle/byte-shuffle/zig-zag/bit-packing as candidate nodes inside the same codec. Each materialized transform is O(N) and adds at least one read + one write; fused decode/query may avoid an O(N) output buffer. Domain-specific claims must stay scoped to matching data. |
| **Enumerative / combinadic coding (2026)** | Enumerative coding approaches order-0 entropy on narrow alphabets with a per-block model overhead; block size changes ratio and reset cost. | Compare only for low-alphabet streams where current Huffman/rANS model overhead is large. Cost includes binomial/count-table work and model bytes; compute `model_bytes/block_bytes` across block sizes before accepting. |
| **Learned compression** | 2025 research demonstrates large models can improve ratios on text/image/audio/video but relies on costly prediction/model inference; it is a different speed/space point from LZ codecs. | Keep as an archival/offline research comparator, not a hot-path candidate unless the user explicitly selects ratio over latency. Report model weights, warmup, memory, and encode/decode latency in totals. |

Primary sources: [OpenZL graph model](https://arxiv.org/abs/2605.09928), [OpenZL LZ performance and wire-format discussion](https://openzl.org/blog/2026-09-29-lz-in-openzl/), [PivCo-Huffman](https://arxiv.org/abs/2606.05765), [ACEAPEX](https://arxiv.org/abs/2606.04268), [match-granularity study](https://arxiv.org/abs/2607.18541), [SEA 2026 approximate LZ77](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2026.16), [GSST](https://doi.org/10.1145/3759441.3759450), [ASPLOS 2025 CPU/GPU scientific compression](https://userweb.cs.txstate.edu/~burtscher/papers/asplos25.pdf), [ICDE 2025 SIMD encoded IoT aggregation](https://sxsong.github.io/doc/25ICDE-SIMD.pdf), and [Zstandard format RFC 8878](https://www.rfc-editor.org/rfc/rfc8878.html).

## Benchmark gaps and next evidence to collect

1. Add L2 no-op insert-present/remove-absent, partial-range demotion, and allocator page/size-class growth cases. Existing matrix covers successful insert/remove, dirty drains, full-range demotion, and hot/cold p32/p64 membership; keep setup outside timed closures.
2. Add adversarial L1/PlainEF key distributions with one heavily populated leaf bucket and report max bucket size, select-word span, and ns/query; this resolves the O(bucket-span) risk.
3. Add internal-stage cost breakdowns for split, LZ match, token/literal emit, histogram/model, Huffman/rANS, pack, entropy decode, LZ expansion, and interleave. Current end-to-end measurements cannot identify which pass costs the largest share.
4. The Linux Hyperfine and Callgrind profiling job now passes. Its 15 JSON/Markdown cases and Valgrind files are preserved in the workflow artifact and summarized in [`rnbw-index-hyperfine-process-2026-11-09.txt`](../benchmarks/rnbw-index-hyperfine-process-2026-11-09.txt). The current report has function-level Callgrind attribution; the next run adds optimized line tables and readable `massif.txt`. Add `perf`/PMU only when a runner exposes counters.
5. Run the opt-in Safetensors harness against an actual model artifact; keep full file and tensor-payload results separate. GPU requires device-resident/batched benchmarks on an actual GPU runner.
6. Keep L1/L2/L3 comparison tied to each workload. Current data contradicts a universal “L3 always faster” claim; do not change the chosen representation based only on a favorable uniform batch.

## Audit coverage and file manifest

The source pass covers all tracked Rust modules, all Criterion benches and support modules, both deterministic mutation fuzz tests, both WGSL shaders, Cargo manifests, README/design RFC, benchmark corpus descriptions, benchmark scripts, and GitHub workflow. Generated corpus/binary files are treated as hashed artifacts rather than source text. A SHA-256 byte manifest is stored beside this ledger in [`rnbw-file-manifest-2026-11-09.tsv`](rnbw-file-manifest-2026-11-09.tsv).

Current benchmark evidence index:

- [`general-encode-decode-store-current-summary-2026-11-09.txt`](../benchmarks/general-encode-decode-store-current-summary-2026-11-09.txt) — current 16-file encode/decode and store build/decode/promotion medians.
- [`store-summary-2026-11-09.txt`](../benchmarks/store-summary-2026-11-09.txt) — 95 L1/L2/L3/roaring Criterion groups with 1,024-probe query cases.
- [`store-trnmap-current-summary-2026-11-09.txt`](../benchmarks/store-trnmap-current-summary-2026-11-09.txt) and [raw output](../benchmarks/store-trnmap-current-2026-11-09.txt) — current bitmap-free store matrix, 186 Criterion rows, full intervals and five-shape memory table.
- [`trnmap.md`](../benchmarks/trnmap.md) and [`trnmap-count-merge-optimization-2026-11-09.txt`](../benchmarks/trnmap-count-merge-optimization-2026-11-09.txt) — full size/shape matrix and newer count/merge paths.
- [`rnbw-codec-results-2026-11-09.txt`](../benchmarks/rnbw-codec-results-2026-11-09.txt) — 367 byte-codec/comparator cases and ratios.
- [`codec-lane-current-2026-11-09.txt`](../benchmarks/codec-lane-current-2026-11-09.txt) — lane and parallel-cutoff sweeps.
- [`general-lz-word-load-paired-baseline-encode-2026-11-09.txt`](../benchmarks/general-lz-word-load-paired-baseline-encode-2026-11-09.txt), [`general-lz-word-load-paired-candidate-encode-2026-11-09.txt`](../benchmarks/general-lz-word-load-paired-candidate-encode-2026-11-09.txt), and matching decode files — inconclusive unaligned-load trial; no source change retained.
- [`rnbw-index-hyperfine-process-2026-11-09.txt`](../benchmarks/rnbw-index-hyperfine-process-2026-11-09.txt) — 45 whole-process comparisons across five 1M-key workloads; includes current Hyperfine means, medians, ranges, ratios, Callgrind counts, and the source artifact link.
