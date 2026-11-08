# RNBW Rust proof

This crate implements the directly queryable lane-implied trie codec for sorted sets of unsigned integers. The root key width is 32 bits; the same encoder supports narrower even-width keys for chunk and group containers.

## Wire and query layout

- Every trie node is a four-bit lane mask. Each set lane implies its two key bits, so edge labels are not stored.
- The encoder evaluates each even split depth and chooses the smallest exact byte layout. Complete top levels are addressed directly.
- Each level packs 16 masks into a u64. A u32 rank sample covers each 128 masks; up to seven packed-word popcounts recover the local child rank.
- Trie leaves index a unary bucket vector with zero-select samples every 64 leaves. Low suffixes are bit-packed in key order.
- The queryable image is one allocation. RnbwView validates and borrows it without constructing a heap directory.
- The parser checks canonical offsets, counts, rank samples, select samples, order, and padding before exposing queries.

Rnbw::encode requires sorted, unique input. Rnbw::encode_unsorted sorts and deduplicates a copy. Rnbw::encode_width supports even widths from 2 through 32 bits. Rnbw::from_bytes_owned validates an owned image; RnbwView::from_bytes validates a borrowed image.

Membership uses a process-wide OnceLock dispatch: AVX-512BW, AVX2, SSE2, ARM64 NEON, then scalar fallback. SIMD scans byte-aligned suffix buckets; other widths use binary search over packed suffixes. The selected backend is available through Rnbw::simd_backend().

## Packed ternary map

`TrnMap` ports Symbolic std's `trnx64`/`trnmap`: 64 trits per 128-bit thermometer-plane word (`p = t >= 1`, `u = t >= 0`), 64-way levels of exact positive/negative/zero presence summaries, and a parallel hierarchical dirty-bit plane. `TriWord` supplies word-parallel Kleene AND/OR/NOT, class masks, popcounts, and ternary dot product. Search and dirty-word iteration descend the summary hierarchy in O(log_64 n); get is O(1); not is O(n/64); set and dirty updates stop propagating when an ancestor summary is unchanged.

The map occupies one contiguous aligned block: 384-byte header, 16 bytes per 64-trit leaf word, 24 bytes per 64 summary entries at each upper level, and 8 bytes per 64 dirty entries at each level. `TrnMap::new` owns the block; `TrnMap::in_buffer` initializes the same layout in borrowed caller bytes. `TrnMap::bytes_for(n)` includes the 7-byte alignment allowance; `storage_bytes_for(n)` is the exact block size. Empty maps are rejected; new maps start at -1 and clean.

The Criterion matrix covers lengths 64, 4,096, 65,536, 1,000,000, and 16,000,000; all-negative, all-zero, deterministic uniform ternary, alternating, clustered, and 1-in-4096 sparse data; hot and randomized-start searches, absent-class search, run/count, get/set, dirty traversal, merges, NOT, and allocation. Scalar byte-array search/count/run baselines are checked against the same workload. Run it with:

    cargo test --locked --lib --test fuzz_trnmap
    cargo bench --locked --bench trnmap

## Benchmarks

The deterministic workloads are uniform, clustered, dense, morton-shell, and ecs. Criterion compares RNBW against whole-universe Elias-Fano, a sorted raw Vec<u32>, Rust Roaring, and LZ4 block compression. It measures build, hit/miss membership, decode, parse, and parse-plus-decode. Dataset construction stays outside timed iterations.

Run tests:

    cargo test --locked --lib --bins
    cargo test --locked --test fuzz_codec --test fuzz_trnmap
    cargo clippy --locked --all-targets -- -D warnings

Run Criterion:

    cargo bench --locked --bench codec
    cargo bench --locked --bench trnmap

The trnmap storage formula, complexity model, local measurements, and workload
limits are recorded in [docs/benchmarks/trnmap.md](docs/benchmarks/trnmap.md).

Run the CLI:

    cargo run --locked --release -- bench --workload clustered --size 20000 --operation decode --engine all

Hyperfine measures full CLI process time, including dataset generation and representation construction:

    bash scripts/hyperfine.sh clustered build 1000000 10

Valgrind runs Callgrind, Massif, and Memcheck on the RNBW build path:

    bash scripts/valgrind.sh clustered 1000000

The GitHub Actions workflow runs format checks, Clippy, tests, both deterministic fuzz harnesses, both Criterion suites, and CLI verification on Ubuntu; the Criterion results are uploaded as an artifact. A second job installs Hyperfine and Valgrind and uploads its reports. CI timings are evidence for that runner only; they are not a portable performance grade. Compare measurements on the same machine, toolchain, workload, and operation.
