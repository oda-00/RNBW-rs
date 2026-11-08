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

## Benchmarks

The deterministic workloads are uniform, clustered, dense, morton-shell, and ecs. Criterion compares RNBW against whole-universe Elias-Fano, a sorted raw Vec<u32>, Rust Roaring, and LZ4 block compression. It measures build, hit/miss membership, decode, parse, and parse-plus-decode. Dataset construction stays outside timed iterations.

Run tests:

    cargo test --locked --lib --bins
    cargo clippy --locked --all-targets -- -D warnings

Run Criterion:

    cargo bench --locked --bench codec

Run the CLI:

    cargo run --locked --release -- bench --workload clustered --size 20000 --operation decode --engine all

Hyperfine measures full CLI process time, including dataset generation and representation construction:

    bash scripts/hyperfine.sh clustered build 1000000 10

Valgrind runs Callgrind, Massif, and Memcheck on the RNBW build path:

    bash scripts/valgrind.sh clustered 1000000

The GitHub Actions workflow runs format checks, Clippy, tests, Criterion, and CLI verification on Ubuntu. A second job installs Hyperfine and Valgrind and uploads its reports. CI timings are evidence for that runner only; they are not a portable performance grade. Compare measurements on the same machine, toolchain, workload, and operation.
