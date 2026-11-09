# Frozen general benchmark inputs

These files keep source-like and executable codec workloads byte-identical
between builds. They are deliberately snapshots: update them only as an
intentional benchmark-corpus revision, then record new hashes and rerun the
full Criterion matrix.

- `native-executable-binary.exe` is a Windows PE snapshot of the optimized
  `files` Criterion benchmark driver. It is not read from `current_exe()` at
  benchmark runtime.
- `repository-rust-source.txt` concatenates the selected Rust source files as
  they existed when this snapshot was captured.
- `repository-technical-prose.txt` snapshots the selected README, benchmark
  notes, and Cargo manifest.
- `graphics-wgsl-source.txt` snapshots the two codec WGSL shader files.

SHA-256 values for the 2026-11-09 corpus revision:

```text
0efd0db327b9ec774d3e6997c96fff2dbb024f474ef12c14ef2c60d2a6d4eae6  graphics-wgsl-source.txt
422134ed2675514059acc11b965da3226c417be767e96172ba534358e9647767  native-executable-binary.exe
902d6388e3f65d040219c63eec13cbebad980014b8eb36c28dbd49bd45bec5db  repository-rust-source.txt
d5af3b2edfada03ef828ddf5b821567b9b7f64ac1a5c7eeb63a89e2cdd3ae76c  repository-technical-prose.txt
```
