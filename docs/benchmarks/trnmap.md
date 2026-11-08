# `trnmap` local measurements

Run date: 2026-11-08. Toolchain: `rustc 1.95.0`. Host identifier reported by the
Windows runner: `AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD`; the exact CPU
SKU and cache sizes were not exposed, so cache-tier claims are intentionally omitted.

The Criterion run used 10 samples with 100 ms warm-up and 100 ms measurement per
case. Values below are Criterion mean estimates with its 95% confidence interval.
These are short same-host comparisons, not a portable performance grade. The
benchmark checks each workload's search, count, and run results against the scalar
array before timing. It covers map lengths 64, 4,096, 65,536, 1,000,000, and
16,000,000, with all-negative, all-zero, deterministic uniform ternary, alternating,
4,096-trit clusters, and one-positive-per-4,096-trits shapes. It also tests 1,024
deterministic distributed search starts. The scalar baseline is a one-byte `i8` per
trit; it is not a second packed-map implementation.

## Exact storage

For `E0 = n`, `E(k+1) = ceil(Ek / 64)` until the top level has one entry, and `L`
levels, the stored block is:

```text
384 + 16*ceil(n/64)
    + 24*sum(ceil(Ek/64), k=1..L-1)
    +  8*sum(ceil(Ek/64), k=0..L-1) bytes
```

The borrowed-buffer budget adds 7 alignment bytes. Measured exact block sizes:

| Trits | Exact bytes | Levels |
|---:|---:|---:|
| 64 | 440 B | 2 |
| 4,096 | 1,984 B | 3 |
| 65,536 | 25,536 B | 4 |
| 1,000,000 | 383,416 B | 5 |
| 16,000,000 | 6,127,456 B | 5 |

## Representative timings

| Operation and workload | `trnmap` | Scalar byte-array baseline |
|---|---:|---:|
| 1M `get`, repeated hot index | 2.10 ns (2.07–2.16) | 0.68 ns (0.68–0.68) |
| 1M uniform `find(+1)`, hot start | 4.57 ns (4.55–4.60) | 2.55 ns (2.54–2.57) |
| 1M uniform `count(+1)` | 101.97 µs (101.64–102.36) | 279.77 µs (279.15–280.78) |
| 65,536 all-zero `find(negative)` absent | 10.38 ns (10.34–10.45) | 22.43 µs (22.36–22.54) |
| 65,536 all-zero `run(0)` | 12.30 ns (12.27–12.32) | 14.995 µs (14.919–15.100) |
| 65,536 all-zero `count(0)` | 6.77 µs (6.69–6.86) | 18.16 µs (18.14–18.19) |
| 16M 1-in-4096 `find(next +1)` | 12.40 ns (12.33–12.49) | 1.441 µs (1.430–1.451) |
| 16M 1-in-4096 `count(+1)` | 75.12 µs (73.54–76.70) | 4.692 ms (4.642–4.748) |
| 16M 1-in-4096 absent `find(0)` | 11.33 ns (11.26–11.41) | 5.763 ms (5.726–5.807) |
| 16M all-negative `run(-1)` | 14.69 ns (14.54–14.94) | 3.790 ms (3.750–3.830) |

The 1,024-start sparse-search batch took 17.24 µs (16.91–17.61) in the hierarchy
and 771.42 µs (765.74–780.20) in the scalar scan. Per-point mutation and dirty
numbers below are isolated/composite map operations, not scalar-baseline wins:

| Operation | Timing |
|---|---:|
| 1M `set` with a changing value | 22.31 ns (22.14–22.54) |
| 1M dirty `mark + next_dirty + take_dirty` | 16.03 ns (15.98–16.09) |
| 1M clustered OR / AND | 165.56 µs (165.28–165.92) / 66.50 µs (65.66–67.32) |
| 1M NOT | 19.41 µs (19.37–19.46) |
| Allocate and initialize 1M map | 10.45 µs (10.36–10.55) |

## Cost interpretation

- `get`: O(1), one packed leaf-word lookup; measured around 2.1 ns on the 1M case.
- `find`: O(log_64 n) summary loads plus a leaf-word select. It wins decisively for
  absent classes and sparse hits; nearby dense hits favor a direct byte scan.
- `run`: O(log_64 n), including all-equal tails; hierarchy avoids scanning long runs.
- `set`, `mark`, `next_dirty`, `take_dirty`: O(log_64 n) worst case, stopping when
  the affected ancestor summary is already correct.
- `count(t)`: O((matching leaf words + 1) * log_64 n), with popcount per matching
  leaf word. Sparse and uniform 1M counts beat the byte scan here; fully dense
  classes narrow that advantage.
- OR/AND visit only right-hand leaf words whose classes can affect the result, then
  repair summaries only for changed words; NOT scans leaf words and swaps summary
  class planes.

The data supports using this structure when callers exploit sparse class search,
long homogeneous runs, or aggregated counts. It is not a faster replacement for a
plain byte array on simple hot-point reads or nearby dense matches. GitHub Actions
runs the same Criterion benchmark and uploads `target/criterion/` as an artifact;
its Ubuntu measurements must be read independently from these Windows results.

## Full Criterion matrix

All cells show the mean and 95% confidence interval, in ns/µs/ms. Paired cells are hierarchical map / scalar `i8` array. Scalar search uses the same input, target, and start. `random-starts` reports a batch of 1,024 searches, not a per-search value.

### Find, run, and count by shape

| Data shape | Trits | Find map / scalar | Run map / scalar | Count map / scalar |
|---|---:|---:|---:|---:|
| All negative | 64 | 4.35 ns (4.34–4.35) / 1.04 ns (1.04–1.05) | 8.37 ns (8.35–8.38) / 16.07 ns (15.74–16.50) | 10.00 ns (9.78–10.33) / 18.25 ns (18.21–18.29) |
| All negative | 4,096 | 4.57 ns (4.54–4.60) / 1.03 ns (1.03–1.04) | 10.54 ns (10.17–11.19) / 965.59 ns (954.79–978.71) | 438.33 ns (437.38–439.20) / 1.14 µs (1.14–1.15) |
| All negative | 65,536 | 4.34 ns (4.32–4.35) / 1.03 ns (1.02–1.03) | 12.84 ns (12.81–12.88) / 15.67 µs (15.18–16.47) | 6.76 µs (6.66–6.92) / 18.18 µs (18.15–18.24) |
| All negative | 1,000,000 | 4.33 ns (4.32–4.33) / 1.07 ns (1.04–1.11) | 14.59 ns (14.14–15.21) / 239.92 µs (236.04–243.74) | 110.36 µs (109.58–111.13) / 295.61 µs (293.10–298.17) |
| All negative | 16,000,000 | 4.37 ns (4.34–4.40) / 1.15 ns (1.13–1.17) | 14.69 ns (14.54–14.94) / 3.790 ms (3.750–3.830) | 1.660 ms (1.640–1.681) / 4.539 ms (4.532–4.549) |
| All zero | 64 | 4.37 ns (4.33–4.43) / 1.14 ns (1.11–1.18) | 8.91 ns (8.90–8.93) / 16.56 ns (15.91–17.54) | 9.74 ns (9.68–9.78) / 18.43 ns (18.40–18.47) |
| All zero | 4,096 | 4.32 ns (4.32–4.33) / 1.04 ns (1.03–1.07) | 10.57 ns (10.55–10.60) / 958.59 ns (948.43–969.43) | 439.55 ns (437.59–441.28) / 1.15 µs (1.15–1.16) |
| All zero | 65,536 | 4.38 ns (4.36–4.41) / 1.03 ns (1.02–1.03) | 12.30 ns (12.27–12.32) / 15.00 µs (14.92–15.10) | 6.77 µs (6.69–6.86) / 18.16 µs (18.14–18.19) |
| All zero | 1,000,000 | 4.37 ns (4.36–4.39) / 1.14 ns (1.09–1.22) | 14.39 ns (14.09–14.93) / 239.50 µs (236.84–241.76) | 108.77 µs (108.07–109.53) / 282.93 µs (281.93–284.14) |
| All zero | 16,000,000 | 4.51 ns (4.44–4.59) / 1.08 ns (1.05–1.11) | 14.03 ns (13.99–14.07) / 3.728 ms (3.703–3.755) | 1.675 ms (1.669–1.682) / 4.497 ms (4.489–4.507) |
| Uniform ternary | 64 | 4.37 ns (4.37–4.38) / 2.59 ns (2.58–2.61) | 5.97 ns (5.95–5.98) / 0.89 ns (0.84–0.94) | 9.72 ns (9.68–9.76) / 18.30 ns (18.28–18.33) |
| Uniform ternary | 4,096 | 4.32 ns (4.32–4.33) / 2.58 ns (2.58–2.59) | 5.93 ns (5.92–5.94) / 0.87 ns (0.81–0.92) | 446.25 ns (443.84–448.69) / 1.14 µs (1.14–1.14) |
| Uniform ternary | 65,536 | 4.33 ns (4.32–4.35) / 2.51 ns (2.51–2.52) | 6.40 ns (6.39–6.40) / 0.90 ns (0.82–0.99) | 6.68 µs (6.67–6.70) / 18.31 µs (18.29–18.33) |
| Uniform ternary | 1,000,000 | 4.57 ns (4.55–4.60) / 2.55 ns (2.54–2.57) | 6.41 ns (6.35–6.47) / 0.96 ns (0.87–1.05) | 101.97 µs (101.64–102.36) / 279.77 µs (279.15–280.78) |
| Uniform ternary | 16,000,000 | 4.42 ns (4.41–4.44) / 2.72 ns (2.72–2.73) | 5.96 ns (5.95–5.97) / 0.89 ns (0.79–1.00) | 1.658 ms (1.649–1.666) / 4.500 ms (4.485–4.526) |
| Alternating −1/+1 | 64 | 4.56 ns (4.54–4.58) / 1.15 ns (1.15–1.16) | 6.02 ns (5.99–6.05) / 0.87 ns (0.80–0.94) | 9.76 ns (9.70–9.82) / 18.80 ns (18.65–18.98) |
| Alternating −1/+1 | 4,096 | 4.35 ns (4.34–4.35) / 1.16 ns (1.14–1.20) | 6.19 ns (6.08–6.28) / 0.93 ns (0.80–1.10) | 450.81 ns (447.84–453.57) / 1.17 µs (1.16–1.17) |
| Alternating −1/+1 | 65,536 | 4.33 ns (4.32–4.34) / 1.22 ns (1.14–1.34) | 6.15 ns (6.14–6.16) / 0.91 ns (0.83–0.99) | 7.07 µs (7.05–7.09) / 18.18 µs (18.16–18.23) |
| Alternating −1/+1 | 1,000,000 | 4.33 ns (4.32–4.36) / 1.20 ns (1.18–1.22) | 6.00 ns (5.99–6.02) / 0.80 ns (0.75–0.86) | 103.48 µs (103.33–103.64) / 284.76 µs (284.22–285.55) |
| Alternating −1/+1 | 16,000,000 | 4.39 ns (4.39–4.39) / 1.22 ns (1.22–1.22) | 6.03 ns (6.00–6.07) / 0.90 ns (0.82–0.98) | 1.634 ms (1.628–1.642) / 4.682 ms (4.664–4.705) |
| 4,096-trit clusters | 64 | 6.81 ns (6.76–6.85) / 28.42 ns (28.13–28.90) | 9.08 ns (8.90–9.32) / 15.88 ns (15.65–16.10) | 9.23 ns (9.20–9.27) / 18.17 ns (18.14–18.20) |
| 4,096-trit clusters | 4,096 | 9.03 ns (9.01–9.05) / 1.54 µs (1.48–1.64) | 10.26 ns (10.18–10.35) / 950.24 ns (941.76–958.48) | 11.96 ns (11.93–12.00) / 1.14 µs (1.14–1.14) |
| 4,096-trit clusters | 65,536 | 13.08 ns (13.05–13.10) / 2.89 µs (2.86–2.92) | 16.34 ns (15.80–16.99) / 984.40 ns (962.93–1015.11) | 2.17 µs (2.17–2.18) / 18.24 µs (18.19–18.30) |
| 4,096-trit clusters | 1,000,000 | 12.28 ns (12.28–12.29) / 2.91 µs (2.90–2.92) | 15.23 ns (14.99–15.50) / 1.01 µs (0.99–1.02) | 37.33 µs (37.21–37.47) / 280.30 µs (279.72–281.03) |
| 4,096-trit clusters | 16,000,000 | 12.37 ns (12.31–12.46) / 3.06 µs (3.05–3.08) | 16.35 ns (16.25–16.45) / 973.63 ns (960.33–986.17) | 603.38 µs (597.66–610.79) / 4.489 ms (4.483–4.495) |
| Positive 1/4,096 | 64 | 6.63 ns (6.62–6.65) / 27.42 ns (27.34–27.51) | 8.88 ns (8.84–8.92) / 15.50 ns (15.27–15.75) | 9.91 ns (9.84–9.98) / 18.22 ns (18.10–18.43) |
| Positive 1/4,096 | 4,096 | 9.07 ns (9.02–9.11) / 1.47 µs (1.43–1.54) | 10.08 ns (10.06–10.10) / 967.15 ns (953.78–984.07) | 18.06 ns (17.83–18.24) / 1.14 µs (1.14–1.14) |
| Positive 1/4,096 | 65,536 | 13.49 ns (13.28–13.66) / 1.45 µs (1.43–1.47) | 15.28 ns (15.07–15.48) / 961.73 ns (956.57–967.10) | 282.10 ns (281.13–283.34) / 19.78 µs (18.50–21.55) |
| Positive 1/4,096 | 1,000,000 | 13.13 ns (13.01–13.25) / 1.61 µs (1.54–1.70) | 15.14 ns (15.07–15.24) / 988.45 ns (976.01–1000.25) | 4.53 µs (4.48–4.60) / 289.19 µs (281.47–303.09) |
| Positive 1/4,096 | 16,000,000 | 12.40 ns (12.33–12.49) / 1.44 µs (1.43–1.45) | 14.70 ns (14.63–14.77) / 976.22 ns (959.08–996.19) | 75.12 µs (73.54–76.70) / 4.692 ms (4.642–4.748) |

### No-match searches

| Data shape | Trits | Hierarchical | Scalar scan |
|---|---:|---:|---:|
| All negative | 64 | 6.71 ns (6.66–6.76) | 27.45 ns (27.33–27.66) |
| All negative | 4,096 | 8.45 ns (8.44–8.46) | 1.43 µs (1.42–1.43) |
| All negative | 65,536 | 10.81 ns (10.78–10.85) | 22.53 µs (22.46–22.62) |
| All negative | 1,000,000 | 12.23 ns (12.22–12.25) | 393.77 µs (364.39–431.92) |
| All negative | 16,000,000 | 12.45 ns (12.24–12.68) | 5.596 ms (5.488–5.744) |
| All zero | 64 | 6.64 ns (6.59–6.72) | 27.40 ns (27.33–27.50) |
| All zero | 4,096 | 8.96 ns (8.94–8.98) | 1.43 µs (1.42–1.43) |
| All zero | 65,536 | 10.38 ns (10.34–10.45) | 22.43 µs (22.36–22.54) |
| All zero | 1,000,000 | 12.15 ns (12.09–12.23) | 350.45 µs (349.94–350.97) |
| All zero | 16,000,000 | 12.20 ns (12.14–12.28) | 5.614 ms (5.552–5.695) |
| Alternating −1/+1 | 64 | 6.93 ns (6.41–7.79) | 27.44 ns (27.36–27.58) |
| Alternating −1/+1 | 4,096 | 8.21 ns (8.14–8.30) | 1.50 µs (1.47–1.52) |
| Alternating −1/+1 | 65,536 | 9.79 ns (9.76–9.84) | 22.43 µs (22.36–22.54) |
| Alternating −1/+1 | 1,000,000 | 12.01 ns (11.95–12.07) | 345.95 µs (344.85–347.20) |
| Alternating −1/+1 | 16,000,000 | 11.56 ns (11.39–11.75) | 5.490 ms (5.483–5.498) |
| Positive 1/4,096 | 64 | 7.11 ns (6.70–7.65) | 28.53 ns (28.22–28.82) |
| Positive 1/4,096 | 4,096 | 8.18 ns (7.98–8.49) | 1.42 µs (1.42–1.42) |
| Positive 1/4,096 | 65,536 | 9.94 ns (9.65–10.36) | 23.12 µs (22.38–24.56) |
| Positive 1/4,096 | 1,000,000 | 11.55 ns (11.52–11.59) | 343.50 µs (342.42–345.25) |
| Positive 1/4,096 | 16,000,000 | 11.33 ns (11.26–11.41) | 5.763 ms (5.726–5.807) |

### 1,024 distributed search starts

| Data shape | Trits | Hierarchical batch | Scalar batch |
|---|---:|---:|---:|
| All negative | 64 | 4.14 µs (4.11–4.17) | 815.48 ns (813.66–817.50) |
| All negative | 4,096 | 3.97 µs (3.97–3.98) | 755.64 ns (744.82–763.20) |
| All negative | 65,536 | 4.06 µs (4.05–4.06) | 749.73 ns (746.85–753.72) |
| All negative | 1,000,000 | 4.05 µs (4.04–4.06) | 1.02 µs (1.02–1.03) |
| All negative | 16,000,000 | 5.12 µs (5.11–5.13) | 1.04 µs (1.04–1.04) |
| All zero | 64 | 3.96 µs (3.96–3.97) | 793.84 ns (792.57–795.49) |
| All zero | 4,096 | 4.17 µs (4.16–4.17) | 710.84 ns (710.44–711.32) |
| All zero | 65,536 | 3.97 µs (3.96–3.99) | 749.74 ns (745.46–755.86) |
| All zero | 1,000,000 | 4.27 µs (4.26–4.29) | 965.52 ns (964.32–967.03) |
| All zero | 16,000,000 | 4.85 µs (4.83–4.88) | 1.05 µs (1.04–1.05) |
| Uniform ternary | 64 | 4.05 µs (4.04–4.06) | 1.49 µs (1.45–1.54) |
| Uniform ternary | 4,096 | 4.16 µs (4.15–4.18) | 1.83 µs (1.83–1.83) |
| Uniform ternary | 65,536 | 4.27 µs (4.26–4.27) | 1.78 µs (1.78–1.79) |
| Uniform ternary | 1,000,000 | 4.34 µs (4.33–4.34) | 1.93 µs (1.93–1.94) |
| Uniform ternary | 16,000,000 | 5.30 µs (5.29–5.31) | 1.96 µs (1.96–1.96) |
| Alternating −1/+1 | 64 | 4.07 µs (4.05–4.09) | 1.08 µs (1.05–1.10) |
| Alternating −1/+1 | 4,096 | 4.24 µs (4.19–4.29) | 984.29 ns (966.81–1002.51) |
| Alternating −1/+1 | 65,536 | 4.05 µs (4.01–4.10) | 1.04 µs (1.01–1.09) |
| Alternating −1/+1 | 1,000,000 | 4.34 µs (4.34–4.35) | 1.30 µs (1.26–1.35) |
| Alternating −1/+1 | 16,000,000 | 4.91 µs (4.87–4.96) | 1.31 µs (1.31–1.32) |
| 4,096-trit clusters | 64 | 6.67 µs (6.65–6.70) | 17.18 µs (16.92–17.44) |
| 4,096-trit clusters | 4,096 | 8.61 µs (8.51–8.78) | 745.96 µs (744.11–748.11) |
| 4,096-trit clusters | 65,536 | 11.31 µs (11.28–11.33) | 970.25 µs (968.00–973.37) |
| 4,096-trit clusters | 1,000,000 | 11.94 µs (11.93–11.96) | 1.028 ms (1.014–1.044) |
| 4,096-trit clusters | 16,000,000 | 13.17 µs (13.02–13.33) | 1.014 ms (1.005–1.024) |
| Positive 1/4,096 | 64 | 6.81 µs (6.80–6.82) | 16.12 µs (15.65–16.72) |
| Positive 1/4,096 | 4,096 | 8.81 µs (8.58–9.08) | 727.73 µs (724.58–731.38) |
| Positive 1/4,096 | 65,536 | 13.03 µs (12.57–13.90) | 742.29 µs (730.00–762.60) |
| Positive 1/4,096 | 1,000,000 | 15.12 µs (13.06–17.50) | 771.10 µs (768.98–773.09) |
| Positive 1/4,096 | 16,000,000 | 17.24 µs (16.91–17.61) | 771.42 µs (765.74–780.20) |

### Point access across sizes

| Trits | Packed get | Byte-array get | Find one value, map | Find one value, scalar |
|---:|---:|---:|---:|---:|
| 64 | 1.97 ns (1.96–1.99) | 0.68 ns (0.68–0.68) | 4.17 ns (4.12–4.23) | 1.37 ns (1.36–1.37) |
| 4,096 | 1.98 ns (1.97–2.00) | 0.68 ns (0.68–0.68) | 4.28 ns (4.28–4.29) | 1.60 ns (1.59–1.62) |
| 65,536 | 2.18 ns (2.15–2.20) | 0.78 ns (0.73–0.83) | 4.11 ns (4.10–4.12) | 4.55 ns (4.54–4.56) |
| 1,000,000 | 2.10 ns (2.07–2.16) | 0.68 ns (0.68–0.68) | 4.24 ns (4.23–4.25) | 1.20 ns (1.18–1.22) |
| 16,000,000 | 2.29 ns (2.21–2.37) | 0.69 ns (0.69–0.69) | 4.13 ns (4.12–4.13) | 2.21 ns (2.18–2.24) |

### Mutation and dirty traversal

| Trits | Changing `set` | `mark + next_dirty + take_dirty` |
|---:|---:|---:|
| 64 | 9.60 ns (9.59–9.62) | 8.72 ns (8.70–8.74) |
| 4,096 | 13.72 ns (13.70–13.74) | 9.61 ns (9.56–9.67) |
| 65,536 | 18.58 ns (18.53–18.63) | 12.80 ns (12.79–12.82) |
| 1,000,000 | 22.31 ns (22.14–22.54) | 16.03 ns (15.98–16.09) |
| 16,000,000 | 24.39 ns (22.39–27.22) | 16.26 ns (16.18–16.36) |

### Whole-map cases (1M trits)

| Operation and workload | Mean (95% CI) |
|---|---:|
| OR, clustered RHS | 165.56 µs (165.28–165.92) |
| AND, clustered RHS | 66.50 µs (65.66–67.32) |
| NOT | 19.41 µs (19.37–19.46) |
| Allocate/init | 10.45 µs (10.36–10.55) |
