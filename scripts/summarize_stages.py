"""Normalize Criterion stage/file captures, retaining full-call intervals and units."""
import argparse
import hashlib
import math
import re
from pathlib import Path

UNITS = {"ps": 0.001, "ns": 1, "µs": 1000, "us": 1000, "ms": 1_000_000, "s": 1_000_000_000}
TIME = re.compile(r"time:\s*\[([\d.]+) (ps|ns|µs|us|ms|s) ([\d.]+) (ps|ns|µs|us|ms|s) ([\d.]+) (ps|ns|µs|us|ms|s)\]")


def parse(path):
    times, sizes = {}, {}
    name = None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("stage dataset=") and " raw_bytes=" in line:
            fields = dict(part.split("=", 1) for part in line.split()[1:])
            sizes[fields["dataset"]] = {k: int(v) for k, v in fields.items() if k != "dataset"}
        if re.match(r"^(stage|file-[\w-]+|histogram-cutoff)/", line):
            name = line.split()[0]
        match = TIME.search(line)
        if match and name:
            values = match.groups()
            if name in times:
                raise ValueError(f"Duplicate timing: {name}")
            times[name] = tuple(float(values[i]) * UNITS[values[i + 1]] for i in (0, 2, 4))
            name = None
    if not times:
        raise ValueError(f"No Criterion timings in {path}")
    return times, sizes


def denominator(name, sizes):
    family, dataset, stage = name.split("/", 2)
    if family == "histogram-cutoff":
        return int(stage.split("/")[0]), "raw-byte"
    info = sizes[dataset]
    if family != "stage" or stage in ("split-reuse", "interleave", "lz-parse-reuse", "lz-expand"):
        return info["raw_bytes"], "raw-byte"
    if stage == "normalize-model":
        return info["lanes"], "model"
    if stage == "sample":
        return info["lanes"], "lane-sample"
    return info["stream_bytes"], "token-stream-byte"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare", type=Path, help="Earlier capture with the same benchmark names")
    args = parser.parse_args()
    timings, sizes = parse(args.capture)
    baseline = parse(args.compare)[0] if args.compare else {}
    rows = [
        f"Capture: {args.capture.name}",
        f"SHA256: {hashlib.sha256(args.capture.read_bytes()).hexdigest()}",
        "Times are Criterion center estimates and reported intervals, converted to ns.",
        "Full-call time includes production allocations/destruction where the kernel returns owned output; reuse cases retain scratch capacity.",
        "Forced LZ/rANS/Huffman kernels may be skipped by the real hybrid selection. Do not sum independent stages or treat a forced-path win as an end-to-end win.",
        "Stage entropy denominators are actual token-stream bytes; file denominators are raw bytes; model and sampling calls use lane count.",
    ]
    if args.compare:
        rows += [f"Compared with: {args.compare.name}",
                 "Interval separation describes these captures only; it is not a paired statistical significance test."]
    rows += ["", "| Path | Full call low / center / high (ns) | Unit count | Normalized low / center / high (ns/unit) | Candidate/baseline | Interval comparison |",
             "|---|---:|---:|---:|---:|---|"]
    ratios = {}
    for name, timing in timings.items():
        count, unit = denominator(name, sizes)
        normalized = tuple(t / count for t in timing)
        ratio_text, verdict = "—", "—"
        if name in baseline:
            earlier = baseline[name]
            ratio = timing[1] / earlier[1]
            ratio_text = f"{ratio:.4f}"
            verdict = "faster intervals" if timing[2] < earlier[0] else "slower intervals" if timing[0] > earlier[2] else "overlap"
            ratios.setdefault(name.split("/")[-1], []).append(ratio)
        rows.append(f"| `{name}` | {' / '.join(f'{t:.3f}' for t in timing)} | {count} {unit} | {' / '.join(f'{t:.6f}' for t in normalized)} | {ratio_text} | {verdict} |")
    rows.append("")
    for stage, values in ratios.items():
        mean = math.exp(sum(math.log(v) for v in values) / len(values))
        rows.append(f"Geomean candidate/baseline, {stage}: {mean:.4f} ({len(values)} matched datasets)")
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(timings)} timing rows to {args.output}")
    for line in rows:
        if line.startswith("Geomean"):
            print(line)


if __name__ == "__main__":
    main()
