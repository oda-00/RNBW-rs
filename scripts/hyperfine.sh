#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

workload="${1:-uniform}"
operation="${2:-build}"
size="${3:-1000000}"
runs="${4:-10}"
binary="target/release/rnbw"
report_dir="reports/hyperfine"

if ! command -v hyperfine >/dev/null 2>&1; then
  echo "hyperfine is required. Install it with your system package manager (Ubuntu: sudo apt-get install hyperfine)." >&2
  exit 2
fi

cargo build --locked --release --bin rnbw
mkdir -p "$report_dir"
hyperfine --warmup 2 --runs "$runs" \
  --export-json "$report_dir/${workload}-${operation}.json" \
  --export-markdown "$report_dir/${workload}-${operation}.md" \
  "./$binary bench --workload $workload --size $size --operation $operation --engine rnbw" \
  "./$binary bench --workload $workload --size $size --operation $operation --engine plain-ef" \
  "./$binary bench --workload $workload --size $size --operation $operation --engine raw" \
  "./$binary bench --workload $workload --size $size --operation $operation --engine roaring" \
  "./$binary bench --workload $workload --size $size --operation $operation --engine lz4"
