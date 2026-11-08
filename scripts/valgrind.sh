#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

workload="${1:-clustered}"
size="${2:-1000000}"
binary="target/release/rnbw"
report_dir="reports/valgrind"

if ! command -v valgrind >/dev/null 2>&1; then
  echo "valgrind is required. On Ubuntu install it with: sudo apt-get install valgrind" >&2
  exit 2
fi

cargo build --locked --release --bin rnbw
mkdir -p "$report_dir"
command=("./$binary" bench --workload "$workload" --size "$size" --operation build --engine rnbw)

valgrind --tool=callgrind --callgrind-out-file="$report_dir/callgrind.out" "${command[@]}"
valgrind --tool=massif --massif-out-file="$report_dir/massif.out" "${command[@]}"
valgrind --tool=memcheck --leak-check=full --error-exitcode=99 "${command[@]}"

echo "Valgrind reports written to $report_dir"
