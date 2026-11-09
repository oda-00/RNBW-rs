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

CARGO_PROFILE_RELEASE_DEBUG=1 cargo build --locked --release --bin rnbw
mkdir -p "$report_dir"
command=("./$binary" bench --workload "$workload" --size "$size" --operation build --engine rnbw)

valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes \
  --callgrind-out-file="$report_dir/callgrind.out" "${command[@]}"
callgrind_annotate --inclusive=yes --tree=both "$report_dir/callgrind.out" \
  > "$report_dir/callgrind.txt"
valgrind --tool=massif --massif-out-file="$report_dir/massif.out" "${command[@]}"
ms_print "$report_dir/massif.out" > "$report_dir/massif.txt"
valgrind --tool=memcheck --leak-check=full --error-exitcode=99 "${command[@]}"

echo "Callgrind and Massif reports written to $report_dir"
