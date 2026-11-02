#!/bin/bash
# Collect each suite's kernels and metadata into $OUT/benchmarks/<suite>/:
#   *.mlir                kernels (Tensorize submodule, or benchmarks/kiriko)
#   outputs.txt           the buffers a kernel returns (default: every buffer it stores to)
#   tensorize_broken.txt  kernels whose Tensorize program is known to be wrong (not timed)
# Usage: scripts/prepare.sh [suite ...]
set -e
source "$(dirname "$0")/env.sh"
[ $# -gt 0 ] && SUITES="$*"
for s in $SUITES; do
  d=$OUT/benchmarks/$s
  rm -rf "$d" && mkdir -p "$d"
  cp "$(suite_dir $s)"/*.mlir "$d"/
  [ -d "$HERE/meta/$s" ] && cp "$HERE/meta/$s"/*.txt "$d"/
  echo "$s: $(ls "$d"/*.mlir | wc -l) kernels"
done
