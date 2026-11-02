#!/bin/bash
# Time Tensorize's programs and Enzyme-JAX's raised StableHLO under JAX (compare_suite.py), on the
# default JAX device (GPU if the CUDA plugin is installed; JAX_PLATFORMS=cpu forces CPU; select a
# GPU with CUDA_VISIBLE_DEVICES). Both run jitted on the same device-resident inputs, interleaved
# call by call; each output is checked against Tensorize's.
# Writes $OUT/timings/<suite>/<kernel>.csv (+ .log); bicg/mvt also as <kernel>_1out.csv (both sides
# computing only Tensorize's output). Kernels in $SKIP are not timed; each kernel gets $TIMEOUT s.
# Usage: scripts/time.sh [suite ...]   (run prepare.sh and raise.sh first)
set -e
source "$(dirname "$0")/env.sh"
[ $# -gt 0 ] && SUITES="$*"
need "$JAX_PY"

for s in $SUITES; do
  mkdir -p "$OUT/timings/$s"
  echo "$s"
  for f in "$OUT/benchmarks/$s"/*.mlir; do
    k=$(basename "$f" .mlir)
    if [[ " $SKIP " == *" $s/$k "* ]]; then echo "  $k skipped (\$SKIP)"; continue; fi
    for v in "" _1out; do
      [ -f "$OUT/raised/$s$v/$k.while.mlir" ] || { [ -z "$v" ] && echo "  $k not raised"; continue; }
      rc=0
      pin timeout "$TIMEOUT" "$JAX_PY" "$HERE/scripts/compare_suite.py" "$f" \
        --raised_dir "$OUT/raised/$s$v" --synth_dir "$TZ_PROGRAMS/$s" --no_unroll --no_numpy \
        --warmup "$WARMUP" --reps "$REPS" --discard "$DISCARD" \
        --out "$OUT/timings/$s/$k$v.csv" > "$OUT/timings/$s/$k$v.log" 2>&1 || rc=$?
      [ $rc = 124 ] && echo "timeout after ${TIMEOUT}s" >> "$OUT/timings/$s/$k$v.log"
      echo "  $k$v rc=$rc"
    done
  done
done
