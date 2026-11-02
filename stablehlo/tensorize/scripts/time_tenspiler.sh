#!/bin/bash
# Time Tenspiler's generated code (programs/tenspiler/<suite>/*_numpy.py, run with numpy -> jax.numpy)
# under JAX with the same protocol as time.sh. Tenspiler exists for llama, blas, dsp and blend only.
# Writes $OUT/timings/tenspiler_<suite>.csv.
# Usage: scripts/time_tenspiler.sh [suite ...]   (default: llama blas dsp blend)
set -e
source "$(dirname "$0")/env.sh"
need "$JAX_PY"
mkdir -p "$OUT/timings"
for s in ${*:-llama blas dsp blend}; do
  pin "$JAX_PY" "$HERE/scripts/time_tenspiler.py" "$TS_PROGRAMS/$s" "$s" 2> "$OUT/timings/tenspiler_$s.log" \
    > "$OUT/timings/tenspiler_$s.csv"
  echo "tenspiler $s: $(($(wc -l < "$OUT/timings/tenspiler_$s.csv") - 1)) kernels"
done
