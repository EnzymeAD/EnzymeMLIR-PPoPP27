#!/bin/bash
# (Optional) Re-run Tensorize synthesis (tensorize/main.py --target numpy) on every kernel instead of
# using the shipped programs in programs/tensorize. Needs the Tensorize build (setup/tensorize.sh).
# Writes $OUT/programs/tensorize/<suite>/<kernel>.out (+ .log) and synth_times.csv; time with
#   TZ_PROGRAMS=$OUT/programs/tensorize scripts/time.sh
# Env: SYNTH_TIMEOUT (s per kernel, default 1800), TENSORIZE_SOLVE_TIMEOUT (wall-clock limit of each
# sympy.solve call, default 1 s; on a loaded machine correct sketches can time out — the shipped
# Kiriko gemm/2mm/syrk programs were synthesized with 20), JOBS (kernels in parallel, default 1).
# Usage: scripts/synth.sh [suite ...]   (run scripts/prepare.sh first)
set -e
source "$(dirname "$0")/env.sh"
[ $# -gt 0 ] && SUITES="$*"
need "$TZ_PY"
SYNTH_TIMEOUT=${SYNTH_TIMEOUT:-1800}
JOBS=${JOBS:-1}
P=$OUT/programs/tensorize
mkdir -p "$P"
[ -f "$P/synth_times.csv" ] || echo "suite,kernel,status,wall_s" > "$P/synth_times.csv"
export TZ_PY TENSORIZE P SYNTH_TIMEOUT
export PYTHONPATH=$TENSORIZE/build/python_packages/synth

synth_one() {  # suite kernel.mlir
  local s=$1 f=$2 k; k=$(basename "$2" .mlir)
  mkdir -p "$P/$s"
  local start status=ok secs; start=$(date +%s.%N)
  (cd "$TENSORIZE" && timeout "$SYNTH_TIMEOUT" "$TZ_PY" tensorize/main.py --program "$f" --target numpy \
    --synth_out "$P/$s/$k.out") > "$P/$s/$k.log" 2>&1 || status=$([ $? -eq 124 ] && echo timeout || echo fail)
  secs=$(echo "$(date +%s.%N) - $start" | bc)
  echo "$s,$k,$status,$secs" >> "$P/synth_times.csv"
  echo "$s/$k $status ${secs}s"
}
export -f synth_one

for s in $SUITES; do
  for f in "$OUT/benchmarks/$s"/*.mlir; do echo "$s $f"; done
done | xargs -P "$JOBS" -L 1 bash -c 'synth_one "$0" "$1"'
