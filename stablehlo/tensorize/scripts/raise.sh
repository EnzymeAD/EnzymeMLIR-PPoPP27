#!/bin/bash
# Raise every kernel to StableHLO with Enzyme-JAX:
#   memrefify.py wraps the kernel in a public @main (enzymexla.jit_call) that returns its output
#   buffers, then raise_pipeline.sh runs Reactant's default CPU pass pipeline (raise=true,
#   prefer_while_raising=true) through enzymexlamlir-opt.
# Writes $OUT/raised/<suite>/<kernel>.while.mlir (+ .in.mlir, .err) and raise_times.csv.
# bicg and mvt are also raised returning only the output Tensorize computes (q, x2) into
# $OUT/raised/<suite>_1out/, for a like-for-like comparison.
# Usage: scripts/raise.sh [suite ...]   (run scripts/prepare.sh first)
set -e
source "$(dirname "$0")/env.sh"
[ $# -gt 0 ] && SUITES="$*"
command -v "$OPT" > /dev/null || need "$OPT"

raise_one() {  # kernel.mlir outdir [--outputs list]
  local mlir=$1 dir=$2 k; k=$(basename "$1" .mlir); shift 2
  mkdir -p "$dir"
  python3 -I "$HERE/scripts/memrefify.py" "$mlir" "$@" > "$dir/$k.in.mlir"
  local start status=ok secs; start=$(date +%s.%N)
  "$HERE/scripts/raise_pipeline.sh" "$dir/$k.in.mlir" -o "$dir/$k.while.mlir" 2> "$dir/$k.while.err" || status=error
  secs=$(echo "$(date +%s.%N) - $start" | bc)
  [ $status = ok ] || rm -f "$dir/$k.while.mlir"
  [ -f "$dir/raise_times.csv" ] || echo "kernel,variant,status,raise_s" > "$dir/raise_times.csv"
  echo "$k,while,$status,$secs" >> "$dir/raise_times.csv"
  echo "  $k $status ${secs}s"
}

for s in $SUITES; do
  bench=$OUT/benchmarks/$s
  [ -d "$bench" ] || { echo "run scripts/prepare.sh first ($bench missing)" >&2; exit 1; }
  echo "$s"
  rm -f "$OUT/raised/$s/raise_times.csv" "$OUT/raised/${s}_1out/raise_times.csv"
  for f in "$bench"/*.mlir; do
    k=$(basename "$f" .mlir)
    outputs=$([ -f "$bench/outputs.txt" ] && awk -v k="$k" '$1 == k { print $2 }' "$bench/outputs.txt" || true)
    raise_one "$f" "$OUT/raised/$s" ${outputs:+--outputs $outputs}
    case $s/$k in
      polybench/bicg | kiriko/bicg) raise_one "$f" "$OUT/raised/${s}_1out" --outputs arg4 ;;
      polybench/mvt | kiriko/mvt) raise_one "$f" "$OUT/raised/${s}_1out" --outputs arg2 ;;
    esac
  done
done
