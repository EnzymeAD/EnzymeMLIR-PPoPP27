#!/bin/bash
# (Optional) Re-run Tenspiler synthesis + code generation instead of using programs/tenspiler:
#   llama, blend: Tenspiler's own drivers (tenspiler/{llama,blend}/holing/driver)
#   blas, dsp:    our ports of the Tensorize kernels (tenspiler_ports/{blas,dsp}: C++ + drivers),
#                 compiled to the LLVM IR Tenspiler's frontend reads with tenspiler_ports/blas/compile_ll.sh
# Synthesis only (Tenspiler's default; VERIFY=1 adds its in-loop cvc5 check, which is vacuous, see
# KNOWN_ISSUES.md). Writes $OUT/programs/tenspiler/<suite>/<kernel>_numpy.py (+ other backends) and
# synth_times.csv; time with  TS_PROGRAMS=$OUT/programs/tenspiler scripts/time_tenspiler.sh
# Usage: scripts/tenspiler_synth.sh [suite ...]   (default: llama blas dsp blend; needs setup/tenspiler.sh)
set -e
source "$(dirname "$0")/env.sh"
T=${TENSPILER:-$BUILD/tenspiler}
[ -f "$T/env.sh" ] || { echo "run setup/tenspiler.sh first" >&2; exit 1; }
source "$T/env.sh"
P=$OUT/programs/tenspiler
TS_TIMEOUT=${TS_TIMEOUT:-3600}
FLAG=$([ "${VERIFY:-0}" = 1 ] || echo --no-verify)
mkdir -p "$P"
[ -f "$P/synth_times.csv" ] || echo "suite,kernel,status,wall_s" > "$P/synth_times.csv"

drivers() {  # suite -> driver paths
  case $1 in
    llama | blend) find "$T/tenspiler/$1/holing/driver" -name '*_driver.py' | sort ;;
    blas | dsp)
      for c in "$HERE/tenspiler_ports/$1/cpp"/*.cc; do
        [ -f "${c%.cc}.ll" ] || TENSPILER=$T bash "$HERE/tenspiler_ports/blas/compile_ll.sh" "$c" > /dev/null 2>&1
      done
      ls "$HERE/tenspiler_ports/$1/drivers"/*_driver.py ;;
  esac
}

for s in ${*:-llama blas dsp blend}; do
  for drv in $(drivers $s); do
    b=$(basename "$drv" _driver.py)
    w=$OUT/tenspiler_work/$s/$b
    rm -rf "$w" && mkdir -p "$w" && ln -s "$T/tenspiler" "$w/tenspiler"
    start=$(date +%s.%N); status=ok
    (cd "$w" && timeout "$TS_TIMEOUT" python "$HERE/tenspiler_ports/tenspiler_run.py" "$drv" "$b" "$w/codegen" $FLAG) \
      > "$w/log.txt" 2>&1 || status=$([ $? -eq 124 ] && echo timeout || echo fail)
    secs=$(echo "$(date +%s.%N) - $start" | bc)
    mkdir -p "$P/$s" && cp "$w/codegen"/* "$P/$s/" 2> /dev/null || true
    echo "$s,$b,$status,$secs" >> "$P/synth_times.csv"
    echo "$s/$b $status ${secs}s"
  done
done
