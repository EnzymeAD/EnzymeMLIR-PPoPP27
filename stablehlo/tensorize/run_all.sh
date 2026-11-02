#!/bin/bash
# Full comparison: collect kernels, raise them with Enzyme-JAX, time Tensorize vs Enzyme-JAX (and
# Tenspiler) under JAX, and write $OUT/tables.md. Uses the shipped Tensorize programs unless
# SYNTH=1 (re-synthesize with Tensorize first; hours, needs setup/tensorize.sh).
# Settings: see scripts/env.sh (SUITES, SKIP, TIMEOUT, WARMUP/REPS/DISCARD, CPUS, OUT, ...).
# Usage: ./run_all.sh [suite ...]
set -e
CDPATH= cd -- "$(dirname "$0")" > /dev/null
source scripts/env.sh
[ $# -gt 0 ] && export SUITES="$*"
mkdir -p "$OUT"
{
  echo "date: $(date -Is)"
  echo "host: $(hostname)  load: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "tensorize: $(git -C "$TENSORIZE" rev-parse --short HEAD) $(git -C "$TENSORIZE" diff --quiet || echo '(+ patches/tensorize-fixes.patch)')"
  echo "enzymexlamlir-opt: $OPT (Enzyme-JAX $ENZYME_JAX_REV)"
  "$JAX_PY" -c 'import jax, enzyme_ad, importlib.metadata as m; print("jax", jax.__version__, "enzyme-ad", m.version("enzyme-ad"), "devices", jax.devices())' 2>/dev/null
  command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  echo "protocol: warmup=$WARMUP reps=$REPS discard=$DISCARD timeout=${TIMEOUT}s skip=[$SKIP] cpus=${CPUS:-all}"
} | tee "$OUT/run_info.txt"

scripts/prepare.sh
scripts/raise.sh
if [ "${SYNTH:-0}" = 1 ]; then
  scripts/synth.sh
  export TZ_PROGRAMS=$OUT/programs/tensorize
fi
scripts/time.sh
tenspiler_suites=
for s in llama blas dsp blend; do
  if [[ " $SUITES " == *" $s "* ]]; then tenspiler_suites="$tenspiler_suites $s"; fi
done
if [ -n "$tenspiler_suites" ]; then scripts/time_tenspiler.sh $tenspiler_suites; fi
python3 scripts/tables.py "$OUT" > "$OUT/tables.md"
echo "wrote $OUT/tables.md"
