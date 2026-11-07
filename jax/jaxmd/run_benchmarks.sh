#!/bin/bash
# Orchestrator script to run the JAX-MD benchmark (EnzymeMLIR vs JAX)
# inside the EnzymeMLIR Docker container.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)

export PATH="/home/evaluator/.local/bin:/home/evaluator/eval-env/bin:$PATH"

EVAL_ENV="${EVAL_ENV:-/home/evaluator/eval-env}"
if [ -d "$EVAL_ENV" ]; then
  PYTHON="$EVAL_ENV/bin/python3"
else
  PYTHON=$(command -v python3)
fi

OUT_DIR="${RESULTS_DIR:-$REPO_ROOT/results/jaxmd}"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/results_JAXMD.json"

JAXMD_WARMUP="${JAXMD_WARMUP:-2}"
JAXMD_REPEAT="${JAXMD_REPEAT:-12}"

echo "================================================================="
echo "Starting JAX-MD Benchmark (warmup=$JAXMD_WARMUP, repeat=$JAXMD_REPEAT)"
echo "Python: $($PYTHON --version 2>&1)"
echo "================================================================="

# Refuse to benchmark on a silent CPU fallback (e.g. CUDA plugin rejected the cuDNN version)
if [ "${ALLOW_CPU:-0}" != "1" ]; then
  "$PYTHON" - <<'EOF'
import sys, jax
devs = jax.devices()
print("JAX", jax.__version__, "devices:", devs)
if not any(d.platform in ("gpu", "cuda") for d in devs):
    sys.exit("ERROR: JAX has no GPU backend (set ALLOW_CPU=1 to override)")
EOF
fi

cd "$HERE"
export TEST_UNDECLARED_OUTPUTS_DIR="$OUT_DIR"
"$PYTHON" main.py --warmup="$JAXMD_WARMUP" --repeat="$JAXMD_REPEAT"

if [ ! -f "$OUT_DIR/results_JAXMD.json" ] && [ -f "results_JAXMD.json" ]; then
  mv results_JAXMD.json "$OUT_DIR/results_JAXMD.json"
fi

if [ ! -s "$OUT_DIR/results_JAXMD.json" ]; then
  echo "ERROR: Expected JAX-MD output file $OUT_DIR/results_JAXMD.json was not generated!" >&2
  exit 1
fi

"$PYTHON" - "$OUT_DIR/results_JAXMD.json" <<'EOF'
import json, pathlib, sys
from statistics import mean, stdev

results_json = pathlib.Path(sys.argv[1])
data = json.loads(results_json.read_text())
if not data:
    raise SystemExit(f"ERROR: {results_json} is empty!")

print(f"\n--- JAX-MD Results ({results_json}) ---")
for row in data:
    times = row.get("time_s", [])
    mems = [m for m in row.get("peak_memory_bytes", []) if m is not None]
    avg_t = mean(times) if times else 0.0
    std_t = stdev(times) if len(times) > 1 else 0.0
    peak_mb = f"{max(mems) / 1e6:.2f} MB" if mems else "n/a"
    print(f"  {row.get('pipeline', ''):<12} {row.get('key', ''):<10} mean={avg_t:.8f}s  stdev={std_t:.8f}s  peak_mem={peak_mb}")
print("VERIFIED: JAX-MD benchmark completed and numerical checks passed!")
EOF
