#!/bin/bash
# Build and time the Futhark LBM reverse-mode AD benchmark on CUDA.
#
# usage: ./run_timings.sh [REPS]     (default: 12 reps; lbm.fut runs 100 steps)
# env:   RESULTS_DIR=timings  CUDA_VISIBLE_DEVICES=0  CUDA_PATH=/usr/local/cuda-12.9
set -euo pipefail
cd "$(dirname "$0")"

REPS=${1:-${REPS:-12}}
STEPS=100
RESULTS_DIR=${RESULTS_DIR:-timings}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-12.9}
if [ ! -d "$CUDA_PATH" ] && [ -d "/usr/local/cuda" ]; then
  CUDA_PATH="/usr/local/cuda"
fi
export PATH="$CUDA_PATH/bin:$PATH"
export CPATH="$CUDA_PATH/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

REPO_ROOT=$(cd ../.. && pwd)
FUTHARK=${FUTHARK:-"$REPO_ROOT/futhark-nightly-linux-x86_64/bin/futhark"}
if [ ! -x "$FUTHARK" ]; then
  FUTHARK=$(command -v futhark)
fi

mkdir -p "$RESULTS_DIR"
JSON_OUT="$RESULTS_DIR/futhark_lbm.json"

echo "== futhark bench lbm.fut (CUDA, $REPS runs)"
"$FUTHARK" bench --backend=cuda --no-convergence-phase -r "$REPS" --json="$JSON_OUT" lbm.fut

SUMMARY="$RESULTS_DIR/summary_${STEPS}.txt"
python3 - "$JSON_OUT" "$STEPS" "$SUMMARY" <<'EOF'
import json, math, sys

json_path, steps, summary_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(json_path) as f:
    data = json.load(f)

runtimes_s = []
for prog_data in data.values():
    for ds_data in prog_data.get("datasets", {}).values():
        runtimes_s.extend(t / 1e6 for t in ds_data.get("runtimes", []))

if not runtimes_s:
    raise SystemExit(f"No runtimes found in {json_path}")

mu = sum(runtimes_s) / len(runtimes_s)
var = sum((t - mu) ** 2 for t in runtimes_s) / len(runtimes_s)
std = math.sqrt(max(0.0, var))
mn = min(runtimes_s)
ts_str = ", ".join(f"{t:.6f}" for t in runtimes_s)

lines = [
    f"{'config':<28} {'mean (s)':>10} {'std (s)':>10} {'min (s)':>10}",
    f"{'Futhark':<28} {mu:10.6f} {std:10.6f} {mn:10.6f}",
    "",
    f'  "LBM": {{ # {steps} Iterations',
    f'    "Futhark": [{ts_str}],',
    "  },",
]
text = "\n".join(lines) + "\n"
print(text, end="")
with open(summary_path, "w") as f:
    f.write(text)
print(f"summary written to {summary_path}")
EOF
