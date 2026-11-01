#!/bin/bash
# JAX vs EnzymeMLIR: value and gradient of the MPAS single-observation 4D-Var cost.
# Each configuration runs in its own process so the reported peak GPU memory is its own.
# usage: ./run_comparison.sh            (GPU picked with CUDA_VISIBLE_DEVICES; default 0)
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/legoESM${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export RESULTS_DIR=${RESULTS_DIR:-results}
PY=${PYTHON:-python}
mkdir -p "$RESULTS_DIR"

run() { # run TAG MODE STEPS [ENV=VALUE ...]
  local tag=$1 mode=$2 steps=$3; shift 3
  echo "== $tag steps=$steps"
  env "$@" RUN_TAG=$tag $PY enzyme_4dvar_grad.py experiment_1h.yaml $mode $steps \
    > "$RESULTS_DIR/${tag}_${steps}.log" 2>&1
  grep -E "^mode=$mode loop_ckpt" "$RESULTS_DIR/${tag}_${steps}.log" \
    || grep -oE "RESOURCE_EXHAUSTED.*|Error.*" "$RESULTS_DIR/${tag}_${steps}.log" | tail -1
}

for steps in 10 60; do
  run jax jax $steps
  run enzyme enzyme $steps
done
for p in 1 2 4 8 16 30 60; do
  run enzyme_periodic$p enzyme 60 ENZYME_LOOP_CKPT=1 ENZYME_CHECKPOINT_PERIOD=$p
done
for p in 2 4 8 16 30 60; do  # binomial requires a budget > 1
  run enzyme_binomial$p enzyme 60 ENZYME_LOOP_CKPT=1 ENZYME_CHECKPOINT_PERIOD=$p ENZYME_BINOMIAL=1
done

echo "== gradient relative difference vs JAX (same steps)"
$PY - "$RESULTS_DIR" <<'EOF'
import sys, pathlib, numpy as np
d = pathlib.Path(sys.argv[1])
for f in sorted(d.glob("grad_enzyme*.npy")):
    steps = f.stem.rsplit("_", 1)[1]
    ref = d / f"grad_jax_{steps}.npy"
    if ref.exists():
        j, e = np.load(ref), np.load(f)
        print(f"{f.stem}: {np.linalg.norm(e - j) / np.linalg.norm(j):.2e}")
EOF
