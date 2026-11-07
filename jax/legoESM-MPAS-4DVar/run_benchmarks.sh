#!/bin/bash
# Orchestrator script to run the legoESM MPAS-4DVar benchmark (EnzymeMLIR vs JAX)
# inside the EnzymeMLIR Docker container.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)

export PATH="/home/evaluator/.local/bin:/home/evaluator/eval-env/bin:$PATH"

EVAL_ENV="${EVAL_ENV:-/home/evaluator/eval-env}"
if [ -d "$EVAL_ENV" ]; then
  RUN_PY="$EVAL_ENV/bin/python3"
elif [ -x "$HERE/.venv/bin/python" ]; then
  RUN_PY="$HERE/.venv/bin/python"
else
  RUN_PY=$(command -v python3)
fi

cd "$HERE"
if [ ! -d "legoESM/packages" ] && [ ! -d "legoESM/src" ]; then
  git submodule update --init --recursive legoESM || true
fi

if [ -d "$EVAL_ENV" ]; then
  if ! "$RUN_PY" -c "import legoesm.da.control_vector, legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas" >/dev/null 2>&1; then
    echo "Installing legoESM federation packages into $EVAL_ENV..."
    "$RUN_PY" legoESM/scripts/experiment/install_federation.py core atmosphere ml
  fi
elif [ ! -x ".venv/bin/python" ] && command -v uv >/dev/null 2>&1; then
  echo "Setting up legoESM environment with uv..."
  if [ -n "${ENZYME_AD_WHL:-}" ]; then
    bash ./setup_env.sh "$ENZYME_AD_WHL"
  else
    bash ./setup_env.sh
  fi
  if [ -x ".venv/bin/python" ]; then
    RUN_PY="$PWD/.venv/bin/python"
  fi
fi

# Check if enzyme-ad 0.0.10 is used; if so, exclude transpose_like_broadcast_concat
ENZYME_VER=$("$RUN_PY" -c "import importlib.metadata as m; print(m.version('enzyme-ad'))" 2>/dev/null || echo "")
if [ "$ENZYME_VER" = "0.0.10" ]; then
  export ENZYME_EXCLUDED_PASSES="${ENZYME_EXCLUDED_PASSES:-transpose_like_broadcast_concat}"
  echo "Detected enzyme-ad 0.0.10; setting ENZYME_EXCLUDED_PASSES=$ENZYME_EXCLUDED_PASSES"
fi

echo "================================================================="
echo "Starting legoESM MPAS-4DVar Benchmark"
echo "Python: $($RUN_PY --version 2>&1) (enzyme-ad=${ENZYME_VER:-unknown})"
echo "================================================================="


# Refuse to benchmark on a silent CPU fallback (e.g. CUDA plugin rejected the cuDNN version)
if [ "${ALLOW_CPU:-0}" != "1" ]; then
  "$RUN_PY" - <<'EOF'
import sys, jax
devs = jax.devices()
print("JAX", jax.__version__, "devices:", devs)
if not any(d.platform in ("gpu", "cuda") for d in devs):
    sys.exit("ERROR: JAX has no GPU backend (set ALLOW_CPU=1 to override)")
EOF
fi

# Ensure background sample data/mpas_nmc_case_009_20170115000000.npz exists
SAMPLE_NPZ="data/mpas_nmc_case_009_20170115000000.npz"
if [ ! -f "$SAMPLE_NPZ" ]; then
  if [ -f "/home/evaluator/jax/legoESM-MPAS-4DVar/$SAMPLE_NPZ" ]; then
    ln -sf "/home/evaluator/jax/legoESM-MPAS-4DVar/$SAMPLE_NPZ" "$SAMPLE_NPZ"
  else
    echo "Note: $SAMPLE_NPZ not found; generating balanced baroclinic-wave MPAS state for benchmark timing/memory..."
    PYTHONPATH="$PWD/legoESM${PYTHONPATH:+:$PYTHONPATH}" "$RUN_PY" - <<'PYEOF'
from pathlib import Path
import numpy as np
from scripts.run.mpas_4dvar_single import common
from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
from legoesm.grids.voronoi import reconstruct_cell_velocity

cfg = common.load_config(Path("experiment_1h.yaml"))
mesh, sigma, _ = common.make_grid_sigma_model(cfg["model"])
state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True, moist=True)
u_cell, v_cell = reconstruct_cell_velocity(state.u.data, mesh)
out_path = Path(cfg["inputs"]["nmc_sample"])
out_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    out_path,
    f24_u=np.asarray(u_cell, dtype=np.float64),
    f24_v=np.asarray(v_cell, dtype=np.float64),
    f24_T=np.asarray(state.T.data, dtype=np.float64),
    f24_p_s=np.asarray(state.p_s.data, dtype=np.float64),
    f24_phis=np.asarray(state.phis.data, dtype=np.float64),
    f24_tracer_q_v=np.asarray(state.tracers["q_v"].data, dtype=np.float64),
)
print(f"Generated synthetic MPAS background sample at {out_path}")
PYEOF
  fi
fi

MPAS_MODE="${MPAS_MODE:-both}"

if [ -z "${GPUS:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd ' ' -)
  fi
  GPUS="${GPUS:-0}"
fi

if [ "$MPAS_MODE" = "comparison" ] || [ "$MPAS_MODE" = "both" ]; then
  COMP_DIR="$REPO_ROOT/results/mpas_4dvar"
  rm -rf "$COMP_DIR"
  mkdir -p "$COMP_DIR"
  echo "=== Running MPAS 4D-Var comparison (RESULTS_DIR=$COMP_DIR) ==="
  RESULTS_DIR="$COMP_DIR" PYTHON="$RUN_PY" bash ./run_comparison.sh
fi

if { [ "$MPAS_MODE" = "binomial_sweep" ] || [ "$MPAS_MODE" = "both" ]; } && [ -f "./run_binomial_sweep.sh" ]; then
  SWEEP_DIR="$REPO_ROOT/results/binomial_sweep"
  rm -rf "$SWEEP_DIR"
  mkdir -p "$SWEEP_DIR"
  echo "=== Running MPAS 4D-Var binomial checkpointing sweep across GPUs [$GPUS] (RESULTS_DIR=$SWEEP_DIR) ==="
  RESULTS_DIR="$SWEEP_DIR" GPUS="$GPUS" RERUN=1 PYTHON="$RUN_PY" bash ./run_binomial_sweep.sh
fi

echo "=== Verifying legoESM MPAS-4DVar Benchmark Outputs ==="
cd "$REPO_ROOT"
"$RUN_PY" - "$MPAS_MODE" <<'EOF'
import csv, math, pathlib, re, sys
import numpy as np

mpas_mode = sys.argv[1]
ok_pat = re.compile(r"^mode=\S+ .*steps=(\d+) cost=(\S+) \|grad\|=(\S+) first_call=(\S+)s second_call=(\S+)s peak=(\S+)GiB", re.MULTILINE)

def verify_dir(d: pathlib.Path, required_ok: list[str]):
    if not d.exists():
        raise SystemExit(f"ERROR: Expected results directory {d} does not exist!")
    logs = sorted(d.glob("*.log"))
    if not logs:
        raise SystemExit(f"ERROR: No .log files found in {d}!")

    succeeded = set()
    for log in logs:
        text = log.read_text(errors="replace")
        m = ok_pat.search(text)
        if m:
            succeeded.add(log.stem)
            print(f"  [OK]  {log.stem}: {m.group(0)}")
        elif "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower():
            print(f"  [OOM] {log.stem}: device memory exhausted (expected for un-checkpointed large rollouts)")
        else:
            print(f"ERROR: Unexpected failure in {log}:", file=sys.stderr)
            print(text[-4000:], file=sys.stderr)
            sys.exit(1)

    missing_req = [r for r in required_ok if r not in succeeded]
    if missing_req:
        print(f"ERROR: Required MPAS-4DVar runs did not succeed in {d}: {missing_req}", file=sys.stderr)
        sys.exit(1)

    grad_files = sorted(d.glob("grad_enzyme*.npy"))
    if not grad_files:
        print(f"ERROR: No grad_enzyme*.npy files produced in {d}!", file=sys.stderr)
        sys.exit(1)

    for gf in grad_files:
        steps = gf.stem.rsplit("_", 1)[1]
        ref = d / f"grad_jax_{steps}.npy"
        if not ref.exists():
            print(f"ERROR: Missing JAX reference gradient {ref} for {gf}!", file=sys.stderr)
            sys.exit(1)
        j = np.load(ref)
        e = np.load(gf)
        rel_diff = float(np.linalg.norm(e - j) / np.linalg.norm(j))
        print(f"  [GRAD] {gf.name} vs {ref.name}: rel_diff = {rel_diff:.3e}")
        if not math.isfinite(rel_diff) or rel_diff > 1e-4:
            print(f"ERROR: Gradient mismatch for {gf.name}: rel_diff={rel_diff:.3e} > 1e-4", file=sys.stderr)
            sys.exit(1)

if mpas_mode in ("comparison", "both"):
    print("\n--- Verifying results/mpas_4dvar ---")
    verify_dir(
        pathlib.Path("results/mpas_4dvar"),
        required_ok=[
            "jax_10",
            "enzyme_10",
            "jax_60",
            "enzyme_periodic8_60",
            "enzyme_binomial8_60",
            "enzyme_binomial16_60",
            "enzyme_binomial60_60",
        ],
    )

if mpas_mode in ("binomial_sweep", "both"):
    print("\n--- Verifying results/binomial_sweep ---")
    sweep_dir = pathlib.Path("results/binomial_sweep")
    verify_dir(
        sweep_dir,
        required_ok=[
            "jax_60",
            "enzyme_binomial8_60",
            "enzyme_binomial16_60",
            "enzyme_binomial_n_60",
        ],
    )
    summary_csv = sweep_dir / "summary.csv"
    if not summary_csv.exists():
        raise SystemExit(f"ERROR: {summary_csv} not generated!")
    with open(summary_csv) as f:
        rows = list(csv.DictReader(f))
    failed_rows = [r for r in rows if r.get("status") != "ok"]
    if failed_rows:
        print(f"ERROR: Non-ok rows in {summary_csv}: {failed_rows}", file=sys.stderr)
        sys.exit(1)

print("\nVERIFIED: All required legoESM MPAS-4DVar configurations succeeded and matched JAX gradients!")
EOF
