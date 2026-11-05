#!/bin/bash
# Sweep EnzymeMLIR binomial checkpointing of the time loop over step counts and
# checkpoint budgets, with JAX as the reference.
# Each run is its own process, so its peak GPU memory is its own.
#
# usage: ./run_binomial_sweep.sh
# env:   STEPS="10 30 60 120 240"  time steps (60 s each; 60 = the 1 h window)
#        BUDGETS="2 4 8 16 n"      binomial checkpoint budgets (> 1); n = the step count,
#                                  i.e. every step checkpointed, like JAX
#        BASELINES="jax"           reference configs per step count ("" for none;
#                                  "jax enzyme" adds un-checkpointed Enzyme)
#        GPUS="0"                  devices; with several, runs are spread over them
#        RESULTS_DIR=results/binomial_sweep
#        RERUN=0                   1: redo runs whose log already has a result
#        PYTHON=python  LEGOESM_DIR=$PWD/legoESM
#        any variable read by enzyme_4dvar_grad.py, e.g. ENZYME_EXCLUDED_PASSES
#        (older Enzyme-JAX wheels such as 0.0.10 need
#        ENZYME_EXCLUDED_PASSES=transpose_like_broadcast_concat, otherwise the
#        Enzyme pass pipeline runs for hours and uses >100 GB of host memory)
# Writes RESULTS_DIR/<tag>_<steps>.log per run and RESULTS_DIR/summary.csv,
# which plot_binomial_sweep.jl reads.
set -u
cd "$(dirname "$0")"
export PYTHONPATH="${LEGOESM_DIR:-$PWD/legoESM}${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export RESULTS_DIR=${RESULTS_DIR:-results/binomial_sweep}
PY=${PYTHON:-python}
STEPS=${STEPS:-10 30 60 120 240}
BUDGETS=${BUDGETS:-2 4 8 16 n}
BASELINES=${BASELINES-jax}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
RERUN=${RERUN:-0}
mkdir -p "$RESULTS_DIR"

# One job per line: TAG MODE STEPS [ENV=VALUE ...]
jobs=()
for steps in $STEPS; do
  for b in $BASELINES; do jobs+=("$b $b $steps"); done
  for b in $BUDGETS; do
    if [ "$b" = n ]; then  # budget = steps: every step checkpointed, like JAX
      jobs+=("enzyme_binomial_n enzyme $steps ENZYME_LOOP_CKPT=1 ENZYME_CHECKPOINT_PERIOD=$steps ENZYME_BINOMIAL=1")
    else
      jobs+=("enzyme_binomial$b enzyme $steps ENZYME_LOOP_CKPT=1 ENZYME_CHECKPOINT_PERIOD=$b ENZYME_BINOMIAL=1")
    fi
  done
done

run() { # run GPU TAG MODE STEPS [ENV=VALUE ...]
  local gpu=$1 tag=$2 mode=$3 steps=$4; shift 4
  local log="$RESULTS_DIR/${tag}_${steps}.log"
  if [ "$RERUN" != 1 ] && grep -q "^mode=$mode loop_ckpt" "$log" 2>/dev/null; then
    echo "== [gpu $gpu] $tag steps=$steps: done, skipping"; return
  fi
  echo "== [gpu $gpu] $tag steps=$steps"
  env "$@" CUDA_VISIBLE_DEVICES=$gpu RUN_TAG=$tag \
    $PY enzyme_4dvar_grad.py experiment_1h.yaml $mode $steps > "$log" 2>&1
  grep -E "^mode=$mode loop_ckpt" "$log" \
    || grep -oE "RESOURCE_EXHAUSTED.*|Error.*" "$log" | tail -1
}

# Job i goes to GPU i mod #GPUS; each GPU works through its share in order.
gpus=($GPUS)
for g in "${!gpus[@]}"; do
  (
    for i in "${!jobs[@]}"; do
      [ $((i % ${#gpus[@]})) = "$g" ] && run "${gpus[$g]}" ${jobs[$i]}
    done
  ) &
done
wait

echo "== summary"
$PY - "$RESULTS_DIR" <<'EOF'
import sys, re, csv, pathlib, numpy as np
d = pathlib.Path(sys.argv[1])
pat = re.compile(r"^mode=\S+ .*steps=(\d+) cost=(\S+) \|grad\|=(\S+) first_call=(\S+)s second_call=(\S+)s peak=(\S+)GiB")
rows = []
for log in sorted(d.glob("*.log")):
    tag, steps = log.stem.rsplit("_", 1)
    text = log.read_text(errors="replace")
    m = next((m for m in map(pat.match, text.splitlines()) if m), None)
    budget = (steps if tag == "enzyme_binomial_n" else tag.removeprefix("enzyme_binomial")) \
        if tag.startswith("enzyme_binomial") else ""
    row = dict(tag=tag, budget=budget, steps=int(steps), status="ok", cost="", grad_norm="",
               first_call_s="", second_call_s="", peak_gib="", grad_rel_diff_vs_jax="")
    if m:
        row.update(cost=m[2], grad_norm=m[3], first_call_s=m[4], second_call_s=m[5], peak_gib=m[6])
        g, ref = d / f"grad_{tag}_{steps}.npy", d / f"grad_jax_{steps}.npy"
        if tag != "jax" and g.exists() and ref.exists():
            j = np.load(ref)
            row["grad_rel_diff_vs_jax"] = f"{np.linalg.norm(np.load(g) - j) / np.linalg.norm(j):.3e}"
    else:
        row["status"] = "oom" if "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower() else "failed"
    rows.append(row)
rows.sort(key=lambda r: (r["steps"], r["tag"]))
with open(d / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["tag"])
    w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['tag']:20s} {r['steps']:5d} {r['status']:6s} second_call={r['second_call_s']:>8s}s "
          f"peak={r['peak_gib']:>6s}GiB rel_diff={r['grad_rel_diff_vs_jax']}")
print(f"wrote {d / 'summary.csv'}")
EOF
