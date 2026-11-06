#!/bin/bash
# Orchestrator script to build and benchmark LBM (C++ EnzymeMLIR/EnzymeLLVM and Futhark)
# inside the EnzymeMLIR Docker container.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/.." && pwd)

LBM_STEPS=${LBM_STEPS:-100}
LBM_REPS=${LBM_REPS:-5}

export PATH="/home/evaluator/.local/bin:$PATH"

if [ -d "/home/evaluator/llvm-project" ]; then
  export HOME=/home/evaluator
  EVAL_HOME=${EVAL_HOME:-/home/evaluator}
else
  EVAL_HOME=${EVAL_HOME:-$HOME}
fi
export CLANG_PATH=${CLANG_PATH:-$EVAL_HOME/llvm-project/build/bin/clang++}
export CLANG_ENZYME=${CLANG_ENZYME:-$EVAL_HOME/Enzyme/build/Enzyme/ClangEnzyme-24.so}
export REACTANT_PATH=${REACTANT_PATH:-$EVAL_HOME/Reactant/build/Enzyme/libClangReactant-24.so}
export LIB_RAISE_PATH=${LIB_RAISE_PATH:-$EVAL_HOME/Enzyme-JAX/bazel-bin/libRaise.so}

# If libRaise.so is not yet built in the container, build :libRaise.so from Enzyme-JAX (5ff3f33)
if [ ! -f "$LIB_RAISE_PATH" ] && [ "${BUILD_LIBRAISE_IF_MISSING:-1}" = "1" ]; then
  echo "libRaise.so not found at $LIB_RAISE_PATH; building :libRaise.so..."
  if [ ! -d "$EVAL_HOME/Enzyme-JAX" ]; then
    git clone https://github.com/EnzymeAD/Enzyme-JAX "$EVAL_HOME/Enzyme-JAX"
    (cd "$EVAL_HOME/Enzyme-JAX" && git checkout 5ff3f33)
  fi
  (cd "$EVAL_HOME/Enzyme-JAX" && CC=clang-18 HERMETIC_PYTHON_VERSION=3.12 bazel build :libRaise.so --jobs="${NJOBS:-$(nproc)}")
fi

if [ -z "${SM_VERSION:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    SM_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  fi
  SM_VERSION=${SM_VERSION:-80}
fi
export SM_VERSION

echo "================================================================="
echo "Starting LBM Benchmarks (SM_VERSION=$SM_VERSION, steps=$LBM_STEPS, reps=$LBM_REPS)"
echo "================================================================="

echo "=== 1. LBM C++ (EnzymeMLIR & EnzymeLLVM) ==="
cd "$HERE/cpp"
if command -v nsys >/dev/null 2>&1 || [ -x "${CUDA_PATH:-/usr/local/cuda-12.9}/bin/nsys" ]; then
  export MEMORY=${MEMORY:-1}
else
  export MEMORY=0
fi
./run_timings.sh "$LBM_STEPS" "$LBM_REPS"

echo "=== 2. LBM Futhark (CUDA) ==="
cd "$HERE/futhark"
./run_timings.sh "$LBM_REPS"

echo "=== 3. Aggregate & Verify LBM Results ==="
cd "$REPO_ROOT"
mkdir -p results
python3 - <<'EOF'
import ast, glob, json, pathlib, re, sys

results = {"runtimes_s": {}, "peak_memory_bytes": {}}
block_pat = re.compile(r'"LBM":\s*\{\s*#([^\n]*)\n([^}]*)\}', re.MULTILINE)

for path in sorted(glob.glob("lbm/*/timings/summary_*.txt")):
    with open(path) as f:
        content = f.read()
    for match in block_pat.finditer(content):
        comment, body = match.group(1), match.group(2)
        target = results["peak_memory_bytes"] if "memory" in comment.lower() else results["runtimes_s"]
        bench_data = target.setdefault("LBM", {})
        for line in body.strip().splitlines():
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip().strip('"')
                v = v.strip().rstrip(",")
                try:
                    bench_data[k] = ast.literal_eval(v)
                except Exception:
                    bench_data[k] = v

out_path = pathlib.Path("results/lbm_results_latest.json")
out_path.write_text(json.dumps(results, indent=2) + "\n")
print(f"Aggregated LBM benchmark results written to {out_path}")
print(json.dumps(results, indent=2))

lbm_times = results.get("runtimes_s", {}).get("LBM", {})
required = ["EnzymeMLIR (mincut)", "EnzymeLLVM", "Futhark"]
missing = [k for k in required if not lbm_times.get(k)]
if missing:
    print(f"ERROR: Missing or failed LBM benchmark configurations: {missing}", file=sys.stderr)
    for log_file in sorted(glob.glob("lbm/cpp/timings/*.log")):
        print(f"--- {log_file} ---", file=sys.stderr)
        print(pathlib.Path(log_file).read_text(errors="replace")[-3000:], file=sys.stderr)
    sys.exit(1)
print("VERIFIED: All required LBM configurations (EnzymeMLIR, EnzymeLLVM, Futhark) succeeded!")
EOF
