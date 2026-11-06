#!/bin/bash
# Build and time the RSBench gradient under EnzymeMLIR and EnzymeLLVM.
#
# usage: ./run_timings.sh [LOOKUPS] [REPS]     (defaults: 10200 lookups, 12 reps)
# env:   WARMUP=1  RESULTS_DIR=timings  SKIP_BUILD=0  CUDA_VISIBLE_DEVICES=0
#        RSBENCH_EXTRA_ARGS=""  (e.g. "-s small")
#        any Makefile variable (CUDA_PATH, SM_VERSION, CLANG_PATH, ...)
set -u
cd "$(dirname "$0")"

LOOKUPS=${1:-${LOOKUPS:-10200}}
REPS=${2:-${REPS:-12}}
WARMUP=${WARMUP:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
RESULTS_DIR=${RESULTS_DIR:-timings}
RSBENCH_EXTRA_ARGS=${RSBENCH_EXTRA_ARGS:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Detect SM_VERSION if not provided
if [ -z "${SM_VERSION:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    SM_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  fi
  SM_VERSION=${SM_VERSION:-80}
fi
export SM_VERSION

# Set up toolchain paths for Docker container layout ($EVAL_HOME or /home/evaluator)
if [ -d "/home/evaluator/llvm-project" ]; then
  EVAL_HOME=${EVAL_HOME:-/home/evaluator}
else
  EVAL_HOME=${EVAL_HOME:-$HOME}
fi
CLANG_PATH=${CLANG_PATH:-$EVAL_HOME/llvm-project/build/bin/clang++}
CLANG_ENZYME=${CLANG_ENZYME:-$EVAL_HOME/Enzyme/build/Enzyme/ClangEnzyme-24.so}
REACTANT_PATH=${REACTANT_PATH:-$EVAL_HOME/Reactant/build/Enzyme/libClangReactant-24.so}
LIB_RAISE_PATH=${LIB_RAISE_PATH:-$EVAL_HOME/Enzyme-JAX/bazel-bin/libRaise.so}

TOOLCHAIN_VARS=(
  "CLANG_PATH=$CLANG_PATH"
  "CLANG_ENZYME=$CLANG_ENZYME"
  "REACTANT_PATH=$REACTANT_PATH"
  "LIB_RAISE_PATH=$LIB_RAISE_PATH"
  "SM_VERSION=$SM_VERSION"
)

mkdir -p "$RESULTS_DIR"

CONFIGS=(
  "mlir|EnzymeMLIR|USE_ENZYME_LLVM=no"
  "llvm|EnzymeLLVM|USE_ENZYME_LLVM=yes"
)

build() {
  local tag=$1; shift
  echo "== build $tag"
  make clean >/dev/null
  if ! { echo "make $*"; make -j"$(nproc)" "$@"; } > "$RESULTS_DIR/build_$tag.log" 2>&1; then
    echo "   build FAILED, see $RESULTS_DIR/build_$tag.log"
    return 1
  fi
  mv rsbench "rsbench_$tag"
}

run_once() {
  # shellcheck disable=SC2086
  ./"rsbench_$1" -m event $RSBENCH_EXTRA_ARGS -l "$LOOKUPS" 2>&1 \
    | tee -a "$RESULTS_DIR/run_$1.log" \
    | awk '/^Runtime:/ { printf "%.6f\n", $2 }'
}

declare -A TIMES
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tag name vars <<< "$cfg"
  if [ "$SKIP_BUILD" != 1 ] || [ ! -x "rsbench_$tag" ]; then
    # shellcheck disable=SC2086
    build "$tag" "${TOOLCHAIN_VARS[@]}" $vars || continue
  fi

  : > "$RESULTS_DIR/run_$tag.log"
  echo "== run $tag: $name, lookups=$LOOKUPS, $WARMUP warmup + $REPS reps"
  for ((i = 0; i < WARMUP; i++)); do run_once "$tag" >/dev/null; done
  times=()
  for ((i = 0; i < REPS; i++)); do
    t=$(run_once "$tag")
    if [ -z "$t" ]; then
      echo "   run FAILED, see $RESULTS_DIR/run_$tag.log"
      times=(); break
    fi
    times+=("$t")
  done
  [ ${#times[@]} -gt 0 ] && TIMES[$tag]="${times[*]}" && echo "   ${times[*]}"
done
make clean >/dev/null

SUMMARY="$RESULTS_DIR/summary_${LOOKUPS}.txt"
{
  printf "%-28s %10s %10s %10s\n" "config" "mean (s)" "std (s)" "min (s)"
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r tag name _ <<< "$cfg"
    [ -n "${TIMES[$tag]:-}" ] || { printf "%-28s %10s\n" "$name" "FAILED"; continue; }
    tr ' ' '\n' <<< "${TIMES[$tag]}" | awk -v n="$name" '
      { s += $1; ss += $1 * $1; if (NR == 1 || $1 < m) m = $1 }
      END { mu = s / NR; v = ss / NR - mu * mu; if (v < 0) v = 0
            printf "%-28s %10.6f %10.6f %10.6f\n", n, mu, sqrt(v), m }'
  done
  echo
  echo "  \"RSBench\": { # $LOOKUPS lookups"
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r tag name _ <<< "$cfg"
    [ -n "${TIMES[$tag]:-}" ] || continue
    echo "    \"$name\": [$(sed 's/ /, /g' <<< "${TIMES[$tag]}")],"
  done
  echo "  },"
} | tee "$SUMMARY"
echo "summary written to $SUMMARY"
