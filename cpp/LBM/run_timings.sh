#!/bin/bash
# Build and time the LBM gradient under each AD configuration.
# Every configuration gets its own binary (lbm_<tag>) since the Makefile does
# not track flag changes, so the objects are rebuilt from clean each time.
#
# usage: ./run_timings.sh [STEPS] [REPS]       (defaults: 100 steps, 12 reps)
# env:   WARMUP=1  RESULTS_DIR=timings  SKIP_BUILD=0  CUDA_VISIBLE_DEVICES=0
#        MEMORY=1       also run each config once under Nsight Systems to get the
#                       peak allocated device memory (cudaMalloc/cudaFree tracking)
#        NSYS           nsys binary (default $CUDA_PATH/bin/nsys)
#        REACTANT_BIN   dir with reactant-clang  (default ../Reactant/enzyme/bazel-bin, next to the repo)
#        ENZYME_ROOT    Enzyme checkout with llvm-project/buildRelease and enzyme/buildRelease
#                       (default ../Enzyme, next to the repo)
#        any Makefile variable (CUDA_PATH, SM_VERSION, ...)
set -u
cd "$(dirname "$0")"
REPO_PARENT=$(cd ../../.. && pwd)

# EnzymeMLIR: Reactant's clang, with the plugin embedded
REACTANT_BIN=${REACTANT_BIN:-$REPO_PARENT/Reactant/enzyme/bazel-bin}
MLIR_VARS=(EMBEDDED_CLANG=yes
  "CLANG_PATH=$REACTANT_BIN/reactant-clang -resource-dir $REACTANT_BIN/reactant-clang-resource")
# EnzymeLLVM: upstream clang + ClangEnzyme plugin built against it
ENZYME_ROOT=${ENZYME_ROOT:-$REPO_PARENT/Enzyme}
LLVM_VARS=(USE_ENZYME_LLVM=yes
  "CLANG_PATH=$ENZYME_ROOT/llvm-project/buildRelease/bin/clang++"
  "CLANG_ENZYME=$ENZYME_ROOT/enzyme/buildRelease/Enzyme/ClangEnzyme-24.so")

STEPS=${1:-${STEPS:-100}}
REPS=${2:-${REPS:-12}}
WARMUP=${WARMUP:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
RESULTS_DIR=${RESULTS_DIR:-timings}
INPUT=datasets/lbm/short/input/120_120_150_ldc.of
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MEMORY=${MEMORY:-1}
NSYS=${NSYS:-${CUDA_PATH:-/usr/local/cuda-12.9}/bin/nsys}
mkdir -p "$RESULTS_DIR"
if [ "$MEMORY" = 1 ] && [ ! -x "$NSYS" ]; then
  echo "nsys not found at $NSYS (set NSYS=... or MEMORY=0)"; exit 1
fi

# tag | display name | toolchain | extra make variables
CONFIGS=(
  "mlir_mincut|EnzymeMLIR (mincut)|mlir|"
  "llvm|EnzymeLLVM|llvm|"
  "binomial4|Binomial b=4|mlir|BINOMIAL=yes BINOMIAL_BUDGET=4"
  "binomial4_nomincut|Binomial b=4 (no mincut)|mlir|BINOMIAL=yes BINOMIAL_BUDGET=4 MINCUT=no"
  "binomial16|Binomial b=16|mlir|BINOMIAL=yes BINOMIAL_BUDGET=16"
  "binomial16_nomincut|Binomial b=16 (no mincut)|mlir|BINOMIAL=yes BINOMIAL_BUDGET=16 MINCUT=no"
)

build() { # build TAG MAKEVARS...
  local tag=$1; shift
  echo "== build $tag"
  make clean >/dev/null
  if ! { echo "make $*"; make -j"$(nproc)" "$@"; } > "$RESULTS_DIR/build_$tag.log" 2>&1; then
    echo "   build FAILED, see $RESULTS_DIR/build_$tag.log"
    return 1
  fi
  mv lbm "lbm_$tag"
}

# Prints the wall time of the kernel loop in seconds, or nothing on failure
run_once() { # run_once TAG
  ./"lbm_$1" "$STEPS" -i "$INPUT" -o "$RESULTS_DIR/out_$1.txt" 2>&1 \
    | tee -a "$RESULTS_DIR/run_$1.log" \
    | awk '/^nt: [0-9]+ took [0-9]+ us/ { printf "%.6f\n", $4 / 1e6 }'
}

# Separate profiled run, so the tracing overhead stays out of the timings.
# Prints "<peak> <baseline> <peak-baseline> <#allocs>" in bytes, see peak_mem.py
profile_mem() { # profile_mem TAG
  local rep="$RESULTS_DIR/nsys_$1"
  "$NSYS" profile -t cuda --cuda-memory-usage=true --export=sqlite -f true -o "$rep" \
    ./"lbm_$1" "$STEPS" -i "$INPUT" -o "$RESULTS_DIR/out_$1.txt" \
    > "$RESULTS_DIR/nsys_$1.log" 2>&1 && python3 peak_mem.py "$rep.sqlite"
}

declare -A TIMES MEM
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tag name tc vars <<< "$cfg"
  if [ "$SKIP_BUILD" != 1 ] || [ ! -x "lbm_$tag" ]; then
    if [ "$tc" = llvm ]; then tcvars=("${LLVM_VARS[@]}"); else tcvars=("${MLIR_VARS[@]}"); fi
    # shellcheck disable=SC2086
    build "$tag" "${tcvars[@]}" $vars || continue
  fi

  : > "$RESULTS_DIR/run_$tag.log"
  echo "== run $tag: $name, steps=$STEPS, $WARMUP warmup + $REPS reps"
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

  if [ "$MEMORY" = 1 ]; then
    echo "== nsys $tag"
    if m=$(profile_mem "$tag") && [ -n "$m" ]; then
      MEM[$tag]=$m
      awk '{ printf "   peak %.1f MB, AD overhead %.1f MB over %d allocations\n", $1/1e6, $3/1e6, $4 }' <<< "$m"
    else
      echo "   nsys FAILED, see $RESULTS_DIR/nsys_$tag.log"
    fi
  fi
done
make clean >/dev/null

# Summary table plus a dict to paste into results/cpp_results.py
SUMMARY="$RESULTS_DIR/summary_${STEPS}.txt"
{
  printf "%-28s %10s %10s %10s %12s %12s\n" "config" "mean (s)" "std (s)" "min (s)" \
    "peak (MB)" "AD mem (MB)"
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r tag name _ <<< "$cfg"
    [ -n "${TIMES[$tag]:-}" ] || { printf "%-28s %10s\n" "$name" "FAILED"; continue; }
    tr ' ' '\n' <<< "${TIMES[$tag]}" | awk -v n="$name" -v mem="${MEM[$tag]:-}" '
      { s += $1; ss += $1 * $1; if (NR == 1 || $1 < m) m = $1 }
      END { mu = s / NR; v = ss / NR - mu * mu; if (v < 0) v = 0
            split(mem, b, " ")
            pk = mem == "" ? "-" : sprintf("%.1f", b[1] / 1e6)
            ad = mem == "" ? "-" : sprintf("%.1f", b[3] / 1e6)
            printf "%-28s %10.6f %10.6f %10.6f %12s %12s\n", n, mu, sqrt(v), m, pk, ad }'
  done
  echo "(peak = all live cudaMalloc bytes; AD mem = peak minus the grids live before the loop)"
  echo
  echo "  \"LBM\": { # $STEPS Iterations"
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r tag name _ <<< "$cfg"
    [ -n "${TIMES[$tag]:-}" ] || continue
    echo "    \"$name\": [$(sed 's/ /, /g' <<< "${TIMES[$tag]}")],"
  done
  echo "  },"
  if [ ${#MEM[@]} -gt 0 ]; then
    echo
    echo "  \"LBM\": { # peak allocated device memory in bytes, $STEPS Iterations"
    for cfg in "${CONFIGS[@]}"; do
      IFS='|' read -r tag name _ <<< "$cfg"
      [ -n "${MEM[$tag]:-}" ] || continue
      echo "    \"$name\": ${MEM[$tag]%% *},"
    done
    echo "  },"
  fi
} | tee "$SUMMARY"
echo "summary written to $SUMMARY"
