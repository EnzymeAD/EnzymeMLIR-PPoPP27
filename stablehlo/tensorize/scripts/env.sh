# Shared settings for the scripts in this directory (source it; every value can be overridden
# from the environment).

HERE=$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]}")/.." > /dev/null && pwd)

# Inputs
TENSORIZE=${TENSORIZE:-$HERE/tensorize}                 # submodule: Tensorize @ 6c25396 (+ patches/)
TZ_PROGRAMS=${TZ_PROGRAMS:-$HERE/programs/tensorize}    # Tensorize programs to time (<suite>/<kernel>.out)
TS_PROGRAMS=${TS_PROGRAMS:-$HERE/programs/tenspiler}    # Tenspiler generated NumPy (<suite>/<kernel>_numpy.py)

# Tools (built by setup/*.sh)
BUILD=${BUILD:-$HERE/build}
ENZYME_JAX_REV=${ENZYME_JAX_REV:-5d830a05e3164dc5b87713b01af986c776efc98b}
OPT=${OPT:-$BUILD/Enzyme-JAX/bazel-bin/enzymexlamlir-opt}
JAX_PY=${JAX_PY:-$BUILD/venv-jax/bin/python}            # jax 0.11.2 + enzyme-ad 0.0.15 (timing)
TZ_PY=${TZ_PY:-$BUILD/venv-tensorize/bin/python}        # Tensorize's own Python env (synthesis only)
export OPT

# Outputs
OUT=${OUT:-$HERE/out}

# Suites: the Tensorize benchmark suite ("polybench"), Kiriko PolyBench/C 3.2, and the Tensorize
# software suites.
ALL_SUITES="polybench kiriko blas blend darknet dsp dspstone llama makespeare mathfu simpl_array utdsp"
SUITES=${SUITES:-$ALL_SUITES}

# Kernels not timed by default: one Enzyme-JAX call takes > 30 min on an RTX 5090 (and 8-17 min on CPU).
SKIP=${SKIP:-"polybench/symm kiriko/symm kiriko/dynprog kiriko/floyd-warshall kiriko/cholesky kiriko/ludcmp kiriko/trmm"}

# Timing protocol: WARMUP untimed calls, then REPS timed calls; the first DISCARD timed calls are
# dropped and the harmonic mean of the rest is reported. Tensorize and Enzyme-JAX are interleaved.
WARMUP=${WARMUP:-1}
REPS=${REPS:-5}
DISCARD=${DISCARD:-1}
TIMEOUT=${TIMEOUT:-1800}       # seconds per kernel (all calls)
CPUS=${CPUS:-}                 # optional taskset core list for the timing runs, e.g. 0-15

suite_dir() {  # directory holding a suite's kernels (*.mlir)
  case $1 in
    polybench) echo "$TENSORIZE/benchmark/polybench" ;;
    kiriko) echo "$HERE/benchmarks/kiriko" ;;
    *) echo "$TENSORIZE/benchmark/software/$1" ;;
  esac
}

pin() {  # run "$@" pinned to $CPUS if set
  if [ -n "$CPUS" ]; then taskset -c "$CPUS" "$@"; else "$@"; fi
}

need() {  # fail early with a clear message if a tool is missing
  for f in "$@"; do [ -x "$f" ] || { echo "missing: $f (see setup/ and README.md)" >&2; exit 1; }; done
}
