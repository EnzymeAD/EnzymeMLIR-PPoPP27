# Tensorize vs Enzyme-JAX (and Tenspiler) on StableHLO, timed under JAX

This folder reproduces the comparison between:

- [Tensorize](https://github.com/alexanderb14/tensorize) (CGO'25), which lifts loop-level MLIR to
  NumPy/JAX by program synthesis;
- Enzyme-JAX's `raise-affine-to-stablehlo` (EnzymeMLIR), which raises the same loop nests to StableHLO
  by compilation;
- [Tenspiler](https://github.com/tenspiler/tenspiler) (ECOOP'24), on the suites where it applies.

All programs run as jitted JAX on the same device-resident inputs. Tensorize's NumPy program runs
with `numpy` replaced by `jax.numpy`, and Enzyme-JAX's StableHLO runs through
`enzyme_ad.jax.hlo_call`. For each kernel the tables report:

- **preparation time:** synthesis, or raising;
- **XLA compile time**;
- **run time:** the harmonic mean of the timed calls.

Each Enzyme-JAX output is checked against Tensorize's.

## Suites

| suite | kernels | source | sizes |
|---|---|---|---|
| `polybench` | 15 | Tensorize's PolyBench subset (`tensorize/benchmark/polybench`) | PolyBench 4.2 LARGE (covariance shrunk to 5x3) |
| `kiriko` | 30 | Kiriko's PolyBench/C 3.2 MLIR ([lac-dcc/Kiriko](https://github.com/lac-dcc/Kiriko) @ `43a6444`), specialized to the sizes each `main()` uses (`benchmarks/kiriko`, made with `scripts/kiriko_specialize.py`) | e.g. 1024 (gemm, lu), 4000 (atax, durbin) |
| `blas`, `blend`, `darknet`, `dsp`, `dspstone`, `llama`, `makespeare`, `mathfu`, `simpl_array`, `utdsp` | 82 | Tensorize's software suites (`tensorize/benchmark/software`) | mostly 1000 / 1000x1000 |

Tenspiler is compared on `llama` and `blend`, which are its own benchmarks, and on `blas` and `dsp`,
which we ported (`tenspiler_ports/`).

## Layout

```
tensorize/            submodule: Tensorize @ 6c25396 (the revision used)
patches/              tensorize-fixes.patch (our fixes to Tensorize, for re-running synthesis),
                      tenspiler-bitwuzla-path.patch
benchmarks/kiriko/    the 30 specialized Kiriko kernels
meta/<suite>/         outputs.txt (buffers a kernel returns), tensorize_broken.txt (known-wrong programs)
programs/tensorize/   the Tensorize programs that were timed (<suite>/<kernel>.out) + synth_times.csv
programs/tenspiler/   Tenspiler's generated code (<suite>/<kernel>_numpy.py) + synth_times.csv
tenspiler_ports/      blas/dsp ported to Tenspiler (C++ + drivers), tenspiler_run.py
setup/                jax.sh, enzyme_jax.sh (needed); tensorize.sh, tenspiler.sh (only to re-synthesize)
scripts/              env.sh (all settings), prepare/raise/time/time_tenspiler.sh, synth.sh,
                      tenspiler_synth.sh, compare_suite.py, tables.py, ...
requirements/         exact Python environments of the runs (jax-cuda12.txt, jax-cpu.txt)
results/              our reference run (tables.md, run_info.txt, raw CSVs)
KNOWN_ISSUES.md       wrong programs, kernels that don't compute what their name says, Enzyme-JAX
                      issues, Tenspiler caveats, timing caveats
```

## Quick start

Requirements:

- Linux x86-64 with Python 3.12 and git, plus [uv](https://docs.astral.sh/uv/) (or pip);
- bazel (bazelisk) and clang, to build Enzyme-JAX, as in the top-level `Dockerfile`;
- for GPU timings, an NVIDIA GPU with a CUDA 12 driver. Our runs used an RTX 5090.

```sh
git submodule update --init stablehlo/tensorize/tensorize
cd stablehlo/tensorize
setup/jax.sh cuda12          # or: setup/jax.sh cpu     -> build/venv-jax (jax 0.11.2, enzyme-ad 0.0.15)
setup/enzyme_jax.sh          # enzymexlamlir-opt @ Enzyme-JAX 5d830a0 (~10 min with bazel's remote cache)
./run_all.sh                 # all suites -> out/tables.md
./run_all.sh blas llama dsp  # or a subset
```

`run_all.sh` does the following:

1. `scripts/prepare.sh` collects the kernels and metadata into `out/benchmarks/`.
2. `scripts/raise.sh` raises each kernel with Enzyme-JAX into `out/raised/`. Each kernel is wrapped
   in a `@main` that returns its output buffers, then put through Reactant's default CPU pass
   pipeline with `prefer_while_raising=true` (`scripts/raise_pipeline.sh`). This takes about 0.1 s
   per kernel.
3. `scripts/time.sh` times the shipped Tensorize programs against the raised StableHLO under JAX
   (`compare_suite.py`) into `out/timings/`.
4. `scripts/time_tenspiler.sh` times Tenspiler's code with the same protocol.
5. `scripts/tables.py` writes `out/tables.md`. Every kernel gets a row with its status: timed,
   broken, skipped, timeout, or no program.

### Settings

All settings are in `scripts/env.sh` and can be overridden from the environment:

- **Device.** The GPU is used if the CUDA plugin is installed. `JAX_PLATFORMS=cpu` forces CPU, and
  `CUDA_VISIBLE_DEVICES` picks a GPU.
- **Protocol.** `WARMUP=1 REPS=5 DISCARD=1`: one untimed call, then five timed calls. The first
  timed call is dropped (see KNOWN_ISSUES.md) and the harmonic mean of the other four is reported.
  Tensorize and Enzyme-JAX are interleaved call by call.
- **`SKIP`.** Kernels where one Enzyme-JAX call takes more than 30 min on GPU are skipped by default:
  `polybench/symm`, and Kiriko `symm`, `dynprog`, `floyd-warshall`, `cholesky`, `ludcmp` and `trmm`.
  Use `SKIP=` to time everything.
- **`TIMEOUT`.** Seconds per kernel (default 1800). `CPUS` pins the timing runs with `taskset`.
- **Other.** `OUT` (output directory), `OPT` (another `enzymexlamlir-opt`), `ENZYME_JAX_REV`.
  `scripts/raise_pipeline.sh` is the pass pipeline of the pinned revision. Newer revisions changed
  some pass options; for example, at `5ff3f33` `lower-jit` accepts only
  `{jit=true dump_final_module=false}`.

### Expected duration

With the default `SKIP`, a full GPU run takes about 55 min on an RTX 5090. Our reference run took
2 h 23 min, because Kiriko cholesky, ludcmp and trmm were then not skipped and each hit the 1800 s
`TIMEOUT`. Most of the remaining time is Kiriko durbin, seidel-2d and trisolv, which take 80-170 s
per call (see KNOWN_ISSUES.md). The software suites take seconds per kernel. On CPU, several Kiriko
kernels take minutes per call.

## Re-running synthesis (optional)

The timings use the programs the tools produced for our runs, in `programs/`. To regenerate them:

- **Tensorize.**
  - `setup/tensorize.sh` applies `patches/tensorize-fixes.patch` and builds Tensorize with its own
    scripts. This builds LLVM/MLIR and MLIR-HLO first, which takes several hours.
  - Then `SYNTH=1 ./run_all.sh`, or run `scripts/synth.sh` followed by
    `TZ_PROGRAMS=out/programs/tensorize scripts/time.sh`.
  - Synthesis takes seconds to 30 min per kernel (`SYNTH_TIMEOUT`, default 1800 s).
  - Use `TENSORIZE_SOLVE_TIMEOUT=20` on a loaded machine (see KNOWN_ISSUES.md).
- **Tenspiler.**
  - `setup/tenspiler.sh` installs Tenspiler @ `9705480` and its toolchain locally, without root:
    Racket 8.7 + Rosette, Bitwuzla 0.7.0, cvc5 1.1.2, LLVM 11.1.
  - Then run `scripts/tenspiler_synth.sh` followed by
    `TS_PROGRAMS=out/programs/tenspiler scripts/time_tenspiler.sh`.
  - Synthesis takes 2-70 s per kernel.
- **Kiriko kernels.** `python3 scripts/kiriko_specialize.py <Kiriko>/polybench-c-mlir-3.2 --opt
  build/Enzyme-JAX/bazel-bin/enzymexlamlir-opt` regenerates `benchmarks/kiriko/`.

## Reference results

`results/` holds a full run of this folder's pipeline on our machine. The setup is recorded in
`results/run_info.txt`: 2x RTX 5090, driver 575.57, jax 0.11.2, enzymexlamlir-opt built by
`setup/enzyme_jax.sh`. The summary is in `results/tables.md`. Absolute times depend on the GPU and
machine load. Kernels under ~0.1 ms are dominated by launch overhead.
