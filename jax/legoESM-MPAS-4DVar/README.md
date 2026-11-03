# legoESM MPAS 4D-Var: JAX vs EnzymeMLIR

Value and gradient of a single-observation 4D-Var cost on the legoESM MPAS
primitive-equation model (icosahedral mesh, 40962 cells, 40 levels), computed
two ways:

- `jax`: `jax.value_and_grad`, with `jax.checkpoint` around each time step
  inside the `lax.scan` time loop (legoESM's own setup).
- `enzyme`: the whole cost lowered to StableHLO and differentiated by
  EnzymeMLIR in one call (reverse mode, primal computed once). The script
  checks that no model loop or arithmetic is differentiated by JAX.

The control vector has 6,594,882 entries. The cost is a background term plus a
500 hPa temperature observation at one cell at the end of the forecast. One
time step is 60 s, so 60 steps is the 1-hour window.

## Contents

- `legoESM/`: submodule, pinned to the revision the results were produced with.
- `enzyme_4dvar_grad.py`: builds the cost exactly as legoESM's
  `scripts/run/mpas_4dvar_single/run_assimilation.py` does, then times two
  calls of value-and-gradient and reports peak GPU memory.
- `experiment_1h.yaml`: legoESM's `config/4DVar_single/experiment.yaml` with
  the window set to 1 hour and the input paths pointing at `data/`.
- `run_comparison.sh`: runs every configuration and compares each Enzyme
  gradient against JAX's.

## Setup

```bash
./setup_env.sh                          # released enzyme-ad 0.0.16 from PyPI
./setup_env.sh path/to/enzyme_ad-*.whl  # or a locally built Enzyme-JAX wheel
```

This creates `.venv/` with legoESM installed from the submodule's `uv.lock`,
CUDA 12 JAX 0.10.0, and Enzyme-JAX. It needs [uv](https://docs.astral.sh/uv/)
and Python 3.13 (uv fetches it if missing). `env_lock.txt` lists the exact
versions of every other package in that environment. A local wheel is built
from the `Enzyme-JAX` submodule at the repository root
(`HERMETIC_PYTHON_VERSION=3.13 bazel build :wheel`).

## Inputs (`data/`)

Both inputs come from legoESM's NMC pipeline, run on ERA5 (NCAR RDA ds633.0,
January to March 2017, fetched from the OSDF mirror
`https://osdf-director.osg-htc.org/ncar/gdex/d633000/`).

- Included: `data/mpas_gen_be.{npz,json}` and `data/mpas_gen_be_manifest.json`,
  the background error statistics. They were fitted in single precision from
  18 NMC cases (cases 63-80, valid 2017-03-10 to 2017-03-27) rather than the
  config's 80. The manifest lists the exact samples used. To refit, generate
  those cases, then run
  `PYTHONPATH=. python scripts/data/fit_mpas_gen_be.py --config config/4DVar_single/nmc.yaml`
  from `legoESM/`.
- To generate: `data/mpas_nmc_case_009_20170115000000.npz` (151 MB, too large
  to commit), the background state. It is one NMC case: 24 h and 48 h
  forecasts valid 2017-01-15 00 UTC. From `legoESM/`:
  ```bash
  LEGOESM_ERA5_RDA_ROOT=<era5 dir> PYTHONPATH=. python scripts/data/generate_mpas_nmc.py \
      --config config/4DVar_single/nmc.yaml --case-offset 8 --count 1
  cp outputs/mpas_nmc_r6_l40/samples/mpas_nmc_case_009_20170115000000.npz ../data/
  ```
  This case needs the ERA5 pressure-level files for 2017-01-13 and 2017-01-14
  and the January 2017 surface files.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON=.venv/bin/python ./run_comparison.sh
```

Per configuration it writes `results/<tag>_<steps>.log`, with one summary line
giving the cost, gradient norm, first-call time (includes compilation),
second-call time and peak memory. It also writes the gradient as
`results/grad_<tag>_<steps>.npy`. The script ends by printing each Enzyme
gradient's relative difference from JAX's.

Peak memory is JAX's allocator high-water mark on the device
(`memory_stats()["peak_bytes_in_use"]`), with preallocation off. Run on an
otherwise idle GPU: other processes on the device change both timing and the
point at which a configuration runs out of memory.

Configurations:

| tag | meaning |
|---|---|
| `jax` | JAX reverse mode, `jax.checkpoint` per step |
| `enzyme` | EnzymeMLIR, no loop checkpointing |
| `enzyme_periodic<N>` | EnzymeMLIR periodic checkpointing of the time loop, `enzyme.checkpoint_period = N` |
| `enzyme_binomial<N>` | EnzymeMLIR binomial checkpointing of the time loop, budget `N` |

Every Enzyme configuration skips the `broadcastindim_licm` pattern. It hoists
broadcasts that grow their operand out of the time loop, which turns each into a
full-size buffer carried through the loop. With enzyme-ad 0.0.16 at 10 steps
this costs 10.1 GB and 1.80 s per call, against 7.5 GB and 1.13 s without it.

Loop checkpointing is requested by tagging the time-stepping `stablehlo.while`
with `enzyme.enable_checkpointing`, `enzyme.checkpoint_period` and, for
binomial, `enzyme.binomial_checkpointing`.

Other switches read by `enzyme_4dvar_grad.py`: `JAX_CHECKPOINT=0` (JAX without
per-step checkpointing), `ENZYME_EXCLUDED_PASSES` (comma-separated
optimization patterns to drop, in addition to `broadcastindim_licm`), `COMPILE_ONLY=1` (print XLA's memory plan
without running), `IR_DUMP_DIR`, `ENZYME_IR_DUMP`, `JAX_PROFILE_DIR`.
