# Oceananigans.jl CPU AD benchmark

This folder measures the cost of one reverse-mode gradient of an Oceananigans.jl
ocean model on a CPU, for the paper's CPU AD cost table (Table 2). It holds the
benchmark code, the reference measurements from TACC Vista, and the record of how
they were produced.

- **Model:** a reentrant channel (80 × 160 × 32 cells, 1000 km × 2000 km, stretched
  vertical grid, immersed-boundary bottom topography, Float64), stepped N times.
- **Objective:** the zonal transport through the mid-channel section after N steps,
  in Sv.
- **Gradient:** with respect to the initial temperature and salinity, the surface
  wind stress (both components), the surface temperature flux and the layer
  thicknesses used in the objective.
- **N:** 4, 8, 16, 32, 64, 128 and 256 time steps.

## Configurations

The 9 table rows are defined in [`bench/configs.toml`](bench/configs.toml):

| Config | Table row | What it is |
|---|---|---|
| `reactant_full` | Fully optimized | Reactant (EnzymeMLIR), default optimization pipeline, mincut on, no checkpointing |
| `reactant_after_enzyme` | Enzyme-only optimization | as above, but optimization passes only after AD |
| `reactant_mincut_off` | Fully optimized, mincut disabled | `reactant_full` without mincut |
| `reactant_periodic` | Fully optimized, periodic checkpointing | periodic checkpointing every √N steps |
| `reactant_periodic_mincut_off` | … periodic checkpointing, mincut disabled | as above, without mincut |
| `reactant_binomial_10` | … binomial checkpointing (b = 10) | binomial checkpointing with 10 snapshots |
| `reactant_binomial_log2` | … binomial checkpointing (b = log₂ N) | binomial checkpointing with log₂ N snapshots |
| `enzyme_naive` | AD only, no prior optimization | Enzyme (LLVM) on plain Julia, no checkpointing |
| `enzyme_revolve_log2` | AD only, binomial checkpointing (b = log₂ N) | Enzyme with Checkpointing.jl Revolve, log₂ N snapshots |

`configs.toml` also defines `enzyme_periodic_isqrt`, which is not in the table.

## Layout

| Path | Contents |
|---|---|
| [`src/`](src/) | The model (`ChannelModel.jl`) and the measurement code (`Measure.jl`) |
| [`bench/`](bench/) | Drivers (`driver_reactant.jl`, `driver_enzyme.jl`), the orchestrator (`run_cases.jl`), checks (`forward_consistency.jl`, `check_gradients.jl`, `perf_gate_test.jl`), and table and export scripts (`make_table.jl`, `merge_batches.jl`, `export_raw.jl`) |
| [`run.sh`](run.sh) | Runs the benchmarks inside the repository's container image, and builds the table from the results |
| [`data/vista_2026-11/`](data/vista_2026-11/) | The reference measurements behind the paper's table (see its README) |
| [`provenance/`](provenance/) | How the reference data was produced: the SLURM scripts (`vista/`) and the exact package environment (`julia_env/`, also used to run the benchmarks natively) |
| `Project.toml` | The package environment (a link to `../docker/Project.toml`, which the container image installs) |

## Running the benchmarks

There are two ways to run the benchmarks, with the same scripts: natively, on any
machine with Julia 1.12 (Linux, including NVIDIA Grace, or macOS), or in the
repository's container image. Run every command from this folder. Results go to
`results/<name>/`, which git ignores.

### Natively

Install Julia 1.12 (for example with [juliaup](https://github.com/JuliaLang/juliaup)),
then:

```bash
# 1. Install the exact package environment of the reference runs (provenance/julia_env)
julia --project=provenance/julia_env -e 'using Pkg; Pkg.instantiate()'

# 2. Check that the Reactant-compiled model matches the plain-CPU model
REACTANT_BACKEND=cpu julia --project=provenance/julia_env bench/forward_consistency.jl 1 --reactant-init

# 3. Smoke test: reactant_full and enzyme_naive at N = 4, 2 measured calls each
julia --project=provenance/julia_env bench/run_cases.jl --out results/smoke \
    --configs reactant_full,enzyme_naive --nsteps 4 --timeout 3600 --core 0 \
    --perf off --skip-existing -- --save-gradients --repeats 2
julia --project=provenance/julia_env bench/make_table.jl --out results/smoke --ci --delta
julia --project=provenance/julia_env bench/check_gradients.jl --out results/smoke
```

On an Apple M1 Max, starting from an empty Julia depot, step 1 takes about 5 min,
step 2 about 4 min and step 3 about 7 min. Step 2 prints the objective on both
backends (relative difference about 1e-12) and confirms that salinity is conserved;
`check_gradients.jl` ends with "All comparisons passed".

For the rest of the table, change the configurations, N values and timeout (seconds
per case) of step 3:

- **All 9 configurations at N = 4 and 8:** `--out results/quick --configs table
  --nsteps 4,8 --timeout 14400`, without `--repeats 2` (the default is 12 measured
  calls per case).
- **The whole table:** `--out results/full --configs table --nsteps
  4,8,16,32,64,128,256 --timeout 172800`. On one Grace core this takes about 80 h,
  and the largest cases need up to about 175 GiB of memory; a case that exceeds the
  machine's memory is recorded as out of memory (`m`), as in the paper.

`--skip-existing` skips cases that already finished, so an interrupted run can be
resumed with the same command. On Linux, the benchmark process is pinned to the core
given by `--core` when `numactl` is installed.

**FLOPs** are counted only on Linux with NVIDIA Grace (Arm Neoverse V2) CPUs, with
`perf` installed and `perf_event_paranoid` at 2 or lower: use `--perf on` instead of
`--perf off`, and `--flops` for `make_table.jl`. Runtime and memory can be measured
on any machine.

### In the container

The repository's container image (`ghcr.io/enzymead/enzymemlir-ppopp27:latest`,
built for x86_64 Linux) has Julia 1.12 and the same direct package pins, installed in
`/opt/julia_project`. Inside the container, use the commands above with
`--project=/opt/julia_project` in place of `--project=provenance/julia_env`, and skip
step 1; the [repository README](../README.md) shows how to start the container. From
the host, [`run.sh`](run.sh) runs the same presets in the container for you:

```bash
./run.sh run smoke          # reactant_full + enzyme_naive at N = 4 (checks the setup)
./run.sh run quick          # all 9 configs at N = 4 and 8
./run.sh run full           # the whole table, N = 4 ... 256
./run.sh report results/quick
```

`report` builds `table.tex` and `table.csv`, checks that all configurations compute
the same gradient (`bench/check_gradients.jl`) and exports the raw data. FLOP counts
need `FLOPS=1` on a Grace host. `./run.sh` with no arguments prints all options.

### Rebuilding the paper's table from the reference data

```bash
julia --project=provenance/julia_env bench/make_table.jl --out data/vista_2026-11 --ci --flops --delta
```

(`--project=/opt/julia_project` in the container.) This rewrites
`data/vista_2026-11/table.tex` and `table.csv` from the committed JSON files;
`table.csv` comes out identical.

## How the reference data was produced

- **Hardware:** TACC Vista `gg` nodes (2× NVIDIA Grace, Arm Neoverse V2, 144 cores,
  237 GiB). The benchmark process is pinned to a single core (`numactl -C 0`).
- **Software:** a frozen container image
  (`sha256:bc5ed0f2d08f3c234360fbd31ff66924ddba22d9231a9fe33c8187412b688d6b`, built
  from commit `3f6c63d` of the original benchmark repository) with Julia 1.12.7,
  Oceananigans 0.116.1, Reactant 0.2.308 (Reactant_jll 0.0.443), Enzyme 0.13.219 and
  Checkpointing.jl 0.13.0, run with `julia -O0` and
  `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_backend_extra_options=xla_cpu_use_multi_output_fusion"`.
  That image is not archived with this repository. Its exact package environment
  (`Project.toml` and `Manifest.toml`) is in
  [`provenance/julia_env/`](provenance/julia_env/); the repository's own image
  (`docker/`) has the same direct package pins, and instantiating that Manifest
  reproduces the full environment.
- **Benchmark code:** the image's `bench/driver_reactant.jl` compiled with Reactant's
  default passes. The driver in this repository additionally sets
  `disable_dot_formation_passes = true` (commit 4fb71d0, #27), which was added after
  these runs; that commit reports it as neutral on the gradient. The orchestrator
  `bench/run_cases.jl` now starts each case in the orchestrator's own Julia
  environment (it used the `Oceananigans/` folder's), which changes nothing that is
  measured. Everything else in `bench/` and `src/` is the code that produced the
  data, apart from comments.
- **Protocol: batches.** Each case (configuration × N) ran as 3 independent
  *batches*: separate SLURM jobs, usually on different nodes, each a fresh process
  with its own compilation, 1 warmup call (excluded from the statistics) and 4
  measured calls, each on a freshly built problem. The 12 measured calls of the 3
  batches are pooled, and the table reports mean ± 95% Student-t confidence interval
  over the pool.
  - In all, 183 batch jobs ran on 46 different nodes.
  - Because the batches ran on different nodes, the confidence intervals include
    node-to-node variation. This is intended: it reflects what a user of the
    machine would see.
  - A case whose first batch ran out of memory got no further batches.
- **Metrics:**
  - **Runtime:** wall-clock time of one gradient call.
  - **Memory:** peak resident memory during the call (Linux `VmHWM`, reset before
    each call).
  - **FLOPs:** the Neoverse V2 floating-point hardware counters `r80C0` + `r80C1`,
    counted only inside each measured call.

The full methodology, including every field of the exported CSV files, is in
[`data/vista_2026-11/export/DATA_README.md`](data/vista_2026-11/export/DATA_README.md).
The SLURM scripts that ran the sweep are kept as a record in
[`provenance/vista/`](provenance/vista/).

## Results at a glance

- **Reactant is faster than Enzyme alone.** `reactant_full` takes 0.67–0.71× the
  time of `enzyme_naive` per call, and executes fewer floating-point operations
  (481 against 601 GFLOP at N = 128).
- **With checkpointing,** `reactant_binomial_log2` is 4–6× faster than
  `enzyme_revolve_log2` for N ≥ 8 (204 s against 1095 s at N = 128), at 1.4–1.5×
  the memory (8.7 against 6.0 GiB).
- **Memory:** binomial checkpointing keeps Reactant's peak memory nearly flat in N
  (8.4–8.8 GiB for N = 8 to 256), while `enzyme_naive` grows to 139 GiB at N = 128
  and runs out of memory at N = 256.

The full table is [`data/vista_2026-11/table.tex`](data/vista_2026-11/table.tex)
(and `table.csv`).

## Correctness

- **Within each case,** every measured call of every batch returned the identical
  objective, and the gradient norms of every field are identical across batches.
- **Across configurations,** every configuration agrees with `reactant_full` at
  every N: Reactant configurations within 1.5e-13, Enzyme-only configurations within
  7.8e-13 (relative; objective and gradient norms).
- **Reactant against plain CPU:** [`bench/forward_consistency.jl`](bench/forward_consistency.jl)
  checks that the Reactant-compiled forward model matches the plain-CPU model
  (objective to about 1e-12).
