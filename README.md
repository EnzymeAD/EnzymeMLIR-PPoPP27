# Artifact for "EnzymeMLIR: Combining Domain-Specific Optimizations With Fast Differentiation" (PPoPP '27)

This repository contains the artifact for the paper:
> **"EnzymeMLIR: Combining Domain-Specific Optimizations With Fast Differentiation"**  
> *ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '27)*

Artifact DOI: [![DOI](https://zenodo.org/badge/1439939997.svg)](https://zenodo.org/badge/latestdoi/1439939997)

The paper introduces EnzymeMLIR, an MLIR-based compiler for automatic differentiation. By operating across different levels of abstraction, including structured/parallel control flow, tensor arithmetic,
and more, EnzymeMLIR combines with both existing and user-defined optimizations, resulting in superior performance. This workflow also enables the ability for custom differentiation-specific transformations
that otherwise would have been infeasible. This repository contains instructions and artifacts for the core evaluations of our paper. Notably, here we provide builds linking EnzymeMLIR into
C++ (via Reactant/Polygeist), Julia (via Reactant.jl) and Python (via JAX). We compare performance of both the impact of MLIR-level optimizations on differentiation, as well as overall performace against
existing tools. Overall we have five key benchmarks: LBM (CUDA C++), RSBench (CUDA C++), Oceananigans (Julia on CPU), JAXMD (JAX CUDA), and legoESM (JAX CUDA). Both runtime and memory usage are compared,
potentially across different sized inputs/iterations, and optimization configurations. The CUDA benchmarks were original run on an NVIDIA 5090 and the CPU benchmarks were run on an NVIDIA Grace CPU, though
similarly specified NVIDIA GPUs or X86 CPUs (from any vendor) should suffice, with potential hardware-specific performance differences. All software needed is available in a docker container built and
deployed within the artifact repository (though can also be built by end users, if they prefer).

---

## 1. Overview & Checked-In Reference Results

EnzymeMLIR is an automatic differentiation (AD) framework integrated directly into the MLIR compiler infrastructure. It generates forward- and reverse-mode derivatives while enabling AD to interleave with high-level domain-specific and AD-specific compiler optimizations (including tensor algebra transformations, polyhedral atomic strength reduction and pipelining, loop-aware caching, region-aware mincut tape minimization, and periodic/binomial checkpointing).

The repository includes the following reference measurements and figures for the AE paper under `results/` and `Oceananigans/data/vista_2026-11/`:

| Benchmark Suite | Paper Section / Figure / Table | Checked-In Reference Data & Figures | Plotting / Formatting Script |
| :--- | :--- | :--- | :--- |
| **Parboil LBM** (`lbm/cpp/`, `lbm/futhark/`) | Sec. 4.3, 5.1–5.2; Fig. 6; Tables 3–4 | `results/cpp_results.py`<br>`results/lbm_benchmark.pdf` | `lbm/cpp/plot_lbm_benchmark_results.jl` |
| **RSBench** (`rsbench/cpp/`, `rsbench/futhark/`) | Sec. 5.3 | `results/cpp_results.py` | `rsbench/cpp/run_timings.sh` |
| **Oceananigans.jl** (`Oceananigans/`) | Sec. 6.1; Table 2 | `Oceananigans/data/vista_2026-11/` | `Oceananigans/bench/make_table.jl` |
| **JAX-MD** (`jax/jaxmd/`) | Sec. 6.3; Fig. 9 | `results/results_JAXMD.json`<br>`results/plot-jaxmd.ipynb` | `results/plot-jaxmd.ipynb` |
| **legoESM MPAS-4DVar** (`jax/legoESM-MPAS-4DVar/`) | Sec. 6.2; Fig. 8 | `results/mpas4dvar_binomial_sweep.csv`<br>`results/mpas4dvar_binomial_sweep.pdf` | `jax/legoESM-MPAS-4DVar/plot_binomial_sweep.jl` |

---

## 2. Hardware and Software Dependencies

### A. GPU Benchmarks (`LBM`, `RSBench`, `JAX-MD`, and `legoESM MPAS-4DVar`)

We assume the target machine has a functional NVIDIA GPU driver installation with CUDA 12.9+ and the NVIDIA Container Toolkit.

| Component | Minimum | Recommended | Paper Evaluation Setup |
| :--- | :--- | :--- | :--- |
| **CPU** | 6 cores (x86_64) | 32 cores (x86_64) | 96-core AMD Ryzen Threadripper PRO 7995WX |
| **GPU** | NVIDIA GPU (CUDA 12.9+) | NVIDIA GPU with 24+ GiB VRAM (CUDA 12.9+) | NVIDIA GeForce RTX 5090 (32 GiB VRAM) |
| **RAM** | 32 GiB | 128 GiB | 512 GiB |
| **Storage** | 30 GiB | 60 GiB | NVMe SSD |
| **OS** | Linux (Ubuntu 22.04+) | Ubuntu 24.04 LTS | Ubuntu 24.04.4 LTS |

### B. CPU Benchmark (`Oceananigans.jl`)

| Component | Minimum | Recommended | Paper Evaluation Setup |
| :--- | :--- | :--- | :--- |
| **CPU** | 8 cores; x86_64 for the container, x86_64 or aarch64 natively | 32+ cores | NVIDIA Grace CPU Superchip (2× 72-core Arm Neoverse V2, TACC Vista `gg` node); each benchmark pinned to a single core |
| **GPU** | None (CPU execution) | None (CPU execution) | None (CPU-only node) |
| **RAM** | 32 GiB (`smoke` / `quick` presets) | 192 GiB+ (`full` preset up to $N=256$) | 237 GiB LPDDR5X |
| **Storage** | 30 GiB | 60 GiB | NVMe SSD |
| **OS** | Linux (container); Linux or macOS (native) | Linux | Linux (TACC Vista) |

The container image is built for x86_64. On aarch64 machines, including NVIDIA Grace (the paper's hardware) and Apple silicon, run the benchmark natively with Julia 1.12 instead, as described in [`Oceananigans/README.md`](Oceananigans/README.md). FLOP counts (Table 2) need Linux on NVIDIA Grace (Arm Neoverse V2) CPUs, so they can only be reproduced natively; runtime and memory can be measured on any machine.

---

## 3. Getting Started

### Step 0: Install NVIDIA Container Toolkit

The toolkit is needed only for the GPU benchmarks; for the CPU-only Oceananigans benchmark, skip this step.

On Ubuntu, the NVIDIA Container Toolkit can be installed as follows:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.20.1-1
sudo apt-get install -y \
   nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
   nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
   libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
   libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
For other Linux distributions, see [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Step 1: Pull (or Build) the Container Image

The prebuilt artifact container is hosted on GitHub Container Registry:
```bash
docker pull ghcr.io/enzymead/enzymemlir-ppopp27:latest
```
*(Alternatively, to build the image locally from `docker/Dockerfile`: `docker build -t ghcr.io/enzymead/enzymemlir-ppopp27:latest -f docker/Dockerfile .`)*

The container includes:
- **Clang 18 + MLIR** (`/home/evaluator/llvm-project/build/bin/clang++`)
- **Enzyme LLVM Plugin** (`/home/evaluator/Enzyme/build/Enzyme/ClangEnzyme-24.so`)
- **EnzymeMLIR C++ Frontend & Raising Plugins** (`/home/evaluator/Reactant/build/Enzyme/libClangReactant-24.so`, `/home/evaluator/Enzyme-JAX/bazel-bin/libRaise.so`)
- **JAX 0.10.0 (CUDA 12) + Enzyme-JAX (`enzyme-ad`)** installed in `/home/evaluator/eval-env`
- **NVIDIA Nsight Systems** (`/usr/local/cuda-12.9/bin/nsys`, used for device memory profiling in LBM)
- **Julia 1.12.7** (`/usr/local/bin/julia`) with precompiled packages in `/opt/julia_project` and `/opt/julia_depot`

### Step 2: Launch the Container & Verify the Installation (~5 minutes)

From the root of the artifact directory (if cloning from Git rather than extracting the artifact `.zip`, ensure submodules are initialized via `git submodule update --init --recursive`), launch the container interactively with GPU access:
```bash
docker run -it --rm --gpus all \
  -v "$(pwd)":/workspaces/EnzymeMLIR-PPoPP27 \
  -w /workspaces/EnzymeMLIR-PPoPP27 \
  ghcr.io/enzymead/enzymemlir-ppopp27:latest bash
```

For the CPU-only Oceananigans benchmark, no GPU or NVIDIA Container Toolkit is needed: launch the same container without `--gpus all` (on an x86_64 machine):
```bash
docker run -it --rm \
  -v "$(pwd)":/workspaces/EnzymeMLIR-PPoPP27 \
  -w /workspaces/EnzymeMLIR-PPoPP27 \
  ghcr.io/enzymead/enzymemlir-ppopp27:latest bash
```

Inside the container, verify all toolchains:
```bash
# 1. Verify Clang & MLIR build
/home/evaluator/llvm-project/build/bin/clang++ --version

# Expected output:
#   clang version 24.0.0git
#   Target: x86_64-unknown-linux-gnu
#   Thread model: posix
#   InstalledDir: /home/evaluator/llvm-project/build/bin

# 2. Verify Futhark compiler
./futhark-nightly-linux-x86_64/bin/futhark --version

# Expected output:
#   Futhark 0.29.0 (prerelease - include info below when reporting bugs).
#   git: 5eb30afc3b28c9acc6c344753483152ffcc87adc
#   ...

# 3. Verify Julia environment
julia --project=/opt/julia_project -e 'using InteractiveUtils; versioninfo()'

# Expected output:
#   Julia Version 1.12.7
#   Commit 6d172b025e4 (2026-08-15 08:05 UTC)
#   ...

# 4. Verify JAX & GPU availability
/home/evaluator/eval-env/bin/python3 -c "import jax; print('JAX devices:', jax.devices())"

# Expected output:
#   JAX devices: [CudaDevice(id=0)]
```
*(Check 4 needs a GPU; on a CPU-only machine, skip it. Check 3 is the one the Oceananigans benchmark needs.)*

---

## 4. Step-by-Step Instructions

All commands in this section are executed **inside the Docker container** starting from `/workspaces/EnzymeMLIR-PPoPP27`.

### A. C++ and Futhark GPU Benchmarks: RSBench (`rsbench/`) and LBM (`lbm/`)

Reference 12-repetition raw runtime arrays are checked into `results/cpp_results.py`, and the LBM runtime and peak memory scaling plot (Figure 6; data in Tables 3–4) is checked into `results/lbm_benchmark.pdf`.

You can run each benchmark suite via its top-level runner script:
```bash
bash lbm/run_benchmarks.sh
bash rsbench/run_benchmarks.sh
```
These scripts detect the GPU compute capability (`SM_VERSION`), build and benchmark both C++ (EnzymeMLIR and EnzymeLLVM) and Futhark (CUDA) implementations, and write summary tables to `lbm/{cpp,futhark}/timings/` and `rsbench/{cpp,futhark}/timings/`.

Alternatively, run each implementation individually:

#### 1. RSBench (`rsbench/cpp/` and `rsbench/futhark/`, Section 5.3)
```bash
# C++ (EnzymeMLIR and EnzymeLLVM, 10,200 lookups, 12 repetitions):
cd /workspaces/EnzymeMLIR-PPoPP27/rsbench/cpp
./run_timings.sh 10200 12
cat timings/summary_10200.txt

# Futhark (CUDA reverse-mode AD on data/smaller.in.gz, 12 repetitions):
cd /workspaces/EnzymeMLIR-PPoPP27/rsbench/futhark
../../futhark-nightly-linux-x86_64/bin/futhark bench --backend=cuda -r 12 rsbench.fut
```

#### 2. Parboil LBM (`lbm/cpp/` and `lbm/futhark/`, Sections 4.3 & 5.1–5.2, Figure 6, Tables 3–4)
```bash
# C++ at 100 steps (EnzymeMLIR, EnzymeLLVM, Binomial b in {4, 16} with/without mincut,
# and EnzymeMLIR with atomics; 12 repetitions + Nsight Systems peak device memory):
cd /workspaces/EnzymeMLIR-PPoPP27/lbm/cpp
./run_timings.sh 100 12
cat timings/summary_100.txt

# Full C++ step-count sweep (25, 50, 100, 200, 400, 800, 1200 steps):
make run

# Regenerate Figure 6 into lbm/cpp/timings/lbm_benchmark.pdf from summary_*.txt:
julia plot_lbm_benchmark_results.jl

# Futhark (CUDA reverse-mode AD, 100 steps on data/120_120_150_ldc.in.gz, 12 repetitions):
cd /workspaces/EnzymeMLIR-PPoPP27/lbm/futhark
../../futhark-nightly-linux-x86_64/bin/futhark bench --backend=cuda -r 12 lbm.fut
```

**Ablation tests**.
Section 4.3 describes an ablation study done on LBM comparing variants of EnzymeMLIR with and without polyhedral atomic removal. This speedup can be verified by comparing the LBM runtimes of variants `EnzymeMLIR (mincut)` and `EnzymeMLIR (with atomics)`. The sole difference between the two is that the former includes atomic removal while the latter does not, leading to approximately a 1.8x speedup from removing atomics.

Section 5.3 describes an ablation of RSBench that compares the impact of a loop-invariant cache optimization on both EnzymeMLIR and EnzymeLLVM. Variants `Enzyme{MLIR,LLVM}` include this optimization while variants `Enzyme{MLIR,LLVM} (no lic)` do not. `EnzymeMLIR` should show approximately a 60x speedup over `EnzymeMLIR (no lic)`, while `EnzymeLLVM` should show a 2.2x speedup over `EnzymeLLVM (no lic)`.

---

### B. Julia Oceananigans Benchmarks (`Oceananigans/`, Section 6.1, Table 2)

Reference measurements collected on TACC Vista (Grace CPU) are checked into `Oceananigans/data/vista_2026-11/` (including `table.tex` and `table.csv`).

The commands below run **inside the container** started in Step 2 (with or without `--gpus all`); `/opt/julia_project` exists only there. To run natively instead, on any machine with Julia 1.12 (including aarch64 machines such as NVIDIA Grace and Apple silicon, and required for FLOP counts), follow [`Oceananigans/README.md`](Oceananigans/README.md): it uses the same commands with `--project=provenance/julia_env`, the exact package environment of the reference runs.

#### 1. Formatting Table 2 from Checked-In Data (Inside Container)
```bash
cd /workspaces/EnzymeMLIR-PPoPP27/Oceananigans
julia --project=/opt/julia_project bench/make_table.jl --out data/vista_2026-11 --ci --flops --delta
```

#### 2. Running the Benchmark Suite Inside the Container
```bash
cd /workspaces/EnzymeMLIR-PPoPP27/Oceananigans

# 1. Forward consistency check (CPU):
REACTANT_BACKEND=cpu julia --project=/opt/julia_project bench/forward_consistency.jl 1 --reactant-init

# 2. Smoke test preset (reactant_full + enzyme_naive at N = 4, 2 measured repeats):
julia --project=/opt/julia_project bench/run_cases.jl \
  --out results/smoke --configs reactant_full,enzyme_naive --nsteps 4 \
  --timeout 3600 --core 0 --perf off --skip-existing -- --save-gradients --repeats 2
julia --project=/opt/julia_project bench/make_table.jl --out results/smoke --ci --delta
julia --project=/opt/julia_project bench/check_gradients.jl --out results/smoke

# 3. Quick preset (all 9 Table 2 configs at N = 4, 8):
julia --project=/opt/julia_project bench/run_cases.jl \
  --out results/quick --configs table --nsteps 4,8 \
  --timeout 14400 --core 0 --perf off --skip-existing -- --save-gradients
julia --project=/opt/julia_project bench/make_table.jl --out results/quick --ci --delta

# 4. Full Table 2 sweep (all 9 configs across N = 4, 8, 16, 32, 64, 128, 256):
julia --project=/opt/julia_project bench/run_cases.jl \
  --out results/full --configs table --nsteps 4,8,16,32,64,128,256 \
  --timeout 172800 --core 0 --perf off --skip-existing -- --save-gradients
julia --project=/opt/julia_project bench/make_table.jl --out results/full --ci --delta
```
*(Note: `Oceananigans/run.sh` is an alternative **host-side** wrapper script that invokes `docker run` itself. If running directly from the host shell outside Docker, you can run `cd Oceananigans && ./run.sh run smoke|quick|full` and `./run.sh report results/<preset>`.)*

---

### C. JAX GPU Workloads: JAX-MD and legoESM MPAS-4DVar (`jax/`)

#### 1. JAX-MD Molecular Dynamics Simulation (`jax/jaxmd/`, Section 6.3, Figure 9)

Reference timings and peak device memory across optimization pipelines (`Jax`, `HLOOpt`, `DefOpt`, `IDefOpt`) are checked into `results/results_JAXMD.json`, and the plotting notebook is in `results/plot-jaxmd.ipynb`.

To run the benchmark inside the container (8,000 Lennard-Jones particles, `jaxmd20`, under a Nose-Hoover thermostat):
```bash
cd /workspaces/EnzymeMLIR-PPoPP27
bash jax/jaxmd/run_benchmarks.sh
```
Or invoke `main.py` directly:
```bash
cd /workspaces/EnzymeMLIR-PPoPP27/jax/jaxmd
TEST_UNDECLARED_OUTPUTS_DIR=../../results /home/evaluator/eval-env/bin/python3 main.py --warmup=2 --repeat=12
```
Open `results/plot-jaxmd.ipynb` in Jupyter to regenerate Figure 9. Set `source` in its input-selection cell to `reference` for `results/results_JAXMD.json` or `new` for freshly collected `results/jaxmd/results_JAXMD.json`. See [plotting data selection](docs/plotting-data.md) for command-line execution and output paths.

#### 2. legoESM MPAS-4DVar Atmospheric Data Assimilation (`jax/legoESM-MPAS-4DVar/`, Section 6.2, Figure 8)

Computes the value and gradient of the single-observation 4D-Var cost on the legoESM MPAS primitive-equation model (icosahedral mesh, 40,962 cells, 40 levels, 6,594,882 control variables). Reference binomial checkpointing sweep measurements and the resulting plot are checked into `results/mpas4dvar_binomial_sweep.csv` and `results/mpas4dvar_binomial_sweep.pdf`.

**Input Data (`jax/legoESM-MPAS-4DVar/data/`):**
- **Included in repository:** `data/mpas_gen_be.{npz,json}` and `data/mpas_gen_be_manifest.json` (background error statistics).
- **Background state (`data/mpas_nmc_case_009_20170115000000.npz`, 151 MB):**
  - *Option 1 (Generate from ERA5):* Fetch the ERA5 pressure-level files for `2017-01-13` and `2017-01-14` and the January 2017 surface files from the NCAR RDA `ds633.0` OSDF mirror (`https://osdf-director.osg-htc.org/ncar/gdex/d633000/`), then run:
    ```bash
    cd /workspaces/EnzymeMLIR-PPoPP27/jax/legoESM-MPAS-4DVar/legoESM
    LEGOESM_ERA5_RDA_ROOT=<era5 dir> PYTHONPATH=. /home/evaluator/eval-env/bin/python3 \
      scripts/data/generate_mpas_nmc.py --config config/4DVar_single/nmc.yaml --case-offset 8 --count 1
    cp outputs/mpas_nmc_r6_l40/samples/mpas_nmc_case_009_20170115000000.npz ../data/
    ```
  - *Option 2 (Automatic synthetic state via `run_benchmarks.sh`):* If `data/mpas_nmc_case_009_20170115000000.npz` is not present, `jax/legoESM-MPAS-4DVar/run_benchmarks.sh` automatically initializes a balanced baroclinic-wave MPAS state on the same 40,962-cell, 40-level mesh so compilation, runtime, peak GPU memory, and Enzyme-vs-JAX gradient equivalence can be evaluated without downloading ERA5.

**Running Inside the Container:**
```bash
cd /workspaces/EnzymeMLIR-PPoPP27

# 1. 4D-Var gradient and peak memory comparison (10 and 60 steps; writes results/mpas_4dvar/):
MPAS_MODE=comparison bash jax/legoESM-MPAS-4DVar/run_benchmarks.sh

# 2. Binomial checkpointing sweep across 10–240 steps and budgets b in {2, 4, 8, 16, n}
#    (writes results/binomial_sweep/summary.csv):
MPAS_MODE=binomial_sweep bash jax/legoESM-MPAS-4DVar/run_benchmarks.sh

# 3. Plot the newly collected sweep (Figure 8) into results/plots/mpas/new/:
PLOT_SOURCE=new julia jax/legoESM-MPAS-4DVar/plot_binomial_sweep.jl

# Plot checked-in reference data into results/plots/mpas/reference/:
PLOT_SOURCE=reference julia jax/legoESM-MPAS-4DVar/plot_binomial_sweep.jl
```
See `jax/legoESM-MPAS-4DVar/README.md` for additional options and standalone host setup via `uv` (`./setup_env.sh`).

---

## 5. Repository Structure

```
EnzymeMLIR-PPoPP27/
├── docker/
│   ├── Dockerfile                       # Unified container specification (Julia 1.12.7 + LLVM 18 + Enzyme + JAX 0.10.0 + Nsight)
│   ├── Project.toml                     # Precompiled Julia dependency specifications
│   └── requirements.txt                 # Python dependencies (JAX 0.10.0 CUDA 12, Jupyter, SciPy)
├── lbm/
│   ├── cpp/                             # Parboil LBM CFD C++/CUDA benchmark suite
│   │   ├── Makefile                     # Build rules and step-count sweep (`make run`)
│   │   ├── run_timings.sh               # LBM execution, timing, and Nsight Systems memory profiler
│   │   ├── peak_mem.py                  # SQLite Nsight Systems peak device allocation parser
│   │   └── plot_lbm_benchmark_results.jl # Pluto notebook generating results/lbm_benchmark.pdf
│   ├── futhark/                         # Futhark LBM reverse-mode AD benchmark (`lbm.fut`)
│   └── run_benchmarks.sh                # Orchestrated LBM C++ & Futhark runner
├── rsbench/
│   ├── cpp/                             # RSBench C++/CUDA benchmark suite
│   │   ├── Makefile                     # Build rules for EnzymeMLIR and EnzymeLLVM
│   │   └── run_timings.sh               # RSBench C++ execution and timing script
│   ├── futhark/                         # Futhark RSBench reverse-mode AD benchmark (`rsbench.fut`)
│   └── run_benchmarks.sh                # Orchestrated RSBench C++ & Futhark runner
├── futhark-nightly-linux-x86_64/        # Prebuilt Futhark compiler distribution (`bin/futhark`)
├── Oceananigans/
│   ├── bench/
│   │   ├── forward_consistency.jl       # Numerical forward-pass consistency test
│   │   ├── driver_enzyme.jl             # EnzymeLLVM benchmark driver
│   │   ├── driver_reactant.jl           # EnzymeMLIR / Reactant.jl benchmark driver
│   │   ├── run_cases.jl                 # In-container benchmark preset runner
│   │   ├── check_gradients.jl           # Cross-configuration gradient agreement verifier
│   │   └── make_table.jl                # Formats Table 2 (Markdown, LaTeX, CSV) from JSON results
│   ├── data/vista_2026-11/              # Pre-recorded Table 2 measurements from Grace CPUs at TACC
│   └── run.sh                           # Host-side Docker wrapper for Table 2 sweeps and reporting
├── jax/
│   ├── jaxmd/                           # JAX-MD Lennard-Jones molecular dynamics
│   │   ├── main.py                      # JAX-MD benchmark script
│   │   ├── test_utils.py                # Enzyme-JAX pass pipeline harness and JSON exporter
│   │   ├── xprof_utils.py               # XProf step-time and peak device memory profiler
│   │   └── run_benchmarks.sh            # Orchestrated JAX-MD runner
│   └── legoESM-MPAS-4DVar/              # MPAS atmospheric 4D-Var data assimilation
│       ├── README.md                    # Detailed MPAS 4D-Var setup, ERA5 data generation, and flags
│       ├── setup_env.sh                 # Standalone host uv setup (.venv with legoESM, JAX 0.10.0, Enzyme-JAX)
│       ├── enzyme_4dvar_grad.py         # 4D-Var value-and-gradient driver (JAX vs EnzymeMLIR)
│       ├── run_comparison.sh            # 4D-Var gradient and peak memory comparison
│       ├── run_binomial_sweep.sh        # Step-count and binomial budget sweep
│       ├── plot_binomial_sweep.jl       # Plot reference or new sweep data into results/plots/mpas/
│       └── run_benchmarks.sh            # Orchestrated legoESM MPAS-4DVar runner
└── results/
    ├── cpp_results.py                   # Recorded 12-run raw timings for LBM and RSBench
    ├── lbm_benchmark.pdf                # LBM runtime and peak memory scaling figure (Figure 6)
    ├── results_JAXMD.json               # Recorded JAX-MD timings and peak memory across pass pipelines
    ├── plot-jaxmd.ipynb                 # Notebook plotting JAX-MD runtime and memory usage (Figure 9)
    ├── mpas4dvar_binomial_sweep.csv     # Recorded legoESM MPAS 4D-Var binomial sweep data
    └── mpas4dvar_binomial_sweep.pdf     # legoESM MPAS 4D-Var runtime and peak memory figure (Figure 8)
