# Artifact for "EnzymeMLIR: Combining Domain-Specific Optimizations With Fast Differentiation" (PPoPP '27)

This repository contains the complete artifact for the paper:
> **"EnzymeMLIR: Combining Domain-Specific Optimizations With Fast Differentiation"**  
> *ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '27)*

---

## 1. Overview & Rebuttal Results

EnzymeMLIR is an automatic differentiation (AD) framework integrated directly into the MLIR compiler infrastructure. It generates forward- and reverse-mode derivatives while enabling AD to interleave with high-level domain-specific and AD-specific compiler optimizations (including tensor algebra transformations, loop-aware caching, region-aware mincut tape minimization, and binomial checkpointing).

### Summary Table (Including Author Rebuttal Results)

| Benchmark | Enzyme MLIR (s) | Enzyme LLVM (s) | Futhark (s) | JAX (s) | Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RSBench** | **0.1160 s** | 2.922 s | 0.0322 s | n/a | GPU (NVIDIA) |
| **LBM** | **0.0939 s** | 0.2344 s | 0.1065 s | n/a | GPU (NVIDIA) |
| **Oceananigans** | **Table 2, except last row** | Last row of Table 2 | n/a | n/a | CPU |
| **legoESM** *(JAX port of Oceananigans)* | **6.6 s** *(6.9 GB peak)* | n/a | n/a | 7.0 s *(16.5 GB peak)* | GPU (NVIDIA) |
| **JAX-MD** | **0.00080359 s** | n/a | n/a | 0.00271422 s | GPU (NVIDIA) |

---

### Table 2 from Paper: Oceananigans.jl Differentiation on CPU (Grace CPUs at TACC)

Below are the complete runtime (seconds) and peak memory (GiB) measurements reported in Table 2 (recorded in `Oceananigans/data/vista_2026-11/`):

| Configuration | Metric | $N = 4$ | $N = 8$ | $N = 16$ | $N = 32$ | $N = 64$ | $N = 128$ | $N = 256$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fully optimized (EnzymeMLIR)** | Runtime (s)<br>Memory (GiB) | 11.12<br>13.80 | 21.27<br>7.01 | 43.87<br>7.60 | 86.51<br>8.68 | 171.72<br>11.42 | 345.43<br>15.92 | **700.57**<br>**27.57** |
| **Enzyme-only optimization** | Runtime (s)<br>Memory (GiB) | 13.56<br>15.13 | 23.30<br>7.93 | 46.42<br>8.92 | 93.95<br>10.93 | 182.82<br>15.67 | 377.80<br>21.71 | 745.52<br>41.53 |
| **Fully optimized, mincut disabled** | Runtime (s)<br>Memory (GiB) | 10.84<br>13.51 | 34.98<br>29.50 | 76.16<br>53.20 | 170.27<br>100.99 | —m<br>—m | —m<br>—m | —m<br>—m |
| **Fully optimized, periodic checkpointing** | Runtime (s)<br>Memory (GiB) | 13.52<br>17.13 | 26.59<br>8.76 | 52.87<br>9.63 | 296.60<br>24.90 | 575.24<br>36.81 | —t<br>—t | —t<br>—t |
| **Fully optimized, periodic checkpointing, mincut disabled** | Runtime (s)<br>Memory (GiB) | 10.98<br>13.78 | 40.79<br>30.38 | 75.93<br>54.15 | —t<br>—t | 887.56<br>47.31 | —t<br>—t | —t<br>—t |
| **Fully optimized, binomial checkpointing ($b=10$)** | Runtime (s)<br>Memory (GiB) | 13.35<br>17.01 | 24.23<br>8.93 | 50.97<br>9.21 | 103.72<br>9.86 | 209.44<br>11.19 | 428.42<br>13.87 | 816.90<br>19.25 |
| **Fully optimized, binomial checkpointing ($b=\log N$)** | Runtime (s)<br>Memory (GiB) | 13.32<br>17.03 | 25.83<br>8.50 | 52.31<br>8.91 | 108.14<br>9.51 | 215.94<br>10.97 | 453.61<br>11.78 | 930.37<br>19.33 |
| **AD only, no prior opt, Checkpointing.jl (Enzyme LLVM)** | Runtime (s)<br>Memory (GiB) | 9.26<br>14.56 | —s<br>—s | 36.44<br>34.63 | 76.11<br>61.37 | 183.95<br>114.92 | 477.75<br>222.02 | **—m**<br>**—m** |

*Note: —m indicates out-of-memory, —s indicates segmentation fault, —t indicates timeout.*

---

## 2. Hardware and Software Dependencies

| Component | Minimum | Recommended | Paper Evaluation Setup |
| :--- | :--- | :--- | :--- |
| **CPU** | 8 cores x86_64 / aarch64 | 32+ cores | AMD Ryzen Threadripper PRO 7995WX (96-core) / Grace CPU |
| **GPU** | NVIDIA GPU (CUDA 12.0+) | NVIDIA GPU with 24+ GiB VRAM | NVIDIA RTX 5090 (32 GiB) / NVIDIA A100 (40/80 GiB) / GH100 |
| **RAM** | 32 GiB | 128 GiB+ | 512 GiB RAM |
| **Storage** | 30 GiB | 60 GiB | NVMe SSD |
| **OS** | Linux (Ubuntu 22.04+) | Ubuntu 24.04 LTS | Ubuntu 24.04 LTS |

---

## 3. Getting Started

### Step 0: NVIDIA Container Toolkit Installation

Ensure Docker and the NVIDIA Container Toolkit are installed:
```bash
# Verify nvidia-smi
nvidia-smi

# Verify docker nvidia runtime
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

### Step 1: Pull the Prebuilt Container

The complete artifact environment is prebuilt and hosted on GitHub Container Registry:
```bash
docker pull ghcr.io/enzymead/enzymemlir-ppopp27:latest
```

The container includes:
- **Clang 18 + MLIR** (`/home/evaluator/llvm-project/build/bin/clang++`)
- **Enzyme LLVM** (`/home/evaluator/Enzyme/build/Enzyme/ClangEnzyme-24.so`)
- **Reactant MLIR Plugin & Raising Library** (`/home/evaluator/Reactant/build/Enzyme/libClangReactant-24.so`, `/home/evaluator/Enzyme-JAX/bazel-bin/libRaise.so`)
- **JAX 0.10.0 (CUDA 12) + Enzyme-JAX (`enzyme-ad`)** installed in `/home/evaluator/eval-env`
- **NVIDIA Nsight Systems** (`/usr/local/cuda-12.9/bin/nsys`, used for device memory profiling in LBM)
- **Julia 1.12.7** (`/usr/local/bin/julia`) with precompiled packages in `/opt/julia_project` and `/opt/julia_depot`

### Step 2: Launch the Container & Smoke Test (~5 minutes)

Clone the repository (including submodules) and launch the container interactively with GPU access:
```bash
git clone --recurse-submodules https://github.com/EnzymeAD/EnzymeMLIR-PPoPP27.git
cd EnzymeMLIR-PPoPP27

docker run -it --rm --gpus all \
  -v "$(pwd)":/workspaces/EnzymeMLIR-PPoPP27 \
  -w /workspaces/EnzymeMLIR-PPoPP27 \
  ghcr.io/enzymead/enzymemlir-ppopp27:latest bash
```

Inside the container, run the quick verification commands:
```bash
# 1. Verify Clang & Enzyme
/home/evaluator/llvm-project/build/bin/clang++ --version

# 2. Verify Futhark compiler
./futhark-nightly-linux-x86_64/bin/futhark --version

# 3. Verify Julia & Oceananigans environment
julia --project=/opt/julia_project -e 'using InteractiveUtils; versioninfo()'

# 4. Verify JAX & GPU availability
/home/evaluator/eval-env/bin/python3 -c "import jax; print('JAX devices:', jax.devices())"
```

---

## 4. Step-by-Step Instructions

All reproduction commands below can be executed inside the Docker container from `/workspaces/EnzymeMLIR-PPoPP27`.

### A. C++ and Futhark Benchmarks: RSBench (`rsbench/`) and LBM (`lbm/`)

Recorded 12-repetition raw runtime arrays are stored in `results/cpp_results.py`, and the LBM scaling and memory sweep figure is in `results/lbm_benchmark.pdf`.

We also provide automated per-benchmark orchestrator scripts:
```bash
bash lbm/run_benchmarks.sh
bash rsbench/run_benchmarks.sh
```
These scripts detect the GPU compute architecture (`SM_VERSION`), build and benchmark both C++ (EnzymeMLIR and EnzymeLLVM) and Futhark (CUDA) implementations, and write summary tables to `lbm/{cpp,futhark}/timings/` and `rsbench/{cpp,futhark}/timings/`.

Alternatively, run each implementation individually:

#### 1. RSBench (`rsbench/cpp/` and `rsbench/futhark/`)
```bash
# C++ (EnzymeMLIR and EnzymeLLVM, 10,200 lookups):
cd rsbench/cpp
./run_timings.sh 10200 12
cat timings/summary_10200.txt

# Futhark (CUDA reverse-mode AD on data/smaller.in.gz):
cd ../futhark
../../futhark-nightly-linux-x86_64/bin/futhark bench --backend=cuda -r 12 rsbench.fut
```
- **Enzyme MLIR:** Hoists loop-invariant push/pop operations out of the simulation loop, eliminating redundant tape allocations and achieving **0.1160 s**.
- **Enzyme LLVM:** Lacking high-level loop and alias abstractions, it cannot hoist tape management across the loop, taking **2.922 s** (~25× slower).
- **Futhark:** Achieves **0.0322 s** on the functional Futhark port (`rsbench/futhark/rsbench.fut`).

#### 2. LBM (Parboil CFD Simulation, `lbm/cpp/` and `lbm/futhark/`)
```bash
# C++ (EnzymeMLIR, Binomial Checkpointing, and EnzymeLLVM, 100 steps + Nsight Systems memory):
cd lbm/cpp
./run_timings.sh 100 12
cat timings/summary_100.txt

# Full C++ step-count sweep (25, 50, 100, 200, 400, 800, 1200 steps):
make run

# Regenerate the runtime & peak memory figure (results/lbm_benchmark.pdf):
julia plot_lbm_benchmark_results.jl

# Futhark (CUDA reverse-mode AD, 100 steps on data/120_120_150_ldc.in.gz):
cd ../futhark
../../futhark-nightly-linux-x86_64/bin/futhark bench --backend=cuda -r 12 lbm.fut
```
- **Enzyme MLIR:** Applies polyhedral atomic strength reduction, memory pipelining, region-aware mincut caching, and binomial checkpointing, achieving **0.0939 s** at 100 steps (faster than both **Futhark** at **0.1065 s** and **Enzyme LLVM** at **0.2344 s**). `run_timings.sh` also evaluates binomial checkpointing budgets ($b \in \{4, 16\}$, with and without mincut) and the `mlir_with_atomics` (`RMATOMICS=no`) atomic-reduction ablation.
- **Enzyme LLVM:** Requires conservative atomic accumulation and un-minimized tape storage, taking **0.2344 s** (~2.5× slower) and running out of memory at higher step counts.

---

### B. Julia Oceananigans Benchmarks (`Oceananigans/`)

#### 1. Forward Consistency & Quick Smoke Test (CPU)
Verify numerical correctness of the forward simulation and AD drivers on CPU:
```bash
cd Oceananigans

# Link precompiled Manifest from the Docker image
ln -sf /opt/julia_project/Manifest.toml Manifest.toml

# Forward consistency check (CPU)
REACTANT_BACKEND=cpu julia --project=/opt/julia_project bench/forward_consistency.jl 1 --reactant-init

# Enzyme smoke run (CPU, 4 steps)
julia --project=/opt/julia_project bench/driver_enzyme.jl --config enzyme_naive --nsteps 4 --repeats 2

# Reactant smoke run (CPU, 4 steps)
REACTANT_BACKEND=cpu julia --project=/opt/julia_project bench/driver_reactant.jl --config reactant_full --nsteps 4 --repeats 2
```

#### 2. Reproducing Table 2
To format Table 2 directly from the recorded TACC Grace CPU measurements in `Oceananigans/data/vista_2026-11/`:
```bash
cd Oceananigans
julia --project=/opt/julia_project bench/make_table.jl --out data/vista_2026-11 --ci --flops --delta
```

To execute a fresh benchmark sweep using `run.sh`:
```bash
# Smoke test (quick execution across configurations):
./run.sh run smoke

# Quick benchmark suite (N = 4, 8):
./run.sh run quick
./run.sh report results/quick

# Full Table 2 reproduction (N = 4, 8, 16, 32, 64, 128, 256):
./run.sh run full
./run.sh report results/full
```

---

### C. JAX Workloads: JAX-MD and legoESM MPAS-4DVar (`jax/`)

We provide automated per-benchmark orchestrator scripts:
```bash
bash jax/jaxmd/run_benchmarks.sh
bash jax/legoESM-MPAS-4DVar/run_benchmarks.sh
```

Alternatively, run each benchmark individually:

#### 1. JAX-MD Molecular Dynamics Simulation (`jax/jaxmd/`)
Simulates 8,000 Lennard-Jones particles (`jaxmd20` on GPU) under a Nose-Hoover thermostat:
```bash
cd jax/jaxmd
TEST_UNDECLARED_OUTPUTS_DIR=../../results /home/evaluator/eval-env/bin/python3 main.py --warmup=2 --repeat=12
```
- **Enzyme MLIR:** Applies StableHLO domain-specific optimizations (scatter-gather simplification, LICM, inlining) prior to AD, achieving **0.00080359 s** (3.38× speedup in the paper; **0.001369 s** and **172.85 MB peak memory** for `HLOOpt PreRev` vs **0.004279 s** and **440.55 MB peak memory** for `Jax BothRev`, or **3.13× speedup** and **2.55× lower peak memory**, in `results/results_JAXMD.json`).
- **JAX native:** Applies AD before domain optimizations, taking **0.00271422 s**.
- **Plotting:** Open `results/plot-jaxmd.ipynb` in Jupyter to regenerate the runtime and peak memory charts from `results/results_JAXMD.json`.

#### 2. legoESM MPAS-4DVar (Atmospheric Model 4D-Var Adjoint, `jax/legoESM-MPAS-4DVar/`)
Computes the value and gradient of the single-observation 4D-Var cost on the legoESM MPAS primitive-equation model (icosahedral mesh, 40,962 cells, 40 levels, 6,594,882 control variables). Inside the Docker container (`/home/evaluator/eval-env`), `run_benchmarks.sh` installs the `legoESM` federation packages, generates a balanced baroclinic-wave MPAS state if `data/*.npz` is not mounted, and verifies gradient agreement against JAX:
```bash
# 4D-Var gradient and peak memory comparison (10 and 60 steps):
MPAS_MODE=comparison bash jax/legoESM-MPAS-4DVar/run_benchmarks.sh

# Binomial checkpointing sweep across 10–240 steps and budgets b in {2, 4, 8, 16, n}
# (recorded in results/mpas4dvar_binomial_sweep.csv and plotted in results/mpas4dvar_binomial_sweep.pdf):
MPAS_MODE=binomial_sweep bash jax/legoESM-MPAS-4DVar/run_benchmarks.sh
julia jax/legoESM-MPAS-4DVar/plot_binomial_sweep.jl
```
*(For standalone host execution outside Docker with `uv`, run `cd jax/legoESM-MPAS-4DVar && ./setup_env.sh` to create `.venv` with `jax[cuda12]==0.10.0` and `enzyme-ad`, then run `CUDA_VISIBLE_DEVICES=0 PYTHON=.venv/bin/python ./run_comparison.sh` or `./run_binomial_sweep.sh`.)*
- **Enzyme MLIR:** Runs the 1-hour (60-step) window in **6.6 s** with **6.9 GB peak memory** (**6.27 s** and **6.59 GiB** with per-step binomial checkpointing $b=n$, down to **3.01 GiB** with $b=2$).
- **JAX native:** Runs in **7.0 s** (**7.05 s**) with **16.5 GB peak memory** (Enzyme MLIR cuts peak memory by **2.5×–5.5×**).

---

## 5. Continuous Integration (CI)

The repository provides automated per-benchmark GitHub Actions workflows triggered on path changes or manual `workflow_dispatch`:

| Workflow | Description | Trigger Paths | Runner |
| :--- | :--- | :--- | :--- |
| **`docker.yml`** | Builds and publishes `ghcr.io/enzymead/enzymemlir-ppopp27:latest` | `docker/**` | `docker` |
| **`test.yml`** | Oceananigans CPU forward consistency and AD smoke tests | `Oceananigans/**` | `ubuntu-latest` (CPU) |
| **`lbm.yml`** | Automated execution of LBM C++ & Futhark benchmarks | `lbm/**` | `linux-x86-a2-48-a100-4gpu` |
| **`rsbench.yml`** | Automated execution of RSBench C++ & Futhark benchmarks | `rsbench/**` | `linux-x86-a2-48-a100-4gpu` |
| **`jaxmd.yml`** | Automated execution of JAX-MD molecular dynamics benchmark | `jax/jaxmd/**` | `linux-x86-a2-48-a100-4gpu` |
| **`legoesm.yml`** | Automated execution of legoESM MPAS-4DVar benchmark | `jax/legoESM-MPAS-4DVar/**` | `linux-x86-a2-48-a100-4gpu` |

---

## 6. Repository Structure

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
│   │   ├── Makefile                     # Build rules for EnzymeMLIR (Reactant) and EnzymeLLVM
│   │   └── run_timings.sh               # RSBench C++ execution and timing script
│   ├── futhark/                         # Futhark RSBench reverse-mode AD benchmark (`rsbench.fut`)
│   └── run_benchmarks.sh                # Orchestrated RSBench C++ & Futhark runner
├── futhark-nightly-linux-x86_64/        # Prebuilt Futhark compiler distribution (`bin/futhark`)
├── Oceananigans/
│   ├── bench/
│   │   ├── forward_consistency.jl       # Numerical forward-pass consistency test
│   │   ├── driver_enzyme.jl             # EnzymeLLVM benchmark driver
│   │   ├── driver_reactant.jl           # EnzymeMLIR / Reactant benchmark driver
│   │   └── make_table.jl                # Formats Table 2 (Markdown, LaTeX, CSV) from JSON results
│   ├── data/vista_2026-11/              # Pre-recorded Table 2 measurements from Grace CPUs at TACC
│   └── run.sh                           # Table 2 sweep and reporting harness
├── jax/
│   ├── jaxmd/                           # JAX-MD Lennard-Jones molecular dynamics
│   │   ├── main.py                      # JAX-MD benchmark script
│   │   ├── test_utils.py                # Enzyme-JAX pass pipeline harness and JSON exporter
│   │   ├── xprof_utils.py               # XProf step-time and peak device memory profiler
│   │   └── run_benchmarks.sh            # Orchestrated JAX-MD runner
│   └── legoESM-MPAS-4DVar/              # MPAS atmospheric 4D-Var data assimilation
│       ├── setup_env.sh                 # Standalone uv setup creating .venv with legoESM, JAX 0.10.0, and Enzyme-JAX
│       ├── enzyme_4dvar_grad.py         # 4D-Var value-and-gradient driver (JAX vs EnzymeMLIR)
│       ├── run_comparison.sh            # 4D-Var gradient and peak memory comparison
│       ├── run_binomial_sweep.sh        # Step-count and binomial budget sweep
│       ├── plot_binomial_sweep.jl       # Pluto notebook generating results/mpas4dvar_binomial_sweep.pdf
│       └── run_benchmarks.sh            # Orchestrated legoESM MPAS-4DVar runner
├── results/
│   ├── cpp_results.py                   # Recorded 12-run raw timings for LBM and RSBench
│   ├── lbm_benchmark.pdf                # LBM runtime and peak memory scaling figure
│   ├── results_JAXMD.json               # Recorded JAX-MD timings and peak memory across pass pipelines
│   ├── plot-jaxmd.ipynb                 # Notebook plotting JAX-MD runtime and memory usage
│   ├── mpas4dvar_binomial_sweep.csv     # Recorded legoESM MPAS 4D-Var binomial sweep data
│   └── mpas4dvar_binomial_sweep.pdf     # legoESM MPAS 4D-Var runtime and peak memory figure
└── .github/workflows/                   # GitHub Actions CI definitions
```
