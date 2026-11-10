# Plotting environment

The artifact image includes a separate Julia plotting project at
`/opt/julia_plotting`, with a checked-in Project/Manifest based on the two
Pluto notebooks and resolved for the container's Julia version. `JULIA_LOAD_PATH` makes its packages available to plain
`julia path/to/plot.jl` commands without modifying the Oceananigans project.
No package installation is needed at runtime, including with networking disabled.

Inside the container, verify the Julia plotting packages:

```sh
julia --project=/opt/julia_plotting -e 'using CairoMakie, DataFrames, Unitful; println("Julia plotting ready")'
```

Run the existing Julia plots with an explicit plotting project if desired:

```sh
julia --project=/opt/julia_plotting lbm/cpp/plot_lbm_benchmark_results.jl
julia --project=/opt/julia_plotting jax/legoESM-MPAS-4DVar/plot_binomial_sweep.jl
```

Each plot still needs its input data. See the notebook's `results_dir` setting
when selecting inputs. These commands use Cairo's headless backend and write
PDF/PNG files; they do not need a display server or GPU.

The Python environment includes JupyterLab, nbconvert and ipykernel alongside
NumPy, pandas and Matplotlib. To open the checked-in notebook from the host:

```sh
docker run --rm -it -p 127.0.0.1:8888:8888 \
  -v "$PWD:/workspaces/EnzymeMLIR-PPoPP27" \
  -w /workspaces/EnzymeMLIR-PPoPP27/results \
  ghcr.io/enzymead/enzymemlir-ppopp27:latest \
  jupyter lab --ip=0.0.0.0 --no-browser --allow-root plot-jaxmd.ipynb
```

Use the token URL printed by Jupyter. The loopback-only port mapping keeps the
server local. For batch execution inside the container, from `results/`:

```sh
MPLBACKEND=Agg jupyter nbconvert --to notebook --execute plot-jaxmd.ipynb \
  --output plot-jaxmd.executed.ipynb
```

For a local Julia installation, instantiate the plotting project once with
`julia --project=docker/plotting -e 'using Pkg; Pkg.instantiate()'`, then use
`--project=docker/plotting` when executing a Julia plot.
