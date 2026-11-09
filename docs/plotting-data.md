# Select reference or freshly collected plot data

The JAX-MD and MPAS plots have an explicit `reference`/`new` selector. Both print
the input path and output directory. They never silently fall back from missing
new data to the checked-in measurements. Paths are independent of the shell's
working directory; explicit relative `--input`/`--output-dir` paths are relative
to that working directory.

| Plot | `--source reference` (default) | `--source new` |
| --- | --- | --- |
| JAX-MD | `results/results_JAXMD.json` | `results/jaxmd/results_JAXMD.json` |
| MPAS | `results/mpas4dvar_binomial_sweep.csv` | `results/binomial_sweep/summary.csv` |

The `new` paths match the top-level benchmark runners. Run these commands inside
the artifact container from the repository root. The Julia plot needs the
CairoMakie/DataFrames/Unitful plotting environment (use `--project` if it is not
on your Julia load path); the Python plot needs NumPy, pandas and Matplotlib.

## Regenerate reference figures without running benchmarks

```sh
MPLBACKEND=Agg python plotting/jaxmd.py --source reference
julia jax/legoESM-MPAS-4DVar/plot_binomial_sweep.jl --source reference
```

## Plot freshly collected data

```sh
bash jax/jaxmd/run_benchmarks.sh
MPLBACKEND=Agg python plotting/jaxmd.py --source new

MPAS_MODE=binomial_sweep bash jax/legoESM-MPAS-4DVar/run_benchmarks.sh
julia jax/legoESM-MPAS-4DVar/plot_binomial_sweep.jl --source new
```

PDF/PNG files are written to `results/plots/jaxmd/{reference,new}/` and
`results/plots/mpas/{reference,new}/`. Original reference JSON/CSV/PDF files are
not overwritten. LBM's existing plot and its input handling are unchanged.

## Custom inputs and outputs

```sh
MPLBACKEND=Agg python plotting/jaxmd.py --source new \
  --input /work/my-jaxmd.json --output-dir /work/figures/jaxmd
julia jax/legoESM-MPAS-4DVar/plot_binomial_sweep.jl --source new \
  --input /work/my-sweep.csv --output-dir /work/figures/mpas
```

Both tools also accept `PLOT_SOURCE`, `PLOT_INPUT`, and `PLOT_OUTPUT_DIR`.
Command-line options take precedence. `--source` labels/selects the dataset,
while `--input` overrides its default file. MPAS continues to accept the legacy
`RESULTS_DIR` input-directory override when `--input`/`PLOT_INPUT` is absent.

In `results/plot-jaxmd.ipynb`, change `SOURCE` in the first code cell and rerun all
cells. The notebook and CLI share the same rendering code and normalization.
In Pluto, set the `PLOT_*` environment variables before starting the notebook;
its command-line arguments are ignored in a Pluto session.

A synthetic MPAS run is a different input experiment from the ERA5 reference.
Selecting `new` identifies where the data came from, not equivalence of the
inputs or hardware. Check the benchmark's logs before comparing performance.

## Checks

```sh
julia tests/plot_data_selection.jl
MPLBACKEND=Agg python -m unittest discover -s tests -p test_plot_data.py -v
```

The checks cover source paths, explicit overrides, missing-new-data errors, the
reference speedup baseline, and headless PDF/PNG rendering.
