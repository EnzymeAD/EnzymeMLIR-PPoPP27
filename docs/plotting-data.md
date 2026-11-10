# Select plotting data

Set `PLOT_SOURCE=reference` (the default) or `PLOT_SOURCE=new` when plotting.
The scripts print the selected input and fail if it is missing; they do not
fall back from new measurements to reference data.

| Plot | Reference input | New input |
| --- | --- | --- |
| JAX-MD | `results/results_JAXMD.json` | `results/jaxmd/results_JAXMD.json` |
| MPAS | `results/mpas4dvar_binomial_sweep.csv` | `results/binomial_sweep/summary.csv` |

Inside the artifact container, from the repository root:

```sh
PLOT_SOURCE=reference julia jax/legoESM-MPAS-4DVar/plot_binomial_sweep.jl
PLOT_SOURCE=new julia jax/legoESM-MPAS-4DVar/plot_binomial_sweep.jl

PLOT_SOURCE=reference MPLBACKEND=Agg jupyter nbconvert --execute --to notebook \
  results/plot-jaxmd.ipynb --output-dir=/tmp --output=jaxmd-reference.ipynb
PLOT_SOURCE=new MPLBACKEND=Agg jupyter nbconvert --execute --to notebook \
  results/plot-jaxmd.ipynb --output-dir=/tmp --output=jaxmd-new.ipynb
```

In Jupyter, change `source` in the notebook's input-selection cell and rerun all
cells. `PLOT_INPUT` overrides the input file and `PLOT_OUTPUT_DIR` overrides the
output directory. Defaults save PDF/PNG plots under
`results/plots/{mpas,jaxmd}/{reference,new}/`, preserving reference files.
MPAS retains its legacy `RESULTS_DIR` input-directory override.

The plotting dependencies are supplied by the Docker plotting environment.
Synthetic MPAS data is a different experiment from the ERA5 reference, even
when selected as `new`.
