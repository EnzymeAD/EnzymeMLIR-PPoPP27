# Reference data: TACC Vista, November 2026

The measurements behind the paper's CPU AD cost table: the cost of one reverse-mode
gradient of the channel model for 9 AD configurations at N = 4, 8, 16, 32, 64, 128
and 256 time steps. What the benchmark is, how this data was produced and what it
shows are described in the [Oceananigans README](../../README.md).

> **Status: complete.** All 63 cases are in: 60 finished (`ok`) and 3 ran out of
> the node's memory (`m`): `reactant_mincut_off` at N = 128 and 256, and
> `enzyme_naive` at N = 256. Every finished case has 12 measured calls from 3
> batches on different nodes.

## Files

| Path | Contents |
|---|---|
| `table.tex`, `table.csv` | The table: FLOPs, runtime, memory and Δmemory per config and N, with CI half-widths |
| `export/raw_runs.csv` | One row per call (warmup + measured) of every batch, with its `batch` and `hostname` |
| `export/cases.csv` | One row per case: outcome, pooled statistics, `n_batches`, versions, flags |
| `export/batches.csv` | One row per batch: outcome, node, SLURM job, per-batch means |
| `export/DATA_README.md` | How each field is collected: the full methodology |
| `<config>/N<nnnn>/result.json` | Pooled record of a case: every call (tagged with its batch and node), statistics, gradient summaries, environment, `batches` |
| `<config>/N<nnnn>/status.json` | Outcome (`ok` / `t` / `m` / `s` / `e`) and the batches behind it |
| `batches/table_jll443_b<K>/<config>/N<nnnn>/` | Batch K as it ran: `result.json`, `status.json`, `progress.json`, `job_info.json` (SLURM job, node, image digest, Apptainer version, script commit; allocation name, SIF path and image registry redacted), `log.txt` |

**Not kept:** two intermediate files per case, which [`run.sh`](../../run.sh)
regenerates (both are excluded by `.gitignore`):
- `perf.csv`: the raw 100 ms hardware-counter intervals. The per-call FLOP counts
  computed from them are kept, in each `result.json` and in the `flops` column of
  `export/raw_runs.csv`, and the table is built from those. `run.sh` writes
  `perf.csv` when run with `FLOPS=1`, which needs a Linux host with NVIDIA Grace
  (Arm Neoverse V2) CPUs; runtime and memory can be reproduced on any machine.
- `gradients.jld2`: the full gradient fields of each case's reference call. Their
  summaries (norms and other statistics of every gradient field) are kept in each
  `result.json`; the full fields are only needed for the element-wise
  cross-configuration check. `run.sh run` always writes them, and `run.sh report`
  runs the check (`bench/check_gradients.jl`).

## Reading the table

- **Out of memory:**
  - `reactant_mincut_off` at N = 128 and 256: XLA reports it directly
    (`Out of memory`).
  - `enzyme_naive` at N = 256: killed by the kernel's out-of-memory killer. This
    case ran without hardware counters from the start (`"perf_mode": "off"` in its
    `job_info.json`): under `perf`, a process killed by the kernel is reported as a
    clean exit, so the case would be recorded as an error (`e`) instead of `m`.
- **Node-to-node spread:** for most cases the batch means agree within a few
  percent (median spread between the fastest and slowest batch: 1.2%). Two cases
  stand out, both `reactant_mincut_off`, the configuration with by far the largest
  memory footprint:
  - N = 8: one batch averaged 20.5 s per call, the other two 12.1–12.2 s;
  - N = 32: one batch averaged 73.0 s, the other two 48.2–49.3 s.

  In both, the slow batch was consistently slow on all its calls and executed the
  same number of FLOPs, so the node, not the computation, was slower. Their wide
  confidence intervals (±2.62 s and ±7.59 s) are this spread.
- **N = 4:** several Reactant configurations use more memory at N = 4 than at N = 8;
  it is a property of the compiled program at very short loops.

## Regenerating

From the `Oceananigans/` directory. Pooling the batches (writes
`batches/table_jll443/`, which reproduces the pooled `<config>/N<nnnn>/` files
here exactly):

```bash
julia --project bench/merge_batches.jl --root data/vista_2026-11/batches --sweep table_jll443
```

The table and the export, from the pooled files:

```bash
julia --project bench/make_table.jl --out data/vista_2026-11 --ci --flops --delta
julia --project bench/export_raw.jl --out data/vista_2026-11
```

These read only the committed JSON files and reproduce `table.csv`, the
`export/` files and `table.tex` exactly (apart from the source path in the first
line of `table.tex`).
