# Running the CPU AD cost table on TACC Vista

> **This directory is a record, not the way to run the benchmarks.** These are
> the SLURM scripts that produced [`data/vista_2026-11/`](../../data/vista_2026-11/),
> kept so the procedure behind the published numbers can be inspected. To run the
> benchmarks on your own machine, use [`run.sh`](../../run.sh) instead.
>
> - The allocation name, notification email, SIF location and image registry in
>   [`env.sh`](env.sh) are redacted. The same fields are redacted in each batch's
>   `job_info.json`.
> - The scripts were run from the original benchmark repository, where they lived
>   in `hpc/vista/`; here they are in `provenance/vista/`. The
>   `hpc_scripts_commit` and `git_commit` fields in the data refer to that
>   repository.
> - The original `env.sh` also had profiles for earlier runs; this copy keeps only
>   `jll443`, the profile that produced the data. The other scripts are unchanged
>   apart from their paths, so they still mention the other profiles, and their
>   default profile (`current`) is not in this copy: pass `--profile jll443`.
> - `submit.sh` refuses to submit unless `IMAGE_COMMIT` is in the git history,
>   which is not the case in this repository.
> - The exact package environment of the image is in
>   [`../julia_env/`](../julia_env/) (`Project.toml` and `Manifest.toml`).

Each table case (config × N) runs as its own SLURM job on a whole `gg` node
(NVIDIA Grace), inside the frozen Apptainer image. The settings (allocation, SIF
path, image digest, result locations) live in [`env.sh`](env.sh).

Run every command from the repo root on a login node. `report.sh` uses the
container, and Vista only allows Apptainer on compute nodes, so submit it as a
short job with `submit.sh --report status` (or `--report final`); its output lands
in `$WORK/channelanigans_results/slurm_logs/ch-report-<mode>.<jobid>.out`. Inside an
`idev` session you can also run `provenance/vista/report.sh status` directly.

## Image profile

The container image is selected with `--profile` on every `submit.sh` call; its
settings are in [`env.sh`](env.sh).

| Profile | Image | Results in `$WORK/channelanigans_results/` | Table rows |
|---|---|---|---|
| `jll443` | Julia 1.12.7, Oceananigans 0.116.1, Reactant 0.2.308 (Reactant_jll 0.0.443), Enzyme 0.13.219 ([`../julia_env/`](../julia_env/)); runs in batches | `table_jll443_b1/`, `_b2/`, … pooled into `table_jll443/` | all 9 |

## Batched sweeps (profile `jll443`)

A profile with `BATCH_REPEATS` (in `env.sh`) runs every case as independent
*batches*: separate jobs, usually on different nodes, each with 1 warmup and
`BATCH_REPEATS` (4) measured calls. Batch K lands in `table_jll443_bK/`. Any number
of finished batches can be pooled at any time; the pool's confidence intervals then
include node-to-node spread.

TACC allows only ~40–50 queued + running jobs per user, fewer than the 63 cases, so
`fill.sh` reruns `submit.sh` every 10 min to keep the queue topped up to
`--max-jobs` (default 40). Every batch-1 case is submitted before any batch 2, and
each batch goes longest case first. A case whose batch 1 ended other than `ok`
(e.g. `m`) gets no later batches. Walltimes come from the timings of an earlier run
([`walltimes_jll443_seed.tsv`](walltimes_jll443_seed.tsv)), so there are no waves.

```bash
provenance/vista/submit.sh --profile jll443 --gate                 # may run alongside batch 1
tmux new -s fill                                                   # reattach on the same login node
provenance/vista/fill.sh --profile jll443 --batches 1,2,3 \
    --walltimes provenance/vista/walltimes_jll443_seed.tsv [--max-jobs 40]
provenance/vista/submit.sh --profile jll443 --report status        # every batch
provenance/vista/submit.sh --profile jll443 --report final         # pool finished batches + table
```

`fill.sh` never resubmits a case that already has a result (it passes `--new-only`);
rerun one by hand with `submit.sh --profile jll443 --batches K --configs C --nsteps N
--walltime HH:MM:SS`. `report final` can be run as often as you like: it pools whatever
has finished (`bench/merge_batches.jl`), and adds `export/batches.csv` with each batch's
node and means.

## Gate (one job, about 1 h)

```bash
provenance/vista/submit.sh --profile jll443 --gate
```

The gate checks the node, the FLOP counters (one matmul with a known count, 2·10⁹)
and the full measurement pipeline (`reactant_full` and `enzyme_naive` at N = 4). The
log ends with `GATE PASSED`. The files land in `$WORK/channelanigans_results/gate/<jobid>/`.

## Finalize

```bash
provenance/vista/submit.sh --profile jll443 --report final
```

This pools the finished batches, builds `table.tex` and `table.csv`, cross-checks
gradients across configs, and writes the raw-data export (`export/raw_runs.csv`,
`cases.csv`, `batches.csv`, `DATA_README.md`, `raw_data.tar.gz`) into
`$WORK/channelanigans_results/table_jll443/`.

## Notes

- **Resubmitting is safe.** Cases that are already `ok` in `$WORK`, or that are
  still queued, are skipped. A failed earlier attempt is moved to `_superseded/`,
  never deleted. Pass `--force` to rerun cases that are already ok.
- **Results are copied to `$WORK`.** Jobs write to `$SCRATCH` and copy each finished
  case to `$WORK`. The orchestrator stops 30 min before the walltime, so the copy
  happens even when a case times out.
- **Every case records its provenance.** Each case directory gets a
  `job_info.json` with the job ID, node, image digest, Apptainer version and the
  commit of these scripts. The benchmark code's own commit is in `result.json`.
