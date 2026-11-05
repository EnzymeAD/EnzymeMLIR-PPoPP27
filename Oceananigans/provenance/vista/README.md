# Running the CPU AD cost table on TACC Vista

> **This directory is a record, not the way to run the benchmarks.** These are
> the SLURM scripts that produced [`data/vista_2026-11/`](../../data/vista_2026-11/),
> kept so the procedure behind the published numbers can be inspected. To run the
> benchmarks on your own machine, use [`run.sh`](../../run.sh) instead.
>
> - The allocation name, notification email and SIF location in
>   [`env.sh`](env.sh) are redacted. The same fields are redacted in each case's
>   `job_info.json`.
> - The scripts were run from the original benchmark repository, where they lived
>   in `hpc/vista/`; here they are in `provenance/vista/`. The
>   `hpc_scripts_commit` and `git_commit` fields in the data refer to that
>   repository.
> - `submit.sh` refuses to submit unless `IMAGE_COMMIT` is in the git history,
>   which is not the case in this repository.

Each table case (config × N) runs as its own SLURM job on a whole `gg` node
(NVIDIA Grace), inside the frozen Apptainer image. The settings (allocation, SIF
path, image digest, result locations) live in [`env.sh`](env.sh).

Run every command from the repo root on a login node. `report.sh` uses the
container, and Vista only allows Apptainer on compute nodes, so submit it as a
short job with `submit.sh --report status` (or `--report final`); its output lands
in `$WORK/channelanigans_results/slurm_logs/ch-report-<mode>.<jobid>.out`. Inside an
`idev` session you can also run `provenance/vista/report.sh status` directly.

## 1. Gate (one job, about 1 h)

```bash
provenance/vista/submit.sh --gate
```

The gate checks the node, the FLOP counters (one matmul with a known count, 2·10⁹)
and the full measurement pipeline (`reactant_full` and `enzyme_naive` at N = 4). The
log ends with `GATE PASSED`. The files land in `$WORK/channelanigans_results/gate/<jobid>/`.

## 2. Wave 1: N = 4 and 8 for every config (18 jobs)

```bash
provenance/vista/submit.sh --nsteps 4,8 --walltime 04:00:00
```

These are final table entries. The planner also uses them to predict walltimes
for larger N.

## 3. Later waves

```bash
provenance/vista/submit.sh --report status   # status of every case; writes provenance/vista/walltimes.tsv
provenance/vista/submit.sh --nsteps 16 --walltimes provenance/vista/walltimes.tsv
# ... then repeat status → submit for 32,64 and 128,256
```

The predictions use the two largest finished N for each config, so going one or
two N values at a time keeps them accurate. Cases are capped at 48 h. A case that
runs out of time or memory is recorded as `t` or `m`; both are legitimate table
entries.

## 4. Finalize

```bash
provenance/vista/submit.sh --report final
```

This builds `table.tex` and `table.csv`, cross-checks gradients across configs,
and writes the raw-data export (`export/raw_runs.csv`, `cases.csv`,
`DATA_README.md`, `raw_data.tar.gz`) into `$WORK/channelanigans_results/table/`.

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
