# Shared settings for the Vista (TACC) runs of the CPU AD cost table.
# Sourced by gate.slurm, case.slurm, submit.sh, fill.sh and report.sh; edit here only.
# Image-specific settings live in the profile block below.

# ---- Allocation / partition ------------------------------------------------
ACCOUNT=YOUR_ALLOCATION      # redacted; set to your SLURM allocation (sbatch -A)
PARTITION=gg                 # Grace-Grace CPU nodes (NVIDIA Grace, Neoverse V2)
MAX_WALLTIME_H=48            # gg limit; cases predicted to need more are capped

# ---- Email notifications (one email per job; empty MAIL_USER = none) ---------
# A case that ends as t (timeout) or m (out of memory) is still a successful SLURM
# job, so use END and check `report.sh status` for the table outcome.
MAIL_USER=                   # redacted; set to your email to get job notifications
MAIL_TYPE=END,FAIL

# ---- Image profile -------------------------------------------------------------
# Which container image a run uses, and where its results go. Selected with
# `submit.sh --profile NAME`, which exports CH_PROFILE to the jobs. This copy keeps
# only `jll443`, the profile that produced data/vista_2026-11 (every command was run
# with `--profile jll443`).
# The SIF was pulled from the registry image named in IMAGE_REF. The tag is mutable,
# the digest is not: the digest identifies the image that produced the results.
#   SIF            the pulled image on Vista
#   IMAGE_COMMIT   repo commit baked into the image (its BUILD_INFO.txt), in the
#                  original benchmark repository
#   SWEEP          results subdirectory (under $SCRATCH_ROOT and $WORK_ROOT)
#   TABLE_CONFIGS  configs for `--configs table`; "table" = the rows in configs.toml
#   JOB_PREFIX     SLURM job-name prefix, so two profiles never see each other's jobs
#   WALLTIMES_FILE where `report.sh status` writes its walltime predictions
#   GATE_WALLTIME / GATE_E2E_TIMEOUT_S   gate job limit / its end-to-end test timeout
#   RUN_JULIA_FLAGS / RUN_XLA_FLAGS     optional: Julia and XLA flags for the drivers;
#                  unset = the orchestrator's defaults (-O0, multi-threaded Eigen off)
#   BATCH_REPEATS  measured calls per batch (see "Batches" below)
#   PERF_OFF_CASES config:N cases that run without perf
CH_PROFILE=${CH_PROFILE:-current}
case $CH_PROFILE in
    jll443)    # Reactant 0.2.308 (Reactant_jll 0.0.443) and Enzyme 0.13.219 (provenance/julia_env)
        # Orchestrator defaults: -O0, multi-threaded Eigen off. Run in batches of 4
        # measured calls (see "Batches" below).
        # Image built 2026-11-09 from commit 3f6c63d (arm64), pulled by digest.
        SIF=$WORK/redacted-2026-11-jll443-cpu.sif   # original location redacted
        IMAGE_REF=redacted-artifact:2026-11-jll443-cpu   # registry redacted
        IMAGE_DIGEST=sha256:bc5ed0f2d08f3c234360fbd31ff66924ddba22d9231a9fe33c8187412b688d6b
        IMAGE_COMMIT=3f6c63da496e9bcdb5acf77948578a963ccbd711
        SWEEP=table_jll443
        TABLE_CONFIGS=table
        JOB_PREFIX=ch-j443
        WALLTIMES_FILE=walltimes_jll443.tsv
        GATE_WALLTIME=01:30:00
        GATE_E2E_TIMEOUT_S=3600
        BATCH_REPEATS=4
        # enzyme_naive N=256 runs out of memory; under perf that is recorded as "e", so
        # run it without perf.
        PERF_OFF_CASES="enzyme_naive:256"
        ;;
    *)
        echo "unknown image profile '$CH_PROFILE' (this copy keeps only: jll443; use --profile jll443)" >&2
        return 2 2>/dev/null || exit 2
        ;;
esac

# ---- Results -----------------------------------------------------------------
# Jobs write to $SCRATCH (fast, purged after inactivity) and copy each finished
# case to $WORK (persistent). Everything downstream reads the $WORK copy.
SCRATCH_ROOT=$SCRATCH/redacted_results
WORK_ROOT=$WORK/redacted_results
LOG_DIR=$WORK_ROOT/slurm_logs

# ---- Benchmark settings (same as the Docker proofing runs) -------------------
CORE=0                       # numactl -C core for the benchmark process
DRIVER_ARGS="--save-gradients"   # per-driver extras; protocol defaults: 1 warmup + 12 measured
TIMEOUT_MARGIN_S=1800        # orchestrator timeout = walltime - margin, so a slow
                             # case is classified "t" and copied back before SLURM kills the job
MAX_JOBS=${MAX_JOBS:-40}     # submit.sh keeps at most this many of your jobs in the queue
                             # (all of them, incl. idev); TACC's per-user limit is ~40-50

# ---- Batches -------------------------------------------------------------------
# A profile that sets BATCH_REPEATS runs every case as several independent batches:
# separate jobs (usually on different nodes), each a fresh process with its own
# compilation, 1 warmup call and BATCH_REPEATS measured calls. Batch K writes to
# ${SWEEP}_bK with job names $JOB_PREFIX-bK-...; `report.sh final` pools the measured
# calls of all finished batches into $SWEEP (bench/merge_batches.jl).
# select_batch K switches SWEEP and JOB_PREFIX to batch K (case.slurm gets K as CH_BATCH).
SWEEP_BASE=$SWEEP
JOB_PREFIX_BASE=$JOB_PREFIX
[[ -n ${BATCH_REPEATS:-} ]] && DRIVER_ARGS="--save-gradients --repeats $BATCH_REPEATS"
select_batch() {
    [[ $1 =~ ^[1-9][0-9]*$ ]] || { echo "bad batch number '$1'" >&2; return 2; }
    SWEEP=${SWEEP_BASE}_b$1
    JOB_PREFIX=$JOB_PREFIX_BASE-b$1
}

# Directory holding these scripts (and the repo checkout they belong to)
HPC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$HPC_DIR/../.." && pwd)

# Apptainer: --cleanenv keeps the host environment (incl. your JULIA_DEPOT_PATH)
# out of the container. A per-job writable depot is stacked in front of the frozen,
# precompiled one. SLURM variables are passed back in so result.json records them.
apptainer_exec() {
    local depot=${TMPDIR:-/tmp}/redacted_depot_${SLURM_JOB_ID:-$$}
    mkdir -p "$depot"
    apptainer exec --cleanenv \
        --bind "$SCRATCH_ROOT:/results" \
        --bind "$WORK_ROOT:/work_results" \
        --env "JULIA_DEPOT_PATH=$depot:/opt/julia_depot" \
        --env "SLURM_JOB_ID=${SLURM_JOB_ID:-},SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}" \
        --env "BENCH_IMAGE_DIGEST=$IMAGE_DIGEST" \
        "$SIF" "$@"
}

# Orchestrator arguments for the profile's Julia/XLA flags, in the array ORCH_FLAGS
# (empty for profiles that use the orchestrator's defaults). $1: subdirectory of
# $SCRATCH_ROOT/_hlo for the XLA HLO dump, used when XLA_DUMP=1.
orchestrator_flag_args() {
    ORCH_FLAGS=()
    [[ -n ${RUN_JULIA_FLAGS+x} ]] && ORCH_FLAGS+=(--julia-flags "$RUN_JULIA_FLAGS")
    if [[ -n ${RUN_XLA_FLAGS+x} ]]; then
        local xla=$RUN_XLA_FLAGS
        [[ ${XLA_DUMP:-0} == 1 ]] && xla="$xla --xla_dump_to=/results/_hlo/$1"
        ORCH_FLAGS+=(--xla-flags "$xla")
    fi
    return 0
}

# Lines of XLA's buffer assignment with "Total bytes used", one per compiled module,
# prefixed with the dump file name. $1: HLO dump directory on the host. Depending on the
# XLA version the line is in *-buffer-assignment.txt or *-buffer-assignment-values.txt
# so search both.
xla_total_bytes() {
    grep -H "Total bytes used" "$1"/*buffer-assignment*.txt 2>/dev/null | sed "s|^$1/||"
    return 0
}

# Lmod's `module` is not safe under `set -u`/`set -e` (it references unset variables,
# which silently kills the calling script), so relax both just for the module load.
load_apptainer() {
    if ! command -v apptainer >/dev/null 2>&1; then
        local saved=$-
        set +eu
        module load tacc-apptainer
        [[ $saved == *u* ]] && set -u
        [[ $saved == *e* ]] && set -e
    fi
    command -v apptainer >/dev/null 2>&1 || { echo "apptainer not found (tried: module load tacc-apptainer)" >&2; exit 2; }
}

# HH:MM:SS → seconds
hms_to_s() { local IFS=:; set -- $1; echo $(( 10#$1 * 3600 + 10#$2 * 60 + 10#${3:-0} )); }
