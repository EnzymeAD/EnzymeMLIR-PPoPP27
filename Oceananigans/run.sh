#!/bin/bash
# Run the Oceananigans CPU AD cost benchmarks on this machine, inside the frozen
# container image (Docker or a compatible engine).
#
# usage: ./run.sh build
#        ./run.sh run PRESET [orchestrator options] [-- driver options]
#        ./run.sh report DIR
#
#   build          build the image for this machine's architecture from ./Dockerfile
#   run PRESET     run a set of benchmark cases; results go to results/PRESET/
#                    smoke   reactant_full + enzyme_naive at N = 4, 2 measured calls
#                            (checks the setup; ~15 min, ~18 GiB peak memory)
#                    quick   all 9 table configs at N = 4, 8, full protocol
#                            (~3 h, ~31 GiB peak memory)
#                    full    the paper's table: 9 configs at N = 4 ... 256
#                            (~80 h, ~175 GiB peak memory; cases that exceed the
#                            machine's memory are recorded as "m")
#                  Times are for one NVIDIA Grace core; other CPUs may be slower.
#                  Extra options are passed to bench/run_cases.jl and override the
#                  preset (e.g. --configs reactant_full --nsteps 16); options after
#                  `--` go to every driver (e.g. --repeats 4).
#                  Cases that already finished ("ok") are skipped, so an
#                  interrupted run can be resumed by running the same command again.
#   report DIR     build the table (table.tex, table.csv), cross-check gradients
#                  across configs, and export the raw data (DIR/export/) for a
#                  results directory, e.g. results/quick or data/vista_2026-11
#
# environment:
#   IMAGE          image tag (default ghcr.io/enzymead/enzymemlir-ppopp27:latest)
#   DOCKER         container engine CLI (default docker)
#   PLATFORM       build/run for another platform, e.g. linux/amd64 (default: native)
#   RESULTS_DIR    results directory for `run` (default results/PRESET)
#   TIMEOUT        per-case limit in seconds, incl. compilation and all calls
#                  (default: smoke 3600, quick 14400, full 172800)
#   CORE           CPU core the benchmark process is pinned to (default 0)
#   FLOPS=1        also count FLOPs with Linux perf. Needs a Linux host with NVIDIA
#                  Grace / Arm Neoverse V2 CPUs and perf_event_paranoid <= 2; the
#                  container then runs with --privileged. Default off.
#   PERF_EVENTS    perf events to count instead of the Neoverse V2 FP events
#   ARCHIVE=1      `report` also packs the results tree into DIR/raw_data.tar.gz
#
# Every `run` writes DIR/host_<UTC time>.json describing the host and image.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${IMAGE:-ghcr.io/enzymead/enzymemlir-ppopp27:latest}
DOCKER=${DOCKER:-docker}
PLATFORM=${PLATFORM:-}
CORE=${CORE:-0}
FLOPS=${FLOPS:-0}
PERF_EVENTS=${PERF_EVENTS:-}
ARCHIVE=${ARCHIVE:-0}

usage() { sed -n '2,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }
die() { echo "run.sh: $*" >&2; exit 2; }

platform_args() { [ -n "$PLATFORM" ] && echo "--platform=$PLATFORM"; }

require_image() {
    "$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1 ||
        die "image '$IMAGE' not found; build it with './run.sh build' (or set IMAGE=...)"
}

# Run a command inside the image with host directories mounted.
# Runs as the calling user so the results are owned by them, not by root.
in_image() {
    local dir=$1; shift
    local extra=""
    [ "$FLOPS" = 1 ] && extra="--privileged"
    local repo_root
    repo_root=$(cd "$HERE/.." && pwd)
    "$DOCKER" run --rm --init $(platform_args) $extra \
        --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -v "$repo_root:/opt/EnzymeMLIR-PPoPP27" \
        -v "$dir:/work/results" -w /opt/EnzymeMLIR-PPoPP27/Oceananigans \
        "$IMAGE" "$@"
}

# Minimal JSON string escaping (backslash, quote, control characters to spaces).
json_str() { printf '"%s"' "$(printf '%s' "$1" | tr '\n\t\r' '   ' | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')"; }

host_info() {
    local os cpu ncpu mem
    if [ "$(uname -s)" = Darwin ]; then
        os="macOS $(sw_vers -productVersion 2>/dev/null)"
        cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)
        ncpu=$(sysctl -n hw.ncpu 2>/dev/null)
        mem=$(sysctl -n hw.memsize 2>/dev/null)
    else
        os=$( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || uname -s)
        cpu=$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)
        ncpu=$(getconf _NPROCESSORS_ONLN 2>/dev/null)
        mem=$(awk '/^MemTotal:/ { print $2 * 1024 }' /proc/meminfo 2>/dev/null)
    fi
    local engine_mem engine_ver image_id image_digests commit
    engine_mem=$("$DOCKER" info --format '{{.MemTotal}}' 2>/dev/null)
    engine_ver=$("$DOCKER" version --format '{{.Server.Version}}' 2>/dev/null)
    image_id=$("$DOCKER" image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null)
    image_digests=$("$DOCKER" image inspect --format '{{join .RepoDigests " "}}' "$IMAGE" 2>/dev/null)
    commit=$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)
    [ "$commit" != unknown ] && ! git -C "$HERE" diff --quiet HEAD -- . 2>/dev/null && commit="$commit-dirty"
    cat <<EOF
{
  "date_utc": $(json_str "$(date -u +%Y-%m-%dT%H:%M:%SZ)"),
  "command": $(json_str "$CMDLINE"),
  "repo_commit": $(json_str "$commit"),
  "os": $(json_str "$os"),
  "kernel": $(json_str "$(uname -srm)"),
  "cpu": $(json_str "$cpu"),
  "ncpu": $(json_str "$ncpu"),
  "memory_bytes": $(json_str "$mem"),
  "engine": $(json_str "$DOCKER $engine_ver"),
  "engine_memory_bytes": $(json_str "$engine_mem"),
  "platform": $(json_str "${PLATFORM:-native}"),
  "image": $(json_str "$IMAGE"),
  "image_id": $(json_str "$image_id"),
  "image_repo_digests": $(json_str "$image_digests"),
  "core": $(json_str "$CORE"),
  "flops": $(json_str "$FLOPS")
}
EOF
}

cmd_build() {
    local repo_root commit
    repo_root=$(cd "$HERE/.." && pwd)
    commit=$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)
    [ "$commit" != unknown ] && ! git -C "$HERE" diff --quiet HEAD -- . 2>/dev/null && commit="$commit-dirty"
    echo "building $IMAGE from $repo_root/docker/Dockerfile (repo commit $commit${PLATFORM:+, platform $PLATFORM})"
    "$DOCKER" build $(platform_args) --build-arg GIT_COMMIT="$commit" -t "$IMAGE" -f "$repo_root/docker/Dockerfile" "$repo_root"
}

cmd_run() {
    [ $# -ge 1 ] || usage
    local preset=$1; shift
    local configs nsteps timeout driver_args need_gib
    case $preset in
        smoke) configs=reactant_full,enzyme_naive; nsteps=4;   timeout=3600;   driver_args="--repeats 2"; need_gib=18 ;;
        quick) configs=table;                      nsteps=4,8; timeout=14400;  driver_args="";            need_gib=31 ;;
        full)  configs=table;                      nsteps=4,8,16,32,64,128,256; timeout=172800; driver_args=""; need_gib=175 ;;
        -h|--help) usage 0 ;;
        *) die "unknown preset '$preset' (smoke, quick or full)" ;;
    esac
    timeout=${TIMEOUT:-$timeout}

    # Split the remaining arguments into orchestrator options and driver options.
    local orch=() drv=() seen_sep=0 a
    for a in "$@"; do
        if [ $seen_sep = 0 ] && [ "$a" = "--" ]; then seen_sep=1; continue; fi
        if [ $seen_sep = 0 ]; then orch+=("$a"); else drv+=("$a"); fi
    done

    require_image
    local dir=${RESULTS_DIR:-$HERE/results/$preset}
    mkdir -p "$dir" || die "cannot create $dir"
    dir=$(cd "$dir" && pwd)

    local engine_mem
    engine_mem=$("$DOCKER" info --format '{{.MemTotal}}' 2>/dev/null)
    if [ -n "$engine_mem" ] && [ "$engine_mem" -lt $(( need_gib * 1024 * 1024 * 1024 )) ] 2>/dev/null; then
        echo "run.sh: warning: the container engine has $(( engine_mem / 1024 / 1024 / 1024 )) GiB of memory;" \
             "preset '$preset' peaks at about $need_gib GiB, so some cases may be recorded as out of memory (m)." >&2
    fi

    local perf="--perf off" events=""
    [ "$FLOPS" = 1 ] && perf="--perf on"
    [ -n "$PERF_EVENTS" ] && events="--perf-events $PERF_EVENTS"

    host_info > "$dir/host_$(date -u +%Y%m%dT%H%M%SZ).json"
    echo "results: $dir"
    # shellcheck disable=SC2086  # word splitting of the preset's option strings is intended
    in_image "$dir" julia --project=/opt/julia_project bench/run_cases.jl \
        --out results --configs "$configs" --nsteps "$nsteps" --timeout "$timeout" \
        --core "$CORE" $perf $events --skip-existing \
        ${orch[@]+"${orch[@]}"} \
        -- --save-gradients $driver_args ${drv[@]+"${drv[@]}"}
    local rc=$?
    [ $rc = 0 ] && echo "done; next: ./run.sh report ${dir#"$HERE"/}"
    return $rc
}

cmd_report() {
    [ $# -eq 1 ] || usage
    [ -d "$1" ] || die "no such directory: $1"
    require_image
    local dir
    dir=$(cd "$1" && pwd)
    # Show the FLOPs row only if the results have FLOP counts.
    local flops=""
    grep -qs '"flops_per_call": [0-9]' "$dir"/*/N*/result.json && flops="--flops"
    local archive=""
    [ "$ARCHIVE" = 1 ] && archive="--archive"
    local rc=0
    echo "== table"
    in_image "$dir" julia --project=/opt/julia_project bench/make_table.jl --out results --ci --delta $flops || rc=1
    echo "== gradient check"
    in_image "$dir" julia --project=/opt/julia_project bench/check_gradients.jl --out results ||
        { echo "run.sh: gradient check reported failures (see above)" >&2; rc=1; }
    echo "== raw-data export"
    in_image "$dir" julia --project=/opt/julia_project bench/export_raw.jl --out results $archive || rc=1
    echo "outputs in $dir (table.tex, table.csv, export/)"
    return $rc
}

CMDLINE="$0 $*"
[ $# -ge 1 ] || usage
command=$1; shift
case $command in
    build)  cmd_build "$@" ;;
    run)    cmd_run "$@" ;;
    report) cmd_report "$@" ;;
    -h|--help|help) usage 0 ;;
    *) die "unknown command '$command' (build, run or report)" ;;
esac
