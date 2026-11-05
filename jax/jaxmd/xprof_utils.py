"""Timing of AoT-compiled JAX functions from xprof traces."""

import glob
import json
import os
import shutil
import tempfile

import jax
from xprof.convert.raw_to_tool_data import xspace_to_tool_data


def profile_compiled_function(
    compiled_fn, args, nrepeat: int, warmup: int = 0
) -> dict[str, list]:
    """
    Run `compiled_fn(*args)` `warmup` times without recording anything, then
    `nrepeat` times, each in its own xprof trace, and return the per-run samples:
        - "time_s": step time in seconds
        - "peak_memory_bytes": peak device memory in use during the run

    The peak memory is the allocator's total bytes in use, so it also counts
    every buffer alive in the process during the run (e.g. `args`), not only the
    memory allocated by `compiled_fn`. It is None if xprof recorded no memory
    profile (e.g. on CPU).
    """
    trace_dir = tempfile.mkdtemp(prefix="jax_profile_")
    try:
        # Warmup runs inside a (discarded) trace, so they also absorb the one-time
        # cost of starting the profiler, which otherwise slows down the first sample.
        if warmup > 0:
            with jax.profiler.trace(os.path.join(trace_dir, "warmup")):
                for _ in range(warmup):
                    jax.block_until_ready(compiled_fn(*args))

        times_s = []
        peak_mems = []
        for i in range(nrepeat):
            repeat_dir = os.path.join(trace_dir, f"repeat_{i}")
            with jax.profiler.trace(repeat_dir):
                # StepTraceAnnotation lets xprof compute the wall-clock step time
                with jax.profiler.StepTraceAnnotation("step", step_num=0):
                    jax.block_until_ready(compiled_fn(*args))
            xplane_file = _find_xplane_file(repeat_dir)
            times_s.append(_extract_step_time(xplane_file))
            peak_mems.append(_extract_peak_memory(xplane_file))
    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)

    return {"time_s": times_s, "peak_memory_bytes": peak_mems}


def _find_xplane_file(trace_dir: str) -> str:
    xplane_files = glob.glob(os.path.join(trace_dir, "plugins/profile/*/*.xplane.pb"))
    if not xplane_files:
        raise RuntimeError(
            f"No xplane.pb file found in {trace_dir}. "
            "Make sure the function executed on a device."
        )
    return max(xplane_files, key=os.path.getmtime)  # most recent one


def _extract_peak_memory(xplane_file: str) -> int | None:
    """
    Peak bytes in use during the trace, taken as the max over the memory
    allocators (e.g. GPU_0_bfc) of xprof's memory_profile. None if there are none.
    """
    memory_profile = json.loads(
        xspace_to_tool_data([xplane_file], "memory_profile", {})[0]
    )
    peaks = [
        int(alloc["profileSummary"]["peakStats"]["peakBytesInUse"])
        for alloc in memory_profile.get("memoryProfilePerAllocator", {}).values()
    ]
    return max(peaks) if peaks else None


def _extract_step_time(xplane_file: str) -> float:
    """
    Step time (in seconds) of a single-step trace.

    Reads the full-precision step time from the overview_page step table. If the
    step table is not available, falls back to the total program time reported
    by op_profile.
    """
    # overview_page returns a list of tables; the second table (index 1)
    # contains step timing information with per-step data in rows
    overview_data = json.loads(
        xspace_to_tool_data([xplane_file], "overview_page", {})[0].decode("utf-8")
    )
    if isinstance(overview_data, list) and len(overview_data) > 1:
        step_table = overview_data[1]
        col_indices = {col["id"]: i for i, col in enumerate(step_table.get("cols", []))}
        step_time_idx = col_indices.get("stepTimeMs")

        if step_time_idx is not None:
            step_times_ms = [
                cells[step_time_idx].get("v", 0.0)
                for row in step_table.get("rows", [])
                if step_time_idx < len(cells := row.get("c", []))
            ]
            step_times_ms = [t for t in step_times_ms if t > 0]
            if step_times_ms:
                return min(step_times_ms) / 1e3

    # Fallback: total device time of the program (single step => step time)
    op_profile = json.loads(
        xspace_to_tool_data([xplane_file], "op_profile", {})[0].decode("utf-8")
    )
    picosec = op_profile["byProgram"]["metrics"]["normalizedTimePs"]
    if picosec <= 0:
        raise RuntimeError(
            f"xprof measured no step time in {xplane_file} "
            "(no device activity was recorded, e.g. when running on CPU)."
        )
    return picosec / 1e12
