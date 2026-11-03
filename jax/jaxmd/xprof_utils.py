"""Utility functions for extracting profiling data from xprof/XLA traces."""

import glob
import json
import os
import shutil
import tempfile
import timeit
import warnings
from typing import Any, Callable
from statistics import mean, stdev

try:
    from xprof.convert.raw_to_tool_data import xspace_to_tool_data

    XPROF_AVAILABLE = True
except ImportError:
    XPROF_AVAILABLE = False


def profile_function(
    fn: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    nrepeat: int = 1,
    warmup: int = 1,
    trace_dir: str | None = None,
) -> dict[str, Any]:
    """
    Profile a JAX function and return timing data.

    Args:
        fn: The function to profile
        args: Positional arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        nrepeat: Number of times to run the function during profiling
        warmup: Number of warmup runs before profiling (to ensure compilation is done)
        trace_dir: Directory to save traces. If None, uses a temporary directory.

    Returns:
        A dictionary containing:
            - 'min_time_us': Minimum runtime per iteration in microseconds
            - 'min_time_ms': Minimum runtime per iteration in milliseconds
            - 'min_time_s': Minimum runtime per iteration in seconds
            - 'max_time_s': Maximum runtime per iteration in seconds
            - 'avg_time_s': Average runtime per iteration in seconds
            - 'stdev_time_s': Standard deviation of the runtime in seconds
            - 'nrepeat': Number of repetitions
            - 'trace_dir': Directory where traces were saved (None if temporary)
            - 'xplane_file': Paths to the xplane.pb files (None if temporary)
    """
    if kwargs is None:
        kwargs = {}

    import jax

    compiled_fn = jax.jit(fn).trace(*args, **kwargs).lower().compile()

    # Warmup runs (ensure compilation is complete)
    for _ in range(warmup):
        jax.block_until_ready(compiled_fn(*args, **kwargs))

    return profile_compiled_function(compiled_fn, args, kwargs, nrepeat, trace_dir)


def profile_compiled_function(
    compiled_fn: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    nrepeat: int = 1,
    trace_dir: str | None = None,
) -> dict[str, Any]:
    """
    Profile a JAX function and return timing data.

    Each of the `nrepeat` runs is profiled in its own xprof trace, so that every
    measurement is extracted independently and the statistics are computed over
    `nrepeat` samples.

    Args:
        fn: The function to profile. We assume that the function was AoT compiled.
        args: Positional arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        nrepeat: Number of times to run the function during profiling
        trace_dir: Directory to save traces. If None, uses a temporary directory
            that is deleted after the timings are extracted.

    Returns:
        A dictionary containing:
            - 'min_time_us': Minimum runtime per iteration in microseconds
            - 'min_time_ms': Minimum runtime per iteration in milliseconds
            - 'min_time_s': Minimum runtime per iteration in seconds
            - 'max_time_s': Maximum runtime per iteration in seconds
            - 'avg_time_s': Average runtime per iteration in seconds
            - 'stdev_time_s': Standard deviation of the runtime in seconds
            - 'nrepeat': Number of repetitions
            - 'trace_dir': Directory where traces were saved (None if temporary)
            - 'xplane_file': Paths to the xplane.pb files, one per repetition
              (None if temporary)
    """
    import jax

    if kwargs is None:
        kwargs = {}

    if not XPROF_AVAILABLE:
        warnings.warn("xprof not found, falling back to timeit for profiling.")
        times_s = []
        for _ in range(nrepeat):
            start = timeit.default_timer()
            jax.block_until_ready(compiled_fn(*args, **kwargs))
            end = timeit.default_timer()
            times_s.append(end - start)
        return _timing_stats(times_s)

    keep_traces = trace_dir is not None
    if trace_dir is None:
        trace_dir = tempfile.mkdtemp(prefix="jax_profile_")

    try:
        times_s = []
        xplane_files = []
        for i in range(nrepeat):
            repeat_dir = os.path.join(trace_dir, f"repeat_{i}")
            with jax.profiler.trace(repeat_dir):
                # StepTraceAnnotation lets xprof compute the wall-clock step time
                with jax.profiler.StepTraceAnnotation("step", step_num=0):
                    jax.block_until_ready(compiled_fn(*args, **kwargs))

            xplane_file = _find_xplane_file(repeat_dir)
            xplane_files.append(xplane_file)
            times_s.append(extract_min_step_time([xplane_file]))
    finally:
        if not keep_traces:
            shutil.rmtree(trace_dir, ignore_errors=True)

    if keep_traces:
        return _timing_stats(times_s, trace_dir=trace_dir, xplane_file=xplane_files)
    return _timing_stats(times_s)


def _timing_stats(
    times_s: list[float],
    trace_dir: str | None = None,
    xplane_file: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize per-repetition runtimes (in seconds)."""
    min_time_s = min(times_s)
    max_time_s = max(times_s)
    avg_time_s = mean(times_s)
    stdev_time_s = stdev(times_s) if len(times_s) > 1 else 0.0
    return {
        "min_time_us": min_time_s * 1e6,
        "min_time_ms": min_time_s * 1e3,
        "min_time_s": min_time_s,
        "max_time_s": max_time_s,
        "avg_time_s": avg_time_s,
        "stdev_time_s": stdev_time_s,
        "nrepeat": len(times_s),
        "trace_dir": trace_dir,
        "xplane_file": xplane_file,
    }


def _find_xplane_file(trace_dir: str) -> str:
    xplane_files = glob.glob(os.path.join(trace_dir, "plugins/profile/*/*.xplane.pb"))
    if not xplane_files:
        raise RuntimeError(
            f"No xplane.pb file found in {trace_dir}. "
            "Make sure the function executed on a device."
        )
    return max(xplane_files, key=os.path.getmtime)  # most recent one


def extract_min_step_time(xplane_files: list[str]) -> float:
    """
    Extract the step time from an xprof trace containing a single step.

    Reads the full-precision per-step times from the overview_page step table
    (taking the minimum if several steps are present). If the step table is not
    available, falls back to the total program time reported by op_profile.

    Args:
        xplane_files: List of paths to xplane.pb files of a single-step trace

    Returns:
        Step time in seconds
    """
    if not XPROF_AVAILABLE:
        raise RuntimeError("xprof is not available.")

    # overview_page returns a list of tables; the second table (index 1)
    # contains step timing information with per-step data in rows
    overview_data = json.loads(
        xspace_to_tool_data(xplane_files, "overview_page", {})[0].decode("utf-8")
    )
    if isinstance(overview_data, list) and len(overview_data) > 1:
        step_table = overview_data[1]
        col_indices = {col["id"]: i for i, col in enumerate(step_table.get("cols", []))}
        step_time_idx = col_indices.get("stepTimeMs")

        if step_time_idx is not None:
            step_times_ms = []
            for row in step_table.get("rows", []):
                cells = row.get("c", [])
                if step_time_idx < len(cells):
                    step_time = cells[step_time_idx].get("v", 0.0)
                    if step_time > 0:
                        step_times_ms.append(step_time)

            if step_times_ms:
                return min(step_times_ms) / 1e3

    # Fallback: total device time of the program (single step => step time)
    op_profile = json.loads(
        xspace_to_tool_data(xplane_files, "op_profile", {})[0].decode("utf-8")
    )
    picosec = op_profile["byProgram"]["metrics"]["normalizedTimePs"]
    if picosec <= 0:
        raise RuntimeError(
            f"xprof measured no step time in {xplane_files} "
            "(no device activity was recorded, e.g. when running on CPU)."
        )
    return picosec / 1e12
