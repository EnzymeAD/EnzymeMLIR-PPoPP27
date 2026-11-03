#!/usr/bin/env python3
"""Peak device memory from an Nsight Systems trace exported to SQLite.

Requires `nsys profile -t cuda --cuda-memory-usage=true --export=sqlite`.
Replays the CUDA_GPU_MEMORY_USAGE_EVENTS allocation/free stream and prints

    <peak bytes> <baseline bytes> <peak - baseline bytes> <#allocations>

where baseline is what is live when the first kernel is launched minus the
allocations made inside the AD region, i.e. what is live right after the last
host-to-device copy that precedes the first kernel (the primal + shadow grids).
"""
import sqlite3
import sys

ALLOC, FREE = 0, 1
DEVICE_KINDS = (2, 4)  # Device, Managed
H2D = 1


def analyze(path):
    db = sqlite3.connect(path)
    events = db.execute(
        "SELECT start, bytes, memoryOperationType FROM CUDA_GPU_MEMORY_USAGE_EVENTS"
        f" WHERE memKind IN {DEVICE_KINDS} ORDER BY start"
    ).fetchall()

    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    first_kernel = None
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        # Kernel rows are GPU times; correlate to the API launch on the host timeline
        first_kernel = db.execute(
            "SELECT MIN(r.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k"
            " JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId"
        ).fetchone()[0]
    t0 = None
    if first_kernel is not None and "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        t0 = db.execute(
            "SELECT MAX(r.end) FROM CUPTI_ACTIVITY_KIND_MEMCPY m"
            " JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = m.correlationId"
            " WHERE m.copyKind = ? AND r.start < ?",
            (H2D, first_kernel),
        ).fetchone()[0]

    live = peak = 0
    baseline = None
    nalloc = 0
    for start, nbytes, op in events:
        if baseline is None and t0 is not None and start > t0:
            baseline = live
        if op == ALLOC:
            live += nbytes
            nalloc += 1
            peak = max(peak, live)
        elif op == FREE:
            live -= nbytes
    if baseline is None:
        baseline = live if t0 is not None else 0
    return peak, baseline, peak - baseline, nalloc


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(*analyze(p))
