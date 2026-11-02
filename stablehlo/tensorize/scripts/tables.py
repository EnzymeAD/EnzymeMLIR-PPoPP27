"""Build markdown tables from a run ($OUT): per suite, Tensorize vs Enzyme-JAX (and Tenspiler where
it exists), with preparation time (synthesis / raising), XLA compile time and run time.

Run time = harmonic mean of the kept timed calls (see time.sh). T/E < 1: Tensorize faster.
Usage: python scripts/tables.py [OUT] > tables.md
"""
import csv
import glob
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OUT", os.path.join(HERE, "out"))
TZ_PROGRAMS = os.environ.get("TZ_PROGRAMS", os.path.join(HERE, "programs", "tensorize"))
TS_PROGRAMS = os.environ.get("TS_PROGRAMS", os.path.join(HERE, "programs", "tenspiler"))
SUITES = os.environ.get("SUITES", "polybench kiriko blas blend darknet dsp dspstone llama makespeare "
                        "mathfu simpl_array utdsp").split()
SKIP = set(os.environ.get("SKIP", "polybench/symm kiriko/symm kiriko/dynprog kiriko/floyd-warshall kiriko/cholesky kiriko/ludcmp kiriko/trmm").split())

# Tenspiler kernel computing what our kernel computes, where the names differ.
TENSPILER_NAME = {("blas", "gemv"): "gemv_as_mlir"}
# Known differences between a Tensorize-suite kernel and Tenspiler's kernel of the same name
# (see KNOWN_ISSUES.md).
TENSPILER_DIFFERS = {
    ("llama", "rmsnorm_part2"): "different computation: ours divides by 3.0, not size",
    ("llama", "transformer_part3"): "different computation: ours is 1/(b(1+e^-a)), not SiLU",
    ("blend", "color_burn_8"): "different computation (our MLIR computes 32 - b/a)",
    ("blend", "color_dodge_8"): "different computation (our MLIR divides by 0 at a = 32)",
    ("blend", "dissolve_blend_8"): "different computation (our MLIR ignores `active`)",
    ("blend", "multiply_blend_8"): "Tenspiler divides as integers (floor), ours in float",
    ("blend", "screen_blend_8"): "Tenspiler divides as integers (floor), ours in float",
    ("blend", "overlay_blend_8"): "Tenspiler divides as integers (floor), ours in float",
}
# Kernels whose definition in the suite is not what its name says (see KNOWN_ISSUES.md).
KERNEL_NOTES = {
    ("blas", "gemv"): "kernel reads x[i], not x[j]: computes x[i]*sum_j A[i][j], not A·x",
    ("polybench", "covariance"): "suite runs it at 5x3",
}


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt(x, digits=3):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.{digits}g}"


def gmean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def timing(suite, kernel):
    """{variant: row} from compare_suite, plus a status for kernels without a CSV."""
    path = os.path.join(OUT, "timings", suite, f"{kernel}.csv")
    rows = {r["variant"]: r for r in read_csv(path)}
    log = path[:-4] + ".log"
    if not rows and os.path.exists(log):
        text = open(log).read()
        rows["_status"] = "timeout" if "timeout after" in text else "error (see log)"
    return rows


synth = {(r["suite"], r["kernel"]): r for r in read_csv(os.path.join(TZ_PROGRAMS, "synth_times.csv"))}
ts_synth = {(r["suite"], r["kernel"]): r for r in read_csv(os.path.join(TS_PROGRAMS, "synth_times.csv"))}
all_ratios = []
print(f"# Tensorize vs Enzyme-JAX{' vs Tenspiler' if glob.glob(os.path.join(OUT, 'timings', 'tenspiler_*.csv')) else ''}\n")
print("Times in ms (harmonic mean of the kept timed calls); prep and XLA compile in s. "
      "T/E < 1: Tensorize faster. — : not available (see status).\n")
for suite in SUITES:
    kernels = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(OUT, "benchmarks", suite, "*.mlir")))
    if not kernels:
        continue
    raise_s = {r["kernel"]: r for r in read_csv(os.path.join(OUT, "raised", suite, "raise_times.csv"))}
    tsp = {r["kernel"]: r for r in read_csv(os.path.join(OUT, "timings", f"tenspiler_{suite}.csv"))}
    cols = ["kernel", "Tensorize synth", "Tensorize XLA", "Tensorize ms", "Enzyme raise", "Enzyme XLA",
            "Enzyme ms", "T/E"]
    if tsp:
        cols += ["Tenspiler synth", "Tenspiler XLA", "Tenspiler ms", "Tenspiler/E"]
    cols += ["status / notes"]
    print(f"## {suite}\n")
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    ratios = []
    for k in kernels:
        notes = [KERNEL_NOTES[(suite, k)]] if (suite, k) in KERNEL_NOTES else []
        rows = timing(suite, k)
        one = timing(suite, f"{k}_1out")
        if one.get("enzyme_while"):  # like-for-like: both sides compute only Tensorize's output
            notes.append("both sides compute only Tensorize's output")
            rows = one
        t, e = rows.get("tensorize_jax", {}), rows.get("enzyme_while", {})
        sy = synth.get((suite, k), {})
        t_ok, e_ok = t.get("status") == "ok", e.get("status") == "ok"
        tms, ems = num(t.get("hmean_ms")) if t_ok else None, num(e.get("hmean_ms")) if e_ok else None
        r = tms / ems if tms and ems else None
        if r:
            ratios.append(r)
            all_ratios.append(r)
        if f"{suite}/{k}" in SKIP:
            notes.append("not timed ($SKIP: > 30 min per Enzyme-JAX call)")
        elif "_status" in rows:
            notes.append(rows["_status"])
        elif not rows:
            notes.append("not timed")
        if t and not t_ok:
            notes.append("Tensorize: " + t.get("status", ""))
        elif not t and sy.get("status") not in (None, "ok"):
            notes.append(f"Tensorize synthesis {sy['status']}")
        if e_ok and t_ok and e.get("err_vs_tensorize") not in ("", None):
            err = num(e["err_vs_tensorize"])
            if err is None or err > 1e-9:
                notes.append(f"outputs differ (rel err {e['err_vs_tensorize']})")
        rs = raise_s.get(k, {})
        if rs.get("status") not in (None, "ok"):
            notes.append("Enzyme-JAX raise " + rs.get("status", ""))
        prep_t = num(sy.get("wall_s")) if sy.get("status") == "ok" else None
        cells = [k, fmt(prep_t, 3), fmt(num(t.get("compile_s")) if t_ok else None, 2), fmt(tms),
                 fmt(num(rs.get("raise_s")), 2), fmt(num(e.get("compile_s")) if e_ok else None, 2), fmt(ems),
                 f"{r:.2f}" if r else ""]
        if tsp:
            name = TENSPILER_NAME.get((suite, k), k)
            s = tsp.get(name, {})
            sms = num(s.get("hmean_ms"))
            sr = sms / ems if sms and ems else None
            cells += [fmt(num(ts_synth.get((suite, name), {}).get("wall_s")), 3), fmt(num(s.get("compile_s")), 2),
                      fmt(sms), f"{sr:.2f}" if sr and (suite, k) not in TENSPILER_DIFFERS else (f"({sr:.2f})" if sr else "")]
            if (suite, k) in TENSPILER_DIFFERS:
                notes.append("Tenspiler: " + TENSPILER_DIFFERS[(suite, k)])
        print("| " + " | ".join(cells + ["; ".join(notes)]) + " |")
    if tsp:  # Tenspiler kernels with no counterpart in the suite
        ours = {TENSPILER_NAME.get((suite, k), k) for k in kernels}
        for name in sorted(set(tsp) - ours):
            s = tsp[name]
            cells = [f"{name} (Tenspiler only)"] + [""] * 7 + [
                fmt(num(ts_synth.get((suite, name), {}).get("wall_s")), 3), fmt(num(s.get("compile_s")), 2),
                fmt(num(s.get("hmean_ms"))), "", "no Tensorize/Enzyme-JAX counterpart"]
            print("| " + " | ".join(cells) + " |")
    if ratios:
        print(f"\nT/E over {len(ratios)} kernels: geomean {gmean(ratios):.2f}, median {statistics.median(ratios):.2f}\n")
    else:
        print()
if all_ratios:
    print(f"**All suites:** T/E over {len(all_ratios)} kernels: geomean {gmean(all_ratios):.2f}, "
          f"median {statistics.median(all_ratios):.2f}")
