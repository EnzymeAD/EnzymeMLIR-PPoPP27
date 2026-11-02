"""Compare Tensorize programs with Enzyme-JAX raised StableHLO on a benchmark suite.

For every kernel, all available variants run in one process on the same inputs:
  tensorize_jax    Tensorize's program with np = jax.numpy, under jax.jit
  tensorize_numpy  the same program with real NumPy
  enzyme_while     out/raised/<k>.while.mlir via enzyme_ad.jax.hlo_call, under jax.jit
  enzyme_unroll    out/raised/<k>.unroll.mlir, likewise
Calls are interleaved round-robin across variants so background load affects them
equally. Pin the process with taskset; thread pools are sized to the pinned CPU set.
Run from .venv-enzyme (enzyme-ad needs jax>=0.8).

Usage:
  taskset -c 0-15 python benchmark/compare_suite.py [benchmark/polybench/*.mlir]
"""
import os

# Size BLAS/OpenMP pools to the CPUs we're pinned to (XLA already respects affinity).
_ncpu = str(len(os.sched_getaffinity(0)))
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, _ncpu)

import argparse  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy  # noqa: E402
from enzyme_ad.jax import hlo_call  # noqa: E402

from jax_harness import instantiate, load_program, make_inputs, read_signature  # noqa: E402

BENCH_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(BENCH_DIR)


def leaves(x):
    return [numpy.asarray(o) for o in jax.tree_util.tree_leaves(x)]


def rel_err(a, b):
    return float(numpy.max(numpy.abs(a - b) / (numpy.abs(b) + 1e-12), initial=0.0))


def read_broken(kernel_path):
    """Map kernel -> reason from the tensorize_broken.txt next to a kernel, if any."""
    table = os.path.join(os.path.dirname(kernel_path), "tensorize_broken.txt")
    if not os.path.exists(table):
        return {}
    lines = [line.split(None, 1) for line in open(table) if line.strip() and not line.startswith("#")]
    return {k: reason.strip() for k, reason in lines}


def build_variants(kernel, inputs, args, broken_reason=None):
    """Return {name: (callable, compile_s)} and {name: error string} for failures."""
    variants, failures = {}, {}

    synth = os.path.join(args.synth_dir, kernel + ".out")
    if broken_reason:
        failures["tensorize_jax"] = failures["tensorize_numpy"] = f"broken: {broken_reason}"
    elif os.path.exists(synth):
        src, fn_name = load_program(synth)
        try:
            start = time.perf_counter()
            fn = jax.jit(instantiate(src, fn_name, jnp, synth)).lower(*inputs).compile()
            variants["tensorize_jax"] = (fn, time.perf_counter() - start)
        except Exception as e:
            failures["tensorize_jax"] = f"{type(e).__name__}: {e}"
        if not args.no_numpy and "tensorize_jax" in variants:
            np_fn = instantiate(src, fn_name, numpy, synth)
            host = [numpy.asarray(x) for x in inputs]  # NumPy gets host arrays, converted once
            variants["tensorize_numpy"] = (lambda *_: np_fn(*host), 0.0)
    else:
        failures["tensorize_jax"] = "no synthesized program"

    for v in ("while",) if args.no_unroll else ("while", "unroll"):
        path = os.path.join(args.raised_dir, f"{kernel}.{v}.mlir")
        if not os.path.exists(path):
            failures[f"enzyme_{v}"] = "not raised"
            continue
        source = open(path).read()
        try:
            start = time.perf_counter()
            fn = jax.jit(lambda *a, s=source: hlo_call(*a, source=s)).lower(*inputs).compile()
            variants[f"enzyme_{v}"] = (fn, time.perf_counter() - start)
        except Exception as e:
            failures[f"enzyme_{v}"] = f"{type(e).__name__}: {str(e)[:200]}"
    return variants, failures


def time_interleaved(variants, inputs, warmup, reps):
    """Interleave the JAX variants round-robin; time NumPy afterwards on its own.

    OpenBLAS worker threads keep spinning after a NumPy call returns, which slows
    whatever XLA computation runs next on the same cores, so NumPy can't be interleaved.
    """
    groups = [[n for n in variants if n != "tensorize_numpy"], [n for n in variants if n == "tensorize_numpy"]]
    times = {name: [] for name in variants}
    for group in groups:
        for rep in range(warmup + reps):
            for name in group:
                start = time.perf_counter()
                jax.block_until_ready(variants[name][0](*inputs))
                if rep >= warmup:
                    times[name].append(time.perf_counter() - start)
    return times


def check(variants, inputs):
    """Errors of each enzyme variant vs Tensorize, and of unroll vs while."""
    outs = {name: leaves(fn(*inputs)) for name, (fn, _) in variants.items() if name != "tensorize_numpy"}
    errs = {}
    ref = outs.get("tensorize_jax")
    for name in ("enzyme_while", "enzyme_unroll"):
        if name not in outs:
            continue
        if ref is not None:
            # Tensorize returns one buffer; match it to the closest same-shaped output.
            cands = [rel_err(o, ref[0]) for o in outs[name] if o.shape == ref[0].shape]
            errs[(name, "vs_tensorize")] = min(cands) if cands else "shape mismatch"
        if name == "enzyme_unroll" and "enzyme_while" in outs:
            pairs = list(zip(outs["enzyme_unroll"], outs["enzyme_while"]))
            errs[(name, "vs_while")] = max(rel_err(a, b) for a, b in pairs) if pairs else ""
    return errs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kernels", nargs="*", help="original MLIR kernels (default: polybench)")
    parser.add_argument("--raised_dir", default=os.path.join(ROOT, "out/raised"))
    parser.add_argument("--synth_dir", default=os.path.join(ROOT, "out/synth"))
    parser.add_argument("--out", default=os.path.join(ROOT, "out/compare.csv"))
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--discard", type=int, default=0,
                        help="timed samples to drop from the start before computing the statistics "
                             "(still listed in samples_ms)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_numpy", action="store_true")
    parser.add_argument("--no_unroll", action="store_true", help="skip the enzyme_unroll variant")
    args = parser.parse_args()
    kernels = args.kernels or sorted(glob.glob(os.path.join(BENCH_DIR, "polybench", "*.mlir")))

    print(f"jax {jax.__version__} on {jax.devices()}, pinned to {_ncpu} CPUs", flush=True)
    fields = ["kernel", "variant", "status", "compile_s", "median_ms", "min_ms", "err_vs_tensorize", "err_vs_while",
              "hmean_ms", "samples_ms"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = open(args.out, "w", newline="")
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for path in kernels:
        print(f"{os.path.basename(path)}: compiling", flush=True)
        kernel = os.path.basename(path).rsplit(".", 1)[0]
        # Device-resident inputs, so calls don't include host-to-device copies (jit drops
        # unused args, hlo_call keeps them all, so copying would penalize one side).
        inputs = [jax.device_put(x) for x in make_inputs(read_signature(path), args.seed)]
        variants, failures = build_variants(kernel, inputs, args, read_broken(path).get(kernel))
        times = time_interleaved(variants, inputs, args.warmup, args.reps)
        errs = check(variants, inputs)

        for name in ("tensorize_jax", "tensorize_numpy", "enzyme_while", "enzyme_unroll"):
            row = {"kernel": kernel, "variant": name}
            if name in variants:
                t = times[name][args.discard:]
                row.update(status="ok", compile_s=variants[name][1],
                           median_ms=statistics.median(t) * 1e3, min_ms=min(t) * 1e3,
                           hmean_ms=statistics.harmonic_mean(t) * 1e3,
                           samples_ms=" ".join(f"{x * 1e3:.6g}" for x in times[name]))
            elif name in failures:
                row.update(status=failures[name])
            else:
                continue
            row["err_vs_tensorize"] = errs.get((name, "vs_tensorize"), "")
            row["err_vs_while"] = errs.get((name, "vs_while"), "")
            writer.writerow(row)
            out.flush()

            if row["status"] == "ok":
                err = row["err_vs_tensorize"]
                err = f"{err:.1e}" if isinstance(err, float) else err
                print(f"{kernel:12s} {name:16s} median {row['median_ms']:10.3f} ms  min {row['min_ms']:10.3f} ms"
                      f"  compile {row['compile_s']:6.2f} s  err {err}", flush=True)
            else:
                print(f"{kernel:12s} {name:16s} {row['status']}", flush=True)

    out.close()
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
