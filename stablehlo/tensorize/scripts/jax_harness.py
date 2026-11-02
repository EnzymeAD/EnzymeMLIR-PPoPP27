"""Time Tensorize-synthesized programs under JAX.

Each synthesized program (the --synth_out file of tensorize/main.py) is executed with
`np` bound to `jax.numpy` and jitted. Inputs are random arrays shaped after the
original MLIR kernel's signature. The same program is also run with `np` bound to real
NumPy, as a baseline and to cross-check the JAX results.

Usage:
  python benchmark/jax_harness.py --synth out/synth/gemm.out \
      --program benchmark/polybench/gemm.mlir
  python benchmark/jax_harness.py --synth_dir out/synth   # all synthesized programs
"""
import argparse
import ast
import csv
import glob
import os
import re
import statistics
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy  # noqa: E402

BENCH_DIR = os.path.dirname(os.path.realpath(__file__))
DTYPES = {"f64": "float64", "f32": "float32", "i64": "int64", "i32": "int32"}


def read_signature(mlir_path):
    """Return [(type, attrs)] for the args of the first func.func in an MLIR file.

    attrs is the arg's attribute dict text, e.g. "{irsynth.symmetric}", or "".
    """
    src = open(mlir_path).read()
    m = re.search(r"func\.func\s+@[\w.$-]+\((.*?)\)\s*(?:->|attributes|\{)", src, re.S)
    return re.findall(r"%\w+:\s*(memref<[^>]*>|[if]\d+|index)\s*(\{[^}]*\})?", m.group(1))


def make_inputs(signature, seed):
    """Random inputs for a read_signature() result, honoring Tensorize's structure
    annotations (irsynth.symmetric, irsynth.lower_triangular) on matrix args."""
    rng = numpy.random.default_rng(seed)
    inputs = []
    for ty, attrs in signature:
        if ty.startswith("memref<"):
            *dims, elem = ty[len("memref<"):-1].split("x")
            shape = tuple(int(d) for d in dims)
        else:
            shape, elem = (), ty
        dtype = numpy.dtype(DTYPES.get(elem, "int64"))
        if dtype.kind == "f":
            value = rng.uniform(0.5, 1.5, size=shape).astype(dtype)
            if "irsynth.symmetric" in attrs:
                value = numpy.tril(value) + numpy.tril(value, -1).T
            elif "irsynth.lower_triangular" in attrs:
                # Unit diagonal, as PolyBench's trmm init_array sets A[i][i] = 1; the loop
                # never reads the diagonal, but Tensorize's synthesized program assumes it.
                value = numpy.tril(value, -1) + numpy.eye(*shape, dtype=dtype)
        else:
            value = numpy.ones(shape, dtype=dtype)
        inputs.append(value if shape else dtype.type(value))
    return inputs


def load_program(path):
    """Parse a synthesized program and return (source, entry function name).

    Works around a Tensorize bug where normalize_identifiers renames the entry
    function's parameters v1..vN to arg0..argN-1 but not their uses in the body.
    """
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)][-1]
    params = [a.arg for a in fn.args.args]
    stored = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    stale = {
        n.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Name) and re.fullmatch(r"v\d+", n.id) and n.id not in stored | set(params)
    }
    if stale and all(re.fullmatch(r"arg\d+", p) for p in params):
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and n.id in stale:
                n.id = "arg%d" % (int(n.id[1:]) - 1)
    return ast.unparse(tree), fn.name


def instantiate(src, name, np_module, filename):
    env = {"np": np_module}
    exec(compile(src, filename, "exec"), env)
    return env[name]


def time_calls(fn, args, reps, sync):
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        sync(fn(*args))
        times.append(time.perf_counter() - start)
    return times


def max_rel_err(a, b):
    errs = [0.0]
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        x, y = numpy.asarray(x, dtype=numpy.float64), numpy.asarray(y, dtype=numpy.float64)
        errs.append(float(numpy.max(numpy.abs(x - y) / (numpy.abs(y) + 1e-12), initial=0.0)))
    return max(errs)


def run_one(synth_path, program_path, args):
    name = os.path.basename(synth_path).rsplit(".", 1)[0]
    src, fn_name = load_program(synth_path)
    inputs = make_inputs(read_signature(program_path), args.seed)
    rows = []

    # JAX: compile once, then time the compiled executable.
    jax_fn = instantiate(src, fn_name, jnp, synth_path)
    dev_inputs = [jax.device_put(x) for x in inputs]
    start = time.perf_counter()
    compiled = jax.jit(jax_fn).lower(*dev_inputs).compile()
    compile_s = time.perf_counter() - start
    time_calls(compiled, dev_inputs, args.warmup, jax.block_until_ready)
    jax_times = time_calls(compiled, dev_inputs, args.reps, jax.block_until_ready)
    jax_out = compiled(*dev_inputs)
    rows.append((name, "jax", jax_times, compile_s))

    err = ""
    if not args.no_numpy:
        np_fn = instantiate(src, fn_name, numpy, synth_path)
        time_calls(np_fn, inputs, args.warmup, lambda r: r)
        np_times = time_calls(np_fn, inputs, args.reps, lambda r: r)
        rows.append((name, "numpy", np_times, 0.0))
        err = max_rel_err(jax_out, np_fn(*inputs))

    return [
        {
            "benchmark": bench,
            "variant": variant,
            "device": jax.devices()[0].platform if variant == "jax" else "cpu",
            "time_median_s": statistics.median(times),
            "time_min_s": min(times),
            "compile_s": compile_s,
            "max_rel_err_vs_numpy": err,
        }
        for bench, variant, times, compile_s in rows
    ]


def find_program(synth_path):
    name = os.path.basename(synth_path).rsplit(".", 1)[0]
    matches = glob.glob(os.path.join(BENCH_DIR, "**", name + ".mlir"), recursive=True)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {name}.mlir under {BENCH_DIR}, found {matches}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synth", help="synthesized program (tensorize --synth_out)")
    parser.add_argument("--program", help="original MLIR kernel (default: looked up under benchmark/)")
    parser.add_argument("--synth_dir", help="run every *.out program in this directory")
    parser.add_argument("--out", default="out/jax_times.csv", help="CSV to write results to")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_numpy", action="store_true", help="skip the NumPy baseline")
    args = parser.parse_args()

    if args.synth_dir:
        jobs = [(p, find_program(p)) for p in sorted(glob.glob(os.path.join(args.synth_dir, "*.out")))]
    elif args.synth:
        jobs = [(args.synth, args.program or find_program(args.synth))]
    else:
        parser.error("pass --synth or --synth_dir")

    print(f"JAX devices: {jax.devices()}")
    results = []
    for synth_path, program_path in jobs:
        try:
            rows = run_one(synth_path, program_path, args)
        except Exception as e:  # keep going over the rest of the suite
            print(f"{os.path.basename(synth_path):24s} FAILED: {type(e).__name__}: {e}")
            continue
        for r in rows:
            print(
                f"{r['benchmark']:24s} {r['variant']:6s} median {r['time_median_s'] * 1e3:10.3f} ms"
                + (f"  compile {r['compile_s']:.2f} s" if r["variant"] == "jax" else "")
                + (f"  max rel err {r['max_rel_err_vs_numpy']:.2e}" if r["variant"] == "numpy" else "")
            )
        results += rows

    if results:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
