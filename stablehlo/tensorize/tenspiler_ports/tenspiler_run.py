"""Run one Tenspiler driver (synthesis, optionally Tenspiler's in-loop cvc5 check), then generate code
for every backend (NumPy, PyTorch, TensorFlow, MLX) and the TensIR program.

Usage: python tenspiler_run.py <driver.py> <benchmark_name> <codegen_out_dir> [--no-verify]
Run from a directory containing a `tenspiler` symlink to the Tenspiler repo's tenspiler/ (its
drivers use repo-relative .ll paths and write ./synthesisLogs/), with the Tenspiler checkout on
PYTHONPATH. For the blas/dsp ports, the driver's directory holds a *_ext.py module that adds a TensIR
op (vec_outer / matrix_matrix_mul); it is imported here so the code generators know the op.
"""
import importlib
import json
import os
import runpy
import sys
import time

import tenspiler.utils.synthesis_utils as su
from tenspiler.codegen.numpy_codegen import numpy_codegen
from tenspiler.codegen.pytorch_codegen import pytorch_codegen
from tenspiler.codegen.tensorflow_codegen import tensorflow_codegen
from tenspiler.codegen.mlx_codegen import mlx_codegen

driver_path, bench, outdir = sys.argv[1:4]
verify = "--no-verify" not in sys.argv
driver_dir = os.path.dirname(os.path.abspath(driver_path))
sys.path.insert(0, driver_dir)
for ext in ("blas_ext", "dsp_ext"):
    if os.path.exists(os.path.join(driver_dir, ext + ".py")):
        importlib.import_module(ext)

orig = su.run_synthesis_algorithm
captured = {}


def patched(**kw):
    kw["no_verify"] = not verify
    captured["data_type"] = kw["data_type"]
    captured["driver"] = kw["driver"]
    t0 = time.time()
    orig(**kw)
    captured["synth_time"] = time.time() - t0


su.run_synthesis_algorithm = patched
t0 = time.time()
runpy.run_path(driver_path, run_name="__main__")
total = time.time() - t0

import pathlib

out = pathlib.Path(outdir)
out.mkdir(parents=True, exist_ok=True)
d = captured["driver"]
ps = d.get_actual_ps_fn_decl()
dt = captured["data_type"]
for name, fn, ext in [
    ("numpy", numpy_codegen, "py"),
    ("pytorch", pytorch_codegen, "py"),
    ("tensorflow", tensorflow_codegen, "py"),
    ("mlx", mlx_codegen, "py"),
]:
    try:
        code = fn(ps, d.synthesized_fns, dt)
        (out / f"{bench}_{name}.{ext}").write_text(code)
    except Exception as e:  # some backends do not support some ops
        print(f"CODEGEN {name} failed: {e!r}")
(out / f"{bench}_tensir.txt").write_text(
    "\n\n".join(str(f) for f in d.synthesized_fns.values())
)
print("RESULT " + json.dumps({"bench": bench, "verified": verify,
                              "synth_verify_s": captured.get("synth_time"), "total_s": total}))
