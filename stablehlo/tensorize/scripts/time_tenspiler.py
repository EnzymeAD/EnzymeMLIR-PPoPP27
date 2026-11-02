"""Time Tenspiler's generated kernels (llama, blas, dsp, blend) under JAX on the default device, with the
same protocol as compare_suite: jit, 1 warm-up + 5 timed calls, first discarded, harmonic mean of the
other 4 (samples_ms lists all 5). Tenspiler has no JAX backend: its generated NumPy code is run with
`numpy` replaced by `jax.numpy`, size/position arguments static. Sizes match the Tensorize kernels
(n = 1000, 1000x1000); llama's attention kernels (no Tensorize counterpart) use 50 tokens, head 3,
head size 64, a 100x512 cache. For dsp, outputs are also checked against a float64 reference of the
Tensorize MLIR (max_rel_err_vs_mlir).

Usage: time_tenspiler.py codegen_dir {llama|blas|dsp|blend} > out.csv
"""
import pathlib
import statistics
import sys
import time
import types

import jax

jax.config.update("jax_enable_x64", True)
import numpy as onp  # noqa: E402

CG = pathlib.Path(sys.argv[1])  # directory of <kernel>_numpy.py files
SUITE = sys.argv[2] if len(sys.argv) > 2 else "llama"
N = 1000
rng = onp.random.default_rng(0)
v = lambda n=N: rng.standard_normal(n)  # noqa: E731
blas_cases = {
    "dot": ((v(), v()), ()),
    "gemv": ((rng.standard_normal((N, N)), v()), ()),
    "gemv_as_mlir": ((rng.standard_normal((N, N)), v()), ()),
    "ger": ((rng.standard_normal((N, N)), v(), v()), ()),
}
M = lambda: rng.standard_normal((N, N))  # noqa: E731
# dsp: (args, static, float64 reference from the MLIR semantics)
dsp_cases = {
    "matadd": ((M(), M()), (), lambda A, B: A + B),
    "matinit": ((M(), 0.7), (), lambda A, c: onp.full_like(A, c)),
    "matmul": ((M(), M()), (), lambda A, B: A @ B),
    "matscal": ((M(), 0.7), (), lambda A, s: A * s),
    "matsub": ((M(), M()), (), lambda A, B: A - B),
    "vadd": ((v(), v()), (), lambda a, b: a + b),
    "vcopy": ((v(),), (), lambda a: a.copy()),
    "vfill": ((v(), 0.7), (), lambda a, c: onp.full_like(a, c)),
    "vmul": ((v(), v()), (), lambda a, b: a * b),
    "vneg": ((v(),), (), lambda a: -a),
    "voffset": ((v(), 0.7), (), lambda a, c: a + c),
    "vrecip": ((v(),), (), lambda a: 1.0 / a),
    "vscal": ((v(), 0.7), (), lambda a, s: a * s),
    "vsub": ((v(), v()), (), lambda a, b: a - b),
    "w_vec": ((v(), v(), 0.7), (), lambda a, b, s: s * a + b),
}
P = lambda *sh: rng.integers(0, 33, sh).astype(onp.float64)  # integer-valued pixels in [0, 32]
blend_cases = {k: ((P(N, N), P(N, N)), ()) for k in (
    "color_burn_8", "color_dodge_8", "darken_blend_8", "lighten_blend_8", "linear_burn_8", "linear_dodge_8",
    "multiply_blend_8", "overlay_blend_8", "screen_blend_8")}
blend_cases["dissolve_blend_8"] = ((P(N, N), P(N, N), 0.5, 99.0), ())
blend_cases["normal_blend_8"] = ((P(N), P(N), 13.0), ())
blend_cases["normal_blend_f"] = ((P(N), P(N), 0.37), ())
cases = {
    "matmul": ((rng.standard_normal((N, N)), v()), ()),
    "rmsnorm_part1": ((v(), v()), ()),
    "rmsnorm_part2": ((v(), v(), 37.5), ()),
    "softmax_part1": ((v(), N), (1,)),
    "softmax_part2": ((v(), N, 0.3), (1,)),
    "softmax_part3": ((v(), N), (1,)),
    "softmax_part4": ((v(), N, 2.5), (1,)),
    "transformer_part1": ((50, 3, 64, rng.standard_normal((100, 512)), v(512)), (0, 1, 2)),
    "transformer_part2": ((50, 3, 64, rng.standard_normal((100, 512)), v(100)), (0, 1, 2)),
    "transformer_part3": ((v(), N), (1,)),
    "transformer_part4": ((v(), v(), N), (2,)),
}
refs = {}
if SUITE == "blas":
    cases = blas_cases
elif SUITE == "blend":
    cases = blend_cases
elif SUITE == "dsp":
    cases = {k: c[:2] for k, c in dsp_cases.items()}
    refs = {k: c[2] for k, c in dsp_cases.items()}
print("kernel,compile_s,hmean_ms,samples_ms" + (",max_rel_err_vs_mlir" if refs else ""))
for b, (args, static) in cases.items():
    p = CG / f"{b}_numpy.py"
    if not p.exists():
        print(f"{b},no codegen,,")
        continue
    m = types.ModuleType(b)
    exec(compile(p.read_text().replace("import numpy as np", "import jax.numpy as np"), str(p), "exec"), m.__dict__)
    jargs = tuple(a if isinstance(a, (int, float)) else jax.device_put(a) for a in args)
    dyn = [a for i, a in enumerate(jargs) if i not in static]
    fn = jax.jit(getattr(m, f"{b}_np"), static_argnums=static)
    s = time.perf_counter()
    lowered = fn.lower(*jargs).compile()
    c = time.perf_counter() - s
    call = lambda: lowered(*dyn)  # noqa: E731
    jax.block_until_ready(call())
    t = []
    for _ in range(5):
        s = time.perf_counter()
        jax.block_until_ready(call())
        t.append((time.perf_counter() - s) * 1e3)
    err = ""
    if b in refs:
        want = refs[b](*args)
        got = onp.asarray(call(), dtype=onp.float64)
        err = "," + (f"{onp.max(onp.abs(got - want)) / max(onp.max(onp.abs(want)), 1e-300):.2e}"
                     if got.shape == want.shape else f"shape {got.shape} vs {want.shape}")
    print(f"{b},{c:.3f},{statistics.harmonic_mean(t[1:]):.6g},{' '.join(f'{x:.4g}' for x in t)}{err}", flush=True)
