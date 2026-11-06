"""4D-Var cost gradient (as in run_assimilation.py), JAX-native vs Enzyme-JAX pipeline.
usage: enzyme_4dvar_grad.py CONFIG MODE(jax|enzyme|enzyme_twocall) STEPS"""
import sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp, jax.extend
from legoesm.da.control_vector import build_control_spec, control_to_increment
from legoesm.da.gen_be import GenBETransform, load_gen_be_params
from scripts.run.mpas_4dvar_single import common

import os
if os.environ.get("ENZYME_IR_DUMP"):
    import enzyme_ad.jax.primitives as _P
    _dump_dir = Path(os.environ["ENZYME_IR_DUMP"]); _dump_dir.mkdir(parents=True, exist_ok=True)
    class _DumpingCall:
        def __init__(self, inner): self._inner, self._n = inner, 0
        def __getattr__(self, k): return getattr(self._inner, k)
        def run_pass_pipeline(self, fns, source, pipeline):
            i = self._n; self._n += 1
            tag = "reverse" if "ReverseModeCombined" in pipeline else "forward" if "mode=ForwardMode" in pipeline else "primal"
            (_dump_dir / f"{i:02d}_{tag}_pipeline.txt").write_text(pipeline + "\n")
            (_dump_dir / f"{i:02d}_{tag}_input.mlir").write_text(str(source))
            name, out = self._inner.run_pass_pipeline(fns, source, pipeline)
            (_dump_dir / f"{i:02d}_{tag}_output.mlir").write_text(str(out))
            print(f"dumped enzyme call {i} ({tag}) to {_dump_dir}", flush=True)
            return name, out
    _P.enzyme_call = _DumpingCall(_P.enzyme_call)

cfg = common.load_config(Path(sys.argv[1])); mode = sys.argv[2]; steps = int(sys.argv[3])
mc, be, ob = cfg["model"], cfg["background_error"], cfg["observation"]
dt = float(mc["time_step_seconds"])
mesh, sigma, model = common.make_grid_sigma_model(mc)
bg = common.state_from_sample(Path(cfg["inputs"]["nmc_sample"]), str(cfg["background"]["sample_state"]))
params = load_gen_be_params(Path(cfg["inputs"]["gen_be_path"]))
params = params._replace(len_scale=params.len_scale * float(be["horizontal_length_scale_multiplier"]))
spec = build_control_spec(bg, grid=mesh, fields=("u", "v", "T", "p_s", "tracers"))
tr = GenBETransform(params, spec, mesh, n_diffusion_iter=int(be["diffusion_iterations"]))
sd = float(be["standard_deviation_multiplier"])
cell = common.nearest_cell(mesh, float(ob["latitude_degrees_north"]), float(ob["longitude_degrees_east"]))
yobs = common.t500(common.rollout(model, common.cell_state_to_edge_state(bg, mesh), dt, steps), sigma)[cell] + float(ob["increment_k"])
err = float(ob["error_standard_deviation_k"])

if os.environ.get("ENZYME_REPORT_CKPT") == "1":
    import enzyme_ad.jax.primitives as _P2
    class _Report:
        def __init__(self, inner): self._inner = inner
        def __getattr__(self, k): return getattr(self._inner, k)
        def run_pass_pipeline(self, fns, source, pipeline):
            name, out = self._inner.run_pass_pipeline(fns, source, pipeline)
            print(f"enzyme pipeline: input has enable_checkpointing={str(source).count('enzyme.enable_checkpointing')}; "
                  f"output has checkpoint_segment={out.count('checkpoint_segment')}, while={out.count('stablehlo.while')}", flush=True)
            return name, out
    _P2.enzyme_call = _Report(_P2.enzyme_call)

# JAX_CHECKPOINT=0: same time loop as common.rollout but without jax.checkpoint around the step
if os.environ.get("JAX_CHECKPOINT", "1") == "0":
    def rollout(m, s, dt_, n):
        return jax.lax.scan(lambda v, _: (m.step(v, dt_), None), s, None, length=n)[0]
else:
    rollout = common.rollout
_rollout_impl = rollout

def rollout(m, s, dt_, n):
    with jax.named_scope("time_rollout"):
        return _rollout_impl(m, s, dt_, n)

def forecast_operator(c):
    a = common.apply_cell_increment(bg, control_to_increment(sd * tr.sqrt_multiply(c), spec, bg))
    return common.t500(rollout(model, common.cell_state_to_edge_state(a, mesh), dt, steps), sigma)[cell]

def cost(c):
    r = forecast_operator(c) - yobs
    return 0.5 * jnp.sum(c * c) + 0.5 * (r / err) ** 2

# Hoisting a broadcast that grows its operand out of the time loop turns it into a full-size
# loop-carried buffer: +2.4 GB and ~1.5x slower at 10 steps on enzyme-ad 0.0.16.
# ENZYME_EXCLUDED_PASSES adds further comma-separated patterns to skip.
EXCLUDED = ["broadcastindim_licm"] + [p for p in os.environ.get("ENZYME_EXCLUDED_PASSES", "").split(",") if p]

if mode == "enzyme_twocall":
    # enzyme_jax_ir + jax.value_and_grad: Enzyme-JAX emits a primal call and a separate reverse call.
    import functools
    import enzyme_ad.jax.primitives as _EP
    from enzyme_ad.jax import enzyme_jax_ir
    from enzyme_ad.jax.primitives import JaXPipeline, full_optimization_pass_pipeline
    # its AD lowering calls module-level optimization_passes() with defaults; make that skip the pattern too
    _EP.optimization_passes = functools.partial(_EP.optimization_passes, excluded_passes=EXCLUDED)
    cost = enzyme_jax_ir(pipeline_options=JaXPipeline(full_optimization_pass_pipeline(excluded_passes=EXCLUDED)))(cost)
elif mode == "enzyme":
    # One Enzyme reverse-mode call returning (cost, gradient). enzyme-wrap returns a primal only for
    # enzyme_dup returns and seeds only enzyme_active ones, so the cost is returned twice: (dup, active).
    from enzyme_ad.jax import ffi_call
    from enzyme_ad.jax.primitives import JaXPipeline, optimization_passes, LANG_MHLO
    _opt = optimization_passes(excluded_passes=EXCLUDED)
    _passes = ("inline{default-pipeline=canonicalize max-iterations=4}," + _opt + ",cse,"
               "enzyme-wrap{infn=main outfn= retTys=enzyme_dup,enzyme_active argTys=enzyme_active mode=ReverseModeCombined},"
               "arith-raise{stablehlo=true},enzyme-batch-to-stablehlo,canonicalize,remove-unnecessary-enzyme-ops,"
               "enzyme-simplify-math," + _opt + ",canonicalize,cse")
    if os.environ.get("ENZYME_BINOMIAL") == "1":
        # binomial checkpointing emits enzyme.binomial_progress; Enzyme-JAX's python pipelines never lower it
        _passes += ",lower-enzymexla-math,canonicalize,cse"
    _primal_cost = cost

    def _cost_twice(c):
        v = _primal_cost(c)
        return v, v

    _LOOP_CKPT = os.environ.get("ENZYME_LOOP_CKPT") == "1"
    _ir_cache = {}

    def _source_with_loop_checkpointing(c):
        # Lower the cost and tag the time-stepping stablehlo.while with Enzyme's loop-checkpointing attributes.
        from jax._src.lib.mlir import ir
        key = (c.shape, str(c.dtype))
        if key not in _ir_cache:
            mod = jax.jit(_cost_twice).lower(jax.ShapeDtypeStruct(c.shape, c.dtype)).compiler_ir("stablehlo")
            whiles = []
            def walk(op):
                for r in op.regions:
                    for b in r.blocks:
                        for o in b.operations:
                            if o.operation.name == "stablehlo.while":
                                whiles.append(o)
                            walk(o)
            walk(mod.operation)
            loc = lambda o: str(o.location)
            tgt = [o for o in whiles if "time_rollout/while" in loc(o) and "time_rollout/while/body" not in loc(o)]
            if len(tgt) != 1:
                raise RuntimeError(f"expected exactly one time-rollout while loop, found {len(tgt)} of {len(whiles)}")
            with mod.context:
                tgt[0].attributes["enzyme.enable_checkpointing"] = ir.BoolAttr.get(True)
                if os.environ.get("ENZYME_BINOMIAL") == "1":
                    tgt[0].attributes["enzyme.binomial_checkpointing"] = ir.UnitAttr.get()
                if os.environ.get("ENZYME_CHECKPOINT_PERIOD"):
                    tgt[0].attributes["enzyme.checkpoint_period"] = ir.IntegerAttr.get(
                        ir.IntegerType.get_signless(64), int(os.environ["ENZYME_CHECKPOINT_PERIOD"]))
            _ir_cache[key] = mod.operation.get_asm(enable_debug_info=True)
            print(f"tagged time loop for Enzyme checkpointing (period={os.environ.get('ENZYME_CHECKPOINT_PERIOD', 'default')}); "
                  f"{len(whiles)} while loops in module", flush=True)
        return _ir_cache[key]

    def _enzyme_value_and_grad(c):
        mfunc = _source_with_loop_checkpointing(c) if _LOOP_CKPT else _cost_twice
        source = (jax.tree_util.tree_structure(((c,), {})), ((0, 0),), ((0, -1), (1, 0)), mfunc, ())
        val, grad = ffi_call(
            c, jnp.ones((), c.dtype),
            out_shapes=(jax.core.ShapedArray((), c.dtype), jax.core.ShapedArray(c.shape, c.dtype)),
            source=source, fn="", argv=(), lang=LANG_MHLO, pipeline_options=JaXPipeline(_passes))
        return val, grad

    @jax.custom_vjp
    def cost(c):
        return _enzyme_value_and_grad(c)[0]

    def _fwd(c):
        return _enzyme_value_and_grad(c)

    def _bwd(grad, ct):
        return (ct * grad,)

    cost.defvjp(_fwd, _bwd)
elif mode != "jax":
    raise ValueError(f"unknown mode {mode!r} (jax | enzyme | enzyme_twocall)")

x = jnp.asarray(np.random.default_rng(0).standard_normal(spec.total_size) * 0.1)

def walk(jaxpr, prims, modes):
    for eqn in jaxpr.eqns:
        prims.add(eqn.primitive.name)
        if eqn.primitive.name.startswith("enzyme_"):
            passes = eqn.params["pipeline_options"].pass_pipeline()
            modes.append("ReverseModeCombined" if "ReverseModeCombined" in passes else
                         "ForwardMode" if "mode=ForwardMode" in passes else "no-AD (primal)")
        else:
            for sub in jax.extend.core.jaxprs_in_params(eqn.params):
                walk(sub, prims, modes)

prims, modes = set(), []
walk(jax.make_jaxpr(jax.value_and_grad(cost))(x).jaxpr, prims, modes)
print(f"mode={mode} gradient program primitives (outside enzyme): {sorted(prims)}; enzyme calls: {modes}", flush=True)
if mode.startswith("enzyme"):
    leaked = prims & {"scan", "while", "checkpoint", "remat", "cond", "scatter-add", "scatter_add", "gather", "dot_general", "transpose"}
    if "ReverseModeCombined" not in modes or leaked:
        raise SystemExit(f"CHECK FAILED: enzyme calls={modes} jax-differentiated model ops={sorted(leaked)}")
    print("CHECK OK: gradient computed by Enzyme reverse mode; no model loop/arithmetic differentiated by JAX", flush=True)

vg = jax.jit(jax.value_and_grad(cost))
if os.environ.get("COMPILE_ONLY") == "1":
    _m = vg.lower(x).compile().memory_analysis()
    G = 2**30
    print(f"MEMPLAN mode={mode} steps={steps} loop_ckpt={os.environ.get('ENZYME_LOOP_CKPT','0')}/{os.environ.get('ENZYME_CHECKPOINT_PERIOD','default')}{'/binomial' if os.environ.get('ENZYME_BINOMIAL')=='1' else ''} "
          f"temp={_m.temp_size_in_bytes/G:.2f} args={_m.argument_size_in_bytes/G:.2f} out={_m.output_size_in_bytes/G:.2f} "
          f"generated_code={_m.generated_code_size_in_bytes/G:.3f} GiB", flush=True)
    raise SystemExit(0)
if os.environ.get("IR_DUMP_DIR"):
    _d = Path(os.environ["IR_DUMP_DIR"]); _d.mkdir(parents=True, exist_ok=True)
    _tag = f"{mode}_{steps}steps_ckpt{os.environ.get('JAX_CHECKPOINT', '1')}"
    _low = vg.lower(x)
    (_d / f"{_tag}.stablehlo.mlir").write_text(
        _low.compiler_ir("stablehlo").operation.get_asm(large_elements_limit=16, enable_debug_info=False))
    (_d / f"{_tag}.xla_optimized.hlo").write_text(_low.compile().as_text())
    print(f"dumped IR for {_tag} to {_d}", flush=True)
t0 = time.time(); v, g = vg(x); jax.block_until_ready(g); t1 = time.time()
v, g = vg(x); jax.block_until_ready(g); t2 = time.time()
if os.environ.get("JAX_PROFILE_DIR"):
    with jax.profiler.trace(os.path.join(os.environ["JAX_PROFILE_DIR"], f"{mode}_{steps}steps"), create_perfetto_trace=True):
        for _ in range(int(os.environ.get("PROFILE_CALLS", "3"))):
            jax.block_until_ready(vg(x))
peak = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30
_res = Path(os.environ.get("RESULTS_DIR", "results")); _res.mkdir(parents=True, exist_ok=True)
np.save(_res / f"grad_{os.environ.get('RUN_TAG', mode)}_{steps}.npy", np.asarray(g))
print(f"mode={mode} loop_ckpt={os.environ.get('ENZYME_LOOP_CKPT','0')}/{os.environ.get('ENZYME_CHECKPOINT_PERIOD','default')}{'/binomial' if os.environ.get('ENZYME_BINOMIAL')=='1' else ''} excluded={EXCLUDED} jax_checkpoint={os.environ.get('JAX_CHECKPOINT', '1')} steps={steps} cost={float(v):.12g} |grad|={float(jnp.linalg.norm(g)):.12g} first_call={t1-t0:.1f}s second_call={t2-t1:.2f}s peak={peak:.2f}GiB", flush=True)
