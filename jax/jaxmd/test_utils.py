import csv
import glob
import os

import jax
import jax.numpy as jnp
from absl.testing import absltest

from xprof_utils import profile_compiled_function


def get_cpp_include_args():
    cpp_dir = "/usr/include/c++"
    if not os.path.isdir(cpp_dir):
        return ()

    versions = [d for d in os.listdir(cpp_dir) if d.isdigit()]
    if not versions:
        return ()

    latest_version = sorted(versions, key=int)[-1]
    paths = [os.path.join(cpp_dir, latest_version)]

    arch_dirs = glob.glob(f"/usr/include/*-linux-gnu/c++/{latest_version}")
    if arch_dirs:
        paths.append(arch_dirs[0])
    else:
        # Fallback to checking for specific known dirs if glob fails
        arch = platform.machine()
        if arch == "x86_64" and os.path.isdir(
            f"/usr/include/x86_64-linux-gnu/c++/{latest_version}"
        ):
            paths.append(f"/usr/include/x86_64-linux-gnu/c++/{latest_version}")
        elif arch == "aarch64" and os.path.isdir(
            f"/usr/include/aarch64-linux-gnu/c++/{latest_version}"
        ):
            paths.append(f"/usr/include/aarch64-linux-gnu/c++/{latest_version}")

    return tuple(f"-I{p}" for p in paths)


argv = get_cpp_include_args()


partialopt = (
    "inline{default-pipeline=canonicalize max-iterations=4}," + """canonicalize,cse,
enzyme-hlo-generate-td{
            patterns=compare_op_canon<16>;
transpose_transpose<16>;
broadcast_in_dim_op_canon<16>;
convert_op_canon<16>;
dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;
chained_dynamic_broadcast_in_dim_canonicalization<16>;
dynamic_broadcast_in_dim_all_dims_non_expanding<16>;
noop_reduce_op_canon<16>;
empty_reduce_op_canon<16>;
dynamic_reshape_op_canon<16>;
get_tuple_element_op_canon<16>;
real_op_canon<16>;
imag_op_canon<16>;
get_dimension_size_op_canon<16>;
gather_op_canon<16>;
reshape_op_canon<16>;
merge_consecutive_reshapes<16>;
transpose_is_reshape<16>;
zero_extent_tensor_canon<16>;
reorder_elementwise_and_shape_op<16>;

cse_broadcast_in_dim<16>;
cse_slice<16>;
cse_transpose<16>;
cse_convert<16>;
cse_pad<16>;
cse_dot_general<16>;
cse_reshape<16>;
cse_mul<16>;
cse_div<16>;
cse_add<16>;
cse_subtract<16>;
cse_min<16>;
cse_max<16>;
cse_neg<16>;
cse_concatenate<16>;

concatenate_op_canon<16>(1024);
select_op_canon<16>(1024);
add_simplify<16>;
sub_simplify<16>;
and_simplify<16>;
reduce_or_and_pad;
max_simplify<16>;
min_simplify<16>;
or_simplify<16>;
mul_simplify<16>;
div_simplify<16>;
rem_simplify<16>;
pow_simplify<16>;
noop_slice<16>;
const_prop_through_barrier<16>;
slice_slice<16>;
shift_right_logical_simplify<16>;
pad_simplify<16>(1024);
negative_pad_to_slice<16>;
slice_simplify<16>;
convert_simplify<16>;
dynamic_slice_to_static<16>;
dynamic_update_slice_elim<16>;
concat_to_broadcast<16>;
reduce_to_reshape<16>;
broadcast_to_reshape<16>;
iota_simplify<16>(1024);
broadcast_in_dim_simplify<16>(1024);
convert_concat<1>;
dynamic_update_to_concat<1>;
slice_of_dynamic_update<1>;
slice_elementwise<1>;
slice_pad<1>;
dot_reshape_dot<1>;
concat_const_prop<1>(1024);
reduce_or_and;
concat_fuse<1>;
pad_reshape_pad<1>;
pad_pad<1>;
concat_push_binop_add<1>;
concat_push_binop_mul<1>;
scatter_to_dynamic_update_slice<1>;
reduce_concat<1>;
slice_concat<1>;

bin_broadcast_splat_add<1>;
bin_broadcast_splat_subtract<1>;
bin_broadcast_splat_div<1>;
bin_broadcast_splat_mul<1>;
slice_reshape<1>;

dot_reshape_pad<1>;
pad_dot_general<1>(1);
broadcast_reduce<1>;
            },
            transform-interpreter,
            enzyme-hlo-remove-transform,cse"""
)


def pipelines():
    from enzyme_ad.jax import JaXPipeline, full_optimization_pass_pipeline

    inline = "inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4},"
    return [
        ("JaXPipe", JaXPipeline()),
        ("Jax", None),
        ("HLOOpt", JaXPipeline(inline + "canonicalize,cse,enzyme-hlo-opt,cse")),
        ("PartOpt", JaXPipeline(partialopt)),
        ("IPartOpt", JaXPipeline(inline + partialopt)),
        ("DefOpt", JaXPipeline(full_optimization_pass_pipeline(inline=False))),
        ("IDefOpt", JaXPipeline(full_optimization_pass_pipeline())),
    ]


def splatjvp(in_fn):
    def fwd(*args):
        n = len(args) // 2
        return jax.jvp(in_fn, tuple(args[:n]), tuple(args[n:]))

    return fwd


def splatvjp(in_fn):
    def rev(dout, *args):
        primals, f_vjp = jax.vjp(in_fn, *args)
        return primals, f_vjp(dout)

    return rev


def to_backend(x, backend):
    return jax.device_put(x, jax.local_devices(backend=backend)[0])


def recursive_check(tester, lhs, rhs, pname):
    def leaves_allclose(leaf1, leaf2):
        legal = jnp.allclose(leaf1, leaf2, atol=tester.atol, rtol=tester.rtol)
        if not legal:
            print("lhs (", pname, ")", leaf1)
            print("rhs", leaf2)
            print("abs", jnp.abs(leaf1 - leaf2))
            print("eq", jnp.abs(leaf1 - leaf2) < tester.atol)
            print("max", jnp.max(jnp.abs(leaf1 - leaf2)))
        tester.assertTrue(legal)
        return legal

    tester.assertTrue(jax.tree.all(jax.tree.map(leaves_allclose, lhs, rhs)))


class EnzymeJaxTest(absltest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeat = 50
        self.atol = 1e-6
        self.rtol = 0.0
        self.results = []

    def setUp(self):
        self.name = None

    def test(self):
        # main.py imports this class, so absltest also collects it; skip it.
        if self.name is None:
            return
        self.harness(self.name, self.fn, self.ins, self.dins, self.douts)
        self.write_results_csv()

    def record(self, name, pname, backend, key, timings):
        row = [
            name,
            pname,
            backend,
            key,
            timings["avg_time_s"],
            timings["stdev_time_s"],
            timings["min_time_s"],
            timings["max_time_s"],
            timings["peak_mem_bytes"],
        ]
        print(
            "{:<20}\t{:<20}\t{:<15}\t{:<10}\t{:<15.8f}\t{:<15.8f}\t{:<15.8f}\t{:<15.8f}\t{:<15.0f}".format(
                *row
            ),
            flush=True,
        )
        self.results.append(row)

    def write_results_csv(self, filename=""):
        if filename == "":
            filename = f"results_{self.__class__.__name__}.csv"

        outdir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", ".")
        outfile = os.path.join(outdir, filename)
        with open(outfile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Benchmark Name", "Pass Pipeline", "Backend", "Key", "Avg Time", "stdev Time", "Min Time", "Max Time", "Peak Memory (bytes)"]
            )
            writer.writerows(self.results)
        print(f"Results written to {outfile}")

    def harness(self, name, in_fn, ins, dins, douts):
        from enzyme_ad.jax import enzyme_jax_ir

        # Pipelines only run on the default backend
        backend = jax.default_backend()
        if backend == "gpu" and "gpu" not in jax._src.xla_bridge.backends():
            backend = "cuda"
        all_pipelines = pipelines()

        ins_b = [to_backend(x, backend) for x in ins]
        dins_b = [to_backend(x, backend) for x in dins]
        if isinstance(douts, jax.Array):
            douts_b = to_backend(douts, backend)
        else:
            douts_b = tuple(to_backend(x, backend) for x in douts)

        # First result of each kind is the reference the others are checked against
        refs = {}

        def check(kind, res, pname):
            if kind in refs:
                recursive_check(self, res, refs[kind], kind + " " + pname)
            else:
                refs[kind] = res

        def compile_and_run(fn, args):
            compiled = jax.jit(fn).trace(*args).lower().compile()
            return compiled, compiled(*args)

        def time(compiled, args, pname, key):
            timings = profile_compiled_function(compiled, args, nrepeat=self.repeat)
            self.record(name, pname, backend, key, timings)

        def enzyme(pipeline, fn):
            return enzyme_jax_ir(pipeline_options=pipeline, argv=argv, inner_jit=False)(fn)

        # Primal
        for pname, pipeline in all_pipelines:
            fn = in_fn if pipeline is None else enzyme(pipeline, in_fn)
            compiled, out = compile_and_run(fn, ins_b)
            check("Primal", out, pname)
            time(compiled, ins_b, pname, "Primal")

        # Forward mode
        args = ins_b + dins_b
        for pname, pipeline in all_pipelines:
            fn = in_fn if pipeline is None else jax.jit(enzyme(pipeline, in_fn))
            compiled, (primals, tangents) = compile_and_run(splatjvp(fn), args)
            check("Primal", primals, pname)
            check("Forward", tangents, pname)
            time(compiled, args, pname, "Forward")

        # Reverse mode
        args = [douts_b] + ins_b
        for pname, pipeline in all_pipelines:
            variants = []
            if pipeline is None:
                variants.append(("BothRev", splatvjp(in_fn)))
            else:
                # Enzyme pipeline applied before, after, or both before and after AD
                variants.append(("PreRev", splatvjp(enzyme(pipeline, in_fn))))
                variants.append(("PostRev", enzyme(pipeline, splatvjp(in_fn))))
                if pipeline.mlir_ad():
                    variants.append(
                        ("BothRev", enzyme(pipeline, splatvjp(enzyme(pipeline, in_fn))))
                    )

            for key, fn in variants:
                compiled, (primals, grads) = compile_and_run(fn, args)
                check("Primal", primals, pname)
                check("Reverse", grads, pname)
                time(compiled, args, pname, key)
