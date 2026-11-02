"""TensIR extension used only by the `ger` port: an outer product `vec_outer(x, y)`.

TensIR has no operator that builds a matrix from two vectors (all matrix-valued ops
take a matrix operand and act elementwise / by scalar / by transpose/slicing), so
A + x y^T is not expressible without it. Defined recursively like the other TensIR ops:
    vec_outer(x, y) = []                                  if len(x) < 1
                    = prepend(vec_scalar_mul(x[0], y), vec_outer(x[1:], y))
Importing this module also registers backend translations (np.outer etc.).
"""
from metalift.ir import Int, Matrix, call, fn_decl_recursive, ite
from metalift.ir import List as mlList
from metalift.vc_util import or_objects
from tenspiler.tenspiler_common import call_vec_scalar_mul

VEC_OUTER = "vec_outer"
_x = mlList(Int, "x")
_y = mlList(Int, "y")


def call_vec_outer(x, y):
    return call(VEC_OUTER, Matrix[Int], x, y)


def _vec_outer_body(x, y):
    cur = call_vec_scalar_mul(x[0], y)
    recursed = call_vec_outer(x[1:], y)
    return ite(or_objects(x.len() < 1), Matrix.empty(Int), recursed.prepend(cur))


vec_outer = fn_decl_recursive(VEC_OUTER, Matrix[Int], _vec_outer_body(_x, _y), _x, _y)

# backend translations
import tenspiler.codegen.numpy_codegen as _np_cg
import tenspiler.codegen.pytorch_codegen as _pt_cg
import tenspiler.codegen.tensorflow_codegen as _tf_cg
import tenspiler.codegen.mlx_codegen as _mlx_cg

_np_cg.translations[VEC_OUTER] = lambda a: f"np.outer({a[0]}, {a[1]})"
_pt_cg.translations[VEC_OUTER] = lambda a: f"torch.outer({a[0]}, {a[1]})"
_tf_cg.translations[VEC_OUTER] = lambda a: f"tf.tensordot({a[0]}, {a[1]}, axes=0)"
_mlx_cg.translations[VEC_OUTER] = lambda a: f"mx.outer({a[0]}, {a[1]})"
