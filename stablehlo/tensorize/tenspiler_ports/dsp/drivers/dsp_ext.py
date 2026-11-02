"""TensIR extension used only by the dsp `matmul` port: matrix_matrix_mul(A, B).

TensIR's only contraction is matrix_vec_mul; every matrix-valued op is elementwise /
scalar / transpose / slicing, so C = A @ B (a matrix built from row-by-matrix products)
is not expressible. Defined recursively in the style of the built-in ops:
    matrix_matrix_mul(A, B) = []                                              if len(A) < 1
                            = prepend(matrix_vec_mul(transpose(B), A[0]), matrix_matrix_mul(A[1:], B))
Importing this module registers np.matmul / torch.matmul / tf.matmul / mx.matmul translations.
"""
from metalift.ir import Int, Matrix, call, fn_decl_recursive, ite
from metalift.vc_util import or_objects
from tenspiler.tenspiler_common import call_matrix_vec_mul

MATRIX_MATRIX_MUL = "matrix_matrix_mul"
_A = Matrix(Int, "matrix_a")
_B = Matrix(Int, "matrix_b")


def call_matrix_matrix_mul(a, b):
    return call(MATRIX_MATRIX_MUL, Matrix[Int], a, b)


def _body(a, b):
    cur = call_matrix_vec_mul(b.transpose(), a[0])
    return ite(or_objects(a.len() < 1), Matrix.empty(Int), call_matrix_matrix_mul(a[1:], b).prepend(cur))


matrix_matrix_mul = fn_decl_recursive(MATRIX_MATRIX_MUL, Matrix[Int], _body(_A, _B), _A, _B)

import tenspiler.codegen.numpy_codegen as _np_cg
import tenspiler.codegen.pytorch_codegen as _pt_cg
import tenspiler.codegen.tensorflow_codegen as _tf_cg
import tenspiler.codegen.mlx_codegen as _mlx_cg

_np_cg.translations[MATRIX_MATRIX_MUL] = lambda a: f"np.matmul({a[0]}, {a[1]})"
_pt_cg.translations[MATRIX_MATRIX_MUL] = lambda a: f"torch.matmul({a[0]}, {a[1]})"
_tf_cg.translations[MATRIX_MATRIX_MUL] = lambda a: f"tf.linalg.matmul({a[0]}, {a[1]})"
_mlx_cg.translations[MATRIX_MATRIX_MUL] = lambda a: f"mx.matmul({a[0]}, {a[1]})"
