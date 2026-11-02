import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp_common import CPP, _go, names  # noqa: E402
from dsp_ext import call_matrix_matrix_mul, matrix_matrix_mul  # noqa: E402

from metalift.frontend.llvm import Driver, InvGrammar  # noqa: E402
from metalift.ir import Int, Matrix, choose  # noqa: E402
from metalift.vc_util import and_objects  # noqa: E402
import tenspiler.tenspiler_common as T  # noqa: E402
import tenspiler.axioms_tenspiler as AX  # noqa: E402


def target_lang():
    return [T.vec_elemwise_mul, T.reduce_sum, T.matrix_vec_mul, matrix_matrix_mul,
            AX.vec_elemwise_mul_axiom, AX.reduce_sum_axiom, AX.matrix_vec_mul_axiom]


def mm(A, B):
    return call_matrix_matrix_mul(choose(A, B), choose(B, A))


def ps(writes, reads, in_scope, relaxed):
    e = names(reads)
    return writes[0] == mm(e["A"], e["B"])


def inv0(writes, reads, in_scope, relaxed):  # i loop
    e, w = names(reads), names(writes)
    A, B, i, out = e["A"], e["B"], w["i"], w["agg.result"]
    return and_objects(i >= 0, i <= A.len(), out == call_matrix_matrix_mul(choose(A[:i], B[:i]), choose(B, A)))


def row_prefix(A, B, i, j):
    # row i of A @ B restricted to columns < j
    return T.call_matrix_vec_mul(choose(B.col_slice(0, j).transpose(), A.col_slice(0, j).transpose()), choose(A[i], B[i]))


def inv1(writes, reads, in_scope, relaxed):  # j loop
    e, w = names(reads), names(writes, in_scope)
    A, B, i, j, out, row = e["A"], e["B"], w["i"], w["j"], w["agg.result"], w["row_vec"]
    return and_objects(
        i >= 0, i < A.len(), j >= 0, j <= B[0].len(),
        row == row_prefix(A, B, i, j),
        out == call_matrix_matrix_mul(choose(A[:i], B[:i]), choose(B, A)),
    )


def inv2(writes, reads, in_scope, relaxed):  # k loop
    e, w = names(reads), names(writes, in_scope)
    A, B = e["A"], e["B"]
    i, j, k, curr, out, row = w["i"], w["j"], w["k"], w["curr"], w["agg.result"], w["row_vec"]
    return and_objects(
        i >= 0, i < A.len(), j >= 0, j < B[0].len(), k >= 0, k <= B.len(),
        curr == T.call_reduce_sum(T.call_vec_elemwise_mul(choose(A[i][:k], B[i][:k]), choose(B[:k].col_vec(j), A[:k].col_vec(j)))),
        row == row_prefix(A, B, i, j),
        out == call_matrix_matrix_mul(choose(A[:i], B[:i]), choose(B, A)),
    )


if __name__ == "__main__":
    driver = Driver()
    fn = driver.analyze(
        llvm_filepath=f"{CPP}/matmul.ll", loops_filepath=f"{CPP}/matmul.loops",
        fn_name="matmul", target_lang_fn=target_lang,
        inv_grammars={
            "matmul_inv0": InvGrammar(inv0, []),
            "matmul_inv1": InvGrammar(inv1, ["i", "agg.result"]),
            "matmul_inv2": InvGrammar(inv2, ["i", "j", "agg.result", "row_vec"]),
        },
        ps_grammar=ps,
    )
    A, B = Matrix(Int, "A"), Matrix(Int, "B")
    driver.add_var_objects([A, B])
    driver.add_precondition(A.len() > 0)
    driver.add_precondition(B.len() > 0)
    driver.add_precondition(B[0].len() > 0)
    driver.add_precondition(A[0].len() == B.len())
    _go(driver, fn, [A, B], "matmul")
