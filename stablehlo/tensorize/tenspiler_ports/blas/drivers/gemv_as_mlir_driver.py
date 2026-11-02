import os
import time
from typing import List, Union

from metalift.frontend.llvm import Driver, InvGrammar
from metalift.ir import Bool, FnDecl, FnDeclRecursive, Int, Matrix, choose
from metalift.ir import List as mlList
from metalift.vc_util import and_objects
from tenspiler.axioms_tenspiler import (
    matrix_vec_mul_axiom,
    reduce_sum_axiom,
    vec_elemwise_mul_axiom,
    vec_scalar_mul_axiom,
)
from tenspiler.codegen.utils import DataType
from tenspiler.tenspiler_common import (
    call_matrix_vec_mul,
    call_reduce_sum,
    call_vec_elemwise_mul,
    call_vec_scalar_add,
    call_vec_scalar_mul,
    matrix_vec_mul,
    reduce_sum,
    vec_elemwise_mul,
    vec_scalar_add,
    vec_scalar_mul,
)
from tenspiler.utils.synthesis_utils import run_synthesis_algorithm

CPP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cpp")

# y[i] = x[i] * sum_j A[i][j].  TensIR has no row-sum / broadcast, so the row sums are
# written as A @ ones, with ones = 1 + 0 * A[0] (pure TensIR: vec_scalar_add/vec_scalar_mul).


def ones_like(v):
    return call_vec_scalar_add(Int(1), call_vec_scalar_mul(Int(0), v))


def gemv_as_mlir_target_lang() -> List[Union[FnDecl, FnDeclRecursive]]:
    return [
        matrix_vec_mul,
        vec_elemwise_mul,
        vec_scalar_mul,
        vec_scalar_add,
        reduce_sum,
        vec_scalar_mul_axiom,
        vec_elemwise_mul_axiom,
        reduce_sum_axiom,
        matrix_vec_mul_axiom,
    ]


def gemv_as_mlir_ps_grammar(writes, reads, in_scope, relaxed) -> Bool:
    ret_val = writes[0]
    A, x = reads
    matrix = choose(A, A.transpose())
    vec = choose(x, x[: A.len()])
    return ret_val == call_vec_elemwise_mul(vec, call_matrix_vec_mul(matrix, ones_like(A[0])))


def gemv_as_mlir_inv0_grammar(writes, reads, in_scope, relaxed) -> Bool:
    A, x = reads
    out, col, curr, row = writes
    matrix = choose(A[:row], A[:row].transpose())
    vec = choose(x[:row], x[:col])
    return and_objects(
        row >= 0,
        row <= A.len(),
        out == call_vec_elemwise_mul(vec, call_matrix_vec_mul(matrix, ones_like(A[0]))),
    )


def gemv_as_mlir_inv1_grammar(writes, reads, in_scope, relaxed) -> Bool:
    col, curr = writes
    A, x = reads
    out, row = in_scope
    matrix = choose(A[:row], A[:row].transpose())
    vec = choose(x[:row], x[:col])
    scalar = choose(x[row], x[col])
    inner = choose(A[row][:col], A[:col].col_vec(row))
    return and_objects(
        row >= 0,
        row < A.len(),
        col >= 0,
        col <= A[0].len(),
        curr == call_reduce_sum(call_vec_scalar_mul(scalar, inner)),
        out == call_vec_elemwise_mul(vec, call_matrix_vec_mul(matrix, ones_like(A[0]))),
    )


if __name__ == "__main__":
    driver = Driver()
    gemv_as_mlir = driver.analyze(
        llvm_filepath=f"{CPP}/gemv_as_mlir.ll",
        loops_filepath=f"{CPP}/gemv_as_mlir.loops",
        fn_name="gemv_as_mlir",
        target_lang_fn=gemv_as_mlir_target_lang,
        inv_grammars={
            "gemv_as_mlir_inv0": InvGrammar(gemv_as_mlir_inv0_grammar, []),
            "gemv_as_mlir_inv1": InvGrammar(gemv_as_mlir_inv1_grammar, ["row", "agg.result"]),
        },
        ps_grammar=gemv_as_mlir_ps_grammar,
    )
    A = Matrix(Int, "A")
    x = mlList(Int, "x")
    driver.add_var_objects([A, x])
    driver.add_precondition(A.len() > 0)
    driver.add_precondition(A[0].len() > 0)
    driver.add_precondition(A.len() == x.len())

    start_time = time.time()
    gemv_as_mlir(A, x)
    run_synthesis_algorithm(
        driver=driver, data_type=DataType.FLOAT, benchmark_name="gemv_as_mlir", has_relaxed=False
    )
    print(f"Synthesis took {time.time() - start_time} seconds")
