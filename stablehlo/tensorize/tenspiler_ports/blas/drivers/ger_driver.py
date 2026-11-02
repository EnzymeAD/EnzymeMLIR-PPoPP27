import os
import sys
import time
from typing import List, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blas_ext import call_vec_outer, vec_outer  # TensIR extension (outer product)

from metalift.frontend.llvm import Driver, InvGrammar
from metalift.ir import Bool, FnDecl, FnDeclRecursive, Int, Matrix, choose
from metalift.ir import List as mlList
from metalift.vc_util import and_objects
from tenspiler.axioms_tenspiler import matrix_elemwise_add_axiom, vec_elemwise_add_axiom
from tenspiler.codegen.utils import DataType
from tenspiler.tenspiler_common import (
    call_matrix_elemwise_add,
    call_vec_elemwise_add,
    call_vec_scalar_mul,
    matrix_elemwise_add,
    vec_elemwise_add,
    vec_scalar_mul,
)
from tenspiler.utils.synthesis_utils import run_synthesis_algorithm

CPP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cpp")


def ger_target_lang() -> List[Union[FnDecl, FnDeclRecursive]]:
    return [
        vec_elemwise_add,
        matrix_elemwise_add,
        vec_scalar_mul,
        vec_outer,
        vec_elemwise_add_axiom,
        matrix_elemwise_add_axiom,
    ]


def ger_ps_grammar(writes, reads, in_scope, relaxed) -> Bool:
    A, x, y = reads
    out = writes[0]
    u = choose(x, y)
    v = choose(x, y)
    outer = choose(call_vec_outer(u, v), call_vec_outer(u, v).transpose())
    return out == call_matrix_elemwise_add(A, outer)


def ger_inv0_grammar(writes, reads, in_scope, relaxed) -> Bool:
    A, x, y = reads
    out, i, j, _, _ = writes
    u = choose(x[:i], y[:i])
    v = choose(x, y)
    return and_objects(
        i >= 0, i <= A.len(), out == call_matrix_elemwise_add(A[:i], call_vec_outer(u, v))
    )


def ger_inv1_grammar(writes, reads, in_scope, relaxed) -> Bool:
    A, x, y = reads
    out, i = in_scope
    j, _, row_vec = writes
    u = choose(x[:i], y[:i])
    v = choose(x, y)
    s = choose(x[i], y[i])
    w = choose(x[:j], y[:j])
    return and_objects(
        i >= 0,
        i < A.len(),
        j >= 0,
        j <= A[0].len(),
        row_vec == call_vec_elemwise_add(A[i][:j], call_vec_scalar_mul(s, w)),
        out == call_matrix_elemwise_add(A[:i], call_vec_outer(u, v)),
    )


if __name__ == "__main__":
    driver = Driver()
    ger = driver.analyze(
        llvm_filepath=f"{CPP}/ger.ll",
        loops_filepath=f"{CPP}/ger.loops",
        fn_name="ger",
        target_lang_fn=ger_target_lang,
        inv_grammars={
            "ger_inv0": InvGrammar(ger_inv0_grammar, []),
            "ger_inv1": InvGrammar(ger_inv1_grammar, ["i", "agg.result"]),
        },
        ps_grammar=ger_ps_grammar,
    )
    A = Matrix(Int, "A")
    x = mlList(Int, "x")
    y = mlList(Int, "y")
    driver.add_var_objects([A, x, y])
    driver.add_precondition(A.len() > 0)
    driver.add_precondition(A[0].len() > 0)
    driver.add_precondition(A.len() == x.len())
    driver.add_precondition(A[0].len() == y.len())

    start_time = time.time()
    ger(A, x, y)
    run_synthesis_algorithm(
        driver=driver, data_type=DataType.FLOAT, benchmark_name="ger", has_relaxed=False
    )
    print(f"Synthesis took {time.time() - start_time} seconds")
