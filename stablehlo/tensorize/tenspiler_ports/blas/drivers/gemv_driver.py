import os
import time
from typing import List, Union

CPP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cpp")

from metalift.frontend.llvm import Driver, InvGrammar
from metalift.ir import Bool, FnDecl, FnDeclRecursive, Int
from metalift.ir import List as mlList
from metalift.ir import Matrix, Object, choose, ite
from metalift.vc_util import and_objects
from tenspiler.codegen.utils import DataType
from tenspiler.tenspiler_common import (
    call_matrix_vec_mul,
    call_reduce_sum,
    call_vec_elemwise_mul,
    call_vec_scalar_mul,
    get_no_arg_bool_fn,
    matrix_vec_mul,
    reduce_sum,
    vec_elemwise_mul,
    vec_scalar_mul,
)
from tenspiler.utils.synthesis_utils import run_synthesis_algorithm
from tenspiler.axioms_tenspiler import (
    vec_scalar_mul_axiom,
    reduce_sum_axiom,
    matrix_vec_mul_axiom,
)

# Some loop functions
matrix_outer_loop_index_first_fn_name = "MATRIX_OUTER_LOOP_INDEX_FIRST"
(
    matrix_outer_loop_index_first_fn_decl,
    matrix_outer_loop_index_first_synth,
    is_matrix_outer_loop_index_first,
) = get_no_arg_bool_fn(matrix_outer_loop_index_first_fn_name)

vector_outer_loop_index_fn_name = "VECTOR_OUTER_LOOP_INDEX"
(
    vector_outer_loop_index_fn_decl,
    vector_outer_loop_index_synth,
    is_vector_outer_loop_index,
) = get_no_arg_bool_fn(vector_outer_loop_index_fn_name)


def gemv_target_lang() -> List[Union[FnDecl, FnDeclRecursive]]:
    return [
        matrix_vec_mul,
        vec_elemwise_mul,
        vec_scalar_mul,
        reduce_sum,
        matrix_outer_loop_index_first_fn_decl,
        vector_outer_loop_index_fn_decl,
        vec_scalar_mul_axiom,
        reduce_sum_axiom,
        matrix_vec_mul_axiom,
    ]


def gemv_ps_grammar(
    writes: List[Object], reads: List[Object], in_scope: List[Object], relaxed: bool
) -> Bool:
    ret_val = writes[0]
    weight, input = reads
    matrix = ite(
        is_matrix_outer_loop_index_first(),
        weight,
        weight[0 : input.len()].col_slice(0, weight.len()),
    )
    matrix = choose(matrix, matrix.transpose())
    vec = ite(is_vector_outer_loop_index(), input[: weight.len()], input)
    return ret_val == call_matrix_vec_mul(matrix, vec)


def gemv_inv0_grammar(
    writes: List[Object], reads: List[Object], in_scope: List[Object], relaxed: bool
) -> Bool:
    weight, input = reads
    out, col, _, row = writes
    matrix = ite(
        is_matrix_outer_loop_index_first(),
        weight[:row],
        weight[0 : input.len()].col_slice(0, row),
    )
    matrix = choose(matrix, matrix.transpose())
    vec = ite(is_vector_outer_loop_index(), input[:row], input)

    return and_objects(
        row >= 0, row <= weight.len(), out == call_matrix_vec_mul(matrix, vec)
    )


def gemv_inv1_grammar(
    writes: List[Object], reads: List[Object], in_scope: List[Object], relaxed: bool
) -> Bool:
    col, curr = writes
    weight, input = reads
    out, row = in_scope
    outer_loop_matrix = ite(
        is_matrix_outer_loop_index_first(),
        weight[:row],
        weight[0 : input.len()].col_slice(0, row),
    )
    outer_loop_matrix = choose(outer_loop_matrix, outer_loop_matrix.transpose())
    outer_loop_vec = ite(is_vector_outer_loop_index(), input[:row], input)

    inner_loop_weight_vec = ite(
        is_matrix_outer_loop_index_first(), weight[row][:col], weight[:col].col_vec(row)
    )
    inner_loop_vec_to_reduce = ite(
        is_vector_outer_loop_index(),
        call_vec_scalar_mul(input[row], inner_loop_weight_vec),
        call_vec_elemwise_mul(inner_loop_weight_vec, input[:col]),
    )

    return and_objects(
        row >= 0,
        row < weight.len(),
        col >= 0,
        col <= input.len(),
        curr == call_reduce_sum(inner_loop_vec_to_reduce),
        out == call_matrix_vec_mul(outer_loop_matrix, outer_loop_vec),
    )


if __name__ == "__main__":
    driver = Driver()
    gemv = driver.analyze(
        llvm_filepath=os.path.join(CPP, "gemv.ll"),
        loops_filepath=os.path.join(CPP, "gemv.loops"),
        fn_name="gemv",
        target_lang_fn=gemv_target_lang,
        inv_grammars={
            "gemv_inv0": InvGrammar(gemv_inv0_grammar, []),
            "gemv_inv1": InvGrammar(gemv_inv1_grammar, ["row", "agg.result"]),
        },
        ps_grammar=gemv_ps_grammar,
    )

    weight_var = Matrix(Int, "A")
    input_var = mlList(Int, "x")
    driver.add_var_objects([weight_var, input_var])
    driver.add_precondition(weight_var.len() > 0)
    driver.add_precondition(weight_var[0].len() > 0)
    driver.add_precondition(weight_var[0].len() == input_var.len())
    driver.fns_synths = [
        matrix_outer_loop_index_first_synth,
        vector_outer_loop_index_synth,
    ]

    start_time = time.time()
    gemv(weight_var, input_var)
    run_synthesis_algorithm(
        driver=driver,
        data_type=DataType.FLOAT,
        benchmark_name="gemv",
        has_relaxed=False,
    )
    end_time = time.time()
    print(f"Synthesis took {end_time - start_time} seconds")
