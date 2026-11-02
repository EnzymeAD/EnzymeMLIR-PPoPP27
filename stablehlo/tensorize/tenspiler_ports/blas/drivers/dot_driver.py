import os
import time
from typing import List, Union

from metalift.frontend.llvm import Driver, InvGrammar
from metalift.ir import Bool, FnDecl, FnDeclRecursive, Int
from metalift.ir import List as mlList
from metalift.ir import Object, choose
from metalift.vc_util import and_objects
from tenspiler.axioms_tenspiler import reduce_sum_axiom, vec_elemwise_mul_axiom
from tenspiler.codegen.utils import DataType
from tenspiler.tenspiler_common import (
    call_reduce_sum,
    call_vec_elemwise_mul,
    reduce_sum,
    vec_elemwise_mul,
)
from tenspiler.utils.synthesis_utils import run_synthesis_algorithm

CPP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cpp")


def dot_target_lang() -> List[Union[FnDecl, FnDeclRecursive]]:
    return [vec_elemwise_mul, reduce_sum, vec_elemwise_mul_axiom, reduce_sum_axiom]


def dot_ps_grammar(writes, reads, in_scope, relaxed) -> Bool:
    ret_val = writes[0]
    a, b = reads
    vec = choose(a, b)
    vec2 = choose(a, b)
    return ret_val == call_reduce_sum(call_vec_elemwise_mul(vec, vec2))


def dot_inv0_grammar(writes, reads, in_scope, relaxed) -> Bool:
    a, b = reads
    i, sum = writes
    vec = choose(a[:i], b[:i])
    vec2 = choose(a[:i], b[:i])
    return and_objects(
        i >= 0, i <= a.len(), sum == call_reduce_sum(call_vec_elemwise_mul(vec, vec2))
    )


if __name__ == "__main__":
    driver = Driver()
    dot = driver.analyze(
        llvm_filepath=f"{CPP}/dot.ll",
        loops_filepath=f"{CPP}/dot.loops",
        fn_name="dot",
        target_lang_fn=dot_target_lang,
        inv_grammars={"dot_inv0": InvGrammar(dot_inv0_grammar, [])},
        ps_grammar=dot_ps_grammar,
    )
    a = mlList(Int, "a")
    b = mlList(Int, "b")
    driver.add_var_objects([a, b])
    driver.add_precondition(a.len() == b.len())
    driver.add_precondition(a.len() > 0)

    start_time = time.time()
    dot(a, b)
    run_synthesis_algorithm(
        driver=driver, data_type=DataType.FLOAT, benchmark_name="dot", has_relaxed=False
    )
    print(f"Synthesis took {time.time() - start_time} seconds")
