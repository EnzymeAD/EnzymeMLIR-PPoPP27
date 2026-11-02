import os
"""Shared driver scaffolding for the dsp ports.

Every elementwise kernel is a single loop `out.push_back(E(a[i], b[i], c))` (vector) or a
double loop `row_vec.push_back(E(A[i][j], B[i][j], c)); out.push_back(row_vec)` (matrix),
exactly like Tenspiler's darknet/llama elementwise benchmarks.  The grammars follow those
drivers: postcondition `out == F(...)`, outer invariant `out == F(prefix)`, inner invariant
`row_vec == F_vec(row prefix)`; each driver supplies F (as a small `choose` over candidates)
for the vector and the matrix level.
"""
import time
from typing import Callable, List

from metalift.frontend.llvm import Driver, InvGrammar
from metalift.ir import Bool, Int, Matrix, choose
from metalift.ir import List as mlList
from metalift.vc_util import and_objects
from tenspiler.codegen.utils import DataType
from tenspiler.utils.synthesis_utils import run_synthesis_algorithm

CPP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cpp")


def names(*lists):
    d = {}
    for l in lists:
        for o in l:
            d[o.var_name()] = o
    return d


def make_args(spec):
    objs = []
    for n, kind in spec:
        objs.append({"vec": lambda n=n: mlList(Int, n), "mat": lambda n=n: Matrix(Int, n),
                     "int": lambda n=n: Int(n)}[kind]())
    return objs


def run_vec(name: str, spec, target_lang: Callable, F: Callable):
    """F(vecs: dict name->vector-or-prefix, ints: dict name->Int) -> mlList expr."""
    vec_names = [n for n, k in spec if k == "vec"]
    int_names = [n for n, k in spec if k == "int"]

    def ps(writes, reads, in_scope, relaxed) -> Bool:
        env = names(reads)
        return writes[0] == F({n: env[n] for n in vec_names}, {n: env[n] for n in int_names})

    def inv0(writes, reads, in_scope, relaxed) -> Bool:
        env = names(reads)
        w = names(writes)
        i, out = w["i"], w["agg.result"]
        first = env[vec_names[0]]
        return and_objects(
            i >= 0, i <= first.len(),
            out == F({n: env[n][:i] for n in vec_names}, {n: env[n] for n in int_names}),
        )

    driver = Driver()
    fn = driver.analyze(
        llvm_filepath=f"{CPP}/{name}.ll", loops_filepath=f"{CPP}/{name}.loops",
        fn_name=name, target_lang_fn=target_lang,
        inv_grammars={f"{name}_inv0": InvGrammar(inv0, [])}, ps_grammar=ps,
    )
    args = make_args(spec)
    driver.add_var_objects(args)
    vecs = [a for a, (n, k) in zip(args, spec) if k == "vec"]
    driver.add_precondition(vecs[0].len() > 0)
    for v in vecs[1:]:
        driver.add_precondition(v.len() == vecs[0].len())
    _go(driver, fn, args, name)


def run_mat(name: str, spec, target_lang: Callable, Fm: Callable, Fv: Callable):
    """Fm(mats, ints) -> Matrix expr ; Fv(rows, ints) -> mlList expr (the same op on rows)."""
    mat_names = [n for n, k in spec if k == "mat"]
    int_names = [n for n, k in spec if k == "int"]

    def ps(writes, reads, in_scope, relaxed) -> Bool:
        env = names(reads)
        return writes[0] == Fm({n: env[n] for n in mat_names}, {n: env[n] for n in int_names})

    def inv0(writes, reads, in_scope, relaxed) -> Bool:
        env = names(reads)
        w = names(writes)
        i, out = w["i"], w["agg.result"]
        first = env[mat_names[0]]
        return and_objects(
            i >= 0, i <= first.len(),
            out == Fm({n: env[n][:i] for n in mat_names}, {n: env[n] for n in int_names}),
        )

    def inv1(writes, reads, in_scope, relaxed) -> Bool:
        env = names(reads)
        w = names(writes, in_scope)
        i, j, out, row_vec = w["i"], w["j"], w["agg.result"], w["row_vec"]
        first = env[mat_names[0]]
        ints = {n: env[n] for n in int_names}
        return and_objects(
            i >= 0, i < first.len(), j >= 0, j <= first[0].len(),
            row_vec == Fv({n: env[n][i][:j] for n in mat_names}, ints),
            out == Fm({n: env[n][:i] for n in mat_names}, ints),
        )

    driver = Driver()
    fn = driver.analyze(
        llvm_filepath=f"{CPP}/{name}.ll", loops_filepath=f"{CPP}/{name}.loops",
        fn_name=name, target_lang_fn=target_lang,
        inv_grammars={f"{name}_inv0": InvGrammar(inv0, []),
                      f"{name}_inv1": InvGrammar(inv1, ["i", "agg.result"])},
        ps_grammar=ps,
    )
    args = make_args(spec)
    driver.add_var_objects(args)
    mats = [a for a, (n, k) in zip(args, spec) if k == "mat"]
    driver.add_precondition(mats[0].len() > 0)
    driver.add_precondition(mats[0][0].len() > 0)
    for m in mats[1:]:
        driver.add_precondition(m.len() == mats[0].len())
        driver.add_precondition(m[0].len() == mats[0][0].len())
    _go(driver, fn, args, name)


def _go(driver, fn, args, name):
    t0 = time.time()
    fn(*args)
    run_synthesis_algorithm(driver=driver, data_type=DataType.FLOAT, benchmark_name=name,
                            has_relaxed=False)
    print(f"Synthesis took {time.time() - t0} seconds")
