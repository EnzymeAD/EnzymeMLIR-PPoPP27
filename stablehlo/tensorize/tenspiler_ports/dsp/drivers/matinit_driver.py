import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp_common import run_mat, run_vec  # noqa: E402

from metalift.ir import Int, choose  # noqa: E402,F401
import tenspiler.tenspiler_common as T  # noqa: E402
import tenspiler.axioms_tenspiler as AX  # noqa: E402


def target_lang():
    return [T.vec_scalar_add, T.vec_scalar_mul, T.matrix_scalar_add, T.matrix_scalar_mul, AX.vec_scalar_add_axiom, AX.vec_scalar_mul_axiom, AX.matrix_scalar_mul_axiom]


def Fm(m, s):
    return T.call_matrix_scalar_add(choose(s["c"], Int(0), Int(1)), T.call_matrix_scalar_mul(choose(Int(0), Int(1)), m["A"]))


def Fv(m, s):
    return T.call_vec_scalar_add(choose(s["c"], Int(0), Int(1)), T.call_vec_scalar_mul(choose(Int(0), Int(1)), m["A"]))


if __name__ == "__main__":
    run_mat("matinit", [("A","mat"),("c","int")], target_lang, Fm, Fv)
