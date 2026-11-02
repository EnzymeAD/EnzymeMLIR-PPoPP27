import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp_common import run_mat, run_vec  # noqa: E402

from metalift.ir import Int, choose  # noqa: E402,F401
import tenspiler.tenspiler_common as T  # noqa: E402
import tenspiler.axioms_tenspiler as AX  # noqa: E402


def target_lang():
    return [T.vec_elemwise_add, T.matrix_elemwise_add, AX.vec_elemwise_add_axiom, AX.matrix_elemwise_add_axiom]


def Fm(m, s):
    return T.call_matrix_elemwise_add(choose(m["A"], m["B"]), choose(m["A"], m["B"]))


def Fv(m, s):
    return T.call_vec_elemwise_add(choose(m["A"], m["B"]), choose(m["A"], m["B"]))


if __name__ == "__main__":
    run_mat("matadd", [("A","mat"),("B","mat")], target_lang, Fm, Fv)
