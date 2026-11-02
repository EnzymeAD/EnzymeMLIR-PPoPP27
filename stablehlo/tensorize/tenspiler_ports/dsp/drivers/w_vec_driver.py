import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp_common import run_mat, run_vec  # noqa: E402

from metalift.ir import Int, choose  # noqa: E402,F401
import tenspiler.tenspiler_common as T  # noqa: E402
import tenspiler.axioms_tenspiler as AX  # noqa: E402


def target_lang():
    return [T.vec_elemwise_add, T.vec_scalar_mul, AX.vec_elemwise_add_axiom, AX.vec_scalar_mul_axiom]


def F(v, s):
    return T.call_vec_elemwise_add(T.call_vec_scalar_mul(s["s"], choose(v["a"], v["b"])), choose(v["a"], v["b"]))


if __name__ == "__main__":
    run_vec("w_vec", [("a","vec"),("b","vec"),("s","int")], target_lang, F)
