import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp_common import run_mat, run_vec  # noqa: E402

from metalift.ir import Int, choose  # noqa: E402,F401
import tenspiler.tenspiler_common as T  # noqa: E402
import tenspiler.axioms_tenspiler as AX  # noqa: E402


def target_lang():
    return [T.vec_scalar_add, T.vec_scalar_sub, AX.vec_scalar_add_axiom]


def F(v, s):
    return choose(T.call_vec_scalar_add(s["c"], v["a"]), T.call_vec_scalar_sub(s["c"], v["a"]))


if __name__ == "__main__":
    run_vec("voffset", [("a","vec"),("c","int")], target_lang, F)
