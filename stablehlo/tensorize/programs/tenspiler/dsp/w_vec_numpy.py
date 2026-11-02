
####### import statements ########
import numpy as np

def w_vec_np (a, b, s):
    return ((s) * (a)) + (b)

def w_vec_np_glued (a, b, s):
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    return w_vec_np(a, b, s)
