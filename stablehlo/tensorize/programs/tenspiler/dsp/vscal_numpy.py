
####### import statements ########
import numpy as np

def vscal_np (a, s):
    return (s) * (a)

def vscal_np_glued (a, s):
    a = np.array(a).astype(np.float32)
    return vscal_np(a, s)
