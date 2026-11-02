
####### import statements ########
import numpy as np

def vmul_np (a, b):
    return (a) * (b)

def vmul_np_glued (a, b):
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    return vmul_np(a, b)
