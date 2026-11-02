
####### import statements ########
import numpy as np

def vadd_np (a, b):
    return (b) + (a)

def vadd_np_glued (a, b):
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    return vadd_np(a, b)
