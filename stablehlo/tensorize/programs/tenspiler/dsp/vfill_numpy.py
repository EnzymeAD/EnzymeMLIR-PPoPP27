
####### import statements ########
import numpy as np

def vfill_np (a, c):
    return (c) + ((0) * (a))

def vfill_np_glued (a, c):
    a = np.array(a).astype(np.float32)
    return vfill_np(a, c)
