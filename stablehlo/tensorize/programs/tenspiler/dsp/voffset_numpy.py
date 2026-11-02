
####### import statements ########
import numpy as np

def voffset_np (a, c):
    return (c) + (a)

def voffset_np_glued (a, c):
    a = np.array(a).astype(np.float32)
    return voffset_np(a, c)
