
####### import statements ########
import numpy as np

def vcopy_np (a):
    return a

def vcopy_np_glued (a):
    a = np.array(a).astype(np.float32)
    return vcopy_np(a)
