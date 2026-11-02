
####### import statements ########
import numpy as np

def matadd_np (A, B):
    return (B) + (A)

def matadd_np_glued (A, B):
    A = np.array(A).astype(np.float32)
    B = np.array(B).astype(np.float32)
    return matadd_np(A, B)
