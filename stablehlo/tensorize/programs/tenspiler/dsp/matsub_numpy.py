
####### import statements ########
import numpy as np

def matsub_np (A, B):
    return (A) - (B)

def matsub_np_glued (A, B):
    A = np.array(A).astype(np.float32)
    B = np.array(B).astype(np.float32)
    return matsub_np(A, B)
