
####### import statements ########
import numpy as np

def matinit_np (A, c):
    return (c) + ((0) * (A))

def matinit_np_glued (A, c):
    A = np.array(A).astype(np.float32)
    return matinit_np(A, c)
