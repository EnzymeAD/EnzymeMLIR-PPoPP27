
####### import statements ########
import numpy as np

def matscal_np (A, s):
    return (s) * (A)

def matscal_np_glued (A, s):
    A = np.array(A).astype(np.float32)
    return matscal_np(A, s)
