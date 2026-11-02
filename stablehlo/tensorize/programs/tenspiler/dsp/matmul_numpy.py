
####### import statements ########
import numpy as np

def matmul_np (A, B):
    return np.matmul(A, B)

def matmul_np_glued (A, B):
    A = np.array(A).astype(np.float32)
    B = np.array(B).astype(np.float32)
    return matmul_np(A, B)
