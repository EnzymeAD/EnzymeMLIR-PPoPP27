
####### import statements ########
import numpy as np

def gemv_np (A, x):
    return np.matmul(A, x)

def gemv_np_glued (A, x):
    A = np.array(A).astype(np.float32)
    x = np.array(x).astype(np.float32)
    return gemv_np(A, x)
