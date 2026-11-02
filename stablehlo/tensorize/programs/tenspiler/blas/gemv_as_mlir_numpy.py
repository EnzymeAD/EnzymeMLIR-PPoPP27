
####### import statements ########
import numpy as np

def gemv_as_mlir_np (A, x):
    return (x) * (np.matmul(A, (1) + ((0) * (A[0]))))

def gemv_as_mlir_np_glued (A, x):
    A = np.array(A).astype(np.float32)
    x = np.array(x).astype(np.float32)
    return gemv_as_mlir_np(A, x)
