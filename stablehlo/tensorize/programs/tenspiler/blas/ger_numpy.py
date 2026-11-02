
####### import statements ########
import numpy as np

def ger_np (A, x, y):
    return (A) + (np.transpose(np.outer(y, x)))

def ger_np_glued (A, x, y):
    A = np.array(A).astype(np.float32)
    x = np.array(x).astype(np.float32)
    y = np.array(y).astype(np.float32)
    return ger_np(A, x, y)
