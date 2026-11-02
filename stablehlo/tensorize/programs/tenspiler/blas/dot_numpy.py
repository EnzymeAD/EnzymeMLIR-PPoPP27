
####### import statements ########
import numpy as np

def dot_np (a, b):
    return np.sum((b) * (a))

def dot_np_glued (a, b):
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    return dot_np(a, b)
