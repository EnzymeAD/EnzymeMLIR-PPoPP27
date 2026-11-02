
####### import statements ########
import numpy as np

def vsub_np (a, b):
    return (a) - (b)

def vsub_np_glued (a, b):
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    return vsub_np(a, b)
