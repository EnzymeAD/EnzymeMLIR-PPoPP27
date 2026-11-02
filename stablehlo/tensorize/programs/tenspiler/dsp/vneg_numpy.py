
####### import statements ########
import numpy as np

def vneg_np (a):
    return ((0) - (1)) * (a)

def vneg_np_glued (a):
    a = np.array(a).astype(np.float32)
    return vneg_np(a)
