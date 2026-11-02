
####### import statements ########
import numpy as np

def vrecip_np (a):
    return (1) / (a)

def vrecip_np_glued (a):
    a = np.array(a).astype(np.float32)
    return vrecip_np(a)
