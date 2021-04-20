import numpy as np

def C_resid(e0 , e1, y, T, option = None):
    d0 = (y - e0).clip()
    d1 = (y - e1).clip()
    C = 1/np.sqrt(T) * ((d0 - d1).sum())
    return C

def test_statistic(C, method = "CM"):
    rslt = None
    return rslt
