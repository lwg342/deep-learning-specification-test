import numpy as np

def C_resid(e0 , e1, y, T, option = None):
    y_mat = np.outer(y, np.ones(T))
    d0 = (y_mat - np.outer(np.ones(len(y)), e0)).clip(0)
    d1 = (y_mat - np.outer(np.ones(len(y)), e1)).clip(0)
    C = 1/np.sqrt(T) * ((d0 - d1).sum(1))
    return C


def test_statistic(T, sigma_hat=1, method="CM"):
    if method == "CM":
        rslt = 3/sigma_hat /T 
    
    
    return rslt
