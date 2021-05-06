import numpy as np
import torch

def C_resid(e0 , e1, y, T, option = None):
    y_mat = np.outer(y, np.ones(T))
    d0 = (y_mat - np.outer(np.ones(len(y)), e0)).clip(0)
    d1 = (y_mat - np.outer(np.ones(len(y)), e1)).clip(0)
    C = (1/np.sqrt(T)) * ((d0 - d1).sum(1))
    if option == "full":    
        return C, d0 ,d1
    else:
        return C


def compute_l(x, y, e, T, method="ols"):
    if method == "ols":
        A = torch.mean(x,0)
        B = (torch.matmul(x.transpose(1,0), x)) /T
        l = (torch.matmul(torch.matmul(A, B), x.transpose(1, 0))) * e
    if method == "deep learning":
        l = e 
    return l

def compute_w(x,y,e0, e1,T, methods = ["ols", "deep learning"]):
    W0 = compute_l(x, y, e0, T, methods[0])
    W1 = compute_l(x, y, e1, T, methods[1])
    W = W0 - W1
    return W, torch.std(W)

def test_statistic(C, T, sigma_hat, method="CM"):
    if method == "CM":
        rslt = 3/(sigma_hat)**2 / T * np.inner(C, C)
    return rslt
