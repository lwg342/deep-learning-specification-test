# %%
"""
This is a different approach, similar to the one based on kernel estimation 
E(e \mid X) = 0 a.e.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils_simulation import generate_sample, ols
import time
from scipy import stats
from utils import DNN, MLP3


# %%

N = 200
k = 2
sigma = 1
def null_estimate(N,k,sigma):
    x,y,e = generate_sample(N,k,sigma)
    resid =  y - ols(y,x,N,k)[0]
    return x,y,e,resid

x,y,e,resid = null_estimate(N,k,sigma)
plt.scatter(x[:, 0], y)
plt.scatter(x[:, 1], y)
plt.figure()
plt.scatter(e,resid)
plt.scatter(e,e)
# %%
M = DNN(k = k, model = MLP3, lr = 0.001, max_epochs=20)
M.net_combine(x, resid)
plt.scatter(M.net(x).detach().squeeze(), resid)
cm_stat = (M.net(x).squeeze()*resid).mean()*np.sqrt(N)


# %%
plt.scatter(x[:,0], M.net(x).detach().squeeze())
plt.scatter(x[:,0], resid, color = 'r')
# plt.scatter(x[:,1], M.net(x).detach().squeeze())
# %%
N,k,sigma = 200, 4, 1
# lst_cm  = []
for j in range(1000):
    start = time.time()
    x,y,e,resid = null_estimate(N,k,sigma)

    M = DNN(k = k, model = MLP3, lr = 0.001, max_epochs=20)
    M.net_combine(x, resid)

    cm = (M.net(x).squeeze()*resid).mean()*np.sqrt(N)
    lst_cm += [cm]
    end = time.time()
    print(f"iter {j} : time elapsed : {end - start :.2f}, cm: {cm:.2f}")
# %%
import seaborn as sns
lst_cm_tensor = torch.tensor(lst_cm)
lst_cm_tensor = lst_cm_tensor[lst_cm_tensor < 8]
ax = sns.histplot(lst_cm_tensor, stat = "density")

print(f"mean {lst_cm_tensor.mean(): .2f}, variance {torch.tensor(lst_cm).var(): .5f}, skew: {stats.skew(lst_cm_tensor.numpy()) : .3f}, kurtosis : {stats.kurtosis(lst_cm_tensor.numpy()) : .3f}")

# %%
import pickle
with open("save.pth" , "wb") as f:
    pickle.dump([save_k_4], f)
# %%
