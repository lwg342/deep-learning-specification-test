# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
import utils_simulation
import utils
from importlib import reload
import seaborn as sns
# %%
# torch.manual_seed(1)    # reproducible
N = 200 
x,y,e = utils_simulation.generate_sample(N, k = 4, sigma = 0.5) 
plt.scatter(x[:,0], y)
plt.scatter(x[:,1], y)

M = utils.DNN(utils.MLP3, lr = 0.001)
# %%

M.net_combine(x,y,batch_size=1000)
print(f'loss {M.loss}, previous loss {M.previous_loss}')

# %%

# Test the Escanciano method
from wl_regression import OLS
z0 = torch.tensor(OLS(x.numpy(), y.numpy()).y_hat())
e0 = (z0 - y).detach()

z = M.net(x)
e1 = (z.squeeze()-y).detach().float()
    
plt.scatter(y,M.net(x).detach())
plt.scatter(y,z0)
plt.figure()

plt.scatter(y,e0)
plt.scatter(y,e1)

plt.figure()
plt.scatter(e0,e1)
plt.scatter(e0,e0)
# from wl_regression import loc_poly
# ll_z = loc_poly(y.numpy(), x.numpy(), x.detach().numpy())
# e2 = (ll_z - y.detach().numpy()[:,0])

# %%
from importlib import reload  
reload(utils)
C, d0, d1 = utils.C_resid(e0, e1, e0, N, "full")
C1 = utils.C_resid(e0, e1, e1, N)
plt.plot(C)
plt.plot(C1)
plt.figure()
sns.histplot(C1)
# %%
W, sigma_hat = utils.compute_w(x,y,e0,e1,N)
# rslt =utils.test_statistic(C, N, sigma_hat)
rslt =utils.test_statistic(C, N, sigma_hat, "KS", C1)
print(sigma_hat, rslt)
# %%
# %%

# %%
