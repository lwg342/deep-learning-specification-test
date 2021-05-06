# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
import utils_simulation
import utils
# %%
# torch.manual_seed(1)    # reproducible
N = 200 
x,y,e = utils_simulation.generate_sample(N, k = 4, sigma = 0.5) 
plt.scatter(x[:,0], y)
plt.scatter(x[:,1], y)

M = utils.wl_nn(utils.MLP4, lr = 0.0001)
# %%

M.net_update(x,y,batch_size=10000)
plt.scatter(y,M.net(x).detach())

# %%

# Test the Escanciano method
from wl_regression import OLS  
z0 = OLS(x.numpy(), y.numpy()).y_hat()
e0 = (z0 - y.detach().numpy()[:,0])

z = M.net(x)
e1 = (z-y).transpose(1,0).detach().numpy()

plt.scatter(y,e0)
plt.scatter(y,e1)

# from wl_regression import loc_poly
# ll_z = loc_poly(y.numpy(), x.numpy(), x.detach().numpy())
# e2 = (ll_z - y.detach().numpy()[:,0])

# %%
from importlib import reload  
reload(utils)
C, d0, d1 = utils.C_resid(e0, e1, e0, N, "full")
plt.plot(C)
# %%
sigma_hat = utils.compute_w(x,y,torch.tensor(e0),torch.tensor(e1),N)[1]
rslt =utils.test_statistic(C, N, sigma_hat)
print(sigma_hat, rslt)
# %%
# %%
cc =np.sort(np.random.chisquare(1, 100))
