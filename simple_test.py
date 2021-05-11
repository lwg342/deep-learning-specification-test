# %% 
"""
This is a simple test, null y = e
alternative: y = a + e
"""
import seaborn as sns
import numpy as np
import torch
from utils import C_resid, compute_w, test_statistic, compute_l
from scipy import stats
import matplotlib.pyplot as plt

N = 1000
lst_tt = []
lst_cc = []
evals_loc = [0]
for j in range(1000):
    x = torch.ones(N)
    e = torch.normal(0, 1, size=[N])
    y = 1 + e

    resid0 = e
    resid = y - y.mean()

    l0 = 0 
    l1 = resid

    cc0 = C_resid(resid0, resid, resid0, N)
    cc = C_resid(resid0, resid, torch.tensor(evals_loc), N)
    lst_cc += [cc]
    sigma_hat = resid.std()

    tt = test_statistic(cc0, len(e), sigma_hat)
    lst_tt += [tt]
    # print(cc, sigma_hat, tt)
        
# %%
def plot_chi(lst_tt):
    lst_tt_tensor = torch.tensor(lst_tt)
    print(f"quantile {lst_tt_tensor.quantile(0.95): .2f}, mean {lst_tt_tensor.mean(): .2f}")
    ax = sns.histplot(lst_tt_tensor, stat = "density")
    xx = np.arange(0.2, +10, 0.001)
    yy = stats.chi2.pdf(xx,df =1 )
    ax.plot(xx, yy, 'r', lw=2)

# %%
"""
Another simple test, null y = a + e
alternative y = a + beta x + e
"""
def ols(y,x, N, k):
    x = x.reshape(N,k)
    y = y.reshape(N,1)
    V = torch.inverse(torch.matmul(x.transpose(1, 0), x)) 
    W = torch.matmul(x.transpose(1,0), y)
    beta_hat = torch.matmul(V, W)
    return beta_hat
# %%    
lst_tt = []
for j in range(100):
    x = torch.ones(N)
    x1 = torch.linspace(-1,1,N)
    e = torch.normal(0, 2, size=[N])
    y = 1 + 0* x1 + e
    
    x = torch.cat((x.reshape(N, 1),x1.reshape(N, 1)), 1)

    resid0 = y - y.mean()
    resid = y - torch.matmul(x, ols(y,x,N,2)).squeeze()
    
    l0 = resid0
    l1 = compute_l(x,y,resid, len(e), method = 'ols')

    lst_cc = C_resid(resid0, resid, resid0, len(e))
    sigma_hat = (l0 - l1).std()
    # print(sigma_hat)

    tt = test_statistic(lst_cc, len(e), sigma_hat)
    lst_tt += [tt]
    # print(cc, sigma_hat, tt)


#   %%

sns.scatterplot(resid0, resid)
sns.scatterplot(resid0, resid0)
# %%
"""III"""
ii = []
lst_cc = []
lst_cc_infi = []
lst_cc_scaled = []
# evals_loc = [-3, -2, -1, 0, 1, 2, 3]
evals_loc = [0]
N = 1000
for j in range(1000):
    x0 = torch.ones(N)
    # x1 = torch.linspace(-1, 1, N)
    x1 = torch.normal(0, 1, [N])
    e = torch.normal(0, 1, size=[N])
    y = 1 + 0 * x1 + e

    x = torch.cat((x0.reshape(N, 1), x1.reshape(N, 1)), 1)

    resid0 = y - y.mean()
    beta_hat = ols(y, x, N, 2)
    resid = y - torch.matmul(x, beta_hat).squeeze()
    
    cc = C_resid(resid0, resid, torch.tensor(evals_loc), N)
    lst_cc += [cc]
    
    resid0_infeasible = e 
    cc_infiseable = C_resid(resid0_infeasible, resid, torch.tensor(evals_loc), N)
    lst_cc_infi += [cc_infiseable]
    
    cc_scaled =  cc/((resid0- resid).std())
    lst_cc_scaled += [cc_scaled]
    
    
    # print(f"{cc}, {cc_infiseable}, beta : {ols(y, x, N, 2)}")
    
# %%
# sns.histplot(torch.tensor(lst_cc))
# sns.histplot(torch.tensor(lst_cc_infi))
plt.figure()
sns.histplot(torch.tensor(lst_cc_infi))
sns.histplot(torch.tensor(lst_cc_scaled), color = "orange")
print(f"mean {torch.tensor(lst_cc_scaled).mean(): .2f}, variance {torch.tensor(lst_cc_scaled ).var(): .5f}, skew: {stats.skew(torch.tensor(lst_cc_scaled).numpy()) : .3f}, kurtosis : {stats.kurtosis(torch.tensor(lst_cc_scaled).numpy()) : .3f}")
print(f"mean {torch.tensor(lst_cc_infi).mean(): .2f}, variance {torch.tensor(lst_cc_infi ).var(): .5f}, skew: {stats.skew(torch.tensor(lst_cc_infi).numpy()) : .3f}, kurtosis : {stats.kurtosis(torch.tensor(lst_cc_infi).numpy()) : .3f}")
# %%
