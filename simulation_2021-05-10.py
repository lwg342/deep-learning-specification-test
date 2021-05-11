# %% [markdown]
""" This file tests the null $y = a + e$ against the alternative that 
$y = a + x \beta + e$"""
# %%
from utils import compute_l
from utils import OLS, C_resid
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from scipy import stats

# %% [markdown]
"""Generate samples. """
# %%


def reg_func(x, N, k):
    y_star = torch.ones([N, 1])
    return y_star


N, k, sigma, sigma_x = 1000, 1, 1, 1


def gen_sample(reg_func, N, k, sigma, sigma_x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.normal(0, sigma_x, [N, k], device=device)
    e = torch.normal(0, sigma, [N, 1], device=device)
    y = reg_func(x, N, k) + e
    return x, e, y


x, e, y = gen_sample(reg_func, N, k, sigma, sigma_x)

# %% [markdown]
"""
Now we follow Escanciano, find the $\hat{e}_{t0}$ and $\hat{e}_{t1}$ and construct $C(y), we plotted some descriptive graphs$
"""
res0 = y - y.mean()
plt.figure()
plt.plot(e)
plt.plot(res0)

y_hat = OLS(x, y, N, k).y_hat(add_intercept=True)
plt.figure()
plt.scatter(x, y)
plt.scatter(x, y_hat)

res1 = y - y_hat
plt.figure()
plt.plot(e)
plt.plot(res1)

res_diff = res0 - res1
plt.figure()
plt.plot(res_diff)

"""Describe the moments"""
print(f"mean {res_diff.mean(): .5f} , variance {res_diff.var(): .5f}")

# %% [markdown]
"""Construct $C(y)$"""
evals = torch.tensor([0.])
cc = C_resid(res0, res1, evals, N)
print(cc)


# %% [markdown]
"""Now we do J iterations and see the distribution of C_resid, under the null hypothesis, it should have a normal distribution but the variance needs estimation"""


def simulate_cc(reg_func, N, k, sigma, sigma_x, gen_sample, J):
    lst_cc = []
    for j in range(J):
        x, e, y = gen_sample(reg_func, N, k, sigma, sigma_x)
        evals = torch.tensor([0.])
        res0 = y - y.mean()
        y_hat = OLS(x, y, N, k).y_hat(add_intercept=True)
        # beta = OLS(x, y, N, k).beta_hat()
        res1 = y - y_hat
        cc = C_resid(res0, res1, evals, N)
        lst_cc += [cc.detach().cpu()]
    return lst_cc


def describe(lst_cc):
    t_cc = torch.tensor(lst_cc)
    fig, ax = plt.subplots()
    ax = sns.histplot(t_cc, stat="density")
    ax.set_title(
        f"sample size : {N}, mean {t_cc.mean(): .2f}, variance {t_cc.var(): .5f}, skew: {stats.skew(t_cc.numpy()) : .3f}, kurtosis : {stats.kurtosis(t_cc.numpy()) : .3f}")


# %%
"""We simulate J times, with sample size N"""

N, k, sigma, sigma_x = int(1e6), 1, 1, 1
J = 1000
lst_cc = simulate_cc(reg_func, N, k, sigma, sigma_x, gen_sample, J)

# %% [markdown]
"""Now we analyse the distribution of the resulting list of C_resids"""
                
describe(lst_cc)

# %%
"""We test with different sample sizes N"""
for N in [1000, 2000, 5000, int(1e4), int(1e5), int(2e5), int(1e6), int(2e6)]:
    lst_cc = simulate_cc(reg_func, N, k, sigma, sigma_x, gen_sample, J)
    describe(lst_cc)


# %% [markdown]
"""We also want to test if we can estimate the variance correctly"""


N, k, sigma, sigma_x = 1000, 1, 1, 1
x, e, y = gen_sample(reg_func, N, k, sigma, sigma_x)

"""we compute l0 based on OLS residual from H_0"""

res0 = y - y.mean()
plt.scatter(x, e)
plt.scatter(x, res0)

l0 = compute_l(torch.ones([N, 1]), y, res0, N, method="ols")
torch.all(l0 == res0)

"""Now we compute l1 based on OLS residual from H_1"""

m = OLS(x, y, N, k)
res1 = y - m.y_hat(add_intercept=True)
l1 = compute_l(m.IX, y, res1, N, "ols")

fig, ax = plt.subplots()
plt.scatter(x, e)
plt.scatter(x, res1)
plt.scatter(x, l1)

# %%
fig, ax = plt.subplots()
plt.scatter(x, l0 - l1)

print((l0 - l1).var())

# %%
""" Now we find the scaled C, takin into account the estimated sigma_hat"""


def simulate_scaled_cc(reg_func, N, k, sigma, sigma_x, gen_sample, J):
    lst_cc_scaled = []
    lst_cc = []
    lst_sigma_hat = []
    for j in range(J):
        x, e, y = gen_sample(reg_func, N, k, sigma, sigma_x)
        evals = torch.tensor([0.])
        res0 = y - y.mean()
        m = OLS(x, y, N, k)
        y_hat = m.y_hat(add_intercept=True)
        res1 = y - y_hat
        
        l0 = compute_l(torch.ones([N, 1]), y, res0, N, method="ols")
        l1 = compute_l(m.IX, y, res1, N, "ols")
        sigma_hat = (l0 - l1).std()
        
        cc = C_resid(res0, res1, evals, N)
        scaled_cc = cc/sigma_hat
        lst_cc += [cc]
        lst_cc_scaled += [scaled_cc.detach().cpu()]
        lst_sigma_hat += [sigma_hat]
    return lst_cc_scaled, lst_sigma_hat, lst_cc

# %%
J = 1000
N = int(1e4)
lst_cc_scaled, lst_sigma_hat, lst_cc = simulate_scaled_cc(reg_func, N, k, sigma, sigma_x, gen_sample, J)
describe(lst_cc_scaled)
describe(lst_sigma_hat)
describe(lst_cc)

# %%
