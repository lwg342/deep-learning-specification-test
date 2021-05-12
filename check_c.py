# %%
from scipy import stats
import torch
import seaborn as sns
from utils import C_resid, compute_w, test_statistic
import matplotlib.pyplot as plt
import numpy as np
# %%


def gen_C(yloc, obs=1000):
    e0 = torch.normal(0, 1, [obs])
    # e1 = torch.normal(0, 1, [obs])
    e1 = e0 - e0.mean()
    C= C_resid(e0, e1, torch.tensor(yloc), len(e0))
    return C

# %%
yloc = [-1]
obs = 1000
a = [gen_C(yloc, obs) for i in range(100)]
# %%
a = torch.tensor(a)
ax = sns.histplot(a, stat = "density")
print(a.mean())
print(a.var())
xx = np.arange(-2, +2, 0.001)
yy = stats.norm.pdf(xx, scale = a.std())
ax.plot(xx, yy, 'r', lw=2)
# %%


# %%
e0 = torch.normal(0, 1, [obs])
e1 = e0 - e0.mean()
# e1 = torch.normal(0, 1, [100000])
sigma_hat = compute_w(0,0, e0, e1, len(e0), methods = ['deep learning', "deep learning"])[1]
print(sigma_hat)
# %%

print(stats.norm.cdf(yloc))
ratio = a.var() / (stats.norm.cdf(yloc)*sigma_hat.numpy())**2
print(ratio)
# %%
import numpy as np
v = []
cdf = []
sigma_hat_lst = []
yrange = np.linspace(-4, 5, 100)
for yloc in yrange:
    a = [gen_C([yloc], 1000) for j in range(1000)]
    a = torch.tensor(a)
    v = v + [a.var().numpy()]
    cdf += [stats.norm.cdf(yloc)]
    sigma_hat_lst += [compute_w(0, 0, e0, e1, len(e0), methods=['deep learning', "deep learning"])]
# %%
cdf = np.array(cdf)
v = np.array(v)
rr = v/cdf**2
print(rr)
plt.plot(yrange,rr)
# %%
plt.plot(cdf, v)

# %%

def gen_cm(obs):
    e0 = torch.normal(0, 6, [obs])
    e1 = e0 - e0.mean()
    C= C_resid(e0, e1, e0, len(e0))
    sigma_hat = compute_w(0, 0, 0, e1, len(e0), methods=[
                        'deep learning', "deep learning"])[1]
    cm = test_statistic(C, obs, 6)
    print(sigma_hat)
    print(e0.std())
    return cm

# %%
lst_cm = torch.tensor([gen_cm(obs= 100) for j in range(10)])
# %%

ax = sns.histplot(lst_cm, stat = "density", binrange= (0,10), binwidth=0.1)
xx = np.arange(0.2, +10, 0.001)
yy = stats.chi2.pdf(xx,df =1 )
zz = stats.chi2.pdf(xx,df =2 )
ax.plot(xx, yy, 'r', lw=2)
ax.plot(xx, zz, 'y', lw=2)
print(lst_cm.mean())

# %%  one-dimensional 
cm_lst = []
for j in range(1000):
# check the standard deviation is correct
    obs = 1000 
    e0 = torch.normal(0, 6, [obs])
    from utils import MLP3, DNN
    class MLP2(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.f1 = torch.nn.Linear(1, 3)
            self.f2 = torch.nn.Linear(3, 5)
            self.predict = torch.nn.Linear(5, 1)

        def forward(self, x):
            x = torch.nn.functional.relu(self.f1(x))
            x = torch.nn.functional.relu(self.f2(x))
            out = self.predict(x)
            return out
    M = DNN(MLP2, max_epochs= 10)
    x = torch.linspace(100,100,obs).reshape(-1, 1) # one-dim 
    # x = torch.normal(0., 1., size = [obs]).reshape(-1, 1)# one-dim 
    y = 4 * x.squeeze() + e0 # one-dim
    M.net_combine(x, y)
    # plt.scatter(M.net(x).detach(), e0.detach())
    res = e0 - M.net(x).detach().squeeze()
    beta = torch.inner(x.squeeze(), y) / torch.inner(x.squeeze(), x.squeeze())
    res0 = y - beta * x.squeeze()
    C = C_resid(res0, res, res0, len(e0))
    sigma_hat = compute_w(x, y, res0, res, len(e0), methods=[
        'ols', "deep learning"])[1]
    cm = test_statistic(C, obs, sigma_hat)
    cm_lst += [cm]
    if j %100 == 0:
        print(j)
    print(f'{sigma_hat}, {cm}, {M.loss}, {beta}')

# %%
import numpy as np
ss = torch.tensor(cm_lst)
ax = sns.histplot(ss/ss.mean(), stat = "density")
xx = np.arange(0.2, +10, 0.001)
yy = stats.chi2.pdf(xx, df=1)
ax.plot(xx, yy, 'r', lw=2)
print(len(cm_lst))
print(ss.mean(), ss.quantile(0.95), (ss/ss.mean()).quantile(0.95))
# %%
# multi-dimensional
cm_lst = []
for j in range(1000):
# check the standard deviation is correct
    obs = 1000 
    e0 = torch.normal(0, 6, [obs])
    from utils import MLP3, DNN
    class MLP2(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.f1 = torch.nn.Linear(2, 3)
            self.f2 = torch.nn.Linear(3, 5)
            self.predict = torch.nn.Linear(5, 1)

        def forward(self, x):
            x = torch.nn.functional.relu(self.f1(x))
            x = torch.nn.functional.relu(self.f2(x))
            out = self.predict(x)
            return out
    M = DNN(MLP2, max_epochs= 5)
    # x = torch.outer(torch.linspace(-3, 3, obs),
                    # torch.ones(2))
    y = torch.matmul(x, torch.tensor([[1.], [2.]])).squeeze() + e0

    M.net_combine(x, y)
    res = e0 - M.net(x).detach().squeeze()
    beta = torch.matmul((torch.inverse(torch.matmul(
        x.transpose(1, 0), x))), torch.matmul(x.transpose(1, 0), y))
    res0 = y - torch.matmul(x, beta).squeeze()
    C = C_resid(res0, res, res0, len(e0))
    sigma_hat = compute_w(x, y, res0, res, len(e0), methods=[
        'ols', "deep learning"])[1]
    cm = test_statistic(C, obs, sigma_hat)
    cm_lst += [cm]
    if j %100 == 0:
        print(j)
    print(f'{sigma_hat}, {cm}, {M.loss}, {beta}')
