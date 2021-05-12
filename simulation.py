# %%
# Google related
from google.colab import drive
from google.colab import files
import sys
drive.mount("/content/drive")
sys.path.append("/content/drive/My Drive/Colab Notebooks/deep-learning-specification-test")
sys.path.append("/conteπnt/drive/My Drive/Colab Notebooks/python-functions")

# %% 

# Import lib
from importlib import reload
import utils
import utils_simulation
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
# %%
reload(utils)

def repeat(j, N = 100, k = 2, sigma = 0.5):
    start = time.time()
    x, y, e = utils_simulation.generate_sample(N = N, k = k, sigma= sigma)
    M = utils.DNN(utils.MLP3, lr=0.001, max_epochs= 30)

    if torch.cuda.is_available():
        x = x.cuda(0)
        y = y.cuda(0)
        M = M.cuda(0)

    M.net_combine(x,y, batch_size = 1000)
    
    
    from wl_regression import OLS
    z0 = torch.tensor(OLS(x.numpy(), y.numpy()).y_hat())
    e0 = (z0 - y).detach().float()

    z = M.net(x)
    e1 = (z.squeeze()-y).detach().float()

    # from wl_regression import loc_poly
    # ll_z = loc_poly(y.numpy(), x.numpy(), x.detach().numpy())
    # e2 = (ll_z - y.detach().numpy()[:,0])

    C = utils.C_resid(e0, e1, e0, N)
    C1 = utils.C_resid(e0, e1, e1, N)
    sigma_hat = utils.compute_w(x, y, e0, e1, N)[1]
    rslt_cm = utils.test_statistic(C, N, sigma_hat, "CM")
    rslt_ks =utils.test_statistic(C, N, sigma_hat, "KS", C1)

    end = time.time()
    print(f"iter {j}, time {end - start: .2f}, loss {M.loss:.2f}, result {rslt_cm: .1f} and {rslt_ks: .1f} sigma_hat : {sigma_hat: .2f}")
    
    return [N, k, sigma, rslt_cm, rslt_ks, M.loss, sigma_hat]

# %%
i = time.strftime("%Y%m%d%H%M")

def new_func(repeat,sigma):
    rslt_repeat = [repeat(j, N = 200, k = 4, sigma = sigma) for j in range(100)]
    with open(f"result-{i}-{sigma}.p", mode='wb') as f:
        pickle.dump(rslt_repeat, f)
    return rslt_repeat

new_func(repeat,sigma = 0.5)
    # files.download(f'result-{i}-{sigma}.p')
# %%
files.download(f'result-{i}-{sigma}.p')
# %%
with open("result-202105072030-0.5.p", "rb") as f:
    a = pickle.load(f)
cm_lst =torch.tensor([i[3] for i in a])
ax = sns.histplot(cm_lst[cm_lst < 20], stat = "density")
from scipy import stats
xx = np.arange(0.2, +10, 0.001)
yy = stats.chi2.pdf(xx, df=1)
ax.plot(xx, yy, 'r', lw=2)
print(f"the mean is {cm_lst.mean()}")
# %%

rslt_repeat = torch.tensor(rslt_repeat)


# cc = np.sort(np.random.chisquare(1, 10000))
cc = np.abs(np.sort(np.random.normal(0, 1, 10000)))

sns.histplot((rslt_repeat[rslt_repeat < 10]), stat="density")
sns.histplot(cc, stat = "density", color= "orange")
# %%
# %%
# %%
import pickle
with open("/Volumes/GoogleDrive/My Drive/Colab Notebooks/deep-learning-specification-test/result-202105070043-0.3.p" , "rb") as f:
    a = pickle.load(f)
# %%
a = torch.tensor(a).detach()
b = a[:, 0].sort()[0]
print(b)
# %%
b[b<20].mean()
cc = np.sort(np.random.chisquare(1, 10000))
sns.histplot(b[b<20], stat = "density",binrange=(0,10))
sns.histplot(cc, stat="density", binrange=(0, 10), color = "orange")
# %%
