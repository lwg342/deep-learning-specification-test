import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def true_output(x, option = "linear"):
    # y = torch.abs(x[:, 0]) + 2 * (x[:,1]) + 3 * x[:,2] + 4*x[:,3]
    if option == "linear":
        y = torch.matmul(x,  (torch.arange(x.shape[1]) + 1).float())
    return y

def generate_sample(N, k, sigma, reg_func = true_output, device = device):
    x = torch.normal(0, 1, [N,k] , device = device)
    e = sigma * torch.randn([N,1], device = device)
    y = (reg_func(x) + e).float()
    return x,y,e 


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
def describe_distribution(lst_cc):
    N = len(lst_cc)
    t_cc = torch.tensor(lst_cc)
    fig, ax = plt.subplots()
    ax = sns.histplot(t_cc, stat="density")
    ax.set_title(
        f"sample size : {N}, mean {t_cc.mean(): .2f}, variance {t_cc.var(): .5f}, skew: {stats.skew(t_cc.numpy()) : .3f}, kurtosis : {stats.kurtosis(t_cc.numpy()) : .3f}")

