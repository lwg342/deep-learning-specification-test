# %%
from importlib import reload
import utils
from google.colab import files
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
from google.colab import drive
drive.mount("/content/drive")
sys.path.append(
    "/content/drive/My Drive/Colab Notebooks/deep-learning-specification-test")
sys.path.append("/content/drive/My Drive/Colab Notebooks/python-functions")
# %%
reload(utils)

N = 200
num_repeat = 20000


def true_output(x):
    # y = x1.pow(2) + 2* torch.sin(x2) + x1*x2 + x3 * x4
    y = x[:, 0] + 2 * x[:, 1] + 3 * x[:, 2] + 4*x[:, 3]
    return y


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = nn.Linear(4, 5)
        self.f2 = nn.Linear(5, 5)
        self.f3 = nn.Linear(5, 5)
        self.f4 = nn.Linear(5, 5)
        self.predict = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.relu(self.f4(x))
        out = self.predict(x)
        return out


def repeat(j):
    start = time.time()
    x = torch.normal(0, 1, size=[N, 4])
    e = 0.2*torch.randn(N)
    y = (true_output(x) + e).unsqueeze(-1).float()

    net = Net()

    if torch.cuda.is_available():
        x = x.cuda(0)
        y = y.cuda(0)
        net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()

    for t in range(num_repeat):

        prediction = net(x)     # input x and predict based on x
        loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    for t in range(num_repeat):

        prediction = net(x)     # input x and predict based on x
        loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        if t == num_repeat-1:
            print(loss)

    # Test the Escanciano method

    if torch.cuda.is_available():
        y = y.cpu()
        z = net(x).cpu()

    from wl_regression import OLS
    z0 = OLS(x.numpy(), y.numpy()).y_hat()
    e0 = (z0 - y.detach().numpy()[:, 0])

    z = net(x)
    e1 = (z-y).detach().numpy()

    # from wl_regression import loc_poly
    # ll_z = loc_poly(y.numpy(), x.numpy(), x.detach().numpy())
    # e2 = (ll_z - y.detach().numpy()[:,0])

    C_resid = utils.C_resid
    C = C_resid(e0, e1, e0, N)
    # print(C)

    test_statistic = utils.test_statistic
    sigma_hat = utils.compute_w(x, y, e0, e1, N)[1]
    rslt = test_statistic(C, N, sigma_hat)

    end = time.time()
    print(f"{j}-th iter in time {end - start}")
    return rslt

# %%


rslt_repeat = [repeat(j) for j in range(100)]

i = time.strftime("%Y%m%d%H")
with open(f"result-{i}.p", mode='wb') as f:
    pickle.dump(rslt_repeat, f)

files.download(f'result-{i}.p')
