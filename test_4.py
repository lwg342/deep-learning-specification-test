# %%
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# %%
torch.manual_seed(1)    # reproducible

N = 200
rng = np.random.default_rng()
x1 = torch.from_numpy(rng.normal(0, 1, N))
x2 = torch.from_numpy(rng.normal(0, 1, N))
x3 = torch.from_numpy(rng.normal(0, 1, N))
x4 = torch.from_numpy(rng.normal(0, 1, N))
e = 0.2*torch.randn(N)


def true_output(x1, x2, x3, x4):
    # y = x1.pow(2) + 2* torch.sin(x2) + x1*x2 + x3 * x4
    y = x1 + 2* x2 + 3 * x3 + 4*x4
    return y


y = (true_output(x1, x2, x3, x4) + e).unsqueeze(-1).float()

plt.scatter(x1, y)
plt.scatter(x2, y)


# %%
x1.unsqueeze(-1).float()
x2.unsqueeze(-1).float()
x3.unsqueeze(-1).float()
x4.unsqueeze(-1).float()
x = torch.stack((x1,x2, x3, x4)).transpose(0,1).float()


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


net = Net()
print(net(x)[0:5])
# %%

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  
# %%

for t in range(10000):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)


    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 100 == 0:
        print(loss)
# %%
plt.plot(y,net(x))

# %%

# Test the Escanciano method
from wl_regression import OLS  
z0 = OLS(x.numpy(), y.numpy()).y_hat()
e0 = (z0 - y.detach().numpy()[:,0])

z = net(x)
e1 = (z-y).detach().numpy()

from wl_regression import loc_poly
ll_z = loc_poly(y.numpy(), x.numpy(), x.detach().numpy())
e2 = (ll_z - y.detach().numpy()[:,0])

# %%
import utils 
from importlib import reload  
reload(utils)
C_resid = utils.C_resid
C = C_resid(e0, e1, e0, N)
plt.plot(C)
print(C)
# %%

test_statistic = utils.test_statistic
rslt = test_statistic(C, N)
print(rslt)
# %%

