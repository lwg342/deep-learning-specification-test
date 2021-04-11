# %%
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# %%
torch.manual_seed(1)    # reproducible

N = 1000

rng = np.random.default_rng()
x1 = torch.from_numpy(rng.normal(0, 1, N))
x2 = torch.from_numpy(rng.normal(0, 1, N))
e = torch.rand(N)


def true_output(x1, x2):
    y = x1.pow(2) + 2* torch.sin(x2) + x1*x2
    return y


y = (true_output(x1, x2) + e).unsqueeze(-1).float()

plt.scatter(x1, y)
plt.scatter(x2, y)

# %%


def threeD_plot(option  = "true"):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax_x = np.linspace(-2, 2, 30)
    ax_y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(ax_x, ax_y)
    if option == "true": 
        Z = true_output(torch.from_numpy(X), torch.from_numpy(Y))
    ax.contour3D(X, Y, Z, 50)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

threeD_plot()

# %%
x1.unsqueeze(-1).float()
x2.unsqueeze(-1).float()
x = torch.stack((x1,x2)).transpose(0,1).float()


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = nn.Linear(2, 5)
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
    #     # plot and show learning process
    #     plt.cla()
    #     plt.scatter(x.data.numpy(), y.data.numpy())
    #     plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    #     plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
    #     plt.pause(0.1)
# %%

fig = plt.figure()
ax = Axes3D(fig)

sequence_x_vals = np.arange(-2,2,0.1)
sequence_y_vals = np.arange(-2,2,0.1)
fl_vals = np.array(np.meshgrid(
    sequence_x_vals, sequence_y_vals)).reshape(2, len(sequence_x_vals)*len(sequence_y_vals))
sequence_z_vals = net(torch.from_numpy(fl_vals).transpose(0,1).float())
ax.scatter(fl_vals[0,:],fl_vals[1,:], sequence_z_vals.detach().numpy())
ax.scatter(x[:,0], x[:,1], y)
fig.show()

# %%
plt.scatter(x[:,0], y)
plt.scatter(x[:,0], net(x).detach().numpy())
# %%

from wl_regression import loc_poly
ll_z = loc_poly(y.numpy(), x.numpy(), fl_vals.transpose())
# %%
fig,ax = plt.subplots()
ax = Axes3D(fig)
ax.scatter(fl_vals[0, :], fl_vals[1, :],ll_z)
fig.show()
# %%
# True Loss 
loss_func(ll_z, true_output(fl_vals[0, :], fl_vals[1, :]))

# %%
