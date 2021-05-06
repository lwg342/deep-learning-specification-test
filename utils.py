import numpy as np
import torch


def C_resid(e0 , e1, y, T, option = None):
    y_mat = np.outer(y, np.ones(T))
    d0 = (y_mat - np.outer(np.ones(len(y)), e0)).clip(0)
    d1 = (y_mat - np.outer(np.ones(len(y)), e1)).clip(0)
    C = (1/np.sqrt(T)) * ((d0 - d1).sum(1))
    if option == "full":    
        return C, d0 ,d1
    else:
        return C


def compute_l(x, y, e, T, method="ols"):
    if method == "ols":
        A = torch.mean(x,0)
        B = (torch.matmul(x.transpose(1,0), x)) /T
        l = (torch.matmul(torch.matmul(A, B), x.transpose(1, 0))) * e
    if method == "deep learning":
        l = e 
    return l

def compute_w(x,y,e0, e1,T, methods = ["ols", "deep learning"]):
    W0 = compute_l(x, y, e0, T, methods[0])
    W1 = compute_l(x, y, e1, T, methods[1])
    W = W0 - W1
    return W, torch.std(W)

def test_statistic(C, T, sigma_hat, method="CM"):
    if method == "CM":
        rslt = 3/(sigma_hat)**2 / T * np.inner(C, C)
    return rslt

# %% This is about constructing the neural net
class MLP4(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = torch.nn.Linear(4, 5)
        self.f2 = torch.nn.Linear(5, 5)
        self.f3 = torch.nn.Linear(5, 5)
        self.f4 = torch.nn.Linear(5, 5)
        self.predict = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.f1(x))
        x = torch.nn.functional.relu(self.f2(x))
        x = torch.nn.functional.relu(self.f3(x))
        x = torch.nn.functional.relu(self.f4(x))
        out = self.predict(x)
        return out


class wl_nn():
    def __init__(self, model=MLP4, lr=0.0001):
        self.net = model()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr= lr)
        self.loss_func = torch.nn.MSELoss()
        self.previous_loss = 0

    def net_update(self, x, y, batch_size = 10000):
        for t in range(batch_size):
            prediction = self.net(x)     # input x and predict based on x
            loss = self.loss_func(prediction, y)     # must be (1. nn output, 2. target)
            loss.backward()         # backpropagation, compute gradients
            self.optimizer.step()        # apply gradients
            self.optimizer.zero_grad()   # clear gradients for next train
        self.loss = loss

    
