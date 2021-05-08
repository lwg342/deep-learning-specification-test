import numpy as np
import torch


def C_resid(resid0 , resid1, evals, N, option = None):
    y_mat = torch.outer(evals, torch.ones(N))
    d0 = (y_mat - torch.outer(torch.ones(len(evals)), resid0)).clip(0)
    d1 = (y_mat - torch.outer(torch.ones(len(evals)), resid1)).clip(0)
    C = (1/np.sqrt(N)) * ((d0 - d1).sum(1))
    if option == "full":    
        return C, d0 ,d1
    else:
        return C


def compute_l(x, y, resid, N, method="ols"):
    if method == "ols":
        A = torch.mean(x,0)
        B = (torch.matmul(x.transpose(1,0), x)) /N
        l = torch.matmul(torch.matmul(torch.eye(len(resid))*resid, x),
                         torch.matmul(A, B))
    # if method == "ols": wrong with e_t
        # B = (torch.matmul(x.transpose(1, 0), x)).float()
        # B2 = torch.matmul(x.transpose(1, 0), resid.reshape(T,1)).float()
        # l = torch.matmul(x,torch.matmul(torch.inverse(B),B2)).squeeze()
    if method == "deep learning":
        l = resid 
    return l
# def compute_l(x, y, e, T, method="ols", **kwargs):
#     if method == "ols":
        
#         B = (torch.matmul(x.transpose(1,0), x)) /T
#         l = (torch.matmul(torch.matmul(x, B), x.transpose(1, 0))) * e
#     if method == "deep learning":
#         l = e 
#     return l

def compute_w(x,y,resid0, resid1,T, methods = ["ols", "deep learning"]):
    W0 = compute_l(x, y, resid0, T, methods[0])
    W1 = compute_l(x, y, resid1, T, methods[1])
    W = W0 - W1
    return [W0,W1,W], torch.std(W)

def test_statistic(C0, T, sigma_hat, method="CM", C1 = None):
    if method == "CM":
        # print(method)
        rslt = 3/(sigma_hat)/sigma_hat/ T * torch.inner(C0, C0)
    elif method == "KS":
        # print(method)
        rslt = torch.max(torch.max(torch.abs(C1)), torch.max(torch.abs(C0)))/sigma_hat
    else:
        print("no such method")
        rslt = None
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

class MLP3(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = torch.nn.Linear(4, 5)
        self.f2 = torch.nn.Linear(5, 5)
        self.f3 = torch.nn.Linear(5, 5)
        self.predict = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.f1(x))
        x = torch.nn.functional.relu(self.f2(x))
        x = torch.nn.functional.relu(self.f3(x))
        out = self.predict(x)
        return out

class DNN():
    def __init__(self, model=MLP4, lr=0.0001, max_epochs = 50):
        self.net = model()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr= lr)
        self.loss_func = torch.nn.MSELoss()
        self.previous_loss = 1000
        self.max_epochs = max_epochs

    def net_update(self, x, y, batch_size = 1000):
        for t in range(batch_size):
            prediction = self.net(x).squeeze()     # input x and predict based on x
            loss = self.loss_func(prediction, y)     # must be (1. nn output, 2. target)
            loss.backward()         # backpropagation, compute gradients
            self.optimizer.step()        # apply gradients
            self.optimizer.zero_grad()   # clear gradients for next train
        self.loss = loss

    def net_combine(self, x, y, batch_size = 1000):
        self.net_update(x,y,batch_size)
        i= 0
        while (self.previous_loss - self.loss > 0.05 or self.loss > 0.5) and (i < self.max_epochs):
            self.previous_loss = self.loss
            self.net_update(x,y,batch_size)
            i = i + 1
        # print(i)
