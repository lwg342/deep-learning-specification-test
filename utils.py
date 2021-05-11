import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def C_resid(res0, res1, evals, N, option=None, device = device):

    evals_mat = evals @ torch.ones([1, N])
    l = len(evals)
    d0 = (evals_mat - (torch.ones([l, 1], device = device) @ res0.T)).clip(0)
    d1 = (evals_mat - (torch.ones([l, 1], device = device) @ res1.T)).clip(0)
    C = (1/np.sqrt(N)) * ((d0 - d1).sum(1))
    if option == "full":
        return C, d0, d1
    else:
        return C


def compute_l(x, y, res, N, method="ols"):
    if method == "ols":
        A = torch.mean(x, 0).reshape([1, x.shape[1]])
        V = (x.T@x) / N
        # l = torch.matmul(torch.matmul(torch.eye(len(res))*res, x),
        #  torch.matmul(A, B))
        V2 = x.T @ torch.eye(N)@(res)
        l = (A@V@V2).T
    if method == "deep learning":
        l = res
    return l


# def compute_w(x, y, resid0, resid1, N, methods=["ols", "deep learning"]):
    # W0 = compute_l(x, y, resid0, N, methods[0])
    # W1 = compute_l(x, y, resid1, N, methods[1])
    # W = W0 - W1
    # return [W0, W1, W], torch.std(W)


def test_statistic(C0, T, sigma_hat, method="CM", C1=None):
    if method == "CM":
        # print(method)
        rslt = 3/(sigma_hat)/sigma_hat / T * torch.inner(C0, C0)
    elif method == "KS":
        # print(method)
        rslt = torch.max(torch.max(torch.abs(C1)),
                         torch.max(torch.abs(C0)))/sigma_hat
    else:
        print("no such method")
        rslt = None
    return rslt


# %% This is about constructing the neural net
class MLP4(torch.nn.Module):
    def __init__(self, k, W) -> None:
        super().__init__()
        self.f1 = torch.nn.Linear(k, W)
        self.f2 = torch.nn.Linear(W, W)
        self.f3 = torch.nn.Linear(W, W)
        self.f4 = torch.nn.Linear(W, W)
        self.predict = torch.nn.Linear(W, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.f1(x))
        x = torch.nn.functional.relu(self.f2(x))
        x = torch.nn.functional.relu(self.f3(x))
        x = torch.nn.functional.relu(self.f4(x))
        out = self.predict(x)
        return out


class MLP3(torch.nn.Module):
    def __init__(self, k, W) -> None:
        super().__init__()
        self.f1 = torch.nn.Linear(k, W)
        self.f2 = torch.nn.Linear(W, W)
        self.f3 = torch.nn.Linear(W, W)
        self.predict = torch.nn.Linear(W, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.f1(x))
        x = torch.nn.functional.relu(self.f2(x))
        x = torch.nn.functional.relu(self.f3(x))
        out = self.predict(x)
        return out


class DNN():
    def __init__(self, k=4, W=4,  model=MLP4, lr=0.001, max_epochs=50, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.net = model(k=k, W=4).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_func = torch.nn.MSELoss()
        self.previous_loss = 1000
        self.max_epochs = max_epochs

    def net_update(self, x, y, batch_size=1000):
        for t in range(batch_size):
            prediction = self.net(x)    # input x and predict based on x
            # must be (1. nn output, 2. target)
            loss = self.loss_func(prediction, y)
            loss.backward()         # backpropagation, compute gradients
            self.optimizer.step()        # apply gradients
            self.optimizer.zero_grad()   # clear gradients for next train
        self.loss = loss

    def net_combine(self, x, y, batch_size=1000, print_details=False):
        self.net_update(x, y, batch_size)
        i = 0
        while (self.previous_loss - self.loss > 0.05 or self.loss > 0.5) and (i < self.max_epochs):
            self.previous_loss = self.loss
            self.net_update(x, y, batch_size)
            i = i + 1
            if print_details:
                print(f"iters {i}: loss {self.loss : .2f}")
        # print(i)


class OLS():
    def __init__(self, X: torch.tensor, Y: torch.tensor, N, k):
        """
        X is N*k dimensional and Y is N*1 dimensional. 
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.N = N
        self.k = k
        self.beta_est = None
        self.y_est = None
        self.X = X
        self.Y = Y
        self.IX = torch.cat(
            [torch.ones([self.N, 1], device=self.device), self.X], 1)

    def beta_hat(self, add_intercept=True):
        if self.beta_est != None:
            return self.beta_est
        else:
            if add_intercept:
                X = self.IX
            else:
                X = self.X
            Y = self.Y
            V = torch.inverse(X.transpose(1, 0)@X)
            W = X.transpose(1, 0)@Y
            beta = torch.matmul(V, W)
            self.beta_est = beta
            return beta

    def y_hat(self, add_intercept=True):
        if add_intercept:
            X = self.IX
        else:
            X = self.X
        beta = self.beta_hat(add_intercept=add_intercept)

        y_hat = X@beta
        return y_hat
