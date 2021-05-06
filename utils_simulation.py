import torch

def true_output(x):
    # y = torch.abs(x[:, 0]) + 2 * (x[:,1]) + 3 * x[:,2] + 4*x[:,3]
    y = (x[:, 0]) + 2 * (x[:,1]) + 3 * x[:,2] + 4*x[:,3]
    return y

def generate_sample(N, k, sigma):
    x = torch.normal(0, 1, [N,k])
    e = sigma * torch.randn(N)
    y = (true_output(x) + e).float()
    return x,y,e 
