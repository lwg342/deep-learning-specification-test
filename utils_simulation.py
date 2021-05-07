import torch

def true_output(x, option = "linear"):
    # y = torch.abs(x[:, 0]) + 2 * (x[:,1]) + 3 * x[:,2] + 4*x[:,3]
    if option == "linear":
        y = torch.matmul(x,  (torch.arange(x.shape[1]) + 1).float())
    return y

def generate_sample(N, k, sigma):
    x = torch.normal(0, 1, [N,k])
    e = sigma * torch.randn(N)
    y = (true_output(x) + e).float()
    return x,y,e 
