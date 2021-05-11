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

# def ols(y,x, N, k):
#     x = x.reshape(N,k)
#     y = y.reshape(N,1)
#     V = torch.inverse(torch.matmul(x.transpose(1, 0), x)) 
#     W = torch.matmul(x.transpose(1,0), y)
#     beta_hat = torch.matmul(V, W)
#     y_hat = torch.matmul(x, beta_hat).squeeze()
#     return y_hat, beta_hat
