# %%
import torch as th 
import matplotlib.pyplot as plt

def compute_l(x, y, e, N, method="ols"):
    if method == "ols":
        B = (th.matmul(x.transpose(1, 0), x)).float()
        B2 = th.matmul(x.transpose(1, 0), e.reshape(N,1)).float()
        l = x.float()*th.matmul(th.inverse(B),B2)
    if method == "deep learning":
        l = e
    return l

# %%
x= th.normal(0,1,[1000])
e= th.normal(0,1,[1000])
N = len(e) 

y = x *2 + e
# %%
beta_hat = (x.matmul(x))**(-1)*(x.matmul(y))
print(beta_hat)
# %%
res = y - x*beta_hat
plt.plot(res)
# %%
compute_l(x,y,res, N)

# %%
