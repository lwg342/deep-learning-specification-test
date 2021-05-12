# %%
from google.colab import drive
from google.colab import files
import sys
drive.mount("/content/drive/")
sys.path.append(
    "/content/drive/My Drive/Colab Notebooks/deep-learning-specification-test")
# %%
import torch
import matplotlib.pyplot as plt
from utils import OLS, C_resid, compute_l
from utils_simulation import describe_distribution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# %% [markdown]
# First we generate the sample, we generate two samples, one is pure white noise and the null is $H_0 : y = e$ and the second sample the null is $H_0 : y = a + bx + e$

# %%
params = {'N' : 1000, 
          'k' : 2,
          'a' : 2,
          'b' : torch.tensor([[0.], [0.]], device = device),
          "sigma" : 1, 
          "sigma_x" : 1,
          }

def reg_func(x, N, k, a, b, device = device, **kwargs):
    y_star = a *torch.ones([N, 1], device = device)  + x @ b
    return y_star

def gen_sample(reg_func, N, k, sigma, sigma_x , **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.normal(0, sigma_x, [N, k], device=device)
    e = torch.normal(0, sigma, [N, 1], device=device)
    y = reg_func(x, N, k, **kwargs) + e
    return x, e, y
# %% 
x, e, y = gen_sample(reg_func, **params)
plt.scatter(x[:,0].cpu(), y.cpu() )
plt.scatter(x[:,1].cpu(), y.cpu() )

# %% [markdown]

# Under the null hypothesis $y = e$, there is no parameter to estimate, we know that $\hat{e}_0 = e$, 
# The alternative should be computed using the deep learning method. 
# Later in this file we will test the null $y = a + xb + e$, where we use oLS to estimate the unknown parameters. 
# %%

from utils import DNN, MLP3
m = DNN( k = params['k'] , W = 3, model = MLP3 , max_epochs = 10)
m.net_combine(x, y, print_details= True)
fig,ax = plt.subplots()
plt.scatter(x[:,1].detach().cpu(), m.net(x).detach().cpu())
plt.scatter(x[:,1].detach().cpu(), y.cpu())
fig.suptitle("y_hat and y against x_1")
fig,ax = plt.subplots()
plt.scatter(x[:,0].detach().cpu(), m.net(x).detach().cpu())
plt.scatter(x[:,0].detach().cpu(), y.cpu())
fig.suptitle("y_hat and y against x_0")

res0 = e
res1 = y - m.net(x)
# %% [markdown]
# Construct $C(y)$ using the Escanciano, et al method by comparing the CDFs of the residuals.
# %%
evals = torch.tensor([0.] , device = device)
cc = C_resid(res0, res1, evals, params['N'])
print(cc)

# %% [markdown]
# Now we do J iterations and see the distribution of C_resid, under the null hypothesis, it should have a normal distribution but the variance needs estimation

# %%
def simulate_cc(reg_func, N, k, evals, sigma, sigma_x, gen_sample = gen_sample, J = 10 , **kwargs):
    lst_cc = []
    for j in range(J):
        x, e, y = gen_sample(reg_func, N, k, sigma, sigma_x, **kwargs)
        
        m = DNN(k=params['k'], W=3, model=MLP3, max_epochs=10)
        m.net_combine(x, y, print_details=False)
        y_hat = m.net(x)
        
        res0 = e
        res1 = y - y_hat
        
        cc = C_resid(res0, res1, evals, N)
        lst_cc += [cc.detach().cpu()]
    return lst_cc

# %%
params['J'] = 1000
params['evals'] = torch.tensor([0.], device = device)
lst_cc = simulate_cc(reg_func, **params)
# %%
print(params)
describe_distribution(lst_cc)
# %% [markdown]
# Now we test the null of linearity. $H_0: y = a + xb + e$. 

# %%
params1 = params.copy()
params1['b'] = torch.tensor([[1.], [3.]], device=device)

x, e, y = gen_sample(reg_func, **params1)
plt.scatter(x[:,0].cpu(), y.cpu() )
plt.scatter(x[:,1].cpu(), y.cpu() )
# %%
params['J'] = 200
params['evals'] = torch.tensor([0.], device=device)
lst_cc = simulate_cc(reg_func, **params1)
describe_distribution(lst_cc)
# %%
params1['J'] = 200
params1['evals'] = torch.tensor([0.], device = device)
def simulate_cc_scaled(reg_func, N, k, evals, sigma, sigma_x, gen_sample = gen_sample, J = 10 , **kwargs):
    lst_cc = []
    lst_cc_scaled = []
    for j in range(J):
        x, e, y = gen_sample(reg_func, N, k, sigma, sigma_x, **kwargs)
        
        m = DNN(k=params['k'], W=3, model=MLP3, max_epochs=10)
        m.net_combine(x, y, print_details=False)
        y_hat = m.net(x)
        
        # res0 = e
        res0 = y - OLS(x, y, N,k).y_hat()
        res1 = y - y_hat

        cc = C_resid(res0, res1, evals, N)
        lst_cc += [cc.detach().cpu()]

        l0 = compute_l(x, y , res0, N, method = "ols")
        l1 = res1
        sigma_hat = (l0-l1).std()
        cc_scaled = cc/ sigma_hat        
        lst_cc_scaled += [cc.detach().cpu()]
        print(j, cc, cc_scaled, sigma_hat)
    return lst_cc


params['J'] = 200
params['evals'] = torch.tensor([0.], device=device)
lst_cc = simulate_cc_scaled(reg_func, **params1)