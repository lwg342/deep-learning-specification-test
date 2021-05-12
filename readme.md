# Descriptions
- [utils](#utils)
- [utils_simulation](#utils_simulation)
- [Simulations](#simulations)
  - [simulation_2021-05-10](#simulation_2021-05-10)
  - [simulation_2021-05-12](#simulation_2021-05-12)
- [实验记录和TODO](#实验记录和todo)

## utils

用于做test的一些函数：
- Class OLS and DNN, with some MLP models
- Function to compute the C(y) statistic in Escanciano, and $l_0, l_1$
- Given std estimation, implement CM and KS statistics. 

## utils_simulation

- function to generate samples
- function to create the truth regression function
- describe_distribution: will plot the histplot of a tensor and show the first four moments. 
- simulate_cc: do J iteration simutions and return the values of $C(y)$. 


## Simulations 

### [simulation_2021-05-10](simulation_2021-05-10.ipynb)

-  test the null $H_0: y = a + e$ against the alternative $H_1: y  = a + bx + e$. Simulated the distribution of $C(y)$ which should be a normal distribution, and after scaling should have variance $\Phi(y)^2$, $\Phi$ being the normal CDF. 
-  目前还没有试最终的test statistics， 但是 $C(y)$的distribution 和 scaled $C(y)$ 的distribution， 当sample size 很大的时候，接近normal with correct variance。
-  不过convergence rate 很慢， 需要sample size ～ 1e6。 Maybe need correction?
-  更大的sample size确实会溢出，这个很难处理。

### [simulation_2021-05-12](simulation_2021-05-12.ipynb)

- Test the null: $H_0 : y = e$, and $H_0 : y = a + bx + e$ against alternative with deep learning. 
- <mark>It's been noticed before that if the null is pure white noise, the convergence is very fast</mark>
- [ ] ❗️ need to change how max-epoch work in DNN class, currently even when the loss is not decreasing, it doesn't stop. 



## 实验记录和TODO

- [ ] 试一下kernel test 和 deep learning test
- [ ] ❗️ need to change how max-epoch work in DNN class, currently even when the loss is not decreasing, it doesn't stop. 
