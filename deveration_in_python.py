# python3
# Author: Scc_hy
# Create date: 2021-09-14
# Func: Deveration
# =================================================================================================


import numpy as np
import torch as t


def logistic_model(x, torch_flag=False):
    if torch_flag:
        return 1 / (1 + t.exp(-x))
    return 1 / (1 + np.exp(-x))

def logit_deveration(y):
    return y * (1 - y)

def mse_loss(y, y_hat):
    return sum((y - y_hat) ** 2)


# scipy deveration
# ----------------------------------------------------------------
from scipy.misc import derivative
x = np.arange(10)
y = logistic_model(x)
derivative(logistic_model, x, n=1, dx=1e-6)
logit_deveration(y)

# torch计算图求导
# ----------------------------------------------------------------
x = t.randn(10, requires_grad=True)
y = logistic_model(x, torch_flag=True)
y.backward(t.ones(10))
x.grad, logit_deveration(y)

# 参数估算
# -----------------------------------------------------------------
from scipy import optimize

def linear_model_fn(params, *args):
    w, b = params
    y, x = args
    y_hat = x * w + b
    return mse_loss(y, y_hat)


x = np.arange(10)
y = x * 20 + 9 + np.random.randn(10)
optimize.fmin_l_bfgs_b(linear_model_fn, x0=np.array([0.001, 0.001]), args=(y, x), approx_grad=True)

