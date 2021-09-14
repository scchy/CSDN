# python3
# Author: Scc_hy
# Create date: 2021-09-14
# Func: Deveration
# =================================================================================================


import numpy as np
import torch as t
import tensorflow as tf


def mse_loss(y, y_hat):
    return sum((y - y_hat) ** 2)




