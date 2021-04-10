# python3
# create date: 2021-04-11
# Author: Scc_hy
# Func: 自相关系数

## 自相关系数

import numpy as np
import matplotlib.pyplot as plt
from numba import jit 
from statsmodels.graphics.tsaplots import plot_acf

@jit(nopython=True)
def self_corr(in_arr, h):
    """
    自相关系数：
    E((X_t-u)(X_{t-h} -u))/var(X)
    简单化简了下
    """
    u = np.mean(in_arr)
    s1 = in_arr[h:]
    s2 = in_arr[:-h] if h !=0  else in_arr
    return np.sum((s1 - u) * (s2 - u) / np.sum((in_arr-u)*(in_arr-u)))

def m_acf_plot(s, lags, ax='no', show_flag=False):
    acf_list = []
    plot_fig = ax
    if ax == 'no':
        plot_fig = plt
        plt.figure(figsize=(10, 6))

    for i in range(0, lags+1):
        p_ = np.round(self_corr(s, i), 5)
        acf_list.append(p_)
        plot_fig.vlines(x=i, ymin=0, ymax=p_, alpha=0.7)

    plot_fig.scatter(list(range(lags+1)), acf_list)
    plot_fig.axhline(y=0, c='steelblue')
    try:
        plot_fig.title('ACF plot')
    except:
        plot_fig.set_title('ACF plot')
    if show_flag:
        plt.show()
    return acf_list


if __name__ == '__main__':
    x = np.arange(600) / 100 * np.pi
    y = np.sin(x)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(y)
    for i in [200, 400]:
        axes[0].vlines(x=i, ymin=-1, ymax=1, linestyle='--', alpha=0.6)

    plot_acf(y, ax=axes[1], lags=300)
    acf_list=m_acf_plot(y, lags=300, ax=axes[2])
    acf_list1=m_acf_plot(np.array(acf_list), lags=300, ax=axes[3])
    plt.show()
