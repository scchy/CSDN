# python3
# Author: Scc_hy
# Func: 自相关系数

## 自相关系数
import numpy as np
import matplotlib.pyplot as plt
def self_corr(in_arr, h):
    u = np.mean(in_arr)
    s1 = in_arr[h:]
    s2 = in_arr[:-h] if h !=0  else in_arr
    return np.sum((s1 - u) * (s2 - u) / np.sum((s-u) * (s-u)))

def m_acf_plot(s, lags):
    acf_list = []
    plt.figure(figsize=(10, 6))
    for i in range(0, lags+1):
        p_ = np.round(self_corr(s, i), 5)
        acf_list.append(p_)
        plt.vlines(x=i, ymin=0, ymax=p_, alpha=0.7)

    print(acf_list)
    plt.scatter(list(range(lags+1)), acf_list)
    plt.axhline(y=0, c='steelblue')
    plt.title('ACF plot')
    plt.show()
    return acf_list
