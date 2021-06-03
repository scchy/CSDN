# Python3
# Create date: 2021-06-03
# Author: Scc_hy
# Func: 保序回归
# Inference: https://zhuanlan.zhihu.com/p/88623159
# ==============================================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, LineCollection
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state


n = 100
x = np.arange(n)
rs = check_random_state(0)
y = rs.randint(-50, 50, size=(n,)) + 50 * np.log1p(x)

ir = IsotonicRegression()
y_ = ir.fit_transform(x, y)
lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)

# Plot result
segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(np.full(n, 0.5))

plt.plot(x, y, 'r.', markersize=12)
plt.plot(x, y_, 'b.-', markersize=12)
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.gca().add_collection(lc)
plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
plt.title('Isotonic regression')
plt.show()

"""
应用：
假设药物使用量为数组x=0,1,2,3,...,99, 病人对药物的反应量为Y=y1, y2, y3, ..., y99,而由于个体的原因
Y不是一个单调函数(存在波动)如果我们按照药物反应排序，对应的X就会成为乱序，失去了研究意义。
而我们的研究的目的是为了观察随着药物使用量的递增，病人的平均反应状况。
在这种情况下，使用保序回归， 即不改变X的排序顺序，又求得Y的平均状况。

从上图中可以看出，最长的蓝线X的取值约30-60之间，这个区间内，Y的平均值一样，那么从经济及
病人抗药性等因素考虑，使用药量30个单位是最理想的。

"""

# =================================================================
from sklearn.datasets import load_digits
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
mnist = load_digits()
x = mnist.data/255
y = (mnist.target == 5) * 1

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3)

lgb = LGBMClassifier()
lr = LogisticRegression()
lgb.fit(x_tr, y_tr)
lr.fit(x_tr, y_tr)

lgb_pred = (lgb.predict_proba(x_te)[:, 1] > 0.5) * 1
accuracy_score(y_te, lgb_pred)

lr_pred = (lr.predict_proba(x_te)[:, 1] > 0.5) * 1
accuracy_score(y_te, lr_pred)

# lr+保序 提升预测准确率
lr_proba = lr.predict_proba(x_tr)[:, 1]
ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
ir.fit(lr_proba, y_tr)

lr_pred_new = ir.predict(lr.predict_proba(x_te)[:, 1])
accuracy_score(y_te, (lr_pred_new > 0.5)*1)


