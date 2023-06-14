# Python3
# Create date: 2023-06-15
# Author: Scc_hy
# Func: 保序回归
# ==============================================================================================

# calibration_curve PLOT
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
np.random.seed(2023)


y_true = np.random.randint(0, 2, size=1000)
y_pred = np.random.binomial(n=200, p=0.19, size=1000)
y_pred = (y_pred - y_pred.min())/(y_pred.max()-y_pred.min())
y_means, proba_means = calibration_curve(
    y_true, 
    y_pred, 
    n_bins=10, 
    strategy='quantile'
)

# 分割图片 2:1
fig = plt.figure(constrained_layout=True, figsize=(16, 4))
gs = fig.add_gridspec(1, 3)
axes1, axes2 = fig.add_subplot(gs[:2]), fig.add_subplot(gs[2]) 
# 绘制分布
sns.histplot(y_pred, alpha=0.7, ax=axes1)
for i in proba_means:
    axes1.axvline(x=i, linestyle='--', color='darkred', alpha=0.7)
axes1.set_title("predict and bin split\nstrategy='quantile'")
axes1.set_xlabel('Predicted probability')
# 绘制对准曲线
axes2.plot([0, 1], [0, 1], linestyle = '--', label = 'Perfect calibration')
axes2.plot(proba_means, y_means, linestyle='-.')
axes2.set_title('Simplr Predict Calibrator')
axes2.legend()
axes2.set_xlabel("Bin's mean of predicted probability")
axes2.set_ylabel("Bin's mean of target variable")
plt.show()


def quick_calibration_plot(y_true, y_pred, title_msg=''):
    y_means, proba_means = calibration_curve(
        y_true, 
        y_pred, 
        n_bins=10, 
        strategy='quantile'
    )
    # 分割图片 2:1
    fig = plt.figure(constrained_layout=True, figsize=(16, 4))
    gs = fig.add_gridspec(1, 3)
    axes1, axes2 = fig.add_subplot(gs[:2]), fig.add_subplot(gs[2]) 
    # 绘制分布
    sns.histplot(y_pred, alpha=0.7, ax=axes1)
    for i in proba_means:
        axes1.axvline(x=i, linestyle='--', color='darkred', alpha=0.7)
    axes1.set_title("predict and bin split\nstrategy='quantile'")
    axes1.set_xlabel('Predicted probability')
    # 绘制对准曲线
    axes2.plot([0, 1], [0, 1], linestyle = '--', label = 'Perfect calibration')
    axes2.plot(proba_means, y_means, linestyle='-.')
    axes2.set_title(f'Simple Predict Calibrator\n{title_msg}')
    axes2.legend()
    axes2.set_xlabel("Bin's mean of predicted probability")
    axes2.set_ylabel("Bin's mean of target variable")
    plt.show()
    

# 校准试验
# -------------------------------
def expected_calibration_error(y, proba, bins = 'fd'):
    bin_count, bin_edges = np.histogram(proba, bins = bins)
    n_bins = len(bin_count)
    bin_edges[0] -= 1e-8 # because left edge is not included
    bin_id = np.digitize(proba, bin_edges, right = True) - 1
    bin_ysum = np.bincount(bin_id, weights = y, minlength = n_bins)
    bin_probasum = np.bincount(bin_id, weights = proba, minlength = n_bins)
    bin_ymean = np.divide(bin_ysum, bin_count, out = np.zeros(n_bins), where = bin_count > 0)
    bin_probamean = np.divide(bin_probasum, bin_count, out = np.zeros(n_bins), where = bin_count > 0)
    ece = np.abs((bin_probamean - bin_ymean) * bin_count).sum() / len(proba)
    return ece


# 模型简单拟合
from sklearn.datasets import make_classification
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
X, y = make_classification(
    n_samples = 15000, 
    n_features = 50, 
    n_informative = 30, 
    n_redundant = 20,
    weights = [.9, .1],
    random_state = 0
)
X_train, X_valid, X_test = X[:5000], X[5000:10000], X[10000:]
y_train, y_valid, y_test = y[:5000], y[5000:10000], y[10000:]
forest = RandomForestClassifier().fit(X_train, y_train)
proba_valid = forest.predict_proba(X_valid)[:, 1]

# 保序回归
iso_reg = IsotonicRegression(y_min = 0, y_max = 1, out_of_bounds = 'clip').fit(proba_valid, y_valid)
test_pred = forest.predict_proba(X_test)[:, 1]
ece_org = expected_calibration_error(y_test, test_pred, bins = 'fd')
quick_calibration_plot(y_test, test_pred, title_msg=f'not  calibration ECE={ece_org:.3f}')

proba_test_forest_isoreg = iso_reg.predict(test_pred)
ece_iosreg = expected_calibration_error(y_test, proba_test_forest_isoreg, bins = 'fd')
quick_calibration_plot(y_test, proba_test_forest_isoreg, title_msg=f'IsotonicRegression ECE={ece_iosreg:.3f}')

# logistic
log_reg = LogisticRegression().fit(proba_valid.reshape(-1, 1), y_valid)
proba_test_forest_logreg = log_reg.predict_proba(test_pred.reshape(-1, 1))[:, 1]

ece_logreg = expected_calibration_error(y_test, proba_test_forest_logreg, bins = 'fd')
quick_calibration_plot(y_test, proba_test_forest_logreg, title_msg=f'IsotonicRegression ECE={ece_logreg:.3f}')