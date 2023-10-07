# python3
# Create Date: 2023-10-07
# Functions:  to overcome the challenge (1 sample - multi class) : Creating training and validation sets is a bit trickier for multlilabel problems because there is no guaranteed balance for all labels.
# reference:
#   1- http://scikit.ml/api/skmultilearn.model_selection.iterative_stratification.html
#   2- A Network Perspective on Stratification of Multi-Label Data
#        paper： http://proceedings.mlr.press/v74/szyma%C5%84ski17a.html
# Tips:
#   1- pip install scikit-multilearn
# ===================================================================================

__doc__ = """
Yet most of the available data sets have been provided in train/test splits that did not account for 
maintaining a distribution of higher-order relationships between labels among splits or folds.
然而，大多数可用的数据集都进行训练集和测试集拆分，但是这些分割并没有考虑到在split或者folds中的label分布的高阶关系

Algo-01: Second Order Iterative Stratification
Input: Set of samples D, labels L, number of folds k, list of desired proportions per fold r

计算每个label对所需的样本数

\All <- { {lambda_i, lambda_j} : (x, y) \in D, lambda_i, lambda_j \in Y }
for e in All:
    D^e <- (x, Y); Y \and e != empty

# 每个折叠r所需比例的列表
for i in range(KFold):
    c_j <- |D| * r_j
    for e in A:
        c^e_j <- |D^e| * r_j
return DistributeOverFolds(D, All, c)


Algo-02: Iterative distribution of samples into folds (DistributeOverFolds)
Input: 
    Set of samples D, 
    set of edges with samples All, 
    percentages of desired sampling from a given edge per fold c

while |{ (x, Y) \in D: Y != empty }| > 0:
    for lambda_i in All:
        D^i <- {(x, Y): Y \and lambda_i != empty}

    l <- argmin_i |D^i|
    for (x, Y) in D^l:
        M <- argmax_j c_j^l 
        if |M| == 0:
            m <- onlyElement(M)
        else:
            M^\prime <- argmax_{j \in M} c_j
            if |M^\prime| == 0:
                m <- onlyElementOf(M^\prime)
            else:
                m <- randomElementOf(M^\prime)
        S_m <- S_m + (x, Y); D <- D \ (x, Y); c_m^l -= 1; c_m -= 1;

"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from skmultilearn.model_selection import iterative_train_test_split
import pandas as pd


rf1 = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, verbose=0)
rf2 = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, verbose=0)
X, y = make_classification(
    n_samples=100000, 
    n_features=20, 
    n_informative=15, 
    n_redundant=2, 
    n_repeated=1, 
    n_classes=5,
    n_clusters_per_class=3,
    weights=np.array([0.1, 0.15, 0.1, 0.25, 0.4]),
    flip_y=0,
    random_state=2023
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lb, cnt = np.unique(y_train, return_counts=True)  
cnt / cnt.sum() # array([0.09978571, 0.14872857, 0.09932857, 0.25191429, 0.40024286])
rf1.fit(X_train, y_train)
pred = rf1.predict(X_test)
org_is = f1_score(y_train, rf1.predict(X_train), average='macro')
f1_ = f1_score(y_test, pred, average='macro')



y_one_hot = pd.get_dummies(y).values
X_train, y_train_one, X_test, y_test_one = iterative_train_test_split(X, y_one_hot, test_size=0.3)

y_train = np.argmax(y_train_one, axis=1)
y_test = np.argmax(y_test_one, axis=1)
lb, cnt = np.unique(y_train, return_counts=True)  
cnt / cnt.sum() # [0.1       , 0.15075714, 0.09995714, 0.25204286, 0.39724286]
rf2.fit(X_train, y_train.flatten())
pred = rf2.predict(X_test)
org_sois = f1_score(y_train.flatten(), rf2.predict(X_train), average='macro')
f1_sois = f1_score(y_test, pred, average='macro')
print(f"IS   tr_f1={org_is:.3f} te_f1={f1_:.3f} \nSOIS tr_f1={org_sois:.3f} te_f1={f1_sois:.3f}")




import scipy.sparse as sp
import itertools
y_test_one.shape
rows = sp.lil_matrix(y_test_one).rows # each of which is a sorted list of column indices of non-zero
len(rows)
rows[:5]
y_test[:5]

for sample_index, label_assignment in enumerate(rows):
    # for every n-th order label combination
    # register combination in maps and lists used later
    for combination in itertools.combinations_with_replacement(label_assignment, 2):
        print(combination)
    break
