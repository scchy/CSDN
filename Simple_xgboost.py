# python3
# Create date: 2021-08-30
# Author: Scc_hy
# Func: xgb 简单实现
# =================================================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from collections import namedtuple
from tqdm import tqdm
import xgboost as xgb


def get_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns='a b c d'.split(' '))
    df['y'] = iris.target
    return train_test_split(df.iloc[:, :-1].values, df.iloc[:, -1].values, test_size=0.2, random_state=42)


class LossFunction:
    """
    损失函数 及 求导
    """
    def __init__(self, objective='mse'):
        self.objective = objective
    
    def softmax(self, pred):
        return np.exp(pred) / np.exp(pred).sum(axis=0)
    
    def logloss(self, y_onehot, y_hat):
        return -y_onehot * np.log(y_hat)

    def _softmax_derivative(self, y_hat):
        return y_hat * ( 1 - y_hat)
    
    def _logloss_derivative(self, y, y_hat):
        g = -y / y_hat
        h = y / y_hat / y_hat * self._softmax_derivative(y_hat) # y / y_hat *  ( 1 - y_hat)
        return g, h
    
    def mse(self, y, y_hat):
        return 0.5 * np.sum((y - y_hat) * (y - y_hat)) / len(y_hat)
    
    def _mse_derivative(self, y, y_hat):
        g = y_hat - y
        h = np.ones_like(g)
        return g, h
    
    def backward(self, y, y_hat):
        if self.objective == 'mse':
            return self._mse_derivative( y, y_hat)
        if self.objective == 'logloss':
            return self._logloss_derivative( y, y_hat)
    


split_info = namedtuple('SplitInfo', 'split_feature split_th node_gain split_gain wl wr')


class XGBFunction:
    def __init__(self, gamma=1, lambda_=0.01, min_split_loss=0, learning_rate=1):
        self.w = 0.5
        self.lambda_ = lambda_
        self.gamma = gamma
        self.split_dict = {}
        self.ntree = 0
        self.min_split_loss = min_split_loss
        self.learning_rate = learning_rate

    def _gain(self, g_list, h_list):
        T = 0
        gain_list = []
        for g, h in zip(g_list, h_list):
            T += 1
            G = np.sum(g, axis=0)
            H = np.sum(h, axis=0)
            gain = G * G / (H + self.lambda_)
            gain_list.append(gain)
        
        if T == 1:
            return -0.5 * gain + self.gamma
        return  -0.5 * np.concatenate(gain_list).sum(axis=0) + self.gamma * T


    def compute_w(self, g, h):
        return - g / (h + self.lambda_)


    def split_node_gain(self, g, h, L, R):
        left = self._gain([g[L]], [h[L]])
        right = self._gain([g[R]], [h[R]])
        total  = self._gain([g], [h])
        return  total - left - right
    
    def _split_data(self, x: np.ndarray, split_feature: int, split_th: float):
        left_bool = x[:, split_feature] < split_th
        return left_bool, ~left_bool
    
    def stupid_split(self, x, g0, h0):
        """
        基于每个特征， 排序后的每个值 逐一尝试，直到获取最佳切分点
        仅仅是一层决策树
        """
        self.ntree += 1
        node_g = self._gain([g0], [h0])
        m, n = x.shape
        gain_max = -np.inf
        for f in tqdm(list(range(n))):
            sorted_unique_num = np.sort(np.unique(x[:, f]))
            for i in tqdm(sorted_unique_num):
                L, R = self._split_data(x, f, i)
                gain_i = self.split_node_gain(g0, h0, L, R)
                if gain_i > gain_max:
                    gain_max = gain_i
                    if gain_max > self.min_split_loss:
                        self.split_dict[self.ntree] = split_info(
                            split_feature = f,
                            split_th = i,
                            node_gain = node_g,
                            split_gain = gain_max,
                            wl = self.compute_w(g0[L], h0[L]).mean() * self.learning_rate ,
                            wr = self.compute_w(g0[R], h0[R]).mean() * self.learning_rate ,
                        )

        return self.split_dict


    def predict(self, x_te):
        base_value = np.array([0.5] * len(x_te))
        idx = np.arange(len(x_te))
        out = np.zeros_like(idx, dtype=np.float64)
        for k, tree_ in self.split_dict.items():
            f = tree_.split_feature
            th = tree_.split_th
            wl = tree_.wl
            wr = tree_.wr
            L, R = self._split_data(x_te, f, th)
            out[idx[L]] += wl
            out[idx[R]] += wr
        return out + base_value
    
    def gain_importance(self):
        imp = {}
        for k, v in self.split_dict.items():
            k_ = f'f{v.split_feature}'
            imp[k_] = imp.get(k_, 0) + v.split_gain
        return imp



# 数据加载
tr_x, te_x, tr_y, te_y = get_data()
loss_f = LossFunction('mse')
xgb_f = XGBFunction(gamma=1, lambda_=0.01, min_split_loss=0, learning_rate=1)
num_boost_rounds = 5
t = 0
base_value = np.array([0.5] * len(tr_y))
g0, h0 = loss_f.backward(tr_y, base_value)

# 1- 计算g0, h0
# 2- 贪婪生成树（这里简单用了一层决策树）
# 3- 计算分支的预测值w
# 4- 计算当前迭代的预测值 y_hat = base_value + w
# 5- 计算更新g0, h0
while t < num_boost_rounds:
    # 贪婪算法分割-一层决策树
    sp = xgb_f.stupid_split(tr_x, g0, h0)
    try:
        f = sp[t+1].split_feature
        th = sp[t+1].split_th
    except KeyError:
        print(f'early stop in num_boost_rounds [{t+1}]')
        break
    L, R = xgb_f._split_data(tr_x, f, th)
    wl = xgb_f.compute_w(g0[L], h0[L]).mean()
    wr = xgb_f.compute_w(g0[R], h0[R]).mean()
    print(f'{t+1}', sp[t+1])
    # w = xgb_f.compute_w(g0, h0)

    tr_x = np.concatenate([tr_x[L], tr_x[R]])
    tr_y = np.concatenate([tr_y[L], tr_y[R]])
    y_hat = np.concatenate([base_value[L] + wl, base_value[R] + wr])

    g0, h0 = loss_f.backward(tr_y, y_hat)
    base_value = y_hat
    print('train mse:', loss_f.mse(tr_y, y_hat))
    te_pred = xgb_f.predict(te_x)
    print('test mse:', loss_f.mse(te_y, te_pred))

    t += 1


from rich.console import Console
cs = Console()
cs.print(xgb_f.split_dict)


te_p = xgb_f.predict(te_x)
loss_f.mse(te_y, te_p)
te_y == np.round(te_p)




xgb_params = {
    'objective' : 'reg:squarederror',
    'gamma' : 1,
    'min_split_loss': 0,
    'max_depth': 1,
    'reg_lambda': 0.01,
    'learning_rate':1
}
tr_mt = xgb.DMatrix(tr_x, label=tr_y)
te_mt = xgb.DMatrix(te_x, label=te_y)
xgb_model = xgb.train(xgb_params, tr_mt, num_boost_round=3)
te_p_xgb = xgb_model.predict(te_mt)
np.round(te_p_xgb) == te_y
loss_f.mse(te_y, te_p_xgb)

xgb_tree = xgb_model.trees_to_dataframe()
print(xgb_tree)
cs.print(xgb_f.split_dict)

xgb_model.get_score(importance_type='gain')
cs.print( xgb_f.gain_importance() )

# =================================================================================================
# SHAP VALUE 值
import shap
ex_xgb = shap.TreeExplainer(xgb_model)
shap_te = ex_xgb(te_x)
# E(f(z)) base value 是训练集的预测值的均值
shap_te.base_values[0], np.mean(xgb_model.predict(tr_mt))
shap_base = np.mean(xgb_model.predict(tr_mt))


# 预测值 = 结点和+model_base(0.5) = 所有特征的贡献度+shap_base(0.99165833)
x1_contribution = xgb_model.predict(te_mt, pred_interactions=True)[0].sum(axis=1)
x1_contribution.sum(), xgb_model.predict(te_mt)[0], 0.987377  +  -0.099507 -0.199235 + 0.5
x1_contribution

# te_x[0] feature contrubition
print(xgb_tree)
# array([6.1, 2.8, 4.7, 1.2])
## F2 贡献度 fx(s U f2) - fx(s)
frac = 80/120
f2_con = (0.987377 - (0.987377*frac + -0.499875*(1-frac))
-0.099507 - (-0.099507*frac +  0.199060*(1-frac)))

## f3 贡献度 fx(s U f2) - fx(s)
frac_3 = 85/120
f3_con = -0.199235 - (-0.199235*frac_3 + 0.483914*(1-frac_3))

