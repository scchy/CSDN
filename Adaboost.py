# python3
# create date: 2021-07-22
# Author: Scc_hy
# Func: sample adaboost
# Tip: 西瓜书-boosting / sklearn.ensemble._weight_boost
# =================================================================================================

__doc__ = """
简单线性叠加模型
- H(x) = \sum{alpha*h(x)}

核心：
- 增大优基分类器的权重
- 增大误分类样本的权重

损失函数：
- 指数损失 exp(-f(x)H(x))


"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import namedtuple
from sklearn.tree import DecisionTreeClassifier


model_info = namedtuple('model_info', 'model weight error')


class BaseFunction:
    def __init__(self, n_estimators=50, learning_rate=1):
        self.n_estimators = n_estimators
        self.boost_models = []
        self.learning_rate = learning_rate

    def exp_loss(self, y_true, y_pred):
        return np.mean(np.exp(-y_true * y_pred))
    
    def model_loss(self, x, y):
        out = self.decision_function(x)
        return self.exp_loss(y, out)

    def iboost_error(self, y_true, iboost_pred, sample_weight):
        incorrect = y_true != iboost_pred
        # print(f'incorrect.shape: {incorrect.shape}')
        return np.average(incorrect.flatten(), weights=sample_weight.flatten())

    def iboost_weight(self, error):
        return 0.5 * np.log( (1-error) / error)

    def flushed_sample_weight(self, y_true, iboost_pred, sample_weight):
        iboost_error = self.iboost_error(y_true, iboost_pred, sample_weight)
        if iboost_error == 0:
            return sample_weight, 1, 0

        iboost_w = self.iboost_weight(iboost_error)
        sample_weight *= np.exp(iboost_w * (y_true != iboost_pred) * (sample_weight > 0))
        return sample_weight / np.sum(sample_weight), iboost_w, iboost_error

    def _boost(self, estimator, X, y, sample_weight, print_flag=False):
        try:
            estimator.fit(X, y, sample_weight=sample_weight.flatten(), verbose=200 if print_flag else np.inf)
        except:
            estimator.fit(X, y, sample_weight=sample_weight.flatten())
        y_pred = (estimator.predict(X) > 0.5).reshape((-1, 1)) * 1
        # 更新权重， 计算当前模型的误差， 计算当前模型的权重
        sample_weight, iboost_w, iboost_error = self.flushed_sample_weight(y, y_pred, sample_weight)
        
        # 模型保存
        iboost_model = model_info( model = estimator, weight = iboost_w * self.learning_rate, error = iboost_error)
        self.boost_models.append(iboost_model)
        return iboost_model, sample_weight

    def decision_function(self, x):
        pred = sum( (m.model.predict(x) > 0.5) * 1 * m.weight for m in self.boost_models )
        # print(f'pred.mean(): {pred.mean():.5f}')
        return pred / sum(m.weight for m in self.boost_models)


class SampleAdaboost(BaseFunction):
    def __init__(self, n_estimators=50, learning_rate=1, base_model=None):
        super(SampleAdaboost, self).__init__(
            n_estimators = n_estimators,
            learning_rate = learning_rate
        )
        self.base_model = base_model

    def fit(self, x, y, verbose=10):
        y = y.reshape((-1, 1))
        sample_weight =  np.ones(len(y), dtype=np.float64) / len(y)
        sample_weight = sample_weight.reshape((-1, 1))
        loss_min = np.inf
        best_models = self.n_estimators
        for i in range(self.n_estimators):
            if self.base_model == 'logistic':
                modeli = BaseLogistic()
            else:
                modeli = DecisionTreeClassifier(max_depth=3)

            print_f = (i % verbose) == 0
            if print_f:
                print('--'*25)
                print(f'Boost [{i}] ...')

            iboost_model, sample_weight = self._boost(modeli, x, y, sample_weight, print_f)
            loss_ = self.model_loss(x, y)
            if print_f:
                print(f'Model loss: {loss_:.5f}')
            
            if loss_min > loss_:
                loss_min = loss_
                best_models = i
            else:
                x, y, sample_weight = self.resample(x, y, sample_weight)

            if iboost_model.error == 0:
                break
            if sum(sample_weight) <= 0:
                break
        
        self.boost_models = self.boost_models[:best_models+1]

    def resample(self, x, y, sample_weight):
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        return x[idx], y[idx], sample_weight[idx]

    def predict(self, x):
        return self.decision_function(x)


class BaseLogistic:
    def __init__(self, batch_size=256, epochs=500, epsilon=1e-15, learning_rate=0.1, early_stopping=20):
        self.epochs = epochs
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
    
    def data_generator(self, x, y, sample_weight):
        n = 0
        st = n * self.batch_size
        ed = (n + 1) * self.batch_size
        yield x[st:ed], y[st:ed], sample_weight[st:ed]
        n += 1

    def log_loss(self, y, p):
        return np.mean(-y * np.log(p + self.epsilon) - (1 - y) * np.log(1 - p + self.epsilon))

    def _loss(self, x, y):
        p = self.predict(x)
        return self.log_loss(y, p)
    
    def fit(self, x, y, sample_weight, verbose=200):
        y = y.reshape((-1, 1))
        sample_weight = sample_weight.reshape((-1, 1))
        self.hist_loss = []
        self.w = np.random.rand( x.shape[1]).reshape((-1, 1))
        need_w = self.w
        n = 0
        loss_min = np.inf
        for ep in range(self.epochs):
            loop = True
            data_g = self.data_generator(x, y, sample_weight)
            while loop:
                try:
                    xi, yi, sample_weighti = next(data_g)
                except StopIteration:
                    loop = False
                self.fit_on_batch(xi, yi, sample_weighti)

            loss_t = self._loss(x, y)
            if np.round(loss_min, 6) > np.round(loss_t, 6):
                loss_min = loss_t
                need_w = self.w
                n = 0
            else:
                n += 1
            
            if n == self.early_stopping:
                self.hist_loss.append((ep, loss_t))
                self.w = need_w
                if verbose < np.inf:
                    print(f'Early stop [{ep}]- loss: {loss_t:.5f}')
                return self

            if (ep + 1) % verbose == 0:
                self.hist_loss.append((ep, loss_t))
                print(f'[{ep}]- loss: {loss_t:.5f}')
        return self


    def fit_on_batch(self, x, y, sample_weight):
        pred = self.predict(x)
        loss_sigmoid_devaration = (-y * 1 / (pred + self.epsilon) + (1-y)* 1 / (1 - pred + self.epsilon)) * sample_weight / sum(sample_weight)
        sigmoid_linear_devaration = pred - pred * pred
        w_d = x.T.dot(loss_sigmoid_devaration * sigmoid_linear_devaration)
        self.w -= self.learning_rate * w_d

    def predict(self, x):
        # (m, n) . (n, 1)
        linear_out = x.dot(self.w)
        return 1 / (1 + np.exp(-linear_out))



def test(n):
    print('--'*25)
    print('loading data ...')
    X, y = make_classification(n_samples=1000, n_features=13, n_informative=8, n_classes=2)
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.25)
    y_tr = y_tr.flatten()
    y_te = y_te.flatten()
    print('--'*25)
    print('Compare my logistic ...')
    m_lr = BaseLogistic(batch_size=256, epochs=500, epsilon=1e-15, learning_rate=0.1, early_stopping=20)
    m_lr.fit(x_tr, y_tr, np.ones(len(y_tr), dtype=np.float64) / len(y_tr), verbose=200)
    pred_te = m_lr.predict(x_te)
    f1_ = f1_score(y_te, pred_te>0.5)

    lr = LogisticRegression()
    lr.fit(x_tr, y_tr)
    lr_pred = lr.predict(x_te)
    f1_lr = f1_score(y_te, lr_pred>0.5)

    print(f'my-lr f1: {f1_:.3f}; lr-f1:{f1_lr:.3f}')

    print('--'*25)
    print('test adaboost ...')
    tree_ = DecisionTreeClassifier(max_depth=3)
    tree_.fit(x_tr, y_tr)
    tree_pred = tree_.predict(x_te)
    tree_adb = f1_score(y_te, tree_pred>0.5)

    adb = SampleAdaboost(50)
    adb.fit(x_tr, y_tr)
    adb_pred = adb.predict(x_te)
    f1_adb = f1_score(y_te, adb_pred>0.5)

    adb_lr = SampleAdaboost(50, base_model='logistic')
    adb_lr.fit(x_tr, y_tr)
    adb_lr_pred = adb_lr.predict(x_te)
    adb_lr_f1 = f1_score(y_te, adb_lr_pred>0.5)

    adb_sk = AdaBoostClassifier()
    adb_sk.fit(x_tr, y_tr)
    adb_sk_pred = adb_sk.predict(x_te)
    adb_sk_f1 = f1_score(y_te, adb_sk_pred>0.5)


    info_out = f'[test-{n}] dtree-f1:{tree_adb:.5f}; my-lr f1: {f1_:.5f}; lr-f1:{f1_lr:.5f}; \n\tmy-adaboost-dtree:{f1_adb:.5f}; my-adaboost-lr:{adb_lr_f1:.5f}; adaboost-sklearn:{adb_sk_f1:.5f}'
    print(info_out)
    print('Done')
    return info_out


if __name__ == '__main__':
    info_list = []
    for i in range(4):
        inf_ = test(i)
        info_list.append(inf_)

    for i in range(4):
        print('\n')
        print(info_list[i])
    print('Done')


"""
[test-0] dtree-f1:0.74740; my-lr f1: 0.68992; lr-f1:0.67213;
        my-adaboost-dtree:0.74740; my-adaboost-lr:0.73333


[test-1] dtree-f1:0.80645; my-lr f1: 0.83465; lr-f1:0.84375;
        my-adaboost-dtree:0.89231; my-adaboost-lr:0.83922


[test-2] dtree-f1:0.76652; my-lr f1: 0.77333; lr-f1:0.76522;
        my-adaboost-dtree:0.85470; my-adaboost-lr:0.78222

[test-0] dtree-f1:0.86463; my-lr f1: 0.86066; lr-f1:0.85477;
        my-adaboost-dtree:0.91983; my-adaboost-lr:0.86885


[test-1] dtree-f1:0.84758; my-lr f1: 0.77043; lr-f1:0.78764;
        my-adaboost-dtree:0.90769; my-adaboost-lr:0.77165


[test-2] dtree-f1:0.80189; my-lr f1: 0.84018; lr-f1:0.84404;
        my-adaboost-dtree:0.90991; my-adaboost-lr:0.83486


[test-3] dtree-f1:0.86525; my-lr f1: 0.87121; lr-f1:0.88321;
        my-adaboost-dtree:0.92481; my-adaboost-lr:0.87218

[test-0] dtree-f1:0.82449; my-lr f1: 0.83465; lr-f1:0.82305;
        my-adaboost-dtree:0.89167; my-adaboost-lr:0.83137; adaboost-sklearn:0.82500


[test-1] dtree-f1:0.74265; my-lr f1: 0.76471; lr-f1:0.77043;
        my-adaboost-dtree:0.84252; my-adaboost-lr:0.76471; adaboost-sklearn:0.79518


[test-2] dtree-f1:0.75090; my-lr f1: 0.81911; lr-f1:0.83154;
        my-adaboost-dtree:0.88727; my-adaboost-lr:0.82712; adaboost-sklearn:0.82090


[test-3] dtree-f1:0.81618; my-lr f1: 0.89062; lr-f1:0.90909;
        my-adaboost-dtree:0.92481; my-adaboost-lr:0.89062; adaboost-sklearn:0.89552
"""