# python3
# Create date: 2021-06-28
# Author: Scc_hy
# Func: Simple Logistic
# =================================================================================

import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def load_xy():
    iris = load_iris()
    x = iris.data[:, -2:]
    y = (iris.target.reshape((-1, 1)) == 2) * 1
    return train_test_split(x, y, test_size=0.25)



def binary_loss(y_true, y_pred):
    """
    交叉熵
    """
    e_ = 1e-15
    return np.mean(-y_true * np.log(y_pred + e_) - (1-y_true) * np.log(1 - y_pred + e_))


def activation(linear_x, method='sigmoid'):
    """
    激活函数， sigmoid / arctan
    """
    if method == 'sigmoid':
        out = np.zeros_like(linear_x, dtype=np.float64)
        biger_bool = np.abs(linear_x) >= 100
        out[biger_bool] = 1 / ( 1 + np.exp(-100 * np.sign(linear_x[biger_bool])) )
        out[~biger_bool] = 1 / ( 1 + np.exp(- linear_x[~biger_bool]))
        return out
    
    if method == 'arctan':
        return np.arctan(linear_x) / np.pi + 1/2


def activation_derivative(nd,  method='sigmoid'):
    """
    激活函数求导
    """
    if method == 'sigmoid': # sigmoid_out
        return nd - nd * nd
    if method == 'arctan': # linear_x
        return 1 / (1 + nd * nd)  / np.pi


def loss_derivative(y_true, y_pred):
    """
    损失函数求导
    """
    e_ = 1e-15
    return - y_true * 1 / (y_pred + e_) + (1-y_true) * 1 / (1 - y_pred + e_)


def data_generator(x, y, batch_size=128):
    """
    数据生成器
    """
    n = 0
    len_ = len(x)
    while n * batch_size < len_:
        yield x[n * batch_size : (n+1) * batch_size], y[n*batch_size : (n+1)*batch_size]
        n += 1


def epoch_train(x, y, w, lr, method='sigmoid', batch_size=128, epoch_num=0, verbose=0):
    """
    训练一次数据
    """
    loop = True
    data_g = data_generator(x, y, batch_size=batch_size)
    loss_list = []
    while loop:
        try:
            xt, yt = next(data_g)
        except StopIteration:
            loop = False

        linear_x = xt.dot(w) # (m, n).dot((n, 1)) -> (m, 1)
        pred_y = activation(linear_x, method=method)
        loss_d = loss_derivative(yt, pred_y)

        if method == 'sigmoid':
            w_d = xt.T.dot(activation_derivative(pred_y, method='sigmoid') * loss_d) / batch_size
        if method == 'arctan':
            # (m, n).T.dot((m, 1)) -> (n, 1)
            w_d = xt.T.dot(activation_derivative(linear_x, method='arctan') * loss_d) / batch_size

        w -= lr * w_d
        # print('w_d:', w_d)
        loss_list.append(binary_loss(yt, activation(xt.dot(w))))

    if verbose:
        print('w:\n', w)
        print(f'Epoch [{epoch_num}] loss: {np.mean(loss_list):.5f}')
        print('--'*25)
    return w


def test(test_epoches=80):
    x_tr, x_te, y_tr, y_te = load_xy()
    # 初始化参数
    w_sigmoid = np.random.normal(0, 1, x_tr.shape[1]).reshape((-1, 1))
    w_arctan = np.random.normal(0, 1, x_tr.shape[1]).reshape((-1, 1))
    for epi in range(test_epoches):
        w_sigmoid = epoch_train(x_tr, y_tr, w_sigmoid, lr=0.6, method='sigmoid', batch_size=20, epoch_num=epi)
        w_arctan = epoch_train(x_tr, y_tr, w_arctan, lr=0.6, method='arctan', batch_size=20, epoch_num=epi)

    lr = LogisticRegression(penalty='none', fit_intercept=False)
    lr.fit(x_tr, y_tr.flatten())

    sk_te_pred = lr.predict_proba(x_te)
    mysigmoid_te_pred = activation(x_te.dot(w_sigmoid), method='sigmoid')
    myarctan_te_pred = activation(x_te.dot(w_arctan), method='arctan')

    mysigmoid_acc = accuracy_score(y_te, 1*(mysigmoid_te_pred > 0.5))
    myartan_acc = accuracy_score(y_te, 1*(myarctan_te_pred > 0.5))
    lr_acc = accuracy_score(y_te, np.argmax(sk_te_pred, axis=1))

    print(f'LogisticRegression acc: {lr_acc*100:.3f}%\n' +
    f'my-sigmoid-lr acc: {mysigmoid_acc*100:.3f}%\n' + 
    f'my-arctan-lr acc: {myartan_acc*100:.3f}%\n'
    )


if  __name__ == '__main__':
    for t_epoches in [10, 40, 60, 100]:
        print('**'*25)
        for test_time in range(3):
            print('--'*25)
            print(f't_epoches-{t_epoches} test_time: {test_time}')
            test(t_epoches)
