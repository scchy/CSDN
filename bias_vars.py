# python 3
# Create date: 2021-01-19
# Author: Scc_hy
# Func: high vars & high bias
# -----------------------------------------------------------------------
# - 普通NN回归效果差
# - 采用GRNN： 参考 https://blog.csdn.net/weixin_41806692/article/details/81453377
# -----------------------------------------------------------------------


import tensorflow as tf
from tensorflow.keras import Model, Input, losses
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons, make_classification, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

loss_func = losses.CategoricalCrossentropy(from_logits=True)
loss_func_reg = losses.MeanAbsoluteError()


def plot_moon(mx, my):
    x_1 = [mx[i][0] for i in range(len(mx))]
    x_2 = [mx[i][1] for i in range(len(mx))]
    plt.scatter(x_1, x_2, c=my)
    plt.show()


def simple_nn(input_shape: tuple,
              output_shape: int,
              hidden_size_list: list = [],
              print_flag=False,
              learning_type: "reg clf" = 'clf'):
    input_layer = Input(shape=input_shape)
    if not hidden_size_list:
        out = Dense(output_shape, activation=tf.nn.softmax if learning_type == 'clf' else 'linear',
                    use_bias=True, bias_initializer='glorot_uniform')(input_layer)
    elif len(hidden_size_list) == 1:
        x = Dense(hidden_size_list[0], activation=tf.nn.leaky_relu)(input_layer)
        out = Dense(output_shape, activation=tf.nn.softmax if learning_type == 'clf' else 'linear',
                    use_bias=True, bias_initializer='glorot_uniform')(x)
    else:
        x = Dense(hidden_size_list[0], activation=tf.nn.leaky_relu)(input_layer)
        for i in hidden_size_list[1:]:
            x = Dense(i, activation=tf.nn.leaky_relu)(x)
        out = Dense(output_shape, activation=tf.nn.softmax if learning_type == 'clf' else 'linear',
                    use_bias=True, bias_initializer='glorot_uniform')(x)

    clf = Model(input_layer, out)
    if print_flag:
        print(clf.summary())
    clf.compile(
        loss='categorical_crossentropy' if learning_type == 'clf' else losses.MeanAbsoluteError(),
        optimizer='adam',
        metrics=['accuracy'] if learning_type == 'clf' else [tf.keras.metrics.MeanSquaredError()]
    )
    return clf

#
# def GRNN(Model):
#     def __init__(self, input_shape, out_shape):
#         super(GRNN, self).__init__()
#         self.input_layer = Dense(input_shape, activation='')


def plot_learning_curves(model_in, x_in, y_in, hidden_size_list=[], learning_type='clf', ax=None):
    x_train, x_test, y_train, y_test = train_test_split(x_in, y_in, test_size=0.25)
    if learning_type == 'clf':
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        l_func = loss_func
    else:
        y_train = y_train.reshape((-1, 1))
        y_test = y_test.reshape((-1, 1))
        l_func = loss_func_reg
    train_errors, test_errors = [], []
    for m in tqdm(range(1, len(x_train), 20)):
        model_t = model_in(input_shape=(x_train.shape[1],),
                           output_shape=y_train.shape[1],
                           hidden_size_list=hidden_size_list,
                           learning_type=learning_type)
        model_t.fit(x_train[:m], y_train[:m], batch_size=100, epochs=8, verbose=0)
        y_train_pre = model_t.predict(x_train[:m])
        y_test_pre = model_t.predict(x_test)

        train_errors.append(l_func(y_train[:m], y_train_pre).numpy())
        test_errors.append(l_func(y_test, y_test_pre).numpy())

    print('y_test_pre.shape:', y_test_pre.shape)
    print(model_t.summary())
    ax.plot(train_errors, 'r-+', lw=2, label='train', alpha=0.6)
    ax.plot(test_errors, 'b-', lw=2, label='test', alpha=0.6)
    ax.legend(loc='upper right')
    ax.set_title(f"hidden_size_list: {hidden_size_list}, total sample: {x_in.shape[0]}")
    return model_t


def sklearn_plot_learning_curves(model_in, x_in, y_in, ax=None):
    x_train, x_test, y_train, y_test = train_test_split(x_in, y_in, test_size=0.25)
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    train_errors, test_errors = [], []
    for m in tqdm(range(1, len(x_train), 20)):
        model_t = model_in.fit(x_train, y_train)
        y_train_pre = model_t.predict(x_train[:m])
        y_test_pre = model_t.predict(x_test)

        train_errors.append(mean_squared_error(y_train[:m], y_train_pre))
        test_errors.append(mean_squared_error(y_test, y_test_pre))

    print('y_test_pre.shape:', y_test_pre.shape)
    ax.plot(train_errors, 'r-+', lw=2, label='train', alpha=0.6)
    ax.plot(test_errors, 'b-', lw=2, label='test', alpha=0.6)
    ax.legend(loc='upper right')
    ax.set_title(f"total sample: {x_in.shape[0]}")
    return model_t


def simple_train(x, y):
    """
    单一特征训练
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    reg_ = plot_learning_curves(simple_nn, x, y,
                                hidden_size_list=[],
                                learning_type='reg',
                                ax=axes[0])
    x_p = np.linspace(0, 2000, 100)
    y_p = reg_.predict(x_p)
    axes[1].scatter(x, y, s=5, alpha=0.6)
    axes[1].scatter(x_p, y_p, s=5, alpha=0.6)
    plt.show()


def sklearn_simple_train(model_in, x, y, add_log=False):
    """
    单一特征训练
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    if add_log:
        reg_ = sklearn_plot_learning_curves(model_in, np.c_[x, np.log1p(x)], y, ax=axes[0])
    else:
        reg_ = sklearn_plot_learning_curves(model_in, x, y, ax=axes[0])
    x_p = np.linspace(0, x.shape[0], 100)
    if add_log:
        y_p = reg_.predict(
            np.c_[x_p.reshape((-1, 1)), np.log1p(x_p.reshape((-1, 1)))]
        )
    else:
        y_p = reg_.predict(x_p.reshape((-1, 1)))

    axes[1].scatter(x, y, s=5, alpha=0.6)
    axes[1].scatter(x_p, y_p, s=5, alpha=0.6)
    axes[1].set_title(f'add_log: {add_log}')
    plt.show()


def add_feature_train(x, y):
    """
    单一特征训练
    :return: nn model
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x_add = np.c_[x, np.log1p(x * x), np.cos(x)]
    reg_ = plot_learning_curves(simple_nn, x_add, y,
                                hidden_size_list=[4, 4],
                                learning_type='reg',
                                ax=axes[0])
    x_p = np.linspace(0, 2000, 100)
    y_p = reg_.predict(np.c_[x_p.reshape((-1, 1)), np.log1p(x_p.reshape((-1, 1))), np.cos(x_p.reshape((-1, 1)))])
    axes[1].scatter(x, y, s=5, alpha=0.6)
    axes[1].scatter(x_p, y_p, s=5, alpha=0.6)
    plt.show()


def add_y_transform_train(x, y):
    """
    单一特征训练
    :return: nn model
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    y_t = np.exp(y)
    reg_ = plot_learning_curves(simple_nn, x, y_t,
                                hidden_size_list=[4, 4, 4, 4, 4],
                                learning_type='reg',
                                ax=axes[0])
    x_p = np.linspace(0, 2000, 100)
    y_p = reg_.predict(x_p)
    axes[1].scatter(x, y, s=5, alpha=0.6)
    axes[1].scatter(x_p, np.log(y_p), s=5, alpha=0.6)
    plt.show()


if __name__ == '__main__':
    # mk_moons = make_moons(n_samples=2000, noise=0.45, random_state=2021)
    # m_x = mk_moons[0]
    # m_y = mk_moons[1]
    # plot_learning_curves(simple_nn, m_x, m_y, hidden_size_list=[64, 32])
    # 生成回归数据
    x = np.array(list(range(8000)) + list(range(50)) + list(range(20)) + list(range(10))) + 1
    y = 2 * np.log(x * x) + np.random.rand(len(x)) * 3 + np.cos(x)
    plt.scatter(x, y, s=5, alpha=0.6)
    plt.show()
    x = x.reshape((-1, 1))
    # simple_train(x, y)
    # add_feature_train(x, y)
    # add_y_transform_train(x, y)

    lr = LinearRegression()
    xgb_lr = XGBRegressor()

    sklearn_simple_train(lr, x, y)
    # 增加训练数据
    sklearn_simple_train(lr, x, y)
    sklearn_simple_train(lr, x, y, add_log=True)

    # xgb
    sklearn_simple_train(xgb_lr, x, y)


