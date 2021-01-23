# python 3
# Create date: 2021-01-19
# Author: Scc_hy
# Func: high vars & high bias

import tensorflow as tf
from tensorflow.keras import Model, Input, losses
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons, make_classification, make_regression

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
        out = Dense(output_shape, activation=tf.nn.softmax if learning_type == 'clf' else None)(input_layer)
    elif len(hidden_size_list) == 1:
        x = Dense(hidden_size_list[0], activation=tf.nn.leaky_relu)(input_layer)
        out = Dense(output_shape, activation=tf.nn.softmax if learning_type == 'clf' else None)(x)
    else:
        x = Dense(hidden_size_list[0], activation=tf.nn.leaky_relu)(input_layer)
        for i in hidden_size_list[1:]:
            x = Dense(i, activation=tf.nn.leaky_relu)(x)
        out = Dense(output_shape, activation=tf.nn.softmax if learning_type == 'clf' else None)(x)

    clf = Model(input_layer, out)
    if print_flag:
        print(clf.summary())
    clf.compile(
        loss='categorical_crossentropy' if learning_type == 'clf' else losses.MeanAbsoluteError(),
        optimizer='adam',
        metrics=['accuracy'] if learning_type == 'clf' else [tf.keras.metrics.MeanSquaredError()]
    )
    return clf


def plot_learning_curves(model_in, x, y, hidden_size_list=[], learning_type='clf'):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    if learning_type == 'clf':
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        l_func = loss_func
    else:
        y_train = y_train.reshape((-1, 1))
        y_test = y_test.reshape((-1, 1))
        l_func = loss_func_reg
    train_errors, test_errors = [], []
    for m in tqdm(range(1, len(x_train), 50)):
        model_t = model_in(input_shape=(x_train.shape[1],),
                           output_shape=y_train.shape[1],
                           hidden_size_list=hidden_size_list,
                           learning_type=learning_type)
        model_t.fit(x_train[:m], y_train[:m], batch_size=80, epochs=10, verbose=0)
        y_train_pre = model_t.predict(x_train[:m])
        y_test_pre = model_t.predict(x_test)

        train_errors.append(l_func(y_train[:m], y_train_pre).numpy())
        test_errors.append(l_func(y_test, y_test_pre).numpy())

    plt.plot(np.sqrt(train_errors), 'r-+', lw=2, label='train')
    plt.plot(np.sqrt(test_errors), 'b-', lw=2, label='test')
    plt.legend(loc='upper right')
    plt.title(f"hidden_size_list: {hidden_size_list}, total sample: {x.shape[0]}")
    # plt.ylim(0, 1)
    plt.show()


if __name__ == '__main__':
    # mk_moons = make_moons(n_samples=2000, noise=0.45, random_state=2021)
    # m_x = mk_moons[0]
    # m_y = mk_moons[1]
    # plot_learning_curves(simple_nn, m_x, m_y, hidden_size_list=[64, 32])
    # 生成回归数据
    x, y = make_regression(
        n_samples=10000, n_features=2, n_informative=10, n_targets=1, bias=0.0
    )
    plot_learning_curves(simple_nn, x, y, hidden_size_list=[], learning_type='reg')

