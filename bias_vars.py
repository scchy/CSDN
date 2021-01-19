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
from sklearn.datasets import make_moons

loss_func = losses.CategoricalCrossentropy(from_logits=True)


def plot_moon(mx, my):
    x_1 = [mx[i][0] for i in range(len(mx))]
    x_2 = [mx[i][1] for i in range(len(mx))]
    plt.scatter(x_1, x_2, c=my)
    plt.show()


def simple_nn(input_shape: tuple, output_shape: int, hidden_size_list: list = [], print_flag=False):
    input_layer = Input(shape=input_shape)
    if not hidden_size_list:
        out = Dense(output_shape, activation=tf.nn.softmax)(input_layer)
    elif len(hidden_size_list) == 1:
        x = Dense(hidden_size_list[0], activation=tf.nn.leaky_relu)(input_layer)
        out = Dense(output_shape, activation=tf.nn.softmax)(x)
    else:
        x = Dense(hidden_size_list[0], activation=tf.nn.leaky_relu)(input_layer)
        for i in hidden_size_list[1:]:
            x = Dense(i, activation=tf.nn.leaky_relu)(x)
        out = Dense(output_shape, activation=tf.nn.softmax)(x)

    clf = Model(input_layer, out)
    if print_flag:
        print(clf.summary())
    clf.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return clf


def plot_learning_curves(model_in, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    train_errors, test_errors = [], []
    range_step = x_train.shape[0] // 100
    for m in tqdm(range(1, len(x_train), range_step)):
        model_in.fit(x_train[:m], y_train[:m], batch_size=50, epochs=5, verbose=0)
        y_train_pre = model_in.predict(x_train[:m])
        y_test_pre = model_in.predict(x_test)
        train_errors.append(loss_func(y_train[:m], y_train_pre).numpy())
        test_errors.append(loss_func(y_test, y_test_pre).numpy())

    plt.plot(np.sqrt(train_errors), 'r-+', lw=2, label='train')
    plt.plot(np.sqrt(test_errors), 'b-', lw=2, label='test')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    mk_moons = make_moons(n_samples=5000, noise=0.25, random_state=2021)
    m_x = mk_moons[0]
    m_y = mk_moons[1]
    model_t = simple_nn(input_shape=(m_x.shape[1],), output_shape=2, hidden_size_list=[10, 4])
    # plot_moon(m_x, m_y)
    # model = simple_nn(input_shape=(m_x.shape[1],), output_shape=2)
    # plot_learning_curves(simple_nn, m_x, m_y, hidden_size_list=[])
    plot_learning_curves(model_t, m_x, m_y)



