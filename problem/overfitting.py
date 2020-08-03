"""
使用权重衰减解决过拟合问题
"""

import tensorflow as tf
from tensorflow.keras import layers, models, initializers, optimizers, regularizers
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# 20 个训练数据，100个测试数据
# 模型 y= b + w1*x1 + ... + w200*x200
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = tf.ones((num_inputs, 1)) * 0.01, 0.05

features = tf.random.normal(shape=(n_train + n_test, num_inputs))
labels = tf.keras.backend.dot(features, true_w) + true_b
labels += tf.random.normal(mean=0.01, shape=labels.shape)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]


# print(train_features.shape)
print(train_labels)
# print(true_w.shape)
# print(true_b)
def init_params():
    w = tf.Variable(tf.random.normal(mean=1, shape=(num_inputs, 1)))
    b = tf.Variable(tf.zeros(shape=(1,)))
    return [w, b]


def l2_penalty(w):
    """
    惩罚项，用于处理数据量较少的情况下的过拟合,这个只是惩罚项，前面还有系数
    :param w:
    :return:
    """
    return tf.reduce_sum((w ** 2)) / 2


def model(X, w, b):
    return tf.matmul(X, w) + b


# mse损失函数
def squared_loss(y_hat, y):
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size, grads):
    """Mini-batch stochastic gradient descent."""
    for i, param in enumerate(params):
        param.assign_sub(lr * grads[i] / batch_size)


def print2(x, y1, y2):
    plt.rcParams['figure.figsize'] = (6, 6)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(x, y1, 'r-')
    plt.plot(x, y2, 'b-')
    plt.show()


batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = model, squared_loss
optimizer = tf.keras.optimizers.SGD(lr)
train_iter = tf.data.Dataset.from_tensor_slices(
    (train_features, train_labels)).batch(batch_size).shuffle(batch_size)


def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape(persistent=True) as tape:
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            grads = tape.gradient(l, [w, b])
            sgd([w, b], lr, batch_size, grads)
        train_ls.append(tf.reduce_mean(loss(net(train_features, w, b),
                                            train_labels)).numpy())
        test_ls.append(tf.reduce_mean(loss(net(test_features, w, b),
                                           test_labels)).numpy())
    print2(range(1, num_epochs + 1), train_ls, test_ls)
    print('L2 norm of w:', tf.norm(w).numpy())


fit_and_plot(lambd=3)
