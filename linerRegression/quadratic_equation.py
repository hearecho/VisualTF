"""
二次方程线性回归
"""
import random

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tensorflow as tf


# 模拟数据生成
def gen_data(nums_example):
    true_w = 1.4
    true_b = 0.4
    features = tf.constant(np.linspace(-1, 1, nums_example)[:, np.newaxis])
    noise = np.random.normal(0, 0.1, size=features.shape)
    labels = np.power(features, 2) * 1.4 + 0.4 + noise
    return features, tf.cast(tf.constant(labels), dtype=tf.float32)


def data_iter(batch_size, X, y):
    num_example = len(X)
    indices = list(range(num_example))
    random.shuffle(indices)
    for i in range(0, num_example, batch_size):
        j = indices[i: min(i + batch_size, num_example)]
        # axis表示按照行计算
        yield tf.gather(X, axis=0, indices=j), tf.gather(y, axis=0, indices=j)


w = tf.Variable(tf.random.normal((1,), stddev=1))
b = tf.Variable(tf.zeros((1,)))


def mse(y_pred, y):
    return (y_pred - tf.reshape(y, y_pred.shape)) ** 2 / 2.0


def sgd(params, lr, batch_size, grads):
    """小批量随机梯度下降，不断的修正，而不是训练一整批之后才进行修正"""
    for i, param in enumerate(params):
        param.assign_sub(lr * grads[i] / batch_size)


def model(features, w, b):
    return np.power(features, 2) * w + b


features, labels = gen_data(100)
# plt.scatter(features,labels)
# plt.show()
# print(labels.shape)
# print(features.shape)
# print(w)
plt.ion()


# 训练模型
def train(data_iter, model, loss, num_epochs, lr):
    for epoch in range(num_epochs):
        for X, y in data_iter(10, features, labels):
            with tf.GradientTape() as t:
                t.watch([w, b])
                l = tf.reduce_sum(loss(model(X, w, b), y))
            grads = t.gradient(l, [w, b])
            sgd([w, b], lr, 10, grads)
        train_l = loss(model(features, w, b), labels)
        y_pred = model(features, w, b)
        plt.cla()
        plt.scatter(features, labels)
        plt.plot(features, y_pred, 'r-', lw=5)
        plt.text(0.5, 2, 'epoch={}loss={:.4f}'.format(epoch, tf.reduce_mean(train_l)),
                 fontdict={'size': 16, 'color': 'red'})
        plt.pause(0.2)
    plt.ioff()
    plt.show()


train(data_iter, model, mse, 20, 0.5)
