"""
多特征值实现
y = w1*x1 + w2*x2 + b
"""
import random

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as numpy
import tensorflow as tf

def gen_data(nums_example,nums_inputs):
    true_w = [2,-3.4]
    true_b = 4.2
    features = tf.random.normal((nums_example,nums_inputs),stddev=1)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += tf.random.normal(labels.shape, stddev=0.01)
    return features,labels


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = indices[i: min(i+batch_size, num_examples)]
        yield tf.gather(features, axis=0, indices=j), tf.gather(labels, axis=0, indices=j)


w = tf.Variable(tf.random.normal((2,1),stddev=1))
b = tf.Variable(tf.zeros((1,)))

def mse(y_pred,y):
    return (y_pred - tf.reshape(y, y_pred.shape)) ** 2 /2

def sgd(params, lr, batch_size, grads):
    """小批量随机梯度下降，不断的修正，而不是训练一整批之后才进行修正"""
    for i, param in enumerate(params):
        param.assign_sub(lr * grads[i] / batch_size)
def linreg(features,w,b):
    return w[0]*features[:,0]+w[1]*features[:,1]+b

features,labels = gen_data(1000,2)
lr = 0.01
num_epochs = 20
net = linreg
loss = mse
for epoch in range(num_epochs):
    for X, y in data_iter(10, features, labels):
        with tf.GradientTape() as t:
            t.watch([w,b])
            l = tf.reduce_sum(loss(net(X, w, b), y))
        grads = t.gradient(l, [w, b])
        sgd([w, b], lr, 10, grads)
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, tf.reduce_mean(train_l)))

print(w.numpy())
print(b.numpy())