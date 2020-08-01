import matplotlib
matplotlib.use("TkAgg")
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

#加载数据集
from tensorflow.keras.datasets import fashion_mnist
batch_size = 256
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
# print(x_train[0])
#处理数据，将图片中的数据变为0~1之间
x_train = tf.cast(x_train,tf.float32)/255
x_test = tf.cast(x_test,tf.float32)/255
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
# print(len(x_train),len(x_test))
#训练数据x是28*28的图像，即不适用三色表示的图片[h,w]，y_train是每个图片对应的标签,后续处理需要将其进行onehot编码

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)))
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

# 展示数据集效果
# X,y = [],[]
# for i in range(9):
#     X.append(x_train[i])
#     y.append(y_train[i])
#     print(y_train[i])
# show_fashion_mnist(X,get_fashion_mnist_labels(y))

#训练模型
num_inputs = 784
num_outputs = 10
W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.01, dtype=tf.float32))
b = tf.Variable(tf.zeros(num_outputs, dtype=tf.float32))
def softmax(logits, axis=-1):
    return tf.exp(logits)/tf.reduce_sum(tf.exp(logits), axis, keepdims=True)

def model(X):
    #矩阵相乘,并先把输入进行了处理，变为一行
    logits = tf.matmul(tf.reshape(X,shape=(-1,W.shape[0])),W)+b
    #softmax用于处理输出结果，将输出结果固定在
    return softmax(logits)
#交叉熵
def cross_entropy(y_hat, y):
    """
    :param y_hat: 预测值
    :param y: 真实值
    :return:
    """
    y = tf.cast(tf.reshape(y, shape=[-1, 1]),dtype=tf.int32)
    y = tf.one_hot(y, depth=y_hat.shape[-1])
    y = tf.cast(tf.reshape(y, shape=[-1, y_hat.shape[-1]]),dtype=tf.int32)
    return -tf.math.log(tf.boolean_mask(y_hat, y)+1e-8)

def accuracy(y_hat, y):
    return np.mean((tf.argmax(y_hat, axis=1) == y))

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for _, (X, y) in enumerate(data_iter):
        y = tf.cast(y,dtype=tf.int64)
        acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64) == y)
        n += y.shape[0]
    return acc_sum / n

num_epochs, lr = 50, 0.05
# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = tf.reduce_sum(loss(y_hat, y))
            grads = tape.gradient(l, params)
            if trainer is None:
                # 如果没有传入优化器，则使用原先编写的小批量随机梯度下降
                for i, param in enumerate(params):
                    param.assign_sub(lr * grads[i] / batch_size)
            else:
                # tf.keras.optimizers.SGD 直接使用是随机梯度下降 theta(t+1) = theta(t) - learning_rate * gradient
                # 这里使用批量梯度下降，需要对梯度除以 batch_size, 对应原书代码的 trainer.step(batch_size)
                trainer.apply_gradients(zip([grad / batch_size for grad in grads], params))

            y = tf.cast(y, dtype=tf.float32)
            train_l_sum += l.numpy()
            train_acc_sum += tf.reduce_sum(tf.cast(tf.argmax(y_hat, axis=1) == tf.cast(y, dtype=tf.int64), dtype=tf.int64)).numpy()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'% (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

# trainer = tf.keras.optimizers.SGD(lr)
# train_ch3(model, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr,trainer)
for X,y in train_iter:
    y_hat = model(X)
    print(cross_entropy(y_hat,y))
    break

