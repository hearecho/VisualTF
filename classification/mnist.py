from __future__ import absolute_import, division, print_function
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# minist数据集 参数
num_classes = 10        #0-9  10个数字
num_features = 784      #28 *28 像素块大小
#训练过程的参数
learning_rate = 0.01
training_steps = 1000
batch_size = 256
display_step = 50

#准备数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train / 255., x_test / 255.

#过滤与处理数据
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

#设置权重，偏差变量
# 28*28 10十个数字 每个数字都是28*28像素的  二维矩阵矩阵值为1
W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")
# 初始化值为偏差为一维矩阵 矩阵内的值为0
b = tf.Variable(tf.zeros([num_classes]), name="bias")

# Logistic regression (Wx + b).
def logistic_regression(x):
    # 应用softmax将logit归一化为概率分布。
    #matmul为矩阵乘法 因为w为784 * 10 大小的矩阵
    return tf.nn.softmax(tf.matmul(x, W) + b)

# 损失函数
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # 计算交叉熵
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred),1))

# 精度指标
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 随机梯度下降优化器
optimizer = tf.optimizers.SGD(learning_rate)


# 优化过程.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = logistic_regression(x)
        loss = cross_entropy(pred, y)

    # Compute gradients.
    gradients = g.gradient(loss, [W, b])

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W, b]))

accuracys = []
# 按照给定的训练次数进行训练
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    pred = logistic_regression(batch_x)
    acc = accuracy(pred, batch_y)
    accuracys.append(acc)
    if step % display_step == 0:
        loss = cross_entropy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

pred = logistic_regression(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# n_images = 5
# test_images = x_test[:n_images]
# predictions = logistic_regression(test_images)
# # Display image and model prediction.
# for i in range(n_images):
#     plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='Greens')
#     plt.show()
#     print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))
xlist = [i for i in range(1,training_steps+1)]
plt.figure()
plt.scatter(np.array(xlist),np.array(accuracys))
plt.show()
