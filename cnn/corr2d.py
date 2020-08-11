"""
二维卷积层
"""
import tensorflow as tf


def corr2d(X, K):
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.cast(tf.reduce_sum(X[i:i + h, j:j + w] * K), dtype=tf.float32))
    return Y


def corr2d_multi_in(X, K):
    """
    多通道卷积，结果是每个通道卷积之后相加。
    :param X:
    :param K:
    :return:
    """
    return tf.reduce_sum([corr2d(X[i],K[i]) for i in range(X.shape[0])],axis=0)

def corr2d_multi_in_out(X, K):
    """
    多输入多输出卷积层
    :param X:
    :param K:
    :return:
    """
    return tf.stack([corr2d_multi_in(X, k) for k in K],axis=0)

def corr2d_multi_in_out_1x1(X, K):
    """
    1*1卷积层 相当于全连接层
    :param X:
    :param K:
    :return:
    """
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = tf.reshape(X,(c_i, h * w))
    K = tf.reshape(K,(c_o, c_i))
    Y = tf.matmul(K, X)
    return tf.reshape(Y, (c_o, h, w))

X = tf.constant([[[0,1,2],[3,4,5],[6,7,8]],
                 [[1,2,3],[4,5,6],[7,8,9]]])
K = tf.constant([[[0,1],[2,3]],
                 [[1,2],[3,4]]])

print(corr2d_multi_in(X, K))
