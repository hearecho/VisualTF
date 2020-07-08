"""
独热编码  One-hot
"""
import tensorflow as tf
y = tf.constant([0,1,2,3])
y = tf.one_hot(y,depth=4)
print(y)

"""
tf.Tensor(
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)
"""