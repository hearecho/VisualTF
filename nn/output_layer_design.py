import tensorflow as tf

x = tf.constant([30.0, 1.0, 50.0, 100.0])
print(tf.sigmoid(x))
"""
归一化到[0,1]
tf.Tensor([1.        0.7310586 1.        1.       ], shape=(4,), dtype=float32)
"""

x = tf.constant([2., 1., 0.1])
print(tf.nn.softmax(x))
"""
归一化到[0,1],且和为1
tf.Tensor([0.6590012  0.24243298 0.09856589], shape=(3,), dtype=float32)
"""
x = tf.linspace(-6.,6.,10)
print(tf.tanh(x))
"""
[-1,1]
[-0.99998784 -0.99982315 -0.9974579  -0.9640276  -0.58278286  0.58278316
  0.9640276   0.99745804  0.99982315  0.99998784], shape=(10,), dtype=float32)
"""