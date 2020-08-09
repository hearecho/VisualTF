"""
自定义层
"""
import tensorflow as tf
import numpy as  np

class DiyLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='w',
                                 shape=[input_shape[-1], self.units], initializer=tf.random_normal_initializer())
        self.b = self.add_weight(name='b',
                                 shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.w) + self.b


X = tf.random.uniform((2,20))
print(X)
layer = DiyLayer(1)
print(layer(X))
# print(layer.get_weights())
