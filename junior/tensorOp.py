from __future__ import print_function
import tensorflow as tf

'''
tensor:张量 可以算作一个矩阵，所以适用于矩阵运算
'''
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(5)

add = tf.add(a,b)
sub = tf.subtract(a,b)
mul = tf.multiply(a,b)
div = tf.divide(a,b)

print("add =", add.numpy())
print("sub =", sub.numpy())
print("mul =", mul.numpy())
print("div =", div.numpy())

#其他操作
mean = tf.reduce_mean([a, b, c])
sum = tf.reduce_sum([a, b, c])

print("mean =", mean.numpy())
print("sum =", sum.numpy())



# 多维
matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[5., 6.], [7., 8.]])
#矩阵乘法
product = tf.matmul(matrix1, matrix2)
print(product)
#得到一个 numpy矩阵
print(product.numpy())
