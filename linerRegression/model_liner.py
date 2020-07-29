"""
使用tf2构建线性模型
"""
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


X = np.linspace(-1,1,100)[:, np.newaxis]
noise = np.random.normal(0,0.1,size=X.shape)
y = X*1.16 + noise + 0.8

#训练点集图像
# plt.figure()
# plt.scatter(X,y)
# plt.show()

class LinerModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.ones_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self,input):
        output = self.dense(input)
        return output
plt.ion()
model = LinerModel()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_mean(tf.square(y_pred - y))

    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    if i % 5 == 0:
        plt.cla()
        plt.scatter(X,y)
        plt.plot(X,y_pred,'r-',lw=5)
        plt.text(0.5, 0, 'step={}Loss={:.4f}'.format(i,loss), fontdict={'size': 16, 'color': 'red'})
        plt.pause(0.2)

plt.ioff()
plt.show()
print(model.variables[0].numpy())

# print(model.variables)