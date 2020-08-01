"""
使用tf2实现fashion-mnist训练模型
"""
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

#定义层数
model = keras.Sequential([
    #用来将数据打平 从[28*28] -> [256,]
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

loss = 'sparse_categorical_crossentropy'
optimizer = tf.keras.optimizers.SGD(0.1)

model.compile(optimizer=tf.keras.optimizers.SGD(0.1),
              loss = loss,
              metrics=['accuracy'])
#开始训练
model.fit(x_train,y_train,epochs=5,batch_size=256)

#测试 使用evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Acc:',test_acc)

