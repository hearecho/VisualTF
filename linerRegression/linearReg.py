from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import matplotlib;
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

learning_rate = 0.01
#训练次数
training_steps = 1000
#展示步数
display_step = 50

#训练数据集
X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
              7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
              2.827,3.465,1.65,2.904,2.42,2.94,1.3])

#w,b  是张量  tensor
w = tf.Variable(np.random.randn(),name="weight")
b = tf.Variable(np.random.randn(),name="bias")

# 画图的参数
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X,Y)
plt.ion()  #更新
plt.show()

def linear_regression(x):
    return w * x + b

def mean_square(y_pred,y_true):
    #预测值 与 真实值相减
    return tf.reduce_mean(tf.square(y_pred-y_true))

optimizer = tf.optimizers.SGD(learning_rate)


# Optimization process.
def run_optimization():
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
    # 计算梯度
    gradients = g.gradient(loss, [w, b])
    # Update W and b following gradients. 更新w和b的值
    optimizer.apply_gradients(zip(gradients, [w, b]))

wlist = []
blist = []
# Run training for the given number of steps.
for step in range(1, training_steps + 1):
    # Run the optimization to update W and b values.
    run_optimization()

    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        wlist.append(w.numpy())
        blist.append(b.numpy())
        # if (step % 100 == 0):
        #     plt.plot(X, np.array(w * X + b), label='Fitted line {}'.format(step))
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, w.numpy(), b.numpy()))

# plt.plot(X, Y, 'ro', label='Original data')
# fig,ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(np.arange(1,21,1), np.array(wlist),label='w change',color="red")
# ax2.plot(np.arange(1,21,1), np.array(blist),label='b change')
# ax1.set_xlabel('traning count')
# ax1.set_ylabel('w',color = 'g')   #设置Y1轴标题
# ax2.set_ylabel('',color = 'b')   #设置Y2轴标题
# fig.legend()
# plt.show()

#拟合过程生成动态图
fig,ax = plt.subplots();
ln1, = ax.plot(X, Y, 'ro', label='Original data',lw=2)
ln2, = ax.plot([],[],'-',label="change",lw=1)

def init():
    x_out = list(X)
    y_out = list(Y)
    return ln1,

def update(i):
    x_in = list(X)
    y_in = [wlist[i]*x+blist[i] for x in x_in]
    ln2.set_data(x_in,y_in)
    return ln2,

ani = animation.FuncAnimation(fig,update,
                              frames=range(len(wlist)),
                              init_func=init,
                              interval=100)
ani.save('linerReg.gif',writer='imagemagick', fps=100)



