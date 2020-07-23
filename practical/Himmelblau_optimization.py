from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


# himmelblau 的3D曲面图
"""
fig = plt.figure()

ax = fig.add_subplot(projection='3d')
x = np.arange(-6,6,0.1)
y = np.arange(-6,6,0.1)
X,Y = np.meshgrid(x,y)
Z = himmelblau([X,Y])

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
ax.view_init(60,-30)

plt.show()
"""

# himmelblau 的等高线图
"""
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
X, Y = np.meshgrid(x, y)
Z = himmelblau([X, Y])
plt.figure()
plt.contourf(X, Y, Z,256, alpha=.75, cmap=plt.cm.hot)
C = plt.contour(X, Y, Z,16, colors='black')
# plt.clabel(C, inline=True, fontsize=10)
plt.xticks(())
plt.yticks(())
plt.show()
"""
# 函数优化实战
import tensorflow as tf

x = tf.constant([4., 0.])
for step in range(200):
    with tf.GradientTape() as tape:
        tape.watch([x])  # 加入梯度观察列表
        y = himmelblau(x)  # 前向传播
        # 反向梯度传播
        grads = tape.gradient(y, [x])[0]
        x -= 0.01 * grads  # 0.01学习率
        if step % 20 == 19:
            print('step {}: x = {},y = {}'.format(step, x.numpy(), y.numpy()))
#结果接近himmelblau函数的局部最小值之一，通过改变初试x值，最终所得到接近的局部最小值也不相同