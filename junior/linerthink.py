"""
线性回归思想,圣经网络的处理思路
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
x_data = []
y_data = []
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

def genData(n):
    """
    生成测试n数据
    :param n:
    :return:
    """
    data = []
    for i in range(n):
        # 均匀分布 获取一个数
        x = np.random.uniform(-10., 10.)
        # 高斯分布 获取一个噪声
        eps = np.random.normal(0., 1)
        # 生成对应的方程式的偏差结果
        y = 1.477 * x + 0.089 + eps
        x_data.append(x)
        y_data.append(y)
        data.append([x, y])
    return np.array(data)


def mse(b, w, points):
    """
    计算均方误差损失值
    :param b: bias  偏差
    :param w: weight 权重
    :param points: 训练使用的点集
    :return:
    """
    totalBias = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalBias += (y - (w * x + b)) ** 2
    return totalBias / float(len(points))


def setp_gradient(cur_w, cur_b, points, lr):
    """
    计算梯度，得出新的weight和bias
    :param cur_b: 当前偏差
    :param cur_w: 当前权重
    :param points:  测试数据点集
    :param lr:      学习率
    :return:
    """
    b_gradient = 0
    w_gradient = 0
    M = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 误差函数对于b的倒数, grad_b = 2(wx+b-y)
        b_gradient += (2 / M) * ((cur_w * x + cur_b) - y)
        # 误差函数对 w 的导数：grad_w = 2(wx+b-y)*x
        w_gradient += (2 / M) * x * ((cur_w * x + cur_b) - y)
    # 返回新的 w，b
    new_w = cur_w - (lr * w_gradient)
    new_b = cur_b - (lr * b_gradient)
    return [new_w,new_b]

def update(points,s_w,s_b,lr,tn):
    """
    循环更新 w,b
    :param points: 点集
    :param s_w: w初始值
    :param s_b: b初始值
    :param lr:  学习率
    :param tn: training_nums  训练次数
    :return:
    """
    w = s_w
    b = s_b
    for step in range(tn):
        w,b = setp_gradient(w,b,np.array(points),lr)
        loss = mse(b,w,points)
        if step %50 == 0:
            print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")
    return [w,b]

def printer(x,y):
    plt.figure()
    plt.scatter(x,y)
    plt.show()

    pass

def main(data):
    lr = 0.01
    init_w = 0
    init_b = 0
    tn = 1000
    [w, b] = update(data, init_w, init_b, lr, tn)
    loss = mse(b, w, data)  # 计算最优数值解 w,b 上的均方差
    print(f'Final loss:{loss}, w:{w}, b:{b}')

if __name__ == '__main__':
    data = genData(100)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()
    main(data)

