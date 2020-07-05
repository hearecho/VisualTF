# VisualTF
可视化tensorflow学习记录，通过可视化方式观察梯度下降以及权重，偏差等变化情况。numpy,matploytlib

### [模拟线性回归](/linerRegression/linearReg.py)
![](https://media.giphy.com/media/JNPAKFsTx4PhRX1V65/giphy.gif)
#### [线性回归思想](/linerRegression/linerthink.py)
> 线性回归问题主要适用于所有数据可以模拟成一条函数曲线，而且一般的线性回归问题可以通过不断的缩减权重weight和偏置bias来进行模拟。
> h(x) = w*x + b
> w,b就是需要不断进行修正。一般都是通过梯度下降的方法将w,b进行修正。
> 梯度下降一般都是使用求导，不过有很多种优化方法。用来加快w,b修正的速度
> 一般的梯度下降方式有:
1. SGD
> 最简单的方式，就是测试数据分批进行神经网络计算。
2. Momentum
> 传统的W参数更新为: W += -Learning rate * dx
> Momentum 则是加上一个惯性，即m = b1 * m-Learning rate *dx W += m
3. AdaGrad
> 对学习率进行更新: v += dx^2 W += -Learning rate * dx/√v v算是一种惩罚措施，逼迫朝着正确的方向进行变化。
4. RMSProp
> 将AdaGrad和Momentum结合起来  v =  b1*v + (1-b1)*dx^2 W += -Learning rate * dx / √v
5. Adam
> m 