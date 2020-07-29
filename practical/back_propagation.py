"""
反向传播实践,不适用框架手动计算模型的梯度
"""
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import cm
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

N_SAMPLES = 2000
TEST_SIZE = 0.3
X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=TEST_SIZE, random_state=42)


def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None, dark=False):
    if (dark):
        plt.style.use('dark_background')
    plt.figure(figsize=(16, 12))
    axes = plt.gca()
    axes.set(xlabel="$x_1$", ylabel="$x_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if (XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=1, cmap=cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    # 绘制散点图，根据标签区分颜色
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='none')
    plt.savefig('dataset.png')


class Layer:
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        """
        :param n_input: 输入节点数
        :param n_neurons: 输出节点数
        :param activation: 激活函数
        :param weights: 权重
        :param bias: 偏差
        """
        self.weights = weights if weights is not None  else np.random.randn(n_input, n_neurons) * np.sqrt(1 / n_neurons)
        self.bias = bias if bias is not None else np.random.rand(n_neurons) * 0.1
        self.activation = activation  # 激活函数类型，如’sigmoid’
        self.last_activation = None  # 激活函数的输出值 o
        self.error = None  # 用于计算当前层的 delta 变量的中间变量
        self.delta = None  # 记录当前层的 delta 变量，用于计算梯度
        pass
    def activate(self,x):
        """
        last_activation 保存当前层的输出值
        :param x:
        :return:
        """
        # 前向传播函数
        r = np.dot(x, self.weights) + self.bias  # X@W+b
        # 通过激活函数，得到全连接层的输出 o
        self.last_activation = self._apply_activation(r)
        return self.last_activation
        pass

    def _apply_activation(self,r):
        """
        根据激活函数对输出值进行处理
        :param r: 输出结果
        :return:
        """
        # 计算激活函数的输出
        if self.activation is None:
            return r  # 无激活函数，直接返回
        # ReLU 激活函数
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        # tanh 激活函数
        elif self.activation == 'tanh':
            return np.tanh(r)
        # sigmoid 激活函数
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r
    def apply_activation_derivative(self, r):
        """
        激活函数的倒数
        :param r: 输出结果
        :return:
        """
        if self.activation is None:
            return np.ones_like(r)
        # ReLU 函数的导数实现
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
        # tanh 函数的导数实现
        elif self.activation == 'tanh':
            return 1 - r ** 2
        # Sigmoid 函数的导数实现
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        return r
        pass

class NeuralNetwork:
    def __init__(self):
        self._layers = []

    def add_layer(self,layer):
        self._layers.append(layer)

    def feed_forward(self,X):
        """
        前向传播，只需要计算各层的前向计算函数,依次向前叠加
        :param X:
        :return:
        """
        for layer in self._layers:
            X = layer.activate(X)
        return X

    def backpropagation(self, X, y, learning_rate):
        """
        反向传播，主要是为了修正参数
        :param X:
        :param y:
        :param learning_rate:
        :return:
        """
        #得到输出值
        output = self.feed_forward(X)
        for i in reversed(range(len(self._layers))):
            #反向传播，所以从最后一层即输出层开始计算
            layer = self._layers[i]
            if layer == self._layers[-1]:
                layer.error = y- output
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i+1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error *layer.apply_activation_derivative(layer.last_activation)
        #更新权重与偏差
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # o_i 为上一网络层的输出
            o_i = np.atleast_2d(X if i == 0 else self._layers[i -1].last_activation)
            # 梯度下降算法，delta 是公式中的负数，故这里用加号
            layer.weights += layer.delta * o_i.T * learning_rate
        pass

    def train(self, X_train, X_test, y_train, y_test, learning_rate,max_epochs):
        """
        训练函数
        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :param learning_rate:
        :param max_epochs:
        :return:
        """
        y_onehot = np.zeros((y_train.shape[0], 2))
        y_onehot[np.arange(y_train.shape[0]),y_train] = 1
        mses = []
        for i in range(max_epochs):  # 训练 1000 个 epoch
            for j in range(len(X_train)):  # 一次训练一个样本
                self.backpropagation(X_train[j], y_onehot[j], learning_rate)
            if i % 10 == 0:
                # 打印出 MSE Loss
                mse = np.mean(np.square(y_onehot - self.feed_forward(X_train)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))
                # 统计并打印准确率
                # print('Accuracy: %.2f%%' % (self.accuracy(self.predict(X_test),y_test.flatten()) * 100))
        return mses

if __name__ == '__main__':
    nn = NeuralNetwork()
    #生成的数据是二分类问题
    nn.add_layer(Layer(2, 25, 'sigmoid'))  # 隐藏层 1, 2=>25
    nn.add_layer(Layer(25, 50, 'sigmoid'))  # 隐藏层 2, 25=>50
    nn.add_layer(Layer(50, 25, 'sigmoid'))  # 隐藏层 3, 50=>25
    nn.add_layer(Layer(25, 2, 'sigmoid'))  # 输出层, 25=>2
    nn.train(X_train,X_test,y_train,y_test,0.001,1000)

