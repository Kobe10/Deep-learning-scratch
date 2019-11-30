# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from common.functions import *
from common.gradient import numerical_gradient


# 手写数字识别的两层神经网络 (隐藏层为1层的网络)   利用MNIST数据集进行学习
# 参考了斯坦福大学CS231n课程提供的Python源代码
class TwoLayerNet:
    # 进行参数初始化 输入层的神经元数量、隐藏层的神经元数量、输出层的神经元数量
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重  保存神经网络的参数的字典型变量(实例变量)   权重参数
        # 权重使用符合高斯 分布的随机数进行初始化，偏置使用0进行初始化。
        self.params = {}
        # 第一层的权重
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 第一层的偏置
        self.params['b1'] = np.zeros(hidden_size)
        # 第二层的权重
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 第二层的偏置
        self.params['b2'] = np.zeros(output_size)

    # 进行识别或者推理   参数x是图像数据
    def predict(self, x):
        # 两个权重参数
        W1, W2 = self.params['W1'], self.params['W2']
        # 两个偏置参数
        b1, b2 = self.params['b1'], self.params['b2']
        # np.dot 表示多维数组的点积
        a1 = np.dot(x, W1) + b1
        # 激活函数
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        # 输出层的设计：回归问题用恒等函数，分类问题用 softmax 函数
        # 这里属于分类问题所以使用 softmax函数
        y = softmax(a2)

        return y

    # x:输入数据, t:监督数据   计算损失函数的值
    # 基于predcict函数的结果和正确标签解计算交叉熵误差
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    # 计算识别精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据 计算权重参数的梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        # 保存梯度的字典型变量(numerical_gradient() 方法的返回值)
        grads = {}
        # 是第 1 层权重的梯度
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        # 第 1 层偏置的梯度
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 第 2 层权重的梯度
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # 第 2 层偏置的梯度
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # 计算权重参数的梯度 numerical_gradient函数的高速版本
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
