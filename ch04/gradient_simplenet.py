# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


# 神经网络的梯度求解
class simpleNet:
    def __init__(self):
        # 用高斯分布进行初始化
        self.W = np.random.randn(2, 3)

    # x表示输入数据 t接收正确解标签
    def predict(self, x):
        return np.dot(x, self.W)

    # 求损失的函数值
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


# 设置初始化的输入值
x = np.array([0.6, 0.9])
# 设置正确解标签
t = np.array([0, 0, 1])

net = simpleNet()
# 权重参数
print("权重参数----")
print(net.W)
# 最大的索引值
print("索引值----")
print(net.predict(x))
# 定义损失函数    lambda的写法更简单   w表示参数列表    :后面的表示函数表达式(函数里面的内容)
f = lambda w: net.loss(x, t)
# 梯度求解 得出权重参数
dW = numerical_gradient(f, net.W)
print("训练的权重参数----")
print(dW)
