# coding: utf-8
import numpy as np


def identity_function(x):
    return x


# 两个函数的对比
# sigmoid 函数是一条平 滑的曲线，输出随着输入发生连续性的变化。而阶跃函数以 0 为界，输出发 生急剧性的变化。sigmoid 函数的平滑性对神经网络的学习具有重要意义。
# 阶跃函数 使用matplotlib库来进行定义
def step_function(x):
    # 使用numpy中的特殊技巧  当x>0  返回值是1  当x<0 返回值是0  dtype是关键 会自动的做转换
    return np.array(x > 0, dtype=np.int)


# 激活函数   就是一个公式   1 / (1 + np.exp(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


# 缺陷:溢出问题 softmax 函数的实现中要进行指 数函数的运算，但是此时指数函数的值很容易变得非常大。比如，e10 的值 会超过 20000，e100 会变成一个后面有 40 多个 0 的超大值，e1000 的结果会返回
# 一个表示无穷大的 inf。如果在这些超大值之间进行除法运算，结果会出现“不 确定”的情况。
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 交叉熵误差的实现函数
# y是神经网络的输出，t是监督数据
def cross_entropy_error(y, t):
    # 针对一个数据集的处理操作  y的维度为1：改变数据的形状：重新reshape
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
