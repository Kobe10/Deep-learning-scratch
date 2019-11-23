# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


X = np.arange(-5.0, 5.0, 0.1)
# 这里sigmoid函数会和np数组中的每个元素进行运算，得到不同的y值
Y = sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()
