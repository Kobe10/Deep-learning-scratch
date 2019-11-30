# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient


# 梯度下降法求解最大值或者最小值
# lr 表示学习率：
# 这里注意： 学习率lr过大或者过小都无法得到好的结果
# init_x 初始值
# step_num 梯度法的重复次数
# 像学习率这样的参数称为超参数。这是一种和神经网络的参数(权重 和偏置)性质不同的参数。相对于神经网络的权重参数是通过训练 数据和学习算法自动获得的，
# 学习率这样的超参数则是人工设定的。 一般来说，超参数需要尝试多个值，以便找到一种可以使学习顺利 进行的设定
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    # numerical_gradient 这个函数会求f函数的梯度，用该梯度乘以学习率得到的值进行更新操作
    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


# 这个是要求极小值或者最小值的函数  x0的平方 + x1的平方
def function_2(x):
    return x[0] ** 2 + x[1] ** 2


init_x = np.array([-3.0, 4.0])

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot([-5, 5], [0, 0], '--b')
plt.plot([0, 0], [-5, 5], '--b')
plt.plot(x_history[:, 0], x_history[:, 1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
