# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    # 这里对数组做了一个特殊处理：如果大于0的x  都是true， 然后利用dtype=int 将true转换为1
    #                      ：如果小于0的x  都是false， 然后利用dtype=int 将true转换为0
    return np.array(x > 0, dtype=np.int)

X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # 指定图中绘制的y轴的范围
plt.show()
