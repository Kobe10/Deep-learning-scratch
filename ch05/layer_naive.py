# coding: utf-8

# 乘法层的实现   乘法层--MulLayer
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # 正向传播  x*y
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    # 反向传播 翻转xy
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


# 加法层的实现   加法层--AddLayer
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
