# coding: utf-8
from layer_naive import *

from ch05.layer_naive import MulLayer

# 实现苹果购买逻辑
# 参数初始化
apple = 100
apple_num = 2
tax = 1.1

# 正向
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

# 正向输出价格
print("正向价格----")
print("price:", int(price))
# 价格关于苹果价格的导数
print("价格关于苹果价格的导数----")
print("dApple:", dapple)
# 数量关于苹果价格的导数
print("数量关于苹果价格的导数------")
print("dApple_num:", int(dapple_num))
# 税率关于苹果价格的导数
print("税率关于苹果价格的导数----")
print("dTax:", dtax)
