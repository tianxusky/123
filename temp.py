# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from mxnet import nd, autograd
x = nd.arange(4).reshape((4,1))
#调用attach_grad函数来申请存储梯度的内存
x.attach_grad()
#调用record函数来记录与梯度有关的运算
with autograd.record():
    y = nd.dot(x.T, x)
#y是一个标量,调用backward函数自动求梯度
y.backward()
print(x.grad, x)
    

