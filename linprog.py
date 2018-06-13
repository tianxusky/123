# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 20:31:11 2018

@author: lenovo
"""
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
num_inputs = 2
num_examples = 1000
true_w = nd.array([2,-3.4])
true_w = true_w.T
true_b = 4.2
#生成数据集
features = nd.random.normal(scale = 1, shape = (num_examples, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal( scale = 0.01, shape = labels.shape)

plt.rcParams['figure.figsize'] = (3.5, 2.5)
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
plt.show()

batch_size = 10
#从训练集中取出batch_size个样例
def data_iter(batch_size, num_examples, features, labels):
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)
#初始化参数
w = nd.random.normal(scale = 0.01, shape = (num_inputs, 1))
b = nd.zeros((1,1))
params = [w,b];
#分配记录梯度的内存
for param in params:
    param.attach_grad()
#定义模型
def linprog(x,w,b):
    return nd.dot(x,w) + b
def loss( y_hat, y ):
    return ( y_hat - y.reshape(y_hat.shape) ) ** 2 / 2
def sgd( params, batch_size, lr ):
    for param in params:
        param[:] = param - lr * param.grad / batch_size
#训练参数
num_epochs = 20;
lr = 0.03;
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size, num_examples, features, labels):
        #记录与求梯度参数有关的运算
        with autograd.record():
            l = loss( linprog(X, w, b), y)
        l.backward()
        sgd(params, batch_size, lr);
    print("eopochs:%d  loss:%f"  % (epoch, l.mean().asnumpy()))
        
print(w,b)
            
    




