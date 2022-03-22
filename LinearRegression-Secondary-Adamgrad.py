# -*- coding = utf8 -*-
# @Time  :2022/2/7 21:15
# @Author : Nico
# File : LinearRegression-Secondary-Adamgrad.py
# @Software: PyCharm


import math
import numpy as np
import matplotlib.pyplot as plt

# 数据集
x = [12.3, 14.3, 14.5, 14.8, 16.1, 16.8, 16.5, 15.3, 17.0, 17.8, 18.7, 20.2, 22.3, 19.3, 15.5, 16.7, 17.2, 18.3, 19.2, 17.3, 19.5, 19.7, 21.2, 23.04, 23.8, 24.6, 25.2, 25.7, 25.9, 26.3]
y = [11.8, 12.7, 13.0, 11.8, 14.3, 15.3, 13.5, 13.8, 14.0, 14.9, 15.7, 18.8, 20.1, 15.0, 14.5, 14.9, 14.8, 16.4, 17.0, 14.8, 15.6, 16.4, 19.0, 19.8, 20.0, 20.3, 21.9, 22.1, 22.4, 22.6]

# print(len(x))

# 分隔训练集和测试集
x_train = x[0: 20]
y_train = y[0: 20]
n_train = len(x_train)

x_test = x[20:]
y_test = y[20:]
n_test = len(x_test)

# Fit model y = w1 * x + w2 * (x^2) + b
# 初始值

# parameters
w1 = -0.1
w2 = 0.3
b = 3

# hype-parameters
# lr = 0.0000001  # 学习率
# Adamgrad
lr_w1 = 0.0
lr_w2 = 0.0
lr_b = 0.0

N = 10000
for j in range(N):
    sum_w1 = 0.0
    sum_w2 = 0.0
    sum_b = 0.0
    for i in range(n_train):
        y_hat = np.array(w1) * x_train[i] + np.array(w2) * (x_train[i] ** 2) + b
        sum_w1 += (y_train[i] - y_hat) * (-x_train[i])
        sum_w2 += (y_train[i] - y_hat) * (-x_train[i] ** 2)
        sum_b += (y_train[i] - y_hat) * (-1)
    det_w1 = 2 * sum_w1
    det_w2 = 2 * sum_w2
    det_b = 2 * sum_b

    lr_w1 = lr_w1 + det_w1 ** 2
    lr_w2 = lr_w2 + det_w2 ** 2
    lr_b = lr_b + det_b ** 2

    w1 = w1 - (1 / math.sqrt(lr_w1) * det_w1)
    w2 = w2 - (1 / math.sqrt(lr_w2) * det_w2)
    b = b - (1 / math.sqrt(lr_b) * det_b)

fig, ax = plt.subplots()
ax.scatter(x_train, y_train)
ax.plot([i for i in range(10, 27)], [w1 * i + w2 * (i ** 2) + b for i in range(10, 27)])
plt.title('y = w1 * x + w2 * (x^2) + b')
plt.legend(('Data Points', 'Model'), loc='upper left')
plt.show()

total_train_loss = 0
for i in range(n_train):
    y_hat = np.array(w1) * x_train[i] + np.array(w2) * (x_train[i] ** 2) + b
    total_train_loss += (y_train[i] - y_hat) ** 2

total_test_loss = 0
for i in range(n_test):
    y_hat = np.array(w1) * x_test[i] + np.array(w2) * (x_test[i] ** 2) + b
    total_test_loss += (y_test[i] - y_hat) ** 2

print("训练集损失值:", total_train_loss)
print("测试集损失值:", total_test_loss)

