# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import os
import time

# 读入数据
path = 'data' + os.sep + 'LogiReg_data.txt'
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

# 划分正负样本
# pos = pdData[pdData['Admitted']==1]
# neg = pdData[pdData['Admitted']==0]

# 数据初始化
pdData.insert(0, 'Ones', 1)  # 添加第一列, 数值为1, 之后与theta0相乘得截距项
print pdData.head()
orig_data = pdData.as_matrix()  # 转化为矩阵

cols = orig_data.shape[1]
X = orig_data[:, 0:cols-1]      # (100, 4), 第一列截距项, 共100个样本
Y = orig_data[:, cols-1:cols]       # (100, 1)
theta = np.zeros((1, (cols-1)))  # 存在一列为标签, 故减去1, 行向量(1, 3)
N = X.shape[0]      # 100

# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(z))

# 预测模型
def model(X, theta):
    # theta是行向量, X是矩阵(每行是一个样本), 结果为行向量, 行数为X的行数
    # (100,3)*(3*1) = (100,1)
    return sigmoid(np.dot(X, theta.T))


# 计算梯度
def gradient(X, Y, theta):
    # 保持X是个矩阵(每行是一个样本), Y是列向量, theta是行向量, 这样可以保持X*theta形式
    grad = np.zeros(theta.shape)   # grad也是一个向量, (1,3)
    error = (Y - model(X, theta)).ravel()  # 计算误差, 得到列向量
    # error = (model(X, theta) - Y).ravel()       # ndarray类型, (1, 100)
    for j in range(len(theta.ravel())):   # j是遍历计算每一个theta中的元素 ! range(len(theta))!
        # theta.ravel()=(3, 1), 则len()得到3
        # print type(X[:, j])
        term = np.multiply(error, X[:, j])   # 对应样本误差与对应特征的乘积, 所以用multiply
        #  X[:, j] 是ndarray类型
        grad[0, j] = - np.sum(term) / len(X)    # len(X)表示样本个数
    return grad

# print theta
# print len(theta)
# print theta.ravel()
gradient(X, Y, theta)

# 损失函数
def cost(X, Y, theta):
    left = np.multiply(-Y, np.log(model(X, theta)))  # np.multiply是对应元素相乘
    right = np.multiply((1-Y), np.log((1-model(X, theta))))
    # print len(X)
    return np.sum(left-right)/len(X)

print cost(X, Y, theta)

# 数据洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]    # 有多少列
    X = data[:, 0:cols-1]
    Y = data[:, cols - 1:]
    return X, Y

# 下降停止
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2
def stopCriterion(type, value, threshold):
    # value和threshold都是向量, 里面保存着每个theta
    if type == STOP_ITER:
        return value > threshold    # 当前值大于阈值
    if type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold    # 最后两个梯度值之差小于阈值
    if type == STOP_GRAD:
        return np.linalg.norm(value) < threshold  # 利用第二范数(默认)进行比较
# 梯度下降
def GradientDescend(Data, theta, batch_size, stopType, threshold, alpha):
    X, Y = shuffleData(Data)
    iter_num = 0    # 迭代次数
    k = 0   # mini_batch的索引
    grad = np.zeros(theta.shape)    # 初始化梯度
    cost_value = [cost(X, Y, theta)]  # 第一次求损失, 用list的原因是后续继续添加cost_value, 把每次的损失值记录下来
    while True:
        grad = gradient(X[k:k+batch_size], Y[k:k+batch_size], theta)    # 拿出数据一部分来求梯度, grad是向量
        k = k + batch_size
        if k >= N:      # 这里指定N为所有样本的数量
            k = 0
            X, Y = shuffleData(Data)    # 对数据重新洗牌
        theta = theta - grad * alpha
        cost_value.append(cost(X, Y, theta))   # 这里计算损失需要全部数据计算损失
        iter_num = iter_num + 1

        if stopType == STOP_ITER: value = iter_num
        elif stopType == STOP_COST: value = cost_value
        elif stopType == STOP_GRAD: value = grad
        if stopCriterion(stopType, value, threshold): break     # 这里判断是否达到停止标准
    return theta, cost_value, iter_num-1, grad      # 此处返回grad是最后一次计算的梯度

theta, cost_value, iter_num, grad = GradientDescend(orig_data, theta, N, STOP_ITER, threshold=5000, alpha=0.000001)
print theta, '\n', '\n', cost_value

def prediction (X, theta):
    return [1 if x > 0.5 else 0 for x in model(X, theta)]

def accuracy(predict, X, theta):
    pred = prediction(X, theta)
    X["pred"] = pred
    correct = [1 if ()]