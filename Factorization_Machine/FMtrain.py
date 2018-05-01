# coding: UTF-8
'''
date: 20180430
author: Liangye
'''

import numpy as np
from random import normalvariate

def loadDataSet(data):
    '''
    导入训练数据
    :param data: data,string, 训练数据
    :return:
    dataMat, list, 特征
    labelMat, list, 标签
    '''
    dataMat = []
    labelMat = []
    fr = open(data, 'r') # 打开文件
    for line in fr.readlines():
        lines = line.strip().split('\t')
        lineArr = []
        for i in xrange(len(lines) - 1):
            lineArr.append(float(lines[i]))
        dataMat.append(lineArr)
        labelMat.append(float(lines[-1]) * 2 -1)    # 将lable转化为{-1, 1}, 原先为{0, 1}
    fr.close()
    return dataMat, labelMat

def sigmoid(x):
    '''
    sigmoid函数
    :param x: 参数x
    :return: 利用sigmoid函数转化参数x
    '''
    return 1.0 / (1 + np.exp(-x))

def initialize_v(n, k):
    '''
    初始化交叉项
    :param n: int, 特征个数
    :param k: int, 超参数
    :return: v, mat, 交叉项的系数权重
    '''
    v = np.mat(np.zeros((n, k)))
    for i in xrange(n):
        for j in xrange(k):
            v[i, j] = normalvariate(0, 0.2)     # 利用正态分布初始化交叉系数
    return v

def getCost(predict, classLabels):
    '''
    计算损失值
    :param predict: list, 预测值
    :param classLabels: list, 标签
    :return: 返回损失值
    '''
    m = len(predict)
    error = 0.0
    for i in xrange(m):
        error = error - np.log(sigmoid(predict[i] * classLabels[i]))
    return error

def getPrediction(dataMatrix, w0, w, v):
    '''
    对样本进行预测
    :param dataMatrix: mat, 特征数据
    :param w0: float, 截距项
    :param w: int, 常数项权重
    :param v: float, 交叉项权重
    :return: list, 预测结果
    '''
    m = np.shape(dataMatrix)[0]     # 样本数量
    result = []
    for x in xrange(m):
        inter_1 = dataMatrix[x] * v  # dataMatrix[x]看作公式中的x_i
        inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
        # 交叉项
        interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2
        p = w0 + dataMatrix[x] * w + interaction  # 计算预测的输出
        pred = sigmoid(p[0,0])
        result.append(pred)
    return result


def stocGradAscent(dataMatrix, classLabels, k, max_iter, alpha):
    '''
    使用随机梯度下降法训练FM模型
    :param dataMatrix: mat, 训练集
    :param classLabels: mat, 标签
    :param k: int, v的维度
    :param max_iter: int, 最大迭代次数
    :param alpha: float, 学习速率
    :return:
    w0, float, 截距项
    w, mat, 权重
    v, mat, 交叉系数
    '''
    m, n = np.shape(dataMatrix)

    # 1. 初始化权重
    w = np.zeros((n, 1))        # n是特征个数
    w0 = 0      # 截距项
    v = initialize_v(n, k)      # 初始化v

    # 2. 训练
    for it in xrange(max_iter):
        # print 'iteration:', it
        for x in xrange(m):
            inter_1 = dataMatrix[x] * v     # dataMatrix[x]看作公式中的x_i
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
            # 交叉项
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2

            p = w0 + dataMatrix[x] * w + interaction        # 计算预测的输出
            loss = sigmoid(classLabels[x] * p[0,0]) - 1     # 依据公式计算

            # 更新w0
            w0 -= alpha * loss * classLabels[x]

            # 更新w和v
            for i in xrange(n):     # 更新n个w_j
                if dataMatrix[x, i] != 0:
                    w[i, 0] -= alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in xrange(k):
                        v[i, j] -= alpha * loss * classLabels[x] * (dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

        # 计算损失值
        if it % 1000 == 0:
            pred = getPrediction(np.mat(dataMatrix), w0, w, v)
            print "\t------- iter: ", it, " , cost: ", getCost(pred, classLabels)

    # 3. 返回模型参数
    return w0, w, v

def getAccuarcy(predict, classLabel):
    '''
    计算预测的准确性
    :param predict: list, 预测结果
    :param classLabel: list, 标签
    :return: float, 预测误差
    '''
    m = len(predict)
    errorNum = 0
    for i in xrange(m):
        if float(predict[i]) < 0.5 and classLabel[i] > 1.0:
            errorNum += 1
        elif float(predict[i]) >= 0.5 and classLabel[i] == -1.0:
            errorNum += 1
        else:
            continue
    return float(errorNum)/float(m)

def save_model(file_name, w0, w, v):
    '''
    保存训练好的模型
    :param file_name: string, 文件名
    :param w0: float, 截距项
    :param w: mat, 一次项权重
    :param v: 交叉项权重
    '''
    f = open(file_name, 'w')
    # 1. 保存w0
    f.write(str(w0) + '\n')
    # 2. 保存w
    w_array = []
    m = np.shape(w)[0]
    for i in xrange(m):
        w_array.append(str(w[i, 0]))
    f.write('\t'.join(w_array) + '\n')
    # 3. 保存v
    m1, n1 = np.shape(v)
    for i in xrange(m1):
        v_tmp = []
        for j in xrange(n1):
            v_tmp.append(str(v[i,j]))
        f.write('\t'.join(v_tmp) + '\n')
    f.close()

if __name__ == '__main__':
    # 1、导入训练数据
    print "---------- 1.load data ---------"
    dataTrain, labelTrain = loadDataSet('data.txt')
    print "---------- 2.learning ---------"
    # 2、利用随机梯度训练FM模型
    w0, w, v = stocGradAscent(np.mat(dataTrain), labelTrain, 3, 10000, 0.01)
    predict = getPrediction(np.mat(dataTrain), w0, w, v)
    print "----------training accuracy: %f" % (1 - getAccuarcy(predict, labelTrain))
    print "---------- 3.save result ---------"
    save_model('weights', w0, w, v)
