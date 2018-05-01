# coding: UTF-8
'''
date: 20180430
author: Liangye
'''
import numpy as np
from FMtrain import getPrediction

def loadTestData(file_name):
    '''
    导入测试数据
    :param file_name: 测试数据名
    :return: data_test, mat
    '''
    dataMat = []
    fr = open(file_name, 'r')
    for line in fr.readlines():
        lines = line.strip().split('\t')
        lineArr = []

        for i in xrange(len(lines)):
            lineArr.append(float(lines[i]))
        dataMat.append(lineArr)
    fr.close()
    return dataMat

def loadModel(model_file):
    '''
    导入模型
    :param model_file: string, 模型文件名
    :return: w0, np.mat(w), np.mat(v)
    '''
    f = open(model_file, 'r')
    w0 = 0.0
    w = []
    v = []
    line_index = 0      # 标志文件读取到哪一行

    for line in f.readlines():
        lines = line.strip().split('\t')
        if line_index == 0:     # w0
            w0 = float(lines[0].strip())
        elif line_index == 1:       # w开始, w只占一行
            for x in lines:
                w.append(float(x.strip()))
        else:
            v_tmp = []      # v
            for x in lines:
                v_tmp.append(float(x.strip()))
            v.append(v_tmp)
        line_index += 1
    f.close()
    return w0, np.mat(w).T, np.mat(v)       # 确保w, v是matrix, 保证w的列向量

def save_result(file_name, result):
    '''
    保存最终结果
    :param file_name: string, 文件名
    :param result: mat, 测试集的预测结果
    '''
    f = open(file_name, 'w')
    f.write('\t'.join(str(x) for x in result))
    f.close()


if __name__ == '__main__':
    # 1. 导入测试数据
    dataTest = loadTestData('test_data.txt')
    # 2. 导入模型
    w0, w, v = loadModel('weights')
    # 3. 预测数据
    result = getPrediction(dataMatrix = dataTest, w0 = w0, w = w, v = v)
    # 4. 保存结果
    save_result('final_result', result)