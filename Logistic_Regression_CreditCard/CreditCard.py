# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import KFold    # 使用这个, 教程中代码才能实现
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, recall_score, classification_report
# from imblearn.over_sampling import SMOTE

# 读取文件
data = pd.read_csv('creditcard.csv')
# print data.head()

# 统计类别
count_classes = pd.value_counts(data['Class']).sort_index()
# print count_classes

# 数据标准化
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape((-1, 1)))
data = data.drop(['Time', 'Amount'], axis=1)

# 数据划分
X = data.loc[:, data.columns != 'Class']
Y = data.loc[:, data.columns == 'Class']
# print X.head()

# 监控类别异常的数据
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)       # 把Class==0数据的索引取出来
# print len(fraud_indices)

# 取出类别正常数据的索引
normal_indices = data[data.Class == 0].index        # pandas.core.indexes.numeric.Int64Index
# print type(normal_indices)

# 下采样
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)       # 对正样本下采样, 采样数量与负样本一致
random_normal_indices = np.array(random_normal_indices)     # 转化为ndarray
# print len(random_normal_indices)

# 合并两种indices, 得新indices
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
# print len(under_sample_indices)

# 根据新的indices, 下采样提取数据
under_sample_data = data.loc[under_sample_indices, :]
# print pd.value_counts(under_sample_data['Class'])

# 下采样后数据集划分
X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']
Y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class']

# 交叉验证, 用X_test测试模型性能
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)        # random_state保证随机切分(即样本被洗牌), 消除样本对模型的影响

# 下采样数据进行交叉验证, X_train_undersample训练模型
X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample = train_test_split(X_undersample,
                                                                                                    Y_undersample,
                                                                                                    test_size=0.3,
                                                                                                    random_state=0)

def printing_Kfold_scores(x_train, y_train):
    # fold = KFold(len(y_train), 5, shuffle=False)
    fold = KFold(5, shuffle=False, random_state=0)
    c_param_range = [0.01, 0.1, 1, 10, 100]     # 惩罚粒度

    # 创建存放结果的DataFrame
    result_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean_recall_score'])
    result_table['C_parameter'] = c_param_range

    j = 0
    # 探索不同惩罚项对模型的影响
    for c_param in c_param_range:
        print '--------------------------------------------'
        print 'C parameter: ', c_param
        print '--------------------------------------------'

        recall_accs = []
        for iteration, train_test_ind in enumerate(fold.split(x_train), start = 1):
            # 给定具体的惩罚项, 实例化模型对象
            lr = LogisticRegression(C=c_param, penalty='l1')
            # 训练模型
            lr.fit(x_train.iloc[train_test_ind[0], :], y_train.iloc[train_test_ind[0], :].values.ravel())
            # 预测
            y_pred_undersample = lr.predict(x_train.iloc[train_test_ind[1], :].values)
            # 计算精确度
            recall_acc = recall_score(y_train.iloc[train_test_ind[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print 'Iteration', iteration, ': recall score = ', recall_acc

        result_table.loc[j, 'Mean_recall_score'] = np.mean(recall_accs)
        j = j + 1
        print " "
        print 'Mean recall score', np.mean(recall_accs)
        print " "

    # 由于result_table的Mean_recall_score类型变化了, 这里做一步转换
    result_table["Mean_recall_score"] = result_table["Mean_recall_score"].astype('float32')

    # 交叉验证得到的最佳惩罚项
    # print result_table
    # print result_table.dtypes
    best_c = result_table.iloc[result_table['Mean_recall_score'].idxmax()]['C_parameter']
    print '*********************************************************************************'
    print '模型中最佳惩罚项是: ', best_c
    print '*********************************************************************************'

    return best_c


# 模型部分
best_c = printing_Kfold_scores(X_train_undersample, Y_train_undersample)

# Recall = TP/(TP+FN)
lr = LogisticRegression(C = best_c, penalty='l1')
lr.fit(X_train_undersample, Y_train_undersample.values.ravel())

Y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)#.ravel()
Y_pred_undersample_proba_recall = np.asmatrix((Y_pred_undersample_proba > 0.7).astype(int))     # [0.3, 0.7]表示判断为0的概率为0.3, 判断为1的概率为0.7

cnf_matrix = confusion_matrix(Y_test_undersample["Class"], Y_pred_undersample_proba_recall[:,1]).astype(float)
# np.set_printoptions(precision=4)
# print cnf_matrix[1,1]
# print (cnf_matrix[1,0]+cnf_matrix[1,1])
print "Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])



