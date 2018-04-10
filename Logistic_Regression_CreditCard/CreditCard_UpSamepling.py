# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import KFold    # 使用这个, 教程中代码才能实现
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from imblearn.over_sampling import SMOTE

credit_card = pd.read_csv('creditcard.csv')
columns = credit_card.columns

features_columns = columns.delete(len(columns)-1)       # 列名删除最后一列
features = credit_card[features_columns]
lables = credit_card['Class']

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

# 数据集划分
X_train, X_test, Y_train, Y_test = train_test_split(features, lables, test_size=0.2, random_state=0)

oversampler = SMOTE(random_state=0)
os_features, os_labels = oversampler.fit_sample(X_train, Y_train)

os_features = pd.DataFrame(os_features)
os_labels = pd.DataFrame(os_labels)

best_c = printing_Kfold_scores(os_features, os_labels)


