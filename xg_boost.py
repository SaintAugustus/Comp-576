# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 23:19:32 2020

@author: Think
"""

from sklearn.multioutput import MultiOutputRegressor as multi
from xgboost import XGBRegressor as XGBR
from sklearn import preprocessing
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

df_train = pd.read_csv('train_set.csv')
df_test = pd.read_csv('test_set.csv')

def get_data(df):
    df_x = df.iloc[:, 2:-2]
    array_x = np.array(df_x)
    df_y = df.iloc[:, -2]
    array_y = np.array(df_y)
    
    return array_x, array_y

x_train, y_train = get_data(df_train)
x_test, y_test = get_data(df_test)

tree_num_list = [i for i in range(50, 1050, 50)]
max_depth_list = [i for i in range(10, 50, 2)]
min_child_weight_list = [i for i in range(3, 10, 1)]
learning_rate_list = [0.001, 0.003, 0.01, 0.03, 0.1]

min_error_record = 100   #record the mini error, initial value should be large
'''best params, total 4000 conbinations'''
best_tree_num = 0   #600
best_max_depth = 0   #6
best_min_child_weight = 0   #3
best_learning_rate = 0   #0.1
count=0   #record the process
avg_error_train_list = []
'''
#search for the best params
for max_depth in max_depth_list:
    model = multi(XGBR(n_estimators=600, max_depth=max_depth)).fit(x_train, y_train.reshape(-1,1))

    y_trained = model.predict(x_train)
    y_trained = np.array(y_trained)
    error_train = (y_train.reshape(-1,1) - y_trained)/y_train.reshape(-1,1)
    avg_error_train = np.mean(np.abs(error_train))
    avg_error_train_list.append(avg_error_train)
    
    count += 1
    print(count, end=' ')
'''
'''
for tree_num in tree_num_list:
    for max_depth in max_depth_list:
        for min_child_weight in min_child_weight_list:
            for learning_rate in learning_rate_list:
                model = multi(XGBR(n_estimators=tree_num, max_depth=max_depth, 
                                   min_child_weight=min_child_weight, 
                                   learning_rate=learning_rate)).fit(x_train, y_train.reshape(-1,1))

                y_trained = model.predict(x_train)
                y_trained = np.array(y_trained)
                error_train = (y_train.reshape(-1,1) - y_trained)/y_train.reshape(-1,1)
                avg_error_train = np.mean(np.abs(error_train))
                
                count += 1
                print(count, end=' ')
                
                if avg_error_train < min_error_record:
                    min_error_record = avg_error_train
                    best_tree_num = tree_num
                    best_max_depth = max_depth
                    best_min_child_weight = min_child_weight
                    best_learning_rate = learning_rate
'''                    


model = multi(XGBR(n_estimators=1000, max_depth=6, 
                   min_child_weight=3, 
                   learning_rate=0.1)).fit(x_train, y_train.reshape(-1,1))

y_trained = model.predict(x_train)
y_trained = np.array(y_trained)
error_train = (y_train.reshape(-1,1) - y_trained)/y_train.reshape(-1,1)
avg_error_train = np.mean(np.abs(error_train))
print('average error trained is {}'.format(avg_error_train))


y_predict = model.predict(x_test)
y_predict = np.array(y_predict)
error = (y_test.reshape(-1,1) - y_predict)/y_test.reshape(-1,1)
error = np.abs(error)

avg_error = np.mean(error)
print('average error is {}'.format(avg_error))

pd_error = pd.DataFrame(error)
pd_error.to_csv('xgboost-error.csv')

importances = model.estimators_[0].feature_importances_
print('每个维度对应的重要性：', importances)
indices = np.argsort(importances)[::-1]  # a[::-1]让a逆序输出
print('按维度重要性排序的维度的序号：', indices)












