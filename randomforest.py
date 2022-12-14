from sklearn.ensemble import RandomForestRegressor
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


n_estimators_list = [i for i in range(10, 1000,20)]
max_depth_list = [i for i in range(8, 30, 1)]
min_samples_split_list = [i for i in range(2, 10, 1)]
min_samples_leaf_list = [i for i in range(1, 10, 1)]

min_error_record = 100  # record the mini error, initial value should be large
#best params, total 4000 conbinations
best_n_estimators = 0  # 500
best_max_depth = 0  # 15
best_min_samples_split = 0  # 2 default
best_min_samples_leaf = 0  # 1 default
count = 0  # record the process

'''
# Search on n_estimatiors first.
avg_error_train_list = []
for max_depth in max_depth_list:
    model = RandomForestRegressor(n_estimators=500,
                                  max_depth=max_depth).fit(x_train, y_train)
    y_trained = model.predict(x_train)
    y_trained = np.array(y_trained)
    error_train = (y_train - y_trained)/y_train
    avg_error_train = np.mean(np.abs(error_train))
    avg_error_train_list.append(avg_error_train)
    
    count += 1
    print(count, end=' ')
'''

'''
#search for the best params
for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        for min_samples_split in min_samples_split_list:
            for min_samples_leaf in min_samples_leaf_list:
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf).fit(x_train, y_train)

                y_trained = model.predict(x_train)
                y_trained = np.array(y_trained)
                error_train = (y_train - y_trained)/y_train
                avg_error_train = np.mean(np.abs(error_train))

                count += 1
                print(count, end=' ')

                if avg_error_train < min_error_record:
                    min_error_record = avg_error_train
                    best_n_estimators = n_estimators
                    best_max_depth = max_depth
                    best_min_samples_split = min_samples_split
                    best_min_samples_leaf = min_samples_leaf

print(best_n_estimators)
print(best_max_depth)
print(best_min_samples_split)
print(best_min_samples_leaf)
'''


forest = RandomForestRegressor(n_estimators=1000, max_features=0.2, max_depth=50)
model = forest.fit(x_train, y_train)

y_trained = model.predict(x_train)
y_trained = np.array(y_trained)
error_train = (y_train - y_trained)/y_train
avg_error_train = np.mean(np.abs(error_train))
y_predict = model.predict(x_test)
y_predict = np.array(y_predict)
error = (y_test - y_predict) / y_test
error = np.abs(error)
print('average error trained is {}'.format(avg_error_train))

avg_error = np.mean(error)
print('average error is {}'.format(avg_error))

pd_error = pd.DataFrame(error)
pd_error.to_csv('rf-error.csv')

importances = forest.feature_importances_
print('每个维度对应的重要性：', importances)
indices = np.argsort(importances)[::-1]  # a[::-1]让a逆序输出
print('按维度重要性排序的维度的序号：', indices)

