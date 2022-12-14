from sklearn.multioutput import MultiOutputRegressor as multi
from lightgbm import LGBMRegressor as LGBM
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

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


model = multi(LGBM(objective='regression',
                    max_depth = 6, learning_rate=0.01, 
                    n_estimators=1000, metric='mse', 
                    bagging_fraction = 0.8, feature_fraction = 0.8))


model.fit(x_train, y_train.reshape(-1,1),
        eval_set=[(x_test, y_test.reshape(-1,1))],
        eval_metric='l1')

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
pd_error.to_csv('LGBM-error.csv')