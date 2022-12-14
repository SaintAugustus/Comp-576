# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:32:14 2020

@author: Think
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
#from matplotlib import pyplot as plt

torch.set_default_tensor_type('torch.FloatTensor')


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

#x_train_tensor, y_train_tensor = torch.tensor(x_train,dtype=torch.float32), torch.tensor(y_train,dtype=torch.float32)
x_train_tensor, y_train_tensor = torch.tensor(x_train,dtype=torch.float32).cuda(), torch.tensor(y_train,dtype=torch.float32).cuda()#GPU
x_test_tensor, y_test_tensor = torch.tensor(x_test,dtype=torch.float32).cuda(),torch.tensor(y_test,dtype=torch.float32).cuda()
y_train_tensor = y_train_tensor.unsqueeze(1)
y_test_tensor = y_test_tensor.unsqueeze(1)
#print(x_train_tensor.size(), y_train_tensor.size())
#x=torch.tensor(input_data,dtype=torch.float32).cuda()
#y=torch.tensor(result_data).cuda()
#print(x_train_tensor.size(), y_train_tensor.size(), y_test_tensor.size())

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden1,n_hidden2,n_hidden3,n_hidden4,n_hidden5,n_output):
        super(Net,self).__init__()
        self.hidden1=torch.nn.Linear(n_features,n_hidden1)#1
        self.hidden2=torch.nn.Linear(n_hidden1,n_hidden2)#2
        self.hidden3=torch.nn.Linear(n_hidden2,n_hidden3)#2
        self.hidden4=torch.nn.Linear(n_hidden3,n_hidden4)#2
        self.hidden5=torch.nn.Linear(n_hidden4,n_hidden5)#2
        self.predict=torch.nn.Linear(n_hidden5,n_output)#3
        
    def forward(self,x):
        x=torch.relu(self.hidden1(x))
        x=torch.relu(self.hidden2(x))
        x=torch.relu(self.hidden3(x))
        x=torch.relu(self.hidden4(x))
        x=torch.relu(self.hidden5(x))
        x=self.predict(x)
        return x

class NewNet(torch.nn.Module):
    def __init__(self,n_input,n_hidden1,n_hidden2,n_hidden3,n_hidden4,n_hidden5,n_output):
        super(NewNet,self).__init__()
        self.layer1=torch.nn.Sequential(torch.nn.Linear(n_input,n_hidden1), 
                                        torch.nn.BatchNorm1d(n_hidden1), torch.nn.Dropout(0.3))
        self.layer2=torch.nn.Sequential(torch.nn.Linear(n_hidden1,n_hidden2), 
                                        torch.nn.BatchNorm1d(n_hidden2), torch.nn.Dropout(0.2))
        self.layer3=torch.nn.Sequential(torch.nn.Linear(n_hidden2,n_hidden3), 
                                        torch.nn.BatchNorm1d(n_hidden3), torch.nn.Dropout(0.1))
        self.layer4=torch.nn.Sequential(torch.nn.Linear(n_hidden3,n_hidden4), 
                                        torch.nn.BatchNorm1d(n_hidden4), torch.nn.Dropout(0.1))
        self.layer5=torch.nn.Sequential(torch.nn.Linear(n_hidden4,n_hidden5), 
                                        torch.nn.BatchNorm1d(n_hidden5), torch.nn.Dropout(0.1))
        self.predict=torch.nn.Linear(n_hidden5,n_output)
        
    def forward(self, x):
        x=torch.relu(self.layer1(x))
        x=torch.relu(self.layer2(x))
        x=torch.relu(self.layer3(x))
        x=torch.relu(self.layer4(x))
        x=torch.relu(self.layer5(x))
        x=self.predict(x)
        return x
        
        
#net = Net(101,200,200,200,50,10,1)
#net = Net(101,500,500,200,100,20,1).cuda()   #Net GPU

net = NewNet(101,512,256,128,64,32,1).cuda()   #NewNet GPU
net.eval()

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)#优化参数
loss_func = torch.nn.MSELoss()

for epoch in range(12000):
    prediction=net(x_train_tensor)
    loss=loss_func(prediction, y_train_tensor)#带有神经网络所有信息
    optimizer.zero_grad()#归零
    loss.backward()#反向传递，optimizer有参数了
    optimizer.step()#优化
    if epoch%100==0:
        print('loss is {} when epoch is {}'.format(loss.data, epoch))

torch.save(net.state_dict(), 'DNN_parameters.pkl')

net.load_state_dict(torch.load('DNN_parameters.pkl'))

train_predict = []
train_error = []
train_prediction =net(x_train_tensor)
for i in range(len(y_train_tensor)):
    train_predict.append((float(train_prediction[i][0]), float(y_train_tensor[i][0])))
    error = abs(float(train_prediction[i][0]) - float(y_train_tensor[i][0]))/float(y_train_tensor[i][0])
    train_error.append(error)
print('train error is {}'.format(np.mean(train_error)))                        

test_predict = []
test_error = []
test_prediction=net(x_test_tensor)
for i in range(len(y_test_tensor)):
    test_predict.append((float(test_prediction[i][0]), float(y_test_tensor[i][0])))
    error = abs(float(test_prediction[i][0]) - float(y_test_tensor[i][0]))/float(y_test_tensor[i][0])
    test_error.append(error)
print('test error is {}'.format(np.mean(test_error)))

test_error = pd.DataFrame(test_error)
test_error.to_csv('test-error-DNN.csv')    



































