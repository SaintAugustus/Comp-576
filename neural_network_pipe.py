# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_tensor_type('torch.FloatTensor')


df_train = pd.read_csv('train_set.csv')
df_test = pd.read_csv('test_set.csv')

def get_data(df):
    df_x = df.iloc[:, 2:-3]
    array_x = np.array(df_x)
    df_tk = df.iloc[:, -3]   #膜厚单独一列
    array_tk = np.array(df_tk)
    df_y = df.iloc[:, -2]
    array_y = np.array(df_y)
    
    return array_x, array_tk, array_y

x_train, tk_train, y_train = get_data(df_train)
x_test, tk_test, y_test = get_data(df_test)

x_train_tensor, y_train_tensor = torch.tensor(x_train,dtype=torch.float32).cuda(), torch.tensor(y_train,dtype=torch.float32).cuda()#GPU
tk_train_tensor = torch.tensor(tk_train,dtype=torch.float32).cuda()
x_test_tensor, y_test_tensor = torch.tensor(x_test,dtype=torch.float32).cuda(),torch.tensor(y_test,dtype=torch.float32).cuda()
tk_test_tensor = torch.tensor(tk_test,dtype=torch.float32).cuda()
x_train_tensor,x_test_tensor = x_train_tensor.unsqueeze(1),x_test_tensor.unsqueeze(1)
tk_train_tensor,tk_test_tensor = tk_train_tensor.unsqueeze(1),tk_test_tensor.unsqueeze(1)
y_train_tensor,y_test_tensor = y_train_tensor.unsqueeze(1),y_test_tensor.unsqueeze(1)
#print(x_train_tensor.size(),tk_train_tensor.size(),y_train_tensor.size())


#输入的每个例子是101x1的行向量
class CNNNet(torch.nn.Module):
    def __init__(self,n_channels,n_output):
        super(CNNNet,self).__init__()
        self.hidden1=torch.nn.Conv1d(n_channels,3,kernel_size=5,stride=1)#输入为100，输出为93
        self.hidden2=torch.nn.Conv1d(3,5,kernel_size=3,stride=1)#输入为295，输出为147
        self.hidden3=torch.nn.Conv1d(5,10,kernel_size=3,stride=1)#输入为146，输出为144
        self.hidden4=torch.nn.Conv1d(10,20,kernel_size=3,stride=1)#输入为142，输出140
        self.hidden5=torch.nn.Conv1d(20,40,kernel_size=3,stride=1)#输入为138，输出68
        self.hidden6=torch.nn.Conv1d(40,80,kernel_size=3,stride=1)#输入为33,输出31
        self.hidden7=torch.nn.Conv1d(80,160,kernel_size=3,stride=2)#输入为29,输出14
        self.linear1=torch.nn.Sequential(torch.nn.Linear(1281,200), 
                                        torch.nn.BatchNorm1d(200), torch.nn.Dropout(0.3))
        self.linear2=torch.nn.Sequential(torch.nn.Linear(200,20), 
                                        torch.nn.BatchNorm1d(20), torch.nn.Dropout(0.2))
        self.predict=torch.nn.Linear(20,n_output)
        
    def forward(self,x,tk):
        x=torch.relu(self.hidden1(x))
        x=torch.max_pool1d(x,kernel_size=3,stride=1)#输入93，输出91
        x=torch.relu(self.hidden2(x))
        x=torch.max_pool1d(x,kernel_size=3,stride=1)#输入87，输出85
        x=torch.relu(self.hidden3(x))
        x=torch.max_pool1d(x,kernel_size=3,stride=1)#输入40，输出39
        x=torch.relu(self.hidden4(x))
        x=torch.max_pool1d(x,kernel_size=3,stride=1)#输入40，输出39
        x=torch.relu(self.hidden5(x))
        x=torch.avg_pool1d(x,kernel_size=3,stride=1)#输入68，输出33
        x=torch.relu(self.hidden6(x))
        x=torch.max_pool1d(x,kernel_size=3,stride=2)#输入31，输出29
        x=torch.relu(self.hidden7(x))
        x=torch.avg_pool1d(x,kernel_size=3,stride=2)#输入14，输出12
        #print(x.size())
        #x与tk合并
        x=x.view(-1,160*8)
        x=torch.cat((x,tk),1)
        
        x=torch.relu(self.linear1(x))
        x=torch.relu(self.linear2(x))
        x=torch.relu(self.predict(x))
        return x


net=CNNNet(1,1).cuda()

#optimizer=torch.optim.Adam(net.parameters(),lr=0.0001)#优化参数
optimizer=torch.optim.SGD(net.parameters(),lr=0.0001)#优化参数
loss_func=torch.nn.MSELoss()

for epoch in range(10000):
    prediction=net(x_train_tensor,tk_train_tensor)
    #print(prediction.size(),y_train_tensor.size())
    loss=loss_func(prediction, y_train_tensor)#带有神经网络所有信息
    optimizer.zero_grad()#归零
    loss.backward()#反向传递，optimizer有参数了
    optimizer.step()#优化
    if epoch%100==0:
        print('loss is {} when epoch is {}'.format(loss.data, epoch))

torch.save(net.state_dict(), 'CNN_parameters.pkl')

net.load_state_dict(torch.load('CNN_parameters.pkl'))

train_predict = []
train_error = []
train_prediction =net(x_train_tensor,tk_train_tensor)
for i in range(len(y_train_tensor)):
    train_predict.append((float(train_prediction[i][0]), float(y_train_tensor[i][0])))
    error = abs(float(train_prediction[i][0]) - float(y_train_tensor[i][0]))/float(y_train_tensor[i][0])
    train_error.append(error)
print('train error is {}'.format(np.mean(train_error)))                        

test_predict = []
test_error = []
test_prediction=net(x_test_tensor,tk_test_tensor)
for i in range(len(y_test_tensor)):
    test_predict.append((float(test_prediction[i][0]), float(y_test_tensor[i][0])))
    error = abs(float(test_prediction[i][0]) - float(y_test_tensor[i][0]))/float(y_test_tensor[i][0])
    test_error.append(error)
print('test error is {}'.format(np.mean(test_error)))

test_error = pd.DataFrame(test_error)
test_error.to_csv('test-error-CNN.csv')    




