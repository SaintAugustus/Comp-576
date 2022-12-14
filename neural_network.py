# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
#from sklearn.multioutput import MultiOutputRegressor as multi
#from xgboost import XGBRegressor as XGBR
import os
import re
#from sklearn import preprocessing
import time
import torch
#import torch.nn.functional as F
#from matplotlib import pyplot as plt
torch.set_default_tensor_type('torch.FloatTensor')

resnet=False#表明是否用resnet
n_data=600#代表时间分段数，理论上段数越多越好
def get_data(n_data,set_name):
    os.chdir(set_name)
    train_files=os.listdir()
    all_data=[]
    target=[]#n行3列
    for i,file in enumerate(train_files):
        df=pd.read_csv(file,header=None,sep='\s',dtype=np.float32,engine='python').values
        if df[-1,0]!=6:
            continue#就不算这个了
        R=float(re.findall(r'R=([\d_]+)M',file)[0].replace('_','.'))
        M=float(re.findall(r'M=([\d_]+)B',file)[0].replace('_','.'))
        B=float(re.findall(r'B=([\d_]+)\.txt',file)[0].replace('_','.'))#这就是近似过以后的值，所以没问题
        #进行处理，变成一行
        if i%100==0:
            print(i)
        all_data.append(process(df,n_data))
        target.append(np.array((R,M,B)))
    os.chdir('..')
    return np.array(all_data),np.array(target)

def process(data,n_data):
    result=np.zeros((n_data))
    time_start,time_end=data[0,0],data[-1,0]
    times=np.linspace(time_start,time_end,n_data)
    for i in range(n_data):
        pos=find_position(times[i],data[:,0])
        grad=(data[pos+1,1]-data[pos,1])/(data[pos+1,0]-data[pos,0])
        result[i]=grad*(times[i]-data[pos,0])+data[pos,1]
    return result
def find_position(data,data_list):#返回前面的位置
    for i in range(len(data_list)-1):
        if data_list[i]<=data and data<=data_list[i+1]:
            return i

#1,8,4
try:
    X_train=np.load('X_train.npy')
    X_test=np.load('X_test.npy')
    x_test,y_test=X_test[:,:-3],X_test[:,-3:]
    x_train,y_train=X_train[:,:-3],X_train[:,-3:]
except:
    x_train,y_train=get_data(n_data,'train')
    X_train=np.hstack((x_train,y_train))
    #scaler = preprocessing.StandardScaler().fit(X_train)                              
    #scaler.transform(X_train)
    x_test,y_test=get_data(n_data,'test')
    X_test=np.hstack((x_test,y_test))
    np.save('X_train.npy',X_train)
    np.save('X_test.npy',X_test)#保存一下
#X_test=scaler.transform(X_test)




x_train,y_train=-torch.tensor(X_train[:,:-3]),torch.tensor(X_train[:,-3:])
x_train=x_train.unsqueeze(1)
x_test,y_test=-torch.tensor(X_test[:,:-3]),torch.tensor(X_test[:,-3:])#都取负号
x_test=x_test.unsqueeze(1)

x_train=x_train.float()
y_train=y_train.float()
x_test=x_test.float()
y_test=y_test.float()
'''
class Residual_Block(torch.nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Residual_Block,self).__init__()
        self.conv1=torch.nn.Conv1d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.pool1=torch.nn.AvgPool1d(kernel_size=2,stride=1,padding=1)
        self.bn1=torch.nn.BatchNorm1d(out_channel)
        self.relu=torch.nn.ReLU()
        self.conv2=torch.nn.Conv1d(in_channels=out_channel,out_channels=in_channel,kernel_size=3,stride=1,padding=1)
        self.pool2=torch.nn.MaxPool1d(kernel_size=2,stride=1,padding=1)
        self.bn2=torch.nn.BatchNorm1d(in_channel)
    def forward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.pool1(out)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.pool2(out)
        out=self.bn2(out)
        out=out+residual
        out=self.relu(out)
        return out

class ResNet(torch.nn.Module):
    def __init__(self,block,layers,num_classes):#layers 是一个列表，表明每层神经元个数,num_classes是输出个数
        super(ResNet,self).__init__()
        self.in_channels=1
        self.conv=torch.nn.Conv1d(in_channels=1,out_channels=8,kernel_size=3,stride=1)#输入2000，输出1998
        self.bn=torch.nn.BatchNorm1d(8)
        self.relu=torch.nn.ReLU(inplace=True)
        self.layer1=self.make_layer(block,8,16,layers[0])
        self.layer2=self.make_layer(block,8,32,layers[1])
        self.layer3=self.make_layer(block,8,64,layers[2])
        self.avg_pool=torch.nn.AvgPool1d(kernel_size=8,stride=2)#输出1998，输出996
        self.fc1=torch.nn.Linear(8*996,500)
        self.fc2=torch.nn.Linear(500,num_classes)
    def make_layer(self,block,in_channels,out_channels,blocks):#blocks=layers,the number of residual block
        layers=[]
        for i in range(1,blocks):#再添加多少个block
            layers.append(block(in_channels,out_channels))
        return torch.nn.Sequential(*layers)# add all of the residual block
    def forward(self,x):
        out = self.conv(x) 
        out = self.bn(out) 
        out = self.relu(out) 
        out = self.layer1(out) 
        out = self.layer2(out) 
        out = self.layer3(out)
        out = self.avg_pool(out) 
        out = out.view(out.size(0), -1) 
        out = self.fc1(out) 
        out = self.fc2(out) 
        return out 
'''

class Net(torch.nn.Module):
    def __init__(self,n_features,n_output):
        super(Net,self).__init__()
        self.hidden1=torch.nn.Conv1d(n_features,3,kernel_size=10,stride=2)#输入为600，输出为296
        self.hidden2=torch.nn.Conv1d(3,5,kernel_size=3,stride=2)#输入为295，输出为147
        self.hidden3=torch.nn.Conv1d(5,10,kernel_size=3,stride=1)#输入为146，输出为144
        self.hidden4=torch.nn.Conv1d(10,20,kernel_size=3,stride=1)#输入为142，输出140
        self.hidden5=torch.nn.Conv1d(20,40,kernel_size=3,stride=2)#输入为138，输出68
        self.hidden6=torch.nn.Conv1d(40,80,kernel_size=3,stride=1)#输入为33,输出31
        self.hidden7=torch.nn.Conv1d(80,160,kernel_size=3,stride=2)#输入为29,输出14
        self.linear1=torch.nn.Linear(160*12,50)#
        self.linear2=torch.nn.Linear(50,n_output)
    def forward(self,x):
        x=torch.relu(self.hidden1(x))
        x=torch.max_pool1d(x,kernel_size=2,stride=1)#输入296，输出295
        x=torch.relu(self.hidden2(x))
        x=torch.avg_pool1d(x,kernel_size=2,stride=1)#输入147，输出146
        x=torch.relu(self.hidden3(x))
        x=torch.max_pool1d(x,kernel_size=3,stride=1)#输入144，输出142
        x=torch.relu(self.hidden4(x))
        x=torch.avg_pool1d(x,kernel_size=3,stride=1)#输入140，输出138
        x=torch.relu(self.hidden5(x))
        x=torch.avg_pool1d(x,kernel_size=3,stride=2)#输入68，输出33
        x=torch.relu(self.hidden6(x))
        x=torch.max_pool1d(x,kernel_size=3,stride=1)#输入31，输出29
        x=torch.relu(self.hidden7(x))
        x=torch.avg_pool1d(x,kernel_size=3,stride=1)#输入14，输出12
        x=x.view(-1,160*12)
        x=torch.relu(self.linear1(x))
        x=torch.relu(self.linear2(x))
        return x

if resnet:
    net=ResNet(Residual_Block,[1,1,1],3).cuda()
else:
    net=Net(1,3).cuda()
    try:
        net.load_state_dict(torch.load('net_parameters.pkl'))#提取参数
    except:
        pass
#print('请暂停')
time.sleep(3)
#x_train,y_train,x_test,y_test=x_train.cpu(),y_train.cpu(),x_test.cpu(),y_test.cpu()
x_train,y_train,x_test,y_test=x_train.cuda(),y_train.cuda(),x_test.cuda(),y_test.cuda()
optimizer=torch.optim.Adam(net.parameters(),lr=0.000000001)#优化参数
#optimizer=torch.optim.SGD(net.parameters(),lr=0.00000005)#优化参数
loss_func=torch.nn.MSELoss()
'''
batch_size=10000#每批数量,总共50000
batch_num=x_train.shape[0]//batch_size
batch_remain=x_train.shape[0]%batch_size
batch_data=[]
for i in range(batch_num+1):
    batch_data.append(i*batch_size)
if batch_remain:
    batch_data.append(x_train.shape[0])
len_batch=len(batch_data)
'''
for t in range(2000):
    loss_epi=0
    y_predict=net(x_train)
    loss=loss_func(y_predict,y_train)#带有神经网络所有信息
    optimizer.zero_grad()#归零
    loss.backward()#反向传递，optimizer有参数了
    optimizer.step()#优化
    loss_epi=loss.data.item()
    if t%10==0:
        print('%d loss:'%t,loss_epi)
        with open('train.txt','a') as f:
            f.write('%.3f\n'%(loss_epi))
        y_predict=net(x_test)
        loss=loss_func(y_predict,y_test)
        print('%d loss:'%t,loss.data.item())
        with open('test.txt','a') as f:
            f.write('%.3f\n'%(loss.data.item()))

#for t in range(100):
#    y_predict=net(x_train)
#    loss=loss_func(y_predict,y_train)#带有神经网络所有信息
#    optimizer.zero_grad()#归零
#    loss.backward()#反向传递，optimizer有参数了
#    optimizer.step()#优化
#    if t%10==0:
#        print('epoch%d:loss:'%t,loss.data.item())
torch.save(net.state_dict(), 'net_parameters.pkl')  # 保存神经网络的参数
y_predict=net(x_test)
loss=loss_func(y_predict,y_test)
print('测试集loss:',loss.data.item())

data_num=len(y_test)
y_test=y_test.cpu().numpy()
y_predict=y_predict.detach().cpu().numpy()
with open('RMB.txt','w') as f:
    for i in range(data_num):
        f.write('R   %.3f   %.3f   M   %.3f   %.3f   B   %.3f   %.3f\n'
                %(y_predict[i,0],y_test[i,0],y_predict[i,1],
                  y_test[i,1],y_predict[i,2],y_test[i,2]))
        #结果写成RMB.txt文件，数据顺序为预测值，再真实值
#输出各自的百分比误差，和标准差

r=0
m=0
b=0
r_var=0
m_var=0
b_var=0
for i in range(data_num):
    r=r+abs(y_predict[i,0]-y_test[i,0])/y_predict[i,0]
    m=m+abs(y_predict[i,1]-y_test[i,1])/y_predict[i,1]
    b=b+abs(y_predict[i,2]-y_test[i,2])/y_predict[i,2]
    r_var=r_var+abs(y_predict[i,0]-y_test[i,0])**2
    m_var=m_var+abs(y_predict[i,1]-y_test[i,1])**2
    b_var=b_var+abs(y_predict[i,2]-y_test[i,2])**2
print('百分比误差分别为%f , %f, %f'%(r/data_num,m/data_num,b/data_num))
print('标准差分别为%f , %f, %f'%(np.sqrt(r_var/data_num),
                           np.sqrt(m_var/data_num),np.sqrt(b_var/data_num)))




    