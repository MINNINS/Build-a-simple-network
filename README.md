# 构建一个神经网络+对MNIST数据集进行可视化分析

## 1.定义一个神经网络并保存
```python

import torch.nn as nn
from collections import OrderedDict
import torch
##################################################################################
#define a network
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3)
        self.conv2=nn.Conv2d(64,64,3)

        self.maxpool1=nn.MaxPool2d(2,2)
        self.features=nn.Sequential(OrderedDict(
            [
                ('conv3',nn.Conv2d(64,128,3)),
                ('conv4',nn.Conv2d(128,128,3)),
                ('relu1',nn.ReLU())
            ]
        ))
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.maxpool1(x)
        x=self.features(x)
        return x
##################################################################################
# SGD optimizer
net=MyNet()
# define optimizer---for train
optimizer=torch.optim.SGD(net.parameters(),  # net.parameters() is important
                          lr=0.001,  #Learning rate
                          momentum=0.9
                          )
#构建一个词典
state={'net':net.state_dict(),'optimizer':optimizer.state_dict()}
torch.save(state,'./model/MyNet.pth')                  # save the model

# Test
print(net)

for idx,m in enumerate(net.modules()): #data-size
    print(idx,"-",m)
print('##################################################')
for p in net.parameters():
    print(type(p.data),p.size())


```
## 2.MNIST数据集可视化
### 2.1 数据概况
```python
# EDA:一般在进行模型训练之前，都要做一个数据集分析的任务
import pandas as pd
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
train_df = pd.read_csv('./dataset/mnist_train.csv')
n_train = len(train_df)
n_pixels = len(train_df.columns) - 1 #有多少列，就是多少个像素（28*28=784）
label = train_df.iloc[0:60000,0]
n_class = len(set(label))#set删除重复项后，就是类别数
print('Number of training samples: {0}'.format(n_train))
print('Number of training pixels: {0}'.format(n_pixels))
print('Number of classes: {0}'.format(n_class))
print('##################################')
# 读取测试集
test_df = pd.read_csv('./dataset/mnist_test.csv')
n_test = len(test_df)
n_pixels = len(test_df.columns)
print('Number of test samples: {0}'.format(n_test))
print('Number of test pixels: {0}'.format(n_pixels))
```
<img src="./image/result01.png">

----
