# Build-a-simple-network
basic learning

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
