# pytorch——模型的构建访问遍历存储
import torch.nn as nn
from collections import OrderedDict
import torch
#定义一个自己的网络MyNet
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3)
        self.conv2=nn.Conv2d(64,64,3)
        self.maxpool1=nn.MaxPool2d(2,2)

        self.features=nn.Sequential(OrderedDict([
            ('conv3',nn.Conv2d(64,128,3)),
            ('conv4',nn.Conv2d(128,128,3)),
            ('relu1',nn.ReLU())
        ]))
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.maxpool1(x)
        x=self.features(x)
        return x

net=MyNet()
print(net)
#用modules()初始化模型各个层的参数
for idx,m in enumerate(net.modules()):
    print(idx,"-",m)

for p in net.parameters():
    print(type(p.data),p.size())
# 输出的是四个卷积层的权重矩阵参数和偏置参数。值得一提的是，对网络进行训练时需要将parameters()作为优化器optimizer的参数。
'''
总之呢，这个parameters()是返回网络所有的参数，主要用在给optimizer优化器用的。
而要对网络的某一层的参数做处理的时候，一般还是使用named_parameters()方便一些。
'''
'''
#2.
训练
优化器==>更新权重和偏置
'''
'''
保存与载入
PyTorch使用torch.save和torch.load方法来保存和加载网络，而且网络结构和参数可以分开的保存和加载。
torch.save(model,'model.pth') # 保存
model = torch.load("model.pth") # 加载
pytorch中网络结构和模型参数是可以分开保存的。上面的方法是两者同时保存到了.pth文件中，
当然，你也可以仅仅保存网络的参数来减小存储文件的大小。
注意：如果你仅仅保存模型参数，那么在载入的时候，是需要通过运行代码来初始化模型的结构的。torch.save(model.state_dict(),"model.pth") # 保存参数

model = MyNet() # 代码中创建网络结构
params = torch.load("model.pth") # 加载参数
model.load_state_dict(params) # 应用到网络结构中
'''
'''
https://blog.csdn.net/weixin_37804469/article/details/108310660
'''
#先建立一个字典，保存三个参数：
optimizer = torch.optim.SGD(net.parameters(),
                            lr = 0.001,
                            momentum=0.9)
model=MyNet()
# state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict()}
# 调用torch.save():
# 其中dir表示保存文件的绝对路径+保存文件名，如’/home/qinying/Desktop/modelpara.pth’
torch.save(state, './model/MyNet.pth')

#当你想恢复某一阶段的训练（或者进行测试）时，那么就可以读取之前保存的网络模型参数等。
# checkpoint = torch.load(dir)
#
# model.load_state_dict(checkpoint['net'])
#
# optimizer.load_state_dict(checkpoint['optimizer'])

# start_epoch = checkpoint['epoch'] + 1



