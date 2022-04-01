import torch
import torch.nn.functional as F
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NiN块：由一个指定卷积核大小的卷积层和两个1×1 卷积层组成
def nin_block(in_channel,out_channel,k,s,p):
    return nn.Sequential(
        nn.Conv2d(in_channel,out_channel,k,s,p),
        nn.ReLU(),
        nn.Conv2d(out_channel,out_channel,1), # 1*1卷积，不改变通道数
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, 1), # 1*1卷积
        nn.ReLU()
    )

# 全局平均池化层代替全连接层
class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
    def forward(self,x):
        x = F.avg_pool2d(x,x.size()[2:])
        return x

# NiN模型: NiN使用窗口形状为11*11,5*5,和3*3的卷积层，输出通道数量与AlexNet中的相同。 每个NiN块后有一个最大汇聚层，汇聚窗口形状为，步幅为2。
class NiN(nn.Module):
    def __init__(self):
        super(NiN, self).__init__()
        self.model = nn.Sequential(
            nin_block(3,96,11,4,0), # RGB图，所以输入是3， 通道数96，kernel_size=11,strides=4,padding=0
            nn.MaxPool2d(3,2),
            nin_block(96, 256, 5, 1, 2), # 5*5 Conv(256),pad 2
            nn.MaxPool2d(3, 2),
            nin_block(256,384,3,1,1), # 3*3 Conv(384,pad 1)
            nn.MaxPool2d(3,2),
            nn.Dropout(0.5), # 标签类别数是10
            nin_block(384,10,3,1,1), # 3*3 Conv(10,pad 1)
            GlobalAvgPool(),
            nn.Flatten(),
        )

    def forward(self,img):
        return self.model(img)

# 创建一个数据样本来查看每个块的输出形状
if __name__ == '__main__':

    net = NiN()
    X = torch.rand(1, 3, 224, 224)
    print(net(X).shape)
