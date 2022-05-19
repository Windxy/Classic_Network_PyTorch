import torch
import torch.nn as nn
import math

# 通道注意力机制
class ChannelAttention(nn.Module):
    # channel表示输入的通道数，ratio表示缩放的比例，用于第一次全连接
    def __init__(self,in_planes,ratio=8):
        # 初始化
        super(ChannelAttention,self).__init__()
        # 平均池化和最大池化操作，输出高和宽都为1,获得长度为特征层通道数的特征长条AvgPool和MaxPool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)

        # 对平均池化和最大池化的结果，利用共享的两个全连接层进行处理
        # 利用1*1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes,in_planes//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//ratio,in_planes,1,bias=False)

        # 相加后再取sigmoid，此时获得了输入特征层每个通道的权值（0-1之间）
        self.sigmoid = nn.Sigmoid()

        #前传部分，out * x放到最后
        def forward(self,x):
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            out = avg_out + max_out
            return self.sigmoid(out)



# 空间注意力机制
class SpatialAttention(nn.Module):
    # 空间注意力机制不需要传入通道数和ratio，但是有卷积核大小3或7
    def __init__(self,kernel_size=7):
        super(SpatialAttention,self).__init__()
        assert kernel_size in(3,7),'kernel size must be 3 or 7'

        # padding = 7 整除2 = 3
        padding = 3 if kernel_size == 7 else 1

        # 输入通道数为2，即一层最大池化，一层平均池化
        # 输出通道数为1；步长为1，即不需要压缩宽高
        self.conv1 = nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()

        # 前传部分，out * x放在最后
        def forward(self,x):
            # 在通道上进行最大池化和平均池化
            # 对于pytorch，其通道在第一维度，在batch_size之后，dim=1
            # 保留通道，所以keepdim=true
            avg_out = torch.mean(x,dim=1,keepdim=True) # keepdim：是否需要保持输出的维度与输入一样
            max_out,_=torch.max(x,dim=1,keepdim=True)

            # 将最大值和平均值进行堆叠
            x = torch.cat([avg_out,max_out],dim=1)

            # 取卷积
            x = self.conv1(x)
            return self.sigmoid(x)


# 结合空间和通道注意力机制
class cbam_block(nn.Module):
    def __init__(self,channel,ratio=8,kernel_size=7):
        super(cbam_block,self).__init__()
        self.channelattention = ChannelAttention(channel,ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
    def forward(self,x):
        x = x * self.channelattention(x) # 通道注意力机制获得输入特征层每一个通道的权值 * 原输入特征层
        x = x * self.spatialattention(x) # 空间注意力机制获得输入特征层每一个特征点的权值 * 原输入特征层
        return x

if __name__ == '__main__':
    x = torch.rand(50,512,7,7)
    model = cbam_block(512)
    output = model(x)
    print(output)
