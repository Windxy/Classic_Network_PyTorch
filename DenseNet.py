# paper link: https://arxiv.org/pdf/1608.06993.pdf
import torch
import torch.nn as nn
import torch.nn.functional as F

""""
DenseNet的密集连接方式需要特征图大小保持一致。所以DenseNet网络中使用DenseBlock和Transition的结构
"""

# DenseNet使用了ResNet改良版的“批量归一化、激活和卷积”结构，在Conv_block_dense函数里实现这个结构
def Conv_block_dense(inchannels,outchannels):   # DenseBlock块内组成
    return nn.Sequential(
        nn.BatchNorm2d(inchannels),
        nn.ReLU(),
        nn.Conv2d(inchannels,outchannels,3,padding=1)
    )

# 全局平均池化
class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
    def forward(self,x):
        x = F.avg_pool2d(x,x.size()[2:])
        return x

# 稠密块由多个 conv_block 组成，每块使用相同的输出通道数。但在前向计算时，我们将每块的输⼊和输出在通道维上连结。
class DenseBlock(nn.Module):   # 定义一个DenseBlock块
    def __init__(self,num_conv,in_channels,out_channels): # 通道数和模块内的卷积数目
        super(DenseBlock, self).__init__()
        net = []
        # 在DenseLayer中输出是相同的，但是输入的维度有来自前面的特征，所以每次输出的维度都是增长的，且增长的速率和输出的维度有关，称为 growth_rate
        for i in range(num_conv):
            in_c = in_channels + i * out_channels  # 每个层都会和前面所有层在channel维度连接起来
            net.append(Conv_block_dense(in_c,out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_conv * out_channels # 计算输出通道数

    def forward(self,X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X,Y),dim=1)  # 在通道维上将输入和输出连结
        return X

# 过渡层：用来控制通道数，使之不过大
def transition_block(in_channel,out_channel):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel,out_channel,1),  # 1×1卷积层来减小通道数
        nn.AvgPool2d(kernel_size=2,stride=2)  # 使用步幅为2的平均池化层减半高和宽，从而进一步降低模型复杂度
    )
    return blk

# DenseNet模型
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), # 7*7 conv,stride 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # 3*3max pool,stride 2

        num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
        num_convs_in_dense_blocks = [4, 4, 4, 4]  # DenseNet使用4个稠密块，设置每个稠密块使用4个卷积层

        #  利用enumerate可以同时迭代序列的索引和元素
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            #  根据更新的num_convs添加DenseBlock
            DB = DenseBlock(num_convs, num_channels, growth_rate)
            net.add_module("DenseBlosk_%d" % i, DB)
            # 上一个稠密块的输出通道数
            num_channels = DB.out_channels  # # 上一个稠密块的输出通道数

            # 在稠密块之间加入通道数减半的过渡层
            if i != len(num_convs_in_dense_blocks) - 1:
                net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        #  同ResNet一样，最后接上全局池化层和全连接层来输出
        net.add_module("BN", nn.BatchNorm2d(num_channels))
        net.add_module("relu", nn.ReLU())
        net.add_module("global_avg_pool", GlobalAvgPool())  # 利用全局平均池化层可以降低模型的参数数量来最小化过拟合效应
        net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(num_channels, 10)))
        self.net = net

    def forward(self,x):
        return self.net(x)

def DenseNet_T():
    net = DenseNet()
    X = torch.rand(1, 3, 224, 224)
    print(net(X))
DenseNet_T()
