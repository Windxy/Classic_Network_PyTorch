import torch
import torch.nn as nn
import torch.nn.functional as F

# BN+ReLu的基本卷积结构
# stride默认为1，padding默认为0
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

#构建inception的一种分解方法：InceptionA
#得到输入大小不变，通道数为224+pool_features的特征图
class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        # 以第一个InceptionA为例，每一个branch的输入都是35 x 35 x 192
        # 卷积计算公式(192-1+2*0)/1+1=192
        self.branch1 = BasicConv2d(in_channels, 64, kernel_size=1)# 35 x 35 x 64
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 48, kernel_size=1),# 35 x 35 x 48
            BasicConv2d(48, 64, kernel_size=5, padding=2)# 35 x 35 x 64
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),# 35 x 35 x 64
            BasicConv2d(64, 96, kernel_size=3, padding=1),# 35 x 35 x 96
            BasicConv2d(96, 96, kernel_size=3, padding=1)# 35 x 35 x 96
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),# 35 x 35 x 192
            BasicConv2d(in_channels, pool_features, kernel_size=1)# 35 x 35 x pool_features
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        # 最后得到(35,35,64+64+96+pool_features)的特征图
        return out

#构建inception的一种分解方法：InceptionB
#得到输入大小减半，通道数+480的特征图
class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        # 每一个branch的输入都是35 x 35 x 288
        # 卷积计算公式(35-3+2*0)/2+1=17
        self.branch1 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)# 17 x 17 x 384
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),# 35 x 35 x 64
            BasicConv2d(64, 96, kernel_size=3, padding=1),# 35 x 35 x 96
            BasicConv2d(96, 96, kernel_size=3, stride=2),# 17 x 17 x 96
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)# 17 x 17 x 288

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        # 最后得到(17,17,384+96+288)的特征图
        return out

#构建inception的一种分解方法：InceptionC
#得到输入大小不变，通道数为768的特征图
class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        c7 = channels_7x7
        # 每一个branch的输入都是17 x 17 x 768
        # 卷积计算公式(17-1+2*0)/1+1=17
        self.branch1 = BasicConv2d(in_channels, 192, kernel_size=1)# 17 x 17 x 192

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, c7, kernel_size=1),# 17 x 17 x c7
            BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),# 17 x 17 x c7
            BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))# 17 x 17 x 192
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, c7, kernel_size=1),# 17 x 17 x c7
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),# 17 x 17 x c7
            BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),# 17 x 17 x c7
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),# 17 x 17 x c7
            BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))# 17 x 17 x 192
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),# 17 x 17 x 768
            BasicConv2d(in_channels, 192, kernel_size=1)# 17 x 17 x 192
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        # 最终得到(17, 17, 192+192+192+192=768)的特征图
        return out

#构建inception的一种分解方法：InceptionD
#得到输入大小减半，通道数+512的特征图
class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        # 每一个branch的输入都是17 x 17 x 768
        # 卷积计算公式(17-1+2*0)/1+1=17
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),# 17 x 17 x 192
            BasicConv2d(192, 320, kernel_size=3, stride=2)# 8 x 8 x 320
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),# 17 x 17 x 192
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),# 17 x 17 x 192
            BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),# 17 x 17 x 192
            BasicConv2d(192, 192, kernel_size=3, stride=2)# 8 x 8 x 192
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)# 8 x 8 x 768

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        # 最终得到(8,8,320+192+768=1280)的特征图
        return out

#构建inception的一种分解方法：InceptionE
#得到输入大小不变，通道数为2048的特征图
class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        # 以第一个InceptionE为例，每一个branch的输入都是8 x 8 x 1280
        # 卷积计算公式(8-1+2*0)/1+1=8
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)# 8 x 8 x 320
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)# 8 x 8 x 384
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))# 8 x 8 x 384
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))# 8 x 8 x 384
        # (8,8,384+384=768)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)# 8 x 8 x 448
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)# 8 x 8 x 384
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))# 8 x 8 x 384
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))# 8 x 8 x 384
        # (8,8,384+384=768)
        # 平均池化 8 x 8 x 1280
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)# 8 x 8 x 192
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        # 最终得到(8, 8, 320+768+768+192=2048)的特征图
        return torch.cat(outputs, 1)

#构建辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001
    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x

# 最终的InceptionV3网络结构
# 输入(229,229,3)的数据，首先归一化输入，经过5个卷积，2个最大池化层
# 然后经过3个InceptionA结构，1个InceptionB，3个InceptionC，1个InceptionD，2个InceptionE，其中辅助分类器AuxLogits以经过最后一个InceptionC的输出为输入。
# InceptionA：得到输入大小不变，通道数为224+pool_features的特征图。
# InceptionB：得到输入大小减半，通道数+480的特征图。
# InceptionC：得到输入大小不变，通道数为768的特征图。
# InceptionD：得到输入大小减半，通道数+512的特征图。
# InceptionE：得到输入大小不变，通道数为2048的特征图。
class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=False):
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        # 输入299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        # 经过3次InceptionA结构
        x = self.Mixed_5b(x)
        # 经过一个InceptionA，得到输入大小不变，通道数为224+pool_features=32的特征图
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 经过一个InceptionA，得到输入大小不变，通道数为224+pool_features=64的特征图
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 经过一个InceptionA，得到输入大小不变，通道数为224+pool_features=64的特征图
        # 35 x 35 x 288
        # 经过1次InceptionB结构
        x = self.Mixed_6a(x)
        # 经过一个InceptionB，得到输入大小减半，通道数+480的特征图
        # 17 x 17 x 768
        # 经过4次InceptionC结构
        x = self.Mixed_6b(x)
        # 经过一个InceptionC，得到输入大小不变，通道数为768的特征图。
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        # 辅助分类器AuxLogits以经过最后一个InceptionC的输出为输入
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        # 经过1次InceptionD
        x = self.Mixed_7a(x)
        # 经过一个InceptionD，得到输入大小减半，通道数+512的特征图
        # 8 x 8 x 1280
        # 经过2次InceptionE
        x = self.Mixed_7b(x)
        # 经过一个InceptionE，得到输入大小不变，通道数为2048的特征图
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 经过一个InceptionE，得到输入大小不变，通道数为2048的特征图
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:#只在训练过程进行辅助分支并输出
            return x, aux
        return x

if __name__ == '__main__':
    # 一个tensor，3通道，图像尺寸299×299
    X = torch.randn(1, 3, 299, 299)
    net = InceptionV3()
    out = net(X)
    print(out.shape)#输出torch.Size([1, 1000])


