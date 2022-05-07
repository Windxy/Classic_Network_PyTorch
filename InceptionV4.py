import torch
import torch.nn as nn
import torch.nn.functional as F

# BN+ReLu的基本卷积结构
# stride默认为1，padding默认为0
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

# InceptionA模块
class InceptionA(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(InceptionA,self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1, stride=1),
            BasicConv2d(in_channels,96,kernel_size=1)
        )
        self.branch2 = BasicConv2d(in_channels,96,kernel_size=1)
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels,64,kernel_size=1),
            BasicConv2d(64,96,kernel_size=3,padding=1)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channels,64,kernel_size=1),
            BasicConv2d(64,96,kernel_size=3,padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1),
        )

    def forward(self,x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

#InceptionB模块
class InceptionB(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(InceptionB, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3,padding=1,stride=1),
            BasicConv2d(in_channels,128,kernel_size=1)
        )
        self.branch2 = BasicConv2d(in_channels,384,kernel_size=1)
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels,192,kernel_size=1),
            BasicConv2d(192,224,kernel_size=(1,7),padding=(0,3)),
            BasicConv2d(224, 256, kernel_size=(1, 7), padding=(0, 3))
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192, 224, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(224, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self,x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

# InceptionC模块
class InceptionC(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(InceptionC, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3,padding=1,stride=1),
            BasicConv2d(in_channels,256,kernel_size=1)
        )
        self.branch2 = BasicConv2d(in_channels,256,kernel_size=1)
        self.branch3_1 = BasicConv2d(in_channels,384,kernel_size=1)
        self.branch3_2_1 = BasicConv2d(384, 256, kernel_size=(1,3),padding=(0,1))
        self.branch3_2_2 = BasicConv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))

        self.branch4_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch4_2 = BasicConv2d(384, 448, kernel_size=(1, 3), padding=(0, 1))
        self.branch4_3 = BasicConv2d(448, 512, kernel_size=(3, 1), padding=(1, 0))
        self.branch4_4_1 = BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch4_4_2 = BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))

    def forward(self,x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3_1 = self.branch3_2_1(self.branch3_1(x))
        out3_2 = self.branch3_2_2(self.branch3_1(x))
        out4_1 = self.branch4_4_1(self.branch4_3(self.branch4_2(self.branch4_1(x))))
        out4_2 = self.branch4_4_2(self.branch4_3(self.branch4_2(self.branch4_1(x))))
        out = torch.cat([out1, out2, out3_1,out3_2, out4_1,out4_2], dim=1)
        return out

# ReductionA模块
class ReductionA(nn.Module):
    def __init__(self,in_channels,out_channels,k,l,m,n):
        super(ReductionA, self).__init__()
        self.branch1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.branch2 = BasicConv2d(in_channels,n,kernel_size=3,stride=2)
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels,k,kernel_size=1),
            BasicConv2d(k,l,kernel_size=3,padding=1),
            BasicConv2d(l,m,kernel_size=3,stride=2)
        )

    def forward(self,x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out

# ReductionB模块
class ReductionB(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ReductionB, self).__init__()
        self.branch1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels,192,kernel_size=1),
            BasicConv2d(192,192,kernel_size=3,stride=2)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels,256,kernel_size=1),
            BasicConv2d(256,256,kernel_size=(1,7),padding=(0,3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(320,320,kernel_size=3,stride=2)
        )

    def forward(self,x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out

# Stem模块
class Stem(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Stem, self).__init__()
        self.conv1 = BasicConv2d(in_channels,32,kernel_size=3,stride=2)
        self.conv2 = BasicConv2d(32,32,kernel_size=3)
        self.conv3 = BasicConv2d(32,64,kernel_size=3,padding=1)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv4 = BasicConv2d(64,96,kernel_size=3,stride=2)

        self.conv5_1_1 = BasicConv2d(160, 64, kernel_size=1)
        self.conv5_1_2 = BasicConv2d(64, 96, kernel_size=3)
        self.conv5_2_1 = BasicConv2d(160, 64, kernel_size=1)
        self.conv5_2_2 = BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0))
        self.conv5_2_3 = BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3))
        self.conv5_2_4 = BasicConv2d(64, 96, kernel_size=3)

        self.conv6 = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool6 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out1_1 = self.maxpool4(self.conv3(self.conv2(self.conv1(x))))
        out1_2 = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        out1 = torch.cat([out1_1, out1_2], 1)

        out2_1 = self.conv5_1_2(self.conv5_1_1(out1))
        out2_2 = self.conv5_2_4(self.conv5_2_3(self.conv5_2_2(self.conv5_2_1(out1))))
        out2 = torch.cat([out2_1, out2_2], 1)

        out3_1 = self.conv6(out2)
        out3_2 = self.maxpool6(out2)
        out3 = torch.cat([out3_1, out3_2], 1)

        return out3

class InceptionV4(nn.Module):
    def __init__(self):
        super(InceptionV4, self).__init__()
        self.stem = Stem(3,384)
        self.icpA = InceptionA(384, 384)
        self.redA = ReductionA(384, 1024, 192, 224, 256, 384)
        self.icpB = InceptionB(1024, 1024)
        self.redB = ReductionB(1024, 1536)
        self.icpC = InceptionC(1536, 1536)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.dropout = nn.Dropout(p=0.8)
        self.linear = nn.Linear(1536, 1000)

    def forward(self, x):
        # 输入299 x 299 x 3
        # Stem Module
        out = self.stem(x)
        # 输出35 x 35 x 384
        # InceptionA Module * 4
        out = self.icpA(self.icpA(self.icpA(self.icpA(out))))
        # 输出35 x 35 x 384
        # ReductionA Module
        out = self.redA(out)
        # 输出17 x 17 x 1024
        # InceptionB Module * 7
        out = self.icpB(self.icpB(self.icpB(self.icpB(self.icpB(self.icpB(self.icpB(out)))))))
        # 输出17 x 17 x 1024
        # ReductionB Module
        out = self.redB(out)
        # 输出8 x 8 x 1536
        # InceptionC Module * 3
        out = self.icpC(self.icpC(self.icpC(out)))
        # 输出8 x 8 x 1536
        # Average Pooling
        out = self.avgpool(out)
        # 1 x 1 x 1536
        out = out.view(out.size(0), -1)
        # 1536
        # Dropout
        out = self.dropout(out)
        # Linear(Softmax)
        out = self.linear(out)

        return out

if __name__ == '__main__':
    # 一个tensor，3通道，图像尺寸299×299
    X = torch.randn(1, 3, 299, 299)
    net = InceptionV4()
    out = net(X)
    print(out.shape)#输出torch.Size([1, 1000])

