import torch
import torch.nn as nn

# 论文https://arxiv.org/abs/1704.04861

# 标准卷积块，Conv + BN + ReLU6
class ConvBlock(nn.Module):
    def __init__(self,inchannel,outchannel,k=3,s=1,group=1):
        super(ConvBlock, self).__init__()
        self.Conv = nn.Conv2d(in_channels=inchannel,
                              out_channels=outchannel,
                              kernel_size = k,
                              stride = s,
                              padding=(k-1)//2, # 保证输出的尺寸与原尺寸一致
                              groups=group,     # 设置为1，则为标准卷积，设置与输入通道数相同，则为depthwise卷积
                            )
        self.BN = nn.BatchNorm2d(outchannel)
        self.act = nn.ReLU6()

    def forward(self,x):
        y = self.Conv(x)
        y = self.BN(y)
        y = self.act(y)
        return y

# 深度可分离卷积块 depthwise + pointwise
class dwpwBlock(nn.Module):
    def __init__(self,inchannel,hidchannel,outchannel,k,s):
        super(dwpwBlock, self).__init__()
        self.depthwise = ConvBlock(inchannel,hidchannel,k,s,group=inchannel)
        self.pointwise = ConvBlock(hidchannel,outchannel,k=1,s=1)

    def forward(self,x):
        y = self.depthwise(x)
        y = self.pointwise(y)
        return y

class MobileNetv1(nn.Module):
    def __init__(self):
        super(MobileNetv1, self).__init__()

        feature = []
        # 1.标准卷积，输入3x224x224，输出32x112x112
        feature.append(ConvBlock(3,32,k=3,s=2))
        # 2.深度可分离卷积，输入32x112x112，输出32x112x112
        feature.append(dwpwBlock(32,32,32,k=3,s=1))
        # 3.标准卷积，输入32x112x112，输出64x112x112
        feature.append(ConvBlock(32,64,k=1,s=1))
        # 4.深度可分离卷积，输入64x112x112，输出64x56x56
        feature.append(dwpwBlock(64,64,64,k=3,s=2))
        # 5.标准卷积，输入64x56x56，输出128x56x56
        feature.append(ConvBlock(64,128,k=1,s=1))
        # 6.深度可分离卷积，输入128x56x56，输出128x56x56
        feature.append(dwpwBlock(128,128,128,k=3,s=1))
        # 7.标准卷积，输入128x56x56，输出128x56x56
        feature.append(ConvBlock(128,128,k=1,s=1))
        # 8.深度可分离卷积，输入128x56x56，输出128x28x28
        feature.append(dwpwBlock(128,128,128,k=3,s=2))
        # 9.标准卷积，输入128x28x28，输出256x28x28
        feature.append(ConvBlock(128,256,k=1,s=1))
        # 10.深度可分离卷积，输入256x28x28，输出256x28x28
        feature.append(dwpwBlock(256,256,256,k=3,s=1))
        # 11.标准卷积，输入256x28x28，输出256x28x28
        feature.append(ConvBlock(256,256,k=1,s=1))
        # 12.深度可分离卷积，输入256x28x28，输出256x14x14
        feature.append(dwpwBlock(256,256,256,k=3,s=2))
        # 13.标准卷积，输入256x14x14，输出512x14x14
        feature.append(ConvBlock(256,512,k=1,s=1))
        # 14.深度可分离卷积+标准卷积 x 5,输入512x14x14，输出512x14x14
        for i in range(5):
            feature += [dwpwBlock(512,512,512,k=3,s=1)]
            feature += [ConvBlock(512,512,k=1,s=1)]
        # 15.深度可分离卷积，输入512x14x14，输出512x7x7
        feature.append(dwpwBlock(512,512,512,k=3,s=2))
        # 16.标准卷积，输入512x7x7，输出1024x7x7
        feature.append(ConvBlock(512,1024,k=1,s=1))
        # 17.深度可分离卷积，输入1024x7x7，输出1024x7x7,(NOTE:论文中写的是s=2，应该是写错了)
        feature.append(dwpwBlock(1024,1024,1024,k=3,s=1))
        # 18.标准卷积，输入1024x7x7，输出1024x7x7
        feature.append(ConvBlock(1024,1024,k=1,s=1))
        feature.append(nn.AvgPool2d(7,7))

        self.feature = nn.Sequential(*feature)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024,1000),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        y = self.feature(x)
        y = self.classifier(y)
        return y

if __name__ == '__main__':
    x = torch.rand(1,3,224,224)
    model = MobileNetv1()
    y = model(x)
    print(y.shape)