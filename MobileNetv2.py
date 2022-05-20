import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self,inchannel,outchannel,k=3,s=1,group=1):
        super(ConvBlock, self).__init__()
        self.Conv = nn.Conv2d(in_channels=inchannel,
                              out_channels=outchannel,
                              kernel_size = k,
                              stride = s,
                              padding = (k-1) // 2, # 保证输出的尺寸与原尺寸一致
                              groups = group,     # 设置为1，则为标准卷积，设置与输入通道数相同，则为depthwise卷积
                            )
        self.BN = nn.BatchNorm2d(outchannel)
        self.act = nn.ReLU6()

    def forward(self,x):
        y = self.Conv(x)
        y = self.BN(y)
        y = self.act(y)
        return y

class InvertedResidualBlock(nn.Module):
    def __init__(self,t,in_channel,out_channel,stride):
        '''
        :param t: expansion factor t 膨大系数（通道扩大倍数）
        :param in_channel:输入通道数
        :param out_channel:输出通道数
        :param times:重复times次
        :param stride:步长
        '''
        super(InvertedResidualBlock, self).__init__()
        # 输入in_channel*h*w
        hid_channel = in_channel*t
        self.conv1 = ConvBlock(in_channel,hid_channel,k=1,s=1)      # pw
        self.conv2 = ConvBlock(hid_channel,hid_channel,k=3,s=stride)# dw
        self.conv3 = nn.Conv2d(hid_channel,out_channel,kernel_size=1,stride=1,bias = False) # pw-linear
        self.BN = nn.BatchNorm2d(out_channel)
        # 输出out_channel*h//s*w//s
    def forward(self,x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.BN(y)
        return y

# MobileNetv2架构
class MobileNetV2(nn.Module):
    def __init__(self,num_classes=10):
        super(MobileNetV2, self).__init__()
        feature = [ConvBlock(3,32,k=3,s=2)]

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        in_channel = 32
        last_channel = 1280
        for t,c,n,s in inverted_residual_setting:
            out_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                feature.append(InvertedResidualBlock(t,in_channel,out_channel,stride))
                in_channel = out_channel
        # 最后一层
        feature.append(ConvBlock(in_channel,last_channel,1,1))

        self.feature = nn.Sequential(*feature)
        # 其他两种方式
        # self.GlobalAvgPooling = nn.AdaptiveAvgPool2d((1,1))
        # self.GlobalAvgPooling2 = nn.AvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel,num_classes)
        )

    # def _forward_impl(self,x):
    #     '''官网解释：TorchScript 不支持继承，所以超类方法需要有一个新的name，而不是“forward”，可以在子类中访问'''
    #     return y

    def forward(self,x):
        # 输入x.shape = batch,3,224,224
        y = self.feature(x)
        y = F.adaptive_avg_pool2d(y,1)  # y = self.GlobalAvgPooling2(y)
        y = y.reshape(x.shape[0],-1)
        y = self.classifier(y)
        return y

if __name__ == '__main__':
    x = torch.rand(1,3,224,224)
    model = MobileNetV2()
    y = model(x)
    print(y.shape)