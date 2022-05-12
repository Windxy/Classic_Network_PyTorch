import torch
import torch.nn as nn
import torch.nn.init as init

# 自定义卷积模块Fire Module
# Fire模块由squeeze部分和expand部分组成
# squeeze部分由一组连续的1×1卷积组成，expand部分由一组连续的1×1卷积和一组连续的3×3卷积cancatnate组成，因此3×3卷积需要使用same卷积
class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels,expand1x1_channels, expand3x3_channels):
        # 参数说明：
        # in_channels：输入通道数
        # squeeze_channels：squeeze层输出通道数
        # expand1x1_channels：expand层1X1卷积模块输出通道数
        # expand3x3_channels：expand层3X3卷积模块输出通道数
        # 经过一次Fire Module,图片尺寸不变，通道数变为(expand1x1_channels+expand3x3_channels)
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)#定义压缩squeeze层，1X1卷积
        self.squeeze_activation = nn.ReLU(inplace=True)#每次卷积后接relu
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels,kernel_size=1)#定义扩展expand层，1X1卷积
        self.expand1x1_activation = nn.ReLU(inplace=True)#每次卷积后接relu
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels,kernel_size=3, padding=1)#定义扩展expand层，3X3卷积
        self.expand3x3_activation = nn.ReLU(inplace=True)#每次卷积后接relu

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

# SqueezeNet网络结构
class SqueezeNet(nn.Module):
    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        # SqueezeNet V1.0版本，原始论文中提出的版本
        if version == 1.0:
            self.features = nn.Sequential(
                # 输入224×224×3
                # 卷积计算公式(224-7+2*0)/2+1=109，向下取整
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                # 109×109×96
                # （与原论文有出入，原论文输出尺寸为111×111×96，但最终输出都为13×13×512）
                nn.ReLU(inplace=True),
                # 池化计算公式(109-3)/2+1=54
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),# ceil_mode=True表示池化操作得到的数值为小数时向上取整
                # 54×54×96
                # 经过一次Fire Module,图片尺寸不变，通道数变为(expand1x1_channels+expand3x3_channels)
                Fire(96, 16, 64, 64),
                # 54×54×128
                Fire(128, 16, 64, 64),
                # 54×54×128
                Fire(128, 32, 128, 128),
                # 54×54×256
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),#(54-3)/2+1=26.5向上取整27
                # 27×27×256
                Fire(256, 32, 128, 128),
                # 27×27×256
                Fire(256, 48, 192, 192),
                # 27×27×256384
                Fire(384, 48, 192, 192),
                # 27×27×384
                Fire(384, 64, 256, 256),
                # 27×27×512
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                # 13×13×512
                Fire(512, 64, 256, 256),
                # 13×13×512
            )
        # SqueezeNet V1.1版本，在V1.0基础上进行了微调
        # SqueezeNet V1.0版本第一层卷积采用96 filters of resolution 7x7，池化操作发生在conv1，fire4和fire8之后。
        # SqueezeNet V1.1版本第一层卷积采用64 filters of resolution 3x3，通过降低卷积核的大小进一步调低网络参数。
        # 其次，前移池化操作，将池化操作移至conv1，fire3和fire5之后。
        # 在精度没有损失的情况下，sqeezenet v1.1在计算量上比v1.0少了2.4倍以上。
        else:
            self.features = nn.Sequential(
                # 输入224×224×3
                # 卷积计算公式(224-3+2*0)/2+1=111，向下取整
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                # 111×111×64
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                # 55×55×64
                Fire(64, 16, 64, 64),
                # 55×55×128
                Fire(128, 16, 64, 64),
                # 55×55×128
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                # 27×27×128
                Fire(128, 32, 128, 128),
                # 27×27×256
                Fire(256, 32, 128, 128),
                # 27×27×256
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                # 13×13×256
                Fire(256, 48, 192, 192),
                # 13×13×384
                Fire(384, 48, 192, 192),
                # 13×13×384
                Fire(384, 64, 256, 256),
                # 13×13×512
                Fire(512, 64, 256, 256),
                # 13×13×512
            )
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,#最后卷积操作，尺寸不变，输出通道数变为self.num_classes=1000
            # 13×13×1000
            nn.ReLU(inplace=True),#relu
            nn.AvgPool2d(13, stride=1)
            # 1×1×1000
        )

        # 权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

if __name__ == '__main__':
    # 一个tensor，3通道，图像尺寸224×224
    X = torch.randn(1, 3, 224, 224)
    net = SqueezeNet()
    out = net(X)
    print(out.shape)#输出torch.Size([1, 1000])
