import torch.nn as nn
import torch
from torch.hub import load_state_dict_from_url  # 直接引入pytorch中已经训练好的预训练权重值
import torchvision

# 预训练模型路径
model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


# resnet34对应的残差块结构
class BasicBlock(nn.Module):
    # BasicBlock类有一个类属性，BasicBlock.expansion这个类属性的值为1，另外在 Bottleneck类中也有这个类属性，值为4
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        """定义BasicBlock残差块类
            参数：
                in_channel (int): 输入的Feature Map的通道数
                out_channel (int): 输入的Feature Map的通道数
                stride (int, optional): 第一个卷积层的步长
                downsample (nn.Sequential, optional): 旁路下采样的操作
                BatchNormalization对网络有很大的提升
            注意：
                残差块输出的Feature Map的通道数是 每一残差块输入通道数*expansion
        """
        super(BasicBlock, self).__init__()  # 调用父类的初始化函数
        # 残差块第一层的卷积，3*3*64
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        # 残差块第二层的卷积，3*3*64
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 把输入x保存到identity
        if self.downsample is not None:  # 如果旁路是虚线结构，则需要进行下采样操作，保证尺寸统一，此时downsample不为None
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 将残差和原始的输入相加
        out = self.relu(out)  # 在融合之后才调用激活函数

        return out


# 定义ResNet网络的结构
class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True):
        """
            block (BasicBlock / Bottleneck): 残差块类型
            blocks_num (list): 每一个stage的残差块的数目，长度为4
            num_classes (int): 类别数目
            include_top (boolean): 是否基于ResNet进行扩展，暂未用到
        """
        super(ResNet, self).__init__()
        self.include_top = include_top
        # 第一个残差块的输入通道数
        self.in_channel = 64
        # ResNet34网络 第一层卷积 [7,7,64]
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化层 [3,3,64]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 各个stage的残差块，layer对应图中的stage，也即Stage1 ~ Stage4
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        # ResNet34网络结构最后的 平均池化层 全连接层（Softmax内置）
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # 生成各个Stage的结构
    def _make_layer(self, block, channel, block_num, stride=1):
        """
            block (BasicBlock / Bottleneck): 残差块结构
            channel (int): 残差块中第一个卷积层的输出通道数
            block_num (int): 当前Stage中的残差块的数目
            stride (int): 残差块中第一个卷积层的步长
        """
        downsample = None
        # 判断旁路分支是否是虚线结构，即是否需要下采样
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 通过 1*1 卷积 调整维度
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        # 第一个block
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride))
        # 根据block的不同，对输入通道大小进行初始化
        self.in_channel = channel * block.expansion

        # 之后的block都是实线结构，不需要下采样，固循环生成即可
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel))

        # 使用Sequential层组合blocks,形成stage
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


# 封装，加载预训练参数
def _resnet(arch, pretrained, progress, num_classes=1000, include_top=True):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


# 构建一个ResNet-34模型
def resnet34(pretrained=False, progress=True, **kwargs):
    # pretrained (bool): 若为True则返回在ImageNet上预训练的模型
    return _resnet('resnet34', pretrained, progress)


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    model = resnet34()
    y = model(x)
    print(y.shape)
