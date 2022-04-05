import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url # 直接引入pytorch中已经训练好的预训练权重值
import torchvision

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

# 封装以下卷积操作，待后面使用
def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=padding, bias=False)
# bias置成false，如果卷积层之后是BatchNormalization层，则可以不用偏执参数，以节省内存

def conv1x1(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# resnet18对应的残差块结构
class BasicBlock(nn.Module): #输出channel和输入chnnel都是64
    # BasicBlock类有一个类属性，BasicBlock.expansion这个类属性的值为1，另外在 Bottleneck类中也有这个类属性，值为4
        expansion = 1
        def __init__(self, inplanes, planes, stride=1, downsample=None,norm_layer=None):
            """定义BasicBlock残差块类

                 参数：
                     inplanes (int): 输入的Feature Map的通道数
                     planes (int): 第一个卷积层输出的Feature Map的通道数
                     stride (int, optional): 第一个卷积层的步长
                     downsample (nn.Sequential, optional): 旁路下采样的操作
                     BatchNormalization对网络有很大的提升
                 注意：
                     残差块输出的Feature Map的通道数是planes*expansion
             """
            super(BasicBlock, self).__init__() # 调用父类的初始化函数
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d # 如果没有指定BatchNormalization，就用pytoch中标准的即可

            self.conv1 = conv3x3(inplanes, planes, stride) # 第一层是3 * 3的卷积，3 * 3 * 64
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes) # 第二层是3 * 3的卷积,第二个卷积不变还是3 * 3 * 64，故都用planes
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

        # forward()函数中才真正的调用层
        def forward(self, x):
            identity = x # 把输入x保存到identity

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.ReLU1(x)

            x = self.conv2(x)
            x = self.bn2(x)

            # 在网络的有的地方的尺寸已经发生了变化，故需要下采样，保证同步
            if self.downsample is not None:
                identity = self.downsample(identity)

            out = x + identity # 将残差和原始的输入相加
            out = self.relu(out) # 在融合之后才调用激活函数

            return out

# 定义ResNet网络的结构
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,norm_layer=None):
        """
                    block (BasicBlock / Bottleneck): 残差块类型
                    layers (list): 每一个stage的残差块的数目，长度为4
                    num_classes (int): 类别数目
        """
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        super(ResNet, self).__init__()
        self.inplanes = 64  # 第一个残差块的输入通道数
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False) #  7 * 7,64,stride=2
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layer对应图中的stage，也即Stage1 ~ Stage4
        self.layer1 = self._make_layer(block, 64, layers[0])  # [3 * 3 , 64]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # [3 * 3 , 128]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # [3 * 3 , 256]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # [3 * 3 , 512]

        self.avgpool = nn.AdaptiveAvgPool1d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 网络参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # 如果发现m是Conv2d类型，使用kaiming_normal_()初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): # 如果发现m是BatchNorm2d类型，使用constant_()初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 定义make_layer在这个方法，作用是生成Stage的结构
    def _make_layer(self, block, planes, blocks, stride=1):
        """
            block (BasicBlock / Bottleneck): 残差块结构
            plane (int): 残差块中第一个卷积层的输出通道数
            bloacks (int): 当前Stage中的残差块的数目
            stride (int): 残差块中第一个卷积层的步长
        """

        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:  # 判断是否需要下采样(stride!=1就说明发生了下采样)
            downsample = nn.Sequential( # 生成downsample
                nn.conv1x1(self.inplanes, planes * block.expansion,stride),
                          norm_layer(planes*block.expansion)  # 1 * 1卷积调整维度
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,norm_layer))  # 第一个block单独处理
        self.inplanes = planes * self.expansion # 记录layerN的channel变化
        for i in range(1, blocks): # 从1开始循环，生成每一个层，第一个模块前面单独处理过了
            layers.append(block(self.inplanes, planes,norm_layer = norm_layer))

        return nn.Sequential(*layers) # 使用Sequential层组合blocks,形成stage

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x

# 封装，加载预训练参数
def _resnet(arch,block,layers,pretrained,progress,**kwargs):
    model = ResNet(block,layers,**kwargs) # 调用网络
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],progress=progress)
        model.load_state_dict(state_dict)
    return model

# 构建一个ResNet-18模型
def resnet18(pretrained=False, progress=True,**kwargs):
    # pretrained (bool): 若为True则返回在ImageNet上预训练的模型
    return _resnet('resnet18',BasicBlock,[2,2,2,2],pretrained,progress,**kwargs)

if __name__ == '__main__':
    x = torch.rand(1,3,224,224)
    model = torchvision.models.resnet18()
    y = model(x)
    print(y.shape)
