import torch
from torch import nn
from torchvision import models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# VGG块
# 由指定卷积层数的网络块最后加窗口为2X2、步幅为2的最大值池化层构成。每一个网络块由一个3X3的卷积核、padding为1的卷积层再加relu激活函数组成
def vgg_block(num_convs,in_channels,out_channels): # num_convs卷积层的个数，输入通道数，输出通道数
    blk = [] # 存储卷积层
    for i in range(num_convs):
        if i == 0 : # 如果这一层是每个block里面的第一层
            blk.append(nn.Conv2d(in_channels,out_channels,3,1,padding=1)) # 每次加入Conv2d的参数
        else:
            blk.append(nn.Conv2d(out_channels,out_channels,3,1,padding=1))
        blk.append(nn.ReLU())   # 给每一个块中每一个卷积层后面都加了一个ReLU激活函数
    blk.append(nn.MaxPool2d(2,2))   # 每一个block最后接一个maxpool，使得宽高减半
    return nn.Sequential(*blk) # 带*表示是通过非关键字的形式（收集参数，以tuple的形式保存）传入

# VGG网络
# 由指定数的VGG块组成，最后加三个全连接层
class vgg(nn.Module):
    def __init__(self,conv_arch,vgg_fc):
        super(vgg,self).__init__()
        self.net = nn.Sequential()
        # 卷积层部分
        # for i,(num_convs,in_convs,out_convs) in enumerate(conv_arch):
        #     self.net.add_module('vgg_block_'.format(i),vgg_block(num_convs,in_convs,out_convs))
        self.net.add_module('vgg_block_1', vgg_block(2, 3, 64))
        self.net.add_module('vgg_block_2', vgg_block(2, 64, 128))
        self.net.add_module('vgg_block_3', vgg_block(3, 128, 256))
        self.net.add_module('vgg_block_4', vgg_block(3, 256, 512))
        self.net.add_module('vgg_block_5', vgg_block(3, 512, 512))

       # 全连接层部分
        self.net.add_module('fc',nn.Sequential(
            nn.Flatten(), # 作用是将连续的几个维度展平成一个tensor（将一些维度合并）
            nn.Linear(vgg_fc[0],vgg_fc[1]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(vgg_fc[1],vgg_fc[1]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(vgg_fc[1],10),
        ))
    def forward(self,img):
        return self.net(img)

if __name__ == '__main__':
    conv_arch = ((2, 3, 64), (2, 64, 128), (3, 128, 256), (3, 256, 512), (3, 512, 512)) # 指定每个块的参数
    vgg_fc_featrues = 512 * 7 * 7 # 输出的通道数 * 7 * 7
    # 输入的是224*224,5个块每一次都会使高宽减半，最终得到7 * 7
    vgg_fc_hidden = 4096
    vgg_fc = (vgg_fc_featrues, vgg_fc_hidden)

    net = vgg(conv_arch,vgg_fc)
    X = torch.rand(1, 3, 224, 224) # 输入一张224*224大小的单通道图片
    print(net(X).shape)
