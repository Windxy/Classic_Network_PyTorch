import torch
from torch import nn
from torchvision import models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# vgg的block
def vgg_block(num_convs,in_channels,out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0 :
            blk.append(nn.Conv2d(in_channels,out_channels,3,1,padding=1))
        else:
            blk.append(nn.Conv2d(out_channels,out_channels,3,1,padding=1))
        blk.append(nn.ReLU())   # 每个卷积层后借一个ReLU
    blk.append(nn.MaxPool2d(2,2))   # 每一个block最后接一个maxpool
    return nn.Sequential(*blk)

class vgg(nn.Module):
    def __init__(self,conv_arch,vgg_fc):
        super(vgg,self).__init__()
        self.net = nn.Sequential()
        for i,(num_convs,in_convs,out_convs) in enumerate(conv_arch):
            self.net.add_module('vgg_block_'.format(i),vgg_block(num_convs,in_convs,out_convs))
        # self.net.add_module('vgg_block_1', vgg_block(2, 3, 64))
        # self.net.add_module('vgg_block_2', vgg_block(2, 64, 128))
        # self.net.add_module('vgg_block_3', vgg_block(3, 128, 256))
        # self.net.add_module('vgg_block_4', vgg_block(3, 256, 512))
        # self.net.add_module('vgg_block_5', vgg_block(3, 512, 512))
        self.net.add_module('fc',nn.Sequential(
            nn.Flatten(),
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
    conv_arch = ((2, 3, 64), (2, 64, 128), (3, 128, 256), (3, 256, 512), (3, 512, 512))
    vgg_fc_featrues = 512 * 7 * 7
    vgg_fc_hidden = 4096
    vgg_fc = (vgg_fc_featrues, vgg_fc_hidden)

    net = vgg(conv_arch,vgg_fc)
    X = torch.rand(1, 3, 224, 224)
    print(net(X).shape)