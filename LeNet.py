import torch
from torch import nn
# from torchstat import stat
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''LeNet的结构比较简单，2次重复的卷积激活池化后接3个全连接层，卷积核大小为5 * 5，池化窗口大小为2 * 2，步幅为2'''
'''论文链接：https://ieeexplore.ieee.org/document/726791'''
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        '''Conv2d(channel2,output,height,width)
        1.第一个卷积层输出通道数为 6 ，第二个卷积层输出通道数则增加到 16 。这是因为第二个卷积层比第一个卷积层的输入的高和宽要小，所以增加输出通道使两个卷积层的参数尺寸类似。
        2.卷积层块的两个最大池化层的窗口形状均为 [公式] ，且步幅为 2 。由于池化窗口与步幅形状相同，池化窗口在输入上每次滑动所覆盖的区域互不重叠。
        '''
        self.conv = nn.Sequential(
            # 第一个卷积层，输入图像通道是1，输出通道数是6，卷积核大小是5
            nn.Conv2d(1, 6, 5), # (32-5) / 1 + 1 = 28
            nn.Sigmoid(),       # 激活函数选用Sigmoid函数
            # 二维平均池化操作，池化窗口大小为2，窗口移动的步长也是2
            nn.AvgPool2d(2,2),  # (28-2)/2+1=14
            # 第二个卷积层，输入图像通道是6，输出通道数是16，卷积核大小是5
            nn.Conv2d(6, 16, 5),#(14-5)/1+1=10
            nn.Sigmoid(),       # 激活函数
            nn.AvgPool2d(2, 2), # 5
        )
        ''' 全连接层块含3个全连接层。它们的输出个数分别是120,84和10，其中10为输出的类别个数
            全连接层的输入形状将变成二维，其中第一维是小批量中的样本大小，第二维是每个样本变平后的向量表示，且向量长度为通道、高和宽的乘积。
        '''
        self.fc = nn.Sequential(
            nn.Linear(5*5*16, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )
    # 在forward()中，在输入全连接层之前，要先feature.view(img.shape[0],-1)做一次reshape
    def forward(self,x):
        feature = self.conv(x)
        out = self.fc(feature.view(x.shape[0],-1))    # 4张图片一批，img.shape[0]为4，第二个就是对应的概率
        return out

# 构建网络
if __name__ == '__main__':
    net = LeNet()
    # stat(net,(1,32,32))
    X = torch.rand(1, 1, 32, 32)
    print(net(X).shape)
