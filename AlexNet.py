import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''AlexNet有8层：5个卷积层，3个全连接层'''
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv = nn.Sequential(
            # 输入图片大小为3*227*227,size为227*227，通道数为3，RGB型图片
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=2), # 11 * 11 Conv(96),stride 4
            # in_channels=3因为图片类型是RGB型，所以通道数是3
            # out_channels=96表示输出的通道数，设定输出通道数的96（可以按自己需求调整）
            # kernel_size=11:表示卷积核的大小是11x11的
            # stride=4:表示的是步长为4
            # padding=2:表示的是填充值的大小为2
            # 输出大小是：N=（227-11+2*0)/4+1=55,卷积后尺寸是96*55*55
            nn.ReLU(),
            nn.MaxPool2d(3,2),  # kernel_size=3,stride=2
            # 输出大小为N=(55-3+2*0)/2+1=27,输出是27*27*96

            # 输入大小是27*27*96
            nn.Conv2d(96,256,3,1,1),
            # N=(27-3+2*1)/1+1=27，卷积后尺寸是256*27*27
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            # N=(27-3+2*0)/2+1=13,输出大小是13*13*256

            # 连续3个卷积层，计算方法以此类推
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            (nn.Conv2d(384,384,3,1,1)),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256*6*6,4096), # Dense(4096)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),  # Dense(4096)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,1000)  # Dense(1000)
        )
    # 在forward()中，在输入全连接层之前，要先feature.view(img.shape[0],-1)做一次reshape
    def forward(self,img):
        # assert img.shape[1]==3
        feature = self.conv(img)
        return self.fc(feature.view(img.shape[0],-1))


# 构建网络
if __name__ == '__main__':
    net = AlexNet()
    # stat(net,(1,32,32))
    X = torch.rand(1, 3, 224, 224)
    print(net(X).shape)