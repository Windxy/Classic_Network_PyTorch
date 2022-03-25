import torch
from torch import nn
from torchstat import stat
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''LeNet'''

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),#(32-5)/1+1=28
            nn.Sigmoid(),
            nn.AvgPool2d(2,2),#(28-2)/2+1=14
            nn.Conv2d(6, 16, 5),#(14-5)/1+1=10
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),#5
        )
        self.fc = nn.Sequential(
            nn.Linear(5*5*16, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )
    def forward(self,x):
        feature = self.conv(x)
        out = self.fc(feature.view(x.shape[0],-1))    #4张图片一批，img.shape[0]为4，第二个就是对应的概率
        return out

if __name__ == '__main__':
    net = LeNet()
    stat(net,(1,32,32))
    X = torch.rand(1, 1, 32, 32)
    print(net(X).shape)