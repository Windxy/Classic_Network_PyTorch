import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''AlexNet'''
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256*6*6,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,1000)
        )

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
