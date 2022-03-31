import torch
import torch.nn.functional as F
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def nin_block(in_channel,out_channel,k,s,p):
    return nn.Sequential(
        nn.Conv2d(in_channel,out_channel,k,s,p),
        nn.ReLU(),
        nn.Conv2d(out_channel,out_channel,1),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, 1),
        nn.ReLU()
    )

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
    def forward(self,x):
        x = F.avg_pool2d(x,x.size()[2:])
        return x

class NiN(nn.Module):
    def __init__(self):
        super(NiN, self).__init__()
        self.model = nn.Sequential(
            nin_block(3,96,11,4,0),
            nn.MaxPool2d(3,2),
            nin_block(96, 256, 5, 1, 2),
            nn.MaxPool2d(3, 2),
            nin_block(256,384,3,1,1),
            nn.MaxPool2d(3,2),
            nn.Dropout(0.5),
            nin_block(384,10,3,1,1),
            GlobalAvgPool(),
            nn.Flatten(),
        )

    def forward(self,img):
        return self.model(img)


if __name__ == '__main__':

    net = NiN()
    X = torch.rand(1, 3, 224, 224)
    print(net(X).shape)