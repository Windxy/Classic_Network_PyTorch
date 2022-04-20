'''HighwayNet''' # ref from: https://github.com/kefirski/pytorch_Highway
# paper link: https://arxiv.org/abs/1505.00387
import torch
from torch import nn
import torch.nn.functional as F

class Highway(nn.Module):
    """
    y = f(x)的一层非线性变换，具体公式为y = T(x, Wt) * x + (1 - T(x, Wt)) * H(x, Wh)
    """
    def __init__(self, input_size, num_layers, f):
        super(Highway, self).__init__()
        size = input_size
        self.num_layers = num_layers # 设置highway的层数
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)]) #
        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """
        for layer in range(self.num_layers):
            # T(x, Wt)
            gate = torch.sigmoid(self.gate[layer](x)) # F.sigmoid 淘汰
            # H(x,Wh)
            nonlinear = self.f(self.nonlinear[layer](x))
            # (x,wh)
            linear = self.linear[layer](x)
            # T(x, Wt) * x + (1 - T(x, Wt)) * H(x, Wh)
            x = gate * nonlinear + (1 - gate) * linear
        return x

# 构建网络
if __name__ == '__main__':
    net = Highway(input_size=10, num_layers=5, f=F.relu)
    X = torch.rand(1, 10)   # batch_size, size
    print(net(X).shape)
