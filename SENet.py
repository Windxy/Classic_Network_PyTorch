import torch
import torch.nn as nn
import math

"""
原理：
    1.对输入进来的特征层进行全局平均池化
    2.进行两次全连接
    3.再取一次sigmoid将值固定到0-1之间
    4.最后将这个权值*原输入特征层
"""
class se_block(nn.Module):
    # channel表示输入进来的通道数；ratio表示缩放的比例，用于第一次全连接
    def __init__(self,channel,ratio=16):
        super(se_block, self).__init__()  # 初始化
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 自适应全局平均池化，输出高宽为1
        # 两次全连接fc
        self.fc = nn.Sequential(
            # 定义第一次全连接，神经元个数减少
            # 输入神经元个数channel个数，输出神经元个数channel//ratio,不使用偏执量
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            # 定义第一次全连接，神经元个数变为原来个数
            # 输入神经元个数个数channel//ratio，输出神经元个数channel,不使用偏执量
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        # 计算输入进来的特征层的size,batch_size,通道数，高，宽，bath_size表示一次训练所抓取的数据样本数量
        b,c,h,w = x.size()
        # 全局平均池化，结果为y(b,c,1,1),即去掉后面的两个维度
        # view(b,c)：进行reshape，即只保留b,c
        y = self.avg_pool(x).view(b,c)
        # 进行两次全连接fc,b,c->b,c//ratio->b,c
        # reshape:b,c->b,c,1,1
        y = self.fc(y).view(b,c,1,1)
        # 把两次全连接的结果*特征层
        return x * y

if __name__ == '__main__':
    x = torch.rand(1,96,112,112)
    model = se_block(96)
    output = model(x)
    print(output)
