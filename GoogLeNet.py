import torch
import torch.nn as nn
import torch.nn.functional as F

class GoogLeNet(nn.Module):
    # aux_logits: 是否使用辅助分类器（训练的时候为True, 验证的时候为False)
    def __init__(self,num_classes=1000,aux_logits=True,init_weigths=False):
        super(GoogLeNet,self).__init__()
        self.aux_logits = aux_logits

        # RGB图像，通道数为3，卷积核个数64，卷积核大小7，步距2
        # padding取3，output_size=(224-7+2*3)/2+1=112.5,在pytorch中默认向下取整，即112
        self.conv1 = BasicConv2d(3,64,kernel_size=7,stride=2,padding=3)
        # ceil_mode=True表示池化操作得到的数值为小数时向上取整
        self.maxpool1 = nn.MaxPool2d(3,stride=2,ceil_mode=True)
        # nn.LocalResponseNorm() #省略此操作

        self.conv2 = BasicConv2d(64,64,kernel_size=1)
        self.conv3 = BasicConv2d(64,192,kernel_size=3,padding=1)
        self.maxpool2 = nn.MaxPool2d(3,stride=2,ceil_mode=True)

        self.inception3a = Inception(192,64,96,128,16,32,32)
        self.inception3b = Inception(256,128,128,192,32,96,64)
        self.maxpool3 = nn.MaxPool2d(3,stride=2,ceil_mode=True)

        self.inception4a = Inception(480,192,96,208,16,48,64)
        self.inception4b = Inception(512,160,112,224,24,64,64)
        self.inception4c = Inception(512,128,128,256,24,64,64)
        self.inception4d = Inception(512,112,144,288,32,64,64)
        self.inception4e = Inception(528,256,160,320,32,128,128)
        self.maxpool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.inception5a = Inception(832,256,160,320,32,128,128)
        self.inception5b = Inception(832,384,192,384,48,128,128)

        if aux_logits:
            self.aux1 = InceptionAux(512,num_classes)
            self.aux2 = InceptionAux(528,num_classes)

        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))#自适应平均池化，保证输出为1*1
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(1024,num_classes)

        if init_weigths:
            self._initialize_weigths()

    def forward(self,x):
        #N*3*224*224
        x = self.conv1(x)
        # N*64*112*112
        x = self.maxpool1(x)
        # N*64*56*56
        x = self.conv2(x)
        # N*64*56*56
        x = self.conv3(x)
        # N*192*56*56
        x = self.maxpool2(x)

        # N*192*28*28
        x = self.inception3a(x)
        # N*256*28*28
        x = self.inception3b(x)
        # N*480*28*28
        x = self.maxpool3(x)
        # N*480*14*14
        x = self.inception4a(x)
        # N*512*14*14
        if self.training and self.aux_logits:  #只在训练过程进行辅助分支
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N*512*14*14
        x = self.inception4c(x)
        # N*512*14*14
        x = self.inception4d(x)
        # N*528*14*14
        if self.training and self.aux_logits:  #只在训练过程进行辅助分支
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N*832*14*14
        x = self.maxpool4(x)
        # N*832*7*7
        x = self.inception5a(x)
        # N*832*7*7
        x = self.inception5b(x)
        # N*1024*7*7

        x = self.avgpool1(x)
        # N*1024*1*1
        x = torch.flatten(x,1)
        # N*1024
        x = self.dropout(x)
        x = self.fc(x)
        #N*1000(num_classes)
        if self.training and self.aux_logits:  #只在训练过程进行辅助分支并输出
            return x,aux1,aux2
        return x

    def _initialize_weigths(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

# 创建 Inception 结构函数（模板）
class Inception(nn.Module):
    def __init__(self,in_channels,ch1x1,ch3x3red,ch3x3,ch5x5red,ch5x5,pool_proj):
        super(Inception,self).__init__()
        # 四个并联结构
        self.branch1 = BasicConv2d(in_channels,ch1x1,kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels,ch3x3red,kernel_size=1),
            # padding设置为1，保证每个分支所得到的特征矩阵高和宽相同，output_size=(input_size-3+2*1)/1+1=input_size
            BasicConv2d(ch3x3red,ch3x3,kernel_size=3,padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            # padding=2,output_size=(input_size - 5 + 2 * 2) / 1 + 1 = input_size
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            # 池化操作不会改变深度in_channels
            BasicConv2d(in_channels,pool_proj,kernel_size=1)
        )

    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1,branch2,branch3,branch4]
        return torch.cat(outputs,1)

# 创建辅助分类器结构函数（模板）
class InceptionAux(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(InceptionAux,self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5,stride=3)
        self.conv = BasicConv2d(in_channels,128,kernel_size=1) #output[batch,128,4,4]

        self.fc1 = nn.Linear(2048,1024) #128*4*4
        self.fc2 = nn.Linear(1024,num_classes)

    def forward(self,x):
        # aux1:N*512*14*14,aux2=N*528*14*14
        x = self.averagePool(x)
        # aux1:N*512*4*4,aux2=N*528*4*4
        x = self.conv(x)
        # N*128*4*4
        x = F.dropout(x,0.7,training=self.training)
        # N*2048
        x = F.relu(self.fc1(x),inplace=True)
        x = F.dropout(x,0.7,training=self.training)
        # N*1024
        x = self.fc2(x)
        # N*num_classes
        return x

# 创建卷积层函数（模板）
class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(BasicConv2d,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,**kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    X = torch.rand(1,3,224,224)
    net = GoogLeNet(num_classes=5,aux_logits=False)
    out = net(X)
    print(out.shape)
    #输出torch.Size([1, 5])
