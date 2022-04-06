# Classic_Network_PyTorch
Using PyTorch to Rebuild Classic Deep Learning Network

用PyTorch重建经典的深度学习网络（注释版）



| 名称 | 时间 | 亮点 | paper链接 | code链接 |
| ------ | ---- | --------- | ---------- | --------- |
| LeNet  | 1998 | 1.**最早提出**的的卷积神经网络模型，应用于手写数字分类任务<br />2.解释了CNN网络的**主要部件**包括，输入层+卷积层+池化层+全连接层+输出层 | [paper-LeNet](https://ieeexplore.ieee.org/document/726791) | [code-LeNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/LeNet.py) |
| AlexNet| 2012 | 1.首次提出**ReLU**激活函数<br />2.引入**局部相应归一化**<br />3.提出**数据增强**（Data augmentation）和**Dropout**来缓解过拟合（Overfitting）<br/>4.使用**双GPU**进行网络的训练 | [paper-AlexNet](https://dl.acm.org/doi/10.5555/2999134.2999257) | [code-AlexNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/AlexNet.py) |
| VGG16  | 2014 | 1.实质是更大更深的AlexNet（使用5组VGG块）<br /> 2.使用可重复使用的卷积块来构建深度卷积神经网络<br /> 3.所有的卷积层全部采用3 * 3的卷积核<br /> 4.VGG训练速度较AlexNet慢，但是准确率有了很大的提升<br />| [paper-VGG16](https://arxiv.org/abs/1409.1556) | [code-VGG16](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/VGG16.py)|
| NiN    | 2014 | 1.无全连接层<br /> 2.使用卷积层加两个1 * 1卷积层，后者对每个像素增加了非线性性<br />3.交替使用NiN块和步幅为2的最大池化层，逐步减小高宽和增大通道数<br /> 4.最后使用全局平均池化来代替VGG和AlexNet中的全连接层，不容易过拟合，且参数更少<br />  | [paper-NiN](https://arxiv.org/abs/1312.4400) |[code-NiN](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/NiN.py)  |
| ResNet18 | 2015 | 1.首次提出残差学习框架，**利用残差结构让网络能够更深、收敛速度更快、优化更容易，同时参数相对之前的模型更少、复杂度更低**<br />2.梯度消失/爆炸已经通过 normalized initialization 等方式得到解决<br />3.解决深网络退化、难以训练的问题<br />4.适用于多种计算机视觉任务<br />  | [paper-ResNet](https://arxiv.org/abs/1512.03385) |[code-ResNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/ResNet18.py) |
