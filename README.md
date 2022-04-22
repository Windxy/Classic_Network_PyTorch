# Classic_Network_PyTorch
Using PyTorch to Rebuild Classic Deep Learning Network

用PyTorch重建经典的深度学习网络（注释版）



| 名称 | 时间 | 亮点 | paper链接 | code链接 |
| ------ | ---- | --------- | ---------- | --------- |
| LeNet  | 1998 | 1.**最早提出**的的卷积神经网络模型，应用于手写数字分类任务<br />2.解释了CNN网络的**主要部件**包括，输入层+卷积层+池化层+全连接层+输出层<br />3.总结CNN**三大特性核心思想**：局部感受野(local receptive fields)、权值共享(shared weights)、下采样(sub-sampling) | [paper-LeNet](https://ieeexplore.ieee.org/document/726791) | [code-LeNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/LeNet.py) |
| AlexNet| 2012 | 1.首次提出**ReLU**激活函数<br />2.引入**局部相应归一化**<br />3.提出**数据增强**（Data augmentation）和**Dropout**来缓解过拟合（Overfitting）<br/>4.使用**双GPU**进行网络的训练 | [paper-AlexNet](https://dl.acm.org/doi/10.5555/2999134.2999257) | [code-AlexNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/AlexNet.py) |
| VGG16  | 2014 | 1.使用了更小的**3x3卷积核**，和更深的网络（两个3x3卷积核的堆叠相对于5x5卷积核的视野，三个3x3卷积核的堆叠相当于7x7卷积核的视野。这样一方面可以有更少的参数；另一方面拥有更多的非线性变换，增加了CNN对特征的学习能力。<br /> 2.使用可重复使用的卷积块来构建深度卷积神经网络**引入1x1的卷积核**，在不影响输入输出维度的情况下，引入非线性变换，增加网络的表达能力，降低计算量）。<br /> 3.采用了**Multi-Scale的方法**来训练和预测。可以增加训练的数据量，防止模型过拟合，提升预测准确率。<br />| [paper-VGG16](https://arxiv.org/abs/1409.1556) | [code-VGG16](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/VGG16.py)|
| NiN    | 2014 | 1.无全连接层<br /> 2.使用卷积层加两个1 * 1卷积层，后者对每个像素增加了非线性性<br />3.交替使用NiN块和步幅为2的最大池化层，逐步减小高宽和增大通道数<br /> 4.最后使用全局平均池化来代替VGG和AlexNet中的全连接层，不容易过拟合，且参数更少<br />  | [paper-NiN](https://arxiv.org/abs/1312.4400) |[code-NiN](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/NiN.py)  |
| ResNet18 | 2015 | 1.首次提出残差学习框架，**利用残差结构让网络能够更深、收敛速度更快、优化更容易，同时参数相对之前的模型更少、复杂度更低**<br />2.梯度消失/爆炸已经通过 normalized initialization 等方式得到解决<br />3.解决深网络退化、难以训练的问题<br />4.适用于多种计算机视觉任务<br />  | [paper-ResNet](https://arxiv.org/abs/1512.03385) |[code-ResNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/ResNet18.py) |
| GoogLeNet | 2015 | 1.引入稀疏特征，提出**Inception结构**，融合不同尺度的特征信息<br />2.**使用1×1的卷积进行降维**同时**降低参数量**，GoogLeNet参数为500万个，为AlexNet参数的1/12，VGG的1/3<br />3.所有卷积层据使用ReLu激活函数；移除全连接层，像NIN一样使用Global Average Pooling，最后添加一个全连接层<br />4.相对浅层的神经网络层对模型效果有较大的贡献，训练阶段通过**对Inception(4a、4d)增加两个额外的辅助分类器**来增强反向传播时的梯度信号，同时避免了梯度消失，辅助分类器是将中间某一层的输出用作分类，并按一个较小的权重（0.3）加到最终分类结果中，相当于做了模型融合，同时给网络增加了反向传播的梯度信号，也提供了额外的正则化，对于整个网络的训练很有裨益。而在实际测试的时候，这两个额外的softmax会被去掉<br />5.使用丢弃概率为0.7的Dropout层<br />  |[paper-GoogLeNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)|[code-GoogLeNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/GoogLeNet.py) |
| HighwayNet | 2015 | 1.受LSTM的灵感，基于门机制引入了**transform gate T（x.WT）**和**carry gate C（x,WT）**,使得训练更深的网络变为可能，并且加快了网络的收敛速度<br />2.借用**随机梯度下降策略**就可以很好地进行训练（而且很快），在反向传播梯度计算的时候，部分参数为一个常系数，避免了梯度的消失，保留了关键的信息 |[paper-HighwayNet](https://arxiv.org/abs/1507.06228)|[code-HighwayNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/HighwatNet.py) |
| DenseNet | 2018 | 1.提出**Dense块**，引入了相同特征图尺寸的任意两层网络的直接连接，特点是看起来非常“密集”，特征重用<br />2.更强的梯度流动。由于密集连接方式，DenseNet提升了梯度的反向传播，使网络更容易训练<br />3.**参数更少**，DenseNet有效的降低了过拟合的出现，易于优化，加强了特征的传播<br /> |[paper-DenseNet](https://arxiv.org/pdf/1608.06993.pdf)|[code-DenseNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/DenseNet.py) |

