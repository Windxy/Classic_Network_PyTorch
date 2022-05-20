# Classic_Network_PyTorch
Using PyTorch to Rebuild Classic Deep Learning Network

用PyTorch重建经典的深度学习网络（注释版）
### 环境 requirement
pytorch == 1.2.0+


| 名称 | 时间 | 亮点 | paper链接 | code链接 |
| ------ | ---- | --------- | ---------- | --------- |
| LeNet  | 1998 | 1.**最早提出**的的卷积神经网络模型，应用于手写数字分类任务<br />2.解释了CNN网络的**主要部件**包括，输入层+卷积层+池化层+全连接层+输出层<br />3.总结CNN**三大特性核心思想**：局部感受野(local receptive fields)、权值共享(shared weights)、下采样(sub-sampling)<br />4.稀疏连接矩阵避免了巨大的计算开销 | [paper-LeNet](https://ieeexplore.ieee.org/document/726791) | [code-LeNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/LeNet.py) |
| AlexNet| 2012 | 1.首次提出**ReLU**激活函数<br />2.引入**局部响应归一化** LRN(Local Response Normalization) 对局部神经元的活动创建竞争机制，使其中响应比较大对值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力<br />3.提出**数据增强**（Data augmentation）和**Dropout**来缓解过拟合（Overfitting）<br/>4.使用**双GPU**进行网络的训练 | [paper-AlexNet](https://dl.acm.org/doi/10.5555/2999134.2999257) | [code-AlexNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/AlexNet.py) |
| VGG16  | 2014 | 1.使用了更小的**3x3卷积核**，和更深的网络（两个3x3卷积核的堆叠相对于5x5卷积核的视野，三个3x3卷积核的堆叠相当于7x7卷积核的视野。这样一方面可以有更少的参数；另一方面拥有更多的非线性变换，增加了CNN对特征的学习能力。<br /> 2.使用可重复使用的卷积块来构建深度卷积神经网络**引入1x1的卷积核**，在不影响输入输出维度的情况下，引入非线性变换，增加网络的表达能力，降低计算量。<br /> 3.采用了**Multi-Scale**和**Multi-Crop**方法来训练和预测。可以增加训练的数据量，防止模型过拟合，提升预测准确率。<br />| [paper-VGG16](https://arxiv.org/abs/1409.1556) | [code-VGG16](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/VGG16.py)|
| NiN    | 2014 | 1.使用全局平均池化来代替全连接层，不容易过拟合，且参数更少<br /> 2.在卷积层后加入两个核大小为1×1卷积层，1×1卷积增加了非线性<br />3.交替使用NiN块和步幅为2的最大池化层，逐步减小高宽和增大通道数<br />  | [paper-NiN](https://arxiv.org/abs/1312.4400) |[code-NiN](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/NiN.py)  |
| ResNet18 | 2015 | 1.首次提出残差学习框架，**利用残差结构让网络能够更深、收敛速度更快、优化更容易，同时参数相对之前的模型更少、复杂度更低** <br />2.解决深网络退化（而非梯度消失/爆炸，这个问题已经通过normalized initialization and intermediate normalization layers等方式得到解决）、难以训练的问题<br />3.适用于多种计算机视觉任务<br />  | [paper-ResNet](https://arxiv.org/abs/1512.03385) |[code-ResNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/ResNet18.py) |
| GoogLeNet | 2015 | 1.引入稀疏特性，提出**Inception结构**，融合不同感受野大小的特征信息，也就意味着不同尺度特征信息的融合<br />2.**使用1×1的卷积进行降维**同时**降低参数量**，GoogLeNet参数为500万个，为AlexNet参数的1/12，VGG的1/3<br />3.所有卷积层据使用ReLu激活函数；移除全连接层，像NIN一样使用Global Average Pooling，最后添加一个全连接层<br />4.相对浅层的神经网络层对模型效果有较大的贡献，训练阶段通过**对Inception(4a、4d)增加两个额外的辅助分类器**来增强反向传播时的梯度信号，同时避免了梯度消失，辅助分类器是将中间某一层的输出用作分类，并按一个较小的权重（0.3）加到最终分类结果中，相当于做了模型融合，同时给网络增加了反向传播的梯度信号，也提供了额外的正则化，对于整个网络的训练很有裨益。而在实际测试的时候，这两个额外的softmax会被去掉<br />  |[paper-GoogLeNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)|[code-GoogLeNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/GoogLeNet.py) |
| HighwayNet | 2015 | 1.受LSTM的灵感，基于门机制引入了**transform gate T（x.WT）**和**carry gate C（x,WT）**,使得训练更深的网络变为可能，并且加快了网络的收敛速度<br />2.借用**随机梯度下降策略**就可以很好地进行训练（而且很快），在反向传播梯度计算的时候，部分参数为一个常系数，避免了梯度的消失，保留了关键的信息 |[paper-HighwayNet](https://arxiv.org/abs/1507.06228)|[code-HighwayNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/HighwatNet.py) |
| DenseNet | 2018 | 1.提出**Dense块**，引入了相同特征图尺寸的任意两层网络的直接连接，特点是看起来非常“密集”，特征重用<br />2.更强的梯度流动。由于密集连接方式，DenseNet提升了梯度的反向传播，使网络更容易训练<br />3.**参数更少**，DenseNet有效的降低了过拟合的出现，易于优化，加强了特征的传播<br /> |[paper-DenseNet](https://arxiv.org/pdf/1608.06993.pdf)|[code-DenseNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/DenseNet.py) |
| Inception-V3 | 2015 | 1.对于GoogLeNet网络中提出的**inception结构（inceptionV1）** 进行改进<br />2.InceptionV2:用两个 3 x 3 的卷积代替 5×5的大卷积。使用**BN (BatchNormalization ）方法**。<br />3.InceptionV3:提出了更多种卷积分解的方法，**把大卷积因式分解为小卷积和非对称卷积**。<br /> 4.引入Label Smoothing，提升模型性能|[paper-InceptionV3](http://cn.arxiv.org/pdf/1512.00567v3)|[code-InceptionV3](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/InceptionV3.py) |
| Inception-V4 | 2016 | 1.整个结构所使用模块和InceptionV3基本一致<br />2.初始的InceptionV3:3个InceptionA+5个InceptionB+3个InceptionC<br />3.InceptionV4:经过4个InceptionA+1个ReductionA+7个InceptionB+1个ReductionB+3个InceptionC<br />4.在InceptionV3原始结构的基础上**增加了ReductionA和ReductionB模块**，这些**缩减块的加入是为了改变特征图的宽度和高度**，ReductionA：尺寸从35×35缩减到17×17;ReductionB尺寸从17×17缩减到8×8。早期的版本并没有明确使用缩减块，但也实现了其功能。<br />|[paper-InceptionV4](https://arxiv.org/pdf/1602.07261.pdf)|[code-InceptionV4](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/InceptionV4.py) |
| SqueezeNet | 2016 | 1.主要目的是为了在达到当前主流网络的识别精度的基础上**降低CNN模型的参数数量，简化网络复杂度**<br />2.SqueezeNet在保持和 AlexNet同样的准确度上，参数比它少50倍<br />3.使用**三个策略**达到目标：1、大量使用1x1卷积核替换3x3卷积核，因为参数可以降低9倍；2、减少3x3卷积核的输入通道数（input channels），因为卷积核参数为：(number of input channels) * (number of filters) * 3 * 3；3、延迟下采样，可以给卷积层提供更大的激活图：更大的激活图保留了更多的信息，可以提供更高的分类准确率。1和2可以显著减少参数数量，3可以在参数数量受限的情况下提高准确率<br />4.定义了自己的**卷积模块Fire Module**，分为squeeze层和expand层，squeeze层只使用1×1卷积（策略1），还可以限制输入通道数量（策略3）<br />|[paper-SqueezeNet](https://arxiv.org/pdf/1602.07360.pdf)|[code-SqueezeNet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/SqueezeNet.py) |
| SE-Net(Squeeze and Excitation Network) | 2017 | 1.研究了通道关系，通过引入一个新的架构单元**SE block**，来对卷积特征的通道之间的相互依赖关系进行建模，以提高网络的表示能力。将SE模块进行堆叠，构建SENet架构。<br />2.操作分为两步，第一步是Squeeze,第二步是Exciation。前者对应一个全局平均池化操作，将c×h×w压缩成c×1×1，得到全局信息；后者进行两次全连接，最终得到权重矩阵<br /> |[paper-SENet](https://arxiv.org/abs/1709.01507)|[code-SENet](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/SENet.py) |
| MobileNetv1 | 2017 | 1.提出**深度可分离卷积**，即DepthWise+PointWise两种卷积方式，在性能没有急剧降低的情况下，大大降低了网络参数量<br />2.引用**ReLU6**作为激活函数，在低精度计算下能够保持更强的鲁棒性 |[paper-Mobilenet](https://arxiv.org/pdf/1704.04861.pdf)| [code-MobileNetv1](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/MobileNet.py) |
| MobileNetv2 | 2018 |                                                              |[paper-Mobilenetv2](https://arxiv.org/abs/1801.04381)| |
| MobileNetv3 |  | 待完成 || |
| ShuffleNet |  | 待完成 || |
| ResNeXt |  | 待完成 || |
| Xception |  | 待完成 || |
| CBAM(Convolutional Block Attention Module) | 2018 | 1.CBAM表示卷积模块的注意力机制模块，是一种**融合通道和空间注意力的注意力模块**，沿着空间和通道两个维度依次推断出注意力权重，再与原图相乘来对特征进行自适应调整<br />2.在SENet或ECANet的基础上，**在通道注意力模块后，接入空间注意力模块**，实现了通道注意力和空间注意力的双机制<br />3.**注意力模块不再采用单一的最大池化或平均池化，而是采用最大池化和平均池化的相加或堆叠**。通道注意力模块采用相加，空间注意力模块采用堆叠方式。 | [paper-CBAM](https://arxiv.org/pdf/1807.06521.pdf) | [code-CBAM](https://github.com/Windxy/Classic_Network_PyTorch/blob/main/CBAM.py)  |

