---
layout: post
tags: [论文阅读]
comments: true
toc: false
---
centerMask,基于CenterNet.

<!-- more -->

# 

## CenterMask: single shot instance segmentation with point representation

paper is available at: https://arxiv.org/abs/2004.04446
code: not publish

论文信息：

<center><B>Yuqing Wang  Zhaoliang Xu     Hao Shen    Baoshan Cheng     Lirong Yang    </B></center>

<center><B>Meituan Dianping Group</center></B>

<center>{wangyuqing06, xuzhaoliang,shenhao04, chengbaoshan02, yanglirong}@meituan.com </center>


### Abstract

单阶段实例分割主要面临2个挑战：区分目标实例（object instances differentiation）和像素级特征对齐（pixel-wise feature alignment）。

作者提出一种single-shot的实例分割模型，将实例分割问题分解为2个并行子任务：局部形状预测（Local shape prediction）和全局显著图生成（ Global Saliency generation），前者用于区分重叠对象，后者对图像做像素级的分割。结合两分支的输出，就可以得到最终的实例掩码（instance mask）。

> 局部形状信息是从目标中心点表示中提取的（the local shape information is adopted from the representation of object center points）：论文中是特征图位置(x,y)处的，长度为channel的向量。

在COCO数据集上实现了34.5的mask AP，在一阶段目标检测器FCOS上加入该分支做实例分割，也有不错的表现。

### 1 Introduction

实例分割是对图像中的每个实例进行定位、分类和分割，所以它具有目标检测和语义分割的特征。

实例分割比目标检测困难，主要在于前者需要得到实例形状，而后者只需要得到边界框。一阶段实例分割主要面临以下2个挑战：

1. 如何区分目标实例，尤其是当它们具有相同类别时。当前方法是提取出图像的全局特征，进过一定后处理来区分不同实例，但是，当同类别的目标发生重叠时，这种方法就不太管用了。
2. 如何保存像素级的位置信息。一种方法是得到固定数量点的轮廓，但这种方法得到的边界很粗糙，另一种如TenserMask设计了复杂的像素对齐操作，但将这种方法应用到单阶段器中，会严重拖慢网络计算速度。

> There are two main challenges for one-stage instance segmentation: (1) how to differentiate object instances, especially when they are in the same category. Some methods [3, 1] extract the global features of the image firstly then post-process them to separate different instances, but these methods struggle when objects overlap. (2) how to preserve pixel-wise location information.

基于以上问题，作者将掩膜表示（mask representation）分解为两个平行分支：局部形状表示（Local Shape representation）和全局显著性图（Global Saliency Map）。

+ 局部形状表示，它给出不同目标粗糙的形状掩码；
+ 全局显著性图，实现了像素对齐（pixel-wise alignment）。

局部形状表示，就是目标粗糙的形状掩膜。（区分实例的过程中，用到了目标中心点表示：其实就是一个向量，即在特征图上，对应目标中心点处的，长度为channel的向量，具体细节在3.3小节，使用中心点的信息来表示目标，是受到了单阶段检测器CenterNet的启发，所以作者将方法叫做CenterNet。）

> (1) a Local Shape representation that predicts a coarse mask for each local area, which can separate different instances automatically. (2) a Global Saliency Map that segment the whole image, which can provide saliency details, and realize pixel-wise alignment. To realize that, the local shape information is extracted from the point representation at object centers. Modeling object as its center point is motivated by the one stage CenterNet [30] detector, thus we call our method CenterMask.

图1：CenterMask说明图

![image-20210716131659611](https://github.com/chgex/chgex.github.io/raw/master/__md__/image-20210716131659611.png)

**Inference**: 输入一张图片，预测出目标中心点位置，然后提取中心点的特征表示（用于区分实例），形成该目标的局部形状表示（即目标的粗糙形状），局部形状表示可以区分相邻目标（或重叠目标）。与此同时，全卷积骨干网产生整张图片的全局显著性图（用于区分前景和背景，是像素级的区分）。最后，将粗糙但感知实例（coarse but instance-aware）的局部形状和精确但不感知实例（precise but instance-unaware）的全局显著性图组合起来（ennn，就是对应位置元素相乘，具体见下一节），就形成了最终的实例掩码（instance masks）。

> Given an input image, the object center point locations are predicted following a keypoint estimation pipeline. Then the feature representation at the center point is extracted to form the local shape, which is represented by a coarse mask that separates the object from close ones. In the meantime, the fully convolutional backbone produces a global saliency map of the whole image, which separates the foreground from the background at pixel level. Finally, the coarse but instance-aware local shapes and the precise but instance-unaware global saliency map are assembled to form the final instance masks.

论文主要贡献： 

+ An anchor-box free and one-stage instance segmentation method.
+ The Local Shape representation of object masks.
+ The Global Saliency Map is proposed to realize pixel-wise feature alignment naturally.

### 3 CenterMask

CenterMask是一个单阶段实例分割方法，单阶段意味着不需要预生成RoIs，CenterMask将实例分割分解为2个分支，第一个分支从中心点表示中，预测目标的粗糙的local shape，local shape用于限制目标区域并区分不同的实例。第二个分支预测整张图的 saliency map，实现精确前景背景分割。最后，将两个分支的输出相集合，就得到了每个实例的分割结果。

#### 3.1 Local Shape Prediction

为了区分实例，作者基于目标中心点，建模目标mask（中心点定义为目标边界框的中心）。由于固定大小的特征，不够表示不同大小的mask，因此作者将目标mask分解为两个部分：mask size和mask shape（size为目标的高和宽，shape为2维二值数组，修正之后的大小和size宽高相同）。

size和shape分支结构，如下图所示：

![image-20210716150508787](https://github.com/chgex/chgex.github.io/raw/master/__md__/image-20210716150508787.png)

其中，$P$ 是特征图（由骨干网提取得到），Local shape head 输出 $F_{shape} \in R^{H×W×S^2}$ ，size head 输出 $F_{szie} \in R^{H×W×2}$ （H和W为特征图的高和宽，$S^2$ 为特征图的channel数）。

对于特征图上的中心点 $(x,y)$  ，shape 特征 $F_{shape}(x,y)=1×1×S^{2}$ ，reshape 为2维数组 $S×S$  。目标中心点对于的目标size $F_{size}(x,y)=(h,w)$ ，（h,w表示为该目标的高和宽），将 size head 和 shape head结合：即将 $S×S$ 的二维特征图 resize 到 $h×w$ 。

即最终得到的 $h×w$ 的二维向量。

#### 3.2 Global Saliency

local shape 向量仅能预测出粗糙的mask，调整到与目标尺寸对应后，会损失一定的空间信息。

作者提出 Global Saliency Map 来解决像素级的特征对齐（pixel level feature alignment）问题，全局显著性图表示整个图像中像素的显著性，即每个像素是否属于对象区域（像素分类方式有两种，一种是全部像素分2类（前景背景），一种是目标对应像素，非2类）。

Global Saliency Map 可以是类无关的（class-agnostic）：整张图的像素分为前景或背景，也可以是类相关的（class-specifific）：head会为每个目标，生成一个二值 mask，以表示该目标对应位置像素的类别。使用 sigmoid 函数做二元分类。

使用全卷积骨干网，Global Saliency 分支和 Local Shape 分支，并行处理图像分割任务。

图3是一个Global Saliency Map的示例：

![image-20210716161421302](https://github.com/chgex/chgex.github.io/raw/master/__md__/image-20210716161421302.png)

> An example of Global Saliency Map is shown in the top of Figure 3, using the class-agnostic setting for visualization convenience. As can be seen in the figure, the map highlights the pixels that have saliency, and achieves pixel wise alignment with the input image.

#### 3.3. Mask Assembly

将Local shape 和 Global Saliency Map 的的结果相结合，就可以得到最终的实例。

令 $L_{k}\in R_{h×w}$ 表示目标的Local shape，$G_{k}\in R_{h×w}$ 表示对应的cropped Saliency Map，则，计算两个矩阵的 Hadamard product（就是对应位置元素相乘），就得到最终的预测mask $M_{k}$ 。

assemble mask 的损失函数：
$$
L_{mask}=\frac{1}{N}\sum_{k=1}^{N}BCE(M_{k},T{k})
\tag{2}
$$
其中，$T_{k}$ 为目标对应的 ground truth mask，$BCE$ 为像素级的二值交叉熵损失函数，N为目标个数。

#### 3.4 Overall pipeline of CenterMask

CenterMask整体结构见之前的图3。

Heatmap Head 用来预测中心点的位置和类别。该head 是在关键点检测 pipeline 之后。具体做法是：输出的特征图中，每个channel 都是对应类别（一共有C个类）的 heatmap，中心点的获取需要搜索每个 heatmap 的峰值（峰值定义为窗口内的局部最大值）。

Offset head用来恢复output stride 造成的误差。

> The overall architecture of CenterMask is shown in Figure 3. The Heatmap head is utilized to predict the positions and categories for center points, following a typical keypoint estimation[24] pipeline. Each channel of the output is a heatmap for the corresponding category. Obtaining the center points requires to search the peaks for each heatmap, which are defined as the local maximums within a window.
>
> The Offset head is utilized to recover the discretization error caused by the output stride.

预测出中心点之后， shape head 和 size head 开始计算该位置对应的 Local shape。

Saliency head 生成Global Saliency map，在类别无关（class agnostic）方式下，输出的全局显著性图 channel 数为1。使用预测的 location 和 size ，剪裁出对应位置的显著性图。在类别相关的方式下，剪裁出对应位置，对应目标的显著性图。最终 mask 由Local shape 和 Saliency Map 做 assemble 得到。

> Given the predicted center points, the Local Shapes for these points are calculated by the outputs of the Shape head and the Size head at the corresponding locations, following the approach in Section 3.1. The Saliency head produces the Global Saliency Map. In the class-agnostic setting, the output channel number is 1, the Saliency map for each instance is obtained by cropping it with the predicted location and size. In the class-specifific setting, the channel of the corresponding predicted category is cropped. The final masks are constructed by assembling the Local Shapes and the Saliency Map.

**Loss function**

整个损失函数由以下4部分组成：e center point loss, the offset loss, the size loss, and the mask loss.

center point loss:  

该损失函数用于像素级逻辑回归，（由focal loss 修正得到）。 
$$
L_{p}=\frac{-1}{N}\sum_{ijc}
\begin{cases}

(1-\hat{Y_{ijc}})^{\alpha} log(\hat{Y}_{ijc}) \;\;\;\;\;\;\;\;if \;Y_{ijc}=1 \\
(1-\hat{Y_{ijc}})^{\beta}(\hat{Y_{ijc}})^{\alpha}
log(1-\hat{Y}_{ijc}) \;\;\;\;o.w

\end{cases}

\tag{3}
$$

> 与沙漏网络（Hourglass network）中的损失函数是一样的。$\alpha,\beta$ 是超参数。

其中，$\hat{Y}_{ijc}$ 表示在 heatmap 中，位置  (i,j) 类别为 c所 对应的预测 score 。$ Y$ 表示 ground-truth heatmap。 

 offset loss 和 size loss 与CenterNet 中的设定相同，都是使用 L1 loss 来惩罚距离。
$$
L_{off}=\frac{1}{N}\sum_{p}|\hat{O}_{\hat{p}} - (\frac{p}{R}-\hat{p}| 
\tag{4}
$$
 其中，$\hat{O}$ 表示预测的 offset，$p$ 表示中心点的ground truth，$R$ 为 output stride，所以低分辨率下 P 等价于 $\hat{p} =\lfloor \frac{p}{R} \rfloor$ 。

>  两个问题：
>
> 1. 中心点的 offset 是什么，怎么得到的；
> 2. 通过 hatmap ，搜索峰值，就能直接得到中心点吗。
>
> 下一篇读centerNet了。

size loss: 
$$
L_{size}=\frac{1}{N}\sum_{k=1}^{N}|\hat{S_k} -S_k |
\tag{5}
$$
其中， $S_k=(h,w)$ 为目标的真实size，$\hat{S_k}$ 为预测size。

所以，总的损失函数就是将上述4个损失函数合在一起，即：
$$
L_{seg}=\lambda_pL_p + \lambda_{off}L_{off} + \lambda_{size}L_{size} +\lambda_{mask}L_{mask}
\tag{6}
$$

#### 3.5 Implementation Details

训练：S=32，输入 512×512；

推理：返回前100个点所对应的 mask （按照score 排序）。

### 4 Experiments

训练集： 115k trainval35k images 

测试集：5k minival images

验证集：20k test-dev.

Ablation Study

作者对 Shape size Selection，Backbone Architectur，Local Shape branch，Global Saliency branch 分别做了消融实验。

然后与两阶段方法：FCIS，Mask RCNN等，单阶段方法：YOLACT，PolarMask，CenterMask 等，做了对比。

