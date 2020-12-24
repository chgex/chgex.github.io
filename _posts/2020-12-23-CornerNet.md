---
layout: post
title: CornerNet
date: 2020-12-22
Author: JIUGE 
tags: [论文阅读]
comments: true
toc: false
---

本文发布于2019年。提出了一种新的目标检测算法：CornerNet，关注于边界框的一对角点。

<!-- more -->

# 论文简读  CornerNet: Detecting Objects as Paired Keypoints

论文信息：

![image-20201223104536090](https://gitee.com/changyv/md-pic/raw/master/20201223104545.png)

论文连接：https://arxiv.org/abs/1808.01244

发布时间：2019年

## Abstract

本文提出一种目标检测新方法：CornerNet。检测边界框(object bounding box)的一对关键点(keypoints)：左上角和右下角，从而省去了设计anchor box。论文介绍了一种新的池化层：corner pooling，来帮助网络更好的定位角点。在MS COCO上，CornerNet实现了42.4%的AP。（该论文发布于2019年）。

## 1 Introduction

基于卷积神经网络的检测器，一个常用部分就是anchor boxes，这些不同大小和纵横比的box，本质上是目标候选框(detection candidates)。一阶段检测器使用大量且密集的anchor box，有两个缺点：

1. 大量的anchor box中只有少量与ground truth box有重合，其它都是负样本。这将造成positive anchor box和negative anchor box不均衡问题。
2. anchor box引入大量超参数和设计项（如box数量，大小，纵横比）。这些设计需要一些启发式的训练。

本文介绍CornerNet，这是一种新的目标检测方法，抛弃了anchor box。使用单一卷积神经网络为所有同类对象实例的左上角和右下角，预测一个heatmap，并为每一个角点，预测一个embedding  vector。embedding用于同类别的目标角点的分类。CornerNet简化了网络输出并取消了anchor box设计，是受到Newell(2017)的影响。 

> Newell(2017, who detect and group keypoints in the context of multiperson human-pose estimation.

CornerNet另一个创新点就是corner pooling，该池化层帮助卷积网络更好的定位边界框(bounding boxes)。边界框的角点通常在目标之外，因此，我们考虑角点的一个半径区域。我们需要水平地向右看目标的最上面的边界，垂直地向底部看目标最左边的边界，基于此，作者设计了一个croner池化层(corner pooling layer)。carner pooling 层接受两个特征图，在每个像素位置，向右max-pooling第一个特征图的所有特征向量，向下max-pooling第二个特征图的所有特征向量，并将两个池化结果加在一起。如图3所示。

![image-20201223142451016](https://gitee.com/changyv/md-pic/raw/master/20201223142452.png)

检测角点比检测边界框效果更好，作者给出了两个原因：

1. box的中心点难以定位。中心点的定位依赖于四条边，而角点只依赖于2条边。而且，corner pooling编码了一些关于角点定位的先验知识。
2. corner为密集空间划分提供了更有效的方式：只需要$O(wh)$个角点就能代替$O(w^2h^2)$ 个可能的anchor box。

CornerNet在MS COCO上实现了42.4%的AP。通过消融实验，作者证明了corner pooling的有效性。

源码连接： https://github.com/princeton-vl/CornerNet.

## 2 Related Work

2.1 two-tage object detectors

R-CNN第一次提出两阶段目标检测（2014）。两阶段检测器产生一个ROI(region of interest, ROI)的稀疏集合，然后，由网络进行分类。R-CNN使用low level vision算法产生ROI，每个region都是从提取自图像并被卷积网络独立处理。这产生了大量的冗余计算。随后，SPP(2014)和Fast-RCNN(2015)设计了一个特殊的池化层改进了R-CNN，该池化层直接从特征图池化每个region。然而，以上两者都依赖区域划分算法，而且不是端到端的训练。Faster-RCNN(2015)抛弃了low level proposal算法，引入区域推荐网络(region propal network, RPN)，该网络从预决定的候选框集合中产生推荐区域，即anchor boxes。使检测器更加有效，并实现了端到端的训练。R-FCN(2016)通过全卷积子检测网络(fully convolutional sub-detection network)替代全连接子检测网络(fully connected sub-detection network)，进一步提升了Faster-R-CNN的效率。等。

2.2 one-stage object detector

YOLO(2016)和SSD(2016)提出了单阶段检测方法，直接移除了ROI 池化的步骤，并在单一网络中检测目标。单阶段检测器比两阶段检测器计算效率更高。

本文提出的方法受Newell(2017)多人姿态估计中的联想嵌入的影响，Newell提出一种检测并分组人体关键点的方法。关键点是通过embeding的距离来分组的。本文第一次将目标检测任务转换为角点检测，并使用embeding对这些角点分组。本文另一个创新点是引入corner pooling帮助定位角点。将hourglass网络结构进行修改，并增加改进后的focal loss来更好的训练网络。

## 3 CornerNet

### 3.1 Overview

 在CoenerNet中，检测目标的一对关键点：边界框的左上角和右下角。一个卷积网络预测两个热图集(heatmap)，热图集用来描述不同类别目标的角点的位置，一个是为左上角点，另一个是为右下角点。同时，该网络为每个角点预测一个embeding向量，以使同一目标的两个角点之间，embeding最小。为了使预测出的边界框更加紧凑，该网络还预测一个偏移量(offset)，对角点位置做轻微调整。最后，使用一个单一后处理算法，来获得最后的边界框。

![image-20201223153130452](https://gitee.com/changyv/md-pic/raw/master/20201223153132.png)

图4显示了CornerNet结构。作者使用了一个沙漏网络(hourglass network, 2016)作为CornerNet的骨干网，随后，紧跟着两个预测模块。一个模块处理左上角点，另一个模块处理右下角点。每个模块都有自己的corner pooling来池化特征，特征是来自于沙漏网络，特征池化在预测热图、embeding向量和偏移之前。不同于其它目标检测，作者没有使用不同尺度的特征来检测不同尺度的目标，而是仅在沙漏网络之后使用两个相同的处理模块。

### 3.2 Detecting Corners

为左上角和右下角分别预测一个热图集。每个热图集有C个通道，C是类别数。大小为$H*W$ ，没有背景通道。每个通道都是一个二进制掩码，来表示同一类别的角点的位置。

对于每个corner，只有一个ground truth是positive location，其它location相对于该corner都是negative location。训练过程中，对正样本半径以内的负样本，减少对它们的惩罚，因为接近真值的样本的box仍然可能与ground-truth box有重叠。如图5。

![image-20201223211324511](https://gitee.com/changyv/md-pic/raw/master/20201223211326.png)给定一个半径，penalty减少量由非正则化2D高斯函数决定：$e^-\frac{x^2+y^2}{2\sigma ^2}$ , 其中$\sigma$为半径的1/3。

令$p_{cij}$ 表示预测出的heat map中location$(i,j)$属于类别c的置信度得分，令$y_{ij}$ 表示使用非正规化高斯分布增强的真值heat map(ground truth heatmap)。于是，设计出focal loss的变体==损失函数==：
$$
L_{det}=\frac{-1}{N}\sum_{c=1}^{C}\sum_{i=1}^{H}\sum_{j=1}^{W}
\begin{cases}
(1-p_{cij})^\alpha log(p_{cij})  &&if&y_{cij}=1\\
(1-y_{cij})^\beta(p_{cij})^\alpha log(1-p_{cij})  &&o.w\\

\end{cases}
\tag{1}
$$

> 其中N是图片中的目标总数，$\alpha$和$\beta$ 是超参数，控制点的分布，实验中，作者分别设定为2和4。
>
> $y_{cij}$ 表示编码的高斯凸点，$(1-y_{cij})$ 表示减少ground truth locations周围点的的惩罚。原文：With the Gaussian bumps encoded in $y_{cij}$ , the $(1-y_{cij})$ term reduces the penalty around the ground truth locations.

大多数网络使用下采样层收集全局信息并减少内存占用，将其应用在卷积中，输出会小于原图。因此，图像中的$(i,j)$映射到heat map中为$(\lfloor \frac{x}{n}\rfloor,\lfloor \frac{y}{n})$ ，其中n为下采样因子。当从heat map映射回输入图像时，会损失一些精度，进而影响小边界框的IoU，为了解决这个问题，作者预测了一个偏移量(offset)，用于轻微调整角点位置：
$$
o_k=(\frac{x_k}{n}-\lfloor \frac{x}{n} \rfloor,\frac{y_k}{n}-\lfloor \frac{y}{n} \rfloor)
\tag{2}
$$

> $o_k$ 表示偏移，$(x_k,y_k)$为角点K的坐标(x,y)的预测坐标。

为所有类别预测两个offset集合，该集合分别由左上角点和右下角点共享。训练损失函数使用平滑L1损失(smooth L1 Loss)：
$$
L_{off}=\frac{1}{N}\sum_{k=1}^{N}SmoothL1Loss(o_k,\hat o_k)
\tag{3}
$$

### 3.3 Grouping Corners 

一张图片可能有多个目标，因此会检测出多个角点。所以需要确定那对角点是属于同一个目标，受Newell的multi-person pose estimation的启发：检测所有的人体关键点，并为每个关键点生成各自的embeding，然后基于embeding之间的距离分组。作者将其应用在网络中：网络为每个检测出的角点分别预测一个==embeding向量==。属于同一边界框的两个角点之间embeding应达到最小，这样就可以基于embeding距离实现分组。embeding的实际值并不重要，仅计算embeding之间的距离来完成分组。

与Newell相同，使用1维embeding。令$e_{tk}$表示目标k的top-left角点的embeding，令$e_{bk}$ 表示目标k的bottom-right角点的embeding。使用pill损失训练网络分组角点，使用push损失划分角点：
$$
L_{pull}=\frac{1}{N}\sum_{k=1}^{N}[(e_{tk}-e_k)^2+(e_{bk}-e_k)^2]
\tag{4}
$$

$$
L_{push}=\frac{1}{N(N-1)}\sum_{k=1}^{N}\sum_{j=1,j\neq k}^{N}\max(0,\Delta-|e_{k}-e_j|)
\tag{5}
$$

其中，$e_k$ 为$e_{tk}$ 和$e_{bk}$ 的平均值，实验中，作者令$\Delta=1$。与offset loss 相同，仅在ground-truth corner location上使用损失。

### 3.4 Corner Pooling

通过编码一些已知的先验信息，corner pooling能更好的定位到角点。

假设需要确定像素位置(i,j)是否为左上角点。令$f_t$ 和$f_l$ 分别表示$H*W$特征图，这将是top-left的corner pooling层的输入。令$f_{tij}$ 和$f_{lij}$ 分别为特征图中的位置(i,j)的向量。corner pooling层首先max-pooling特征图ft中所有(i,j)到(i,h)之间的特征向量，生成特征向量$t_{ij}$ ，max-pooling特征图fl中所有(i,j)到(W,j)之间的特征向量，生成特征向量$l_{ij}$ ，最后，将两个特征向量$t_{ij}$和$l_{ij}$ 相加。该计算可以表示为：
$$
t_{ij}=
\begin{cases}
max(f_{tij},t_{(i+1)j}),   & if\space i<H\\
f_{tHj},  &o.w
\end{cases}
\tag{6}
$$

$$
l_{ij}=
\begin{cases}
max(f_{lij},l_{i(j+1)}),   & if\space i<H\\
f_{liW},  &o.w
\end{cases}
\tag{7}
$$

通过取最大，$t_{ij}$和$l_{ij}$ 可通过动态规划程序有效计算出来。以同样的方式，定义右下角的corner pooling层。过程如下图。

![image-20201224161901238](https://gitee.com/changyv/md-pic/raw/master/20201224161910.png)

预测模型的结构如下图所示。模型的第一部分是修改过的残差模块，在该残差模块中，将$3*3$ 的卷积模块替换为corner pooling模块，该模块使用2个128通道的$3*3$卷积模块处理来自骨干网络的特征并应用于corner 池化层。残差模块之后，还跟一个通道数为256的$3*3$的conv-BN层，和3 Conv-ReLU-Conv层来生成heatmaps，embeding和offset。==第6页==

![image-20201224162046569](https://gitee.com/changyv/md-pic/raw/master/20201224162048.png)

### 3.5 Hourglass Network

CornerNet使用沙漏(hourglass)网络作为骨干网。作者使用2个hourglass，并做了一些修改。第一个沙漏网络的输入和输出各采用了1个Conv-BN模块，随后通过元素加和(element-wise addition)将它们合并，随后就是ReLU和残差模块，该残差模块有256个通道并用于第二个沙漏模块的输入。沙漏网络的深度是104，使用整个网络最后一层的特征来做预测。

## 4 Experiments

作者在pytorch上应用CornerNet，使用pytorch默认设置随机初始化网络，不适用任何预训练。训练过程中，网络输入大小为$511*511$，输出大小为$128*128$。为防止过拟合，使用了标准数据增强技术，包括随机水平翻转，随机采样，随机剪裁和随机颜色抖动jin（包括调整光线、饱和度）。==并在输入图片上应用PCA==。

使用Adam来优化full训练损失：
$$
L=L_{det}+\alpha L_{PULL}+\beta L_{push}+\gamma L_{off}
\tag{8}
$$
其中，$\alpha$ ，$\beta$和$\gamma$ 为pull, push, offset各自的权重。作者设置$\alpha=0.1，\beta=0.1 ,\gamma=1$。因为更高的$\alpha，\beta$ 网络表现并不好。

### 4.2 Testing Details

训练过程中，使用一个后处理算法从heatmap、embedding和offset生成边界框。==在角点heatmap上使用一个$3*3$的最大池化层来应用非极大值抑制(non-maximal suppression,NMS)== ，然后从heatmap中选出100个top-left角点和bottom-right角点。每对角点通过相关的offset进行调整。计算两个角点(top-left和bottom-right)之间embeding的==L1距离==，距离超过0.5或包含有不同类的角点对将被抛弃。角点对的平均得分将作为检测得分。

作者保持输入图片的原始分辨率，在输入到CornerNet网络之前，使用0填充。

### 4.3 MS COCO

作者在MS COCO上评估CornerNet，80K图像作为训练集，40K作为验证集，20K作为测试集。

<img src="https://gitee.com/changyv/md-pic/raw/master/20201224173159.png" alt="image-20201224173157772">

### 4.4 ABlation

作者证明了corner pooling的有效性，见下表。

![image-20201224174001110](https://gitee.com/changyv/md-pic/raw/master/20201224174003.png)

并且能有效提升检测精度，尤其对中大型物体，正确率分别提高了2.4%和3.6%，见下表1。

![image-20201224173829582](https://gitee.com/changyv/md-pic/raw/master/20201224173831.png)

之后作者证明了corner pooling大区域的稳定性，减少对半径内负样本惩罚，沙漏网络的有效性，与其他网络的比较。

## 5 Conclusion

本文提出一种新的检测方法：CornerNet，并在MS COCO上做了评估。



以上。

> 部分暂时未能弄明白，相关知识积累不够，暂留白。