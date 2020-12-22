---
layout: post
title: Focal Loss
date: 2020-12-16
Author: JIUGE 
tags: [论文阅读]
comments: true
toc: false
---

本文介绍一种新的损失函数：focal loss，通过重构标准交叉熵损失函数和降低易分类样本的损失权重，解决单阶段检测器类不均衡问题。

<!-- more -->

# [论文简读] Focal Loss for Dense Object Detection

论文信息：

![image-20201215221136821](https://gitee.com/changyv/md-pic/raw/master/20201215221146.png)

论文链接：https://arxiv.org/abs/1708.02002

## Abstract

基于R-cnn推广的双阶段检测器有很高精度，其在condidate object location稀疏集合上使用了分类器。相比之下，单阶段检测器在possible object location上进行常规的密集采样，但精度比不过两阶段检测器，在论文中，作者发现：训练期间，foreground-bacground class的极度不均衡是阻碍精度的主要原因。因此，作者提出通过重构(reshape)标准交叉熵损失函数(standard cross entropy loss)，降低易分类样本（well-classified examples）的损失权重，来解决类不均衡问题。作者提出的Focal loss 专注于训练难分样本（hard-examples），并防止了大量的易分样本压倒（overwhelming）检测器。为了评估focal loss的有效性，设计并训练了一个简单密集检测器RetinaNet。使用focal loss训练的RetinaNet达到了先前单阶段检测器的速度，并在精度上超过了目前所有的双阶段检测器（该论文发表于2018年）。

## Introduction

当前最优的双阶段检测器都是基于区域推荐机制(proposal-driven mechanism)。像R-CNN框架一样，第一阶段产生候选定位（candidate object locations），第二阶段，使用卷积神经网络将它们分类为前景和背景。通过一些改进，这种双阶段框架在COCO上一直占据着top精度。

由此引出一个问题：单阶段检测器能实现同样的精度？单阶段检测器用于对物体位置（object location）、大小和宽高比进行常规、密集的采样。最近的单阶段检测器如YOLO,SSD，最快检测速度下，精度已经达到了最优双阶段检测器精度的10-40%。

本文提出的单阶段检测器，在COCO上第一次达到了两阶段检测器所达到的最优精度。这些两阶段检测器大都复杂，如特征金字塔网络（FPN）或faster rcnn的变体Mask R-CNN。本文将训练时的类不平衡问题作为阻碍单阶段检测器精度的主要障碍，提出了一个新的损失函数来消除这个障碍，最终实现实现了该成果。

两阶段检测器如R-CNN，常使用级联（cascade）和启发式采样（heuristic sample）来解决类不均衡问题。在一阶段的proposal阶段，使用如选择性搜索，edgeboses，deepmask,RPN，能快速缩小候选对象（candidate object location）的数量（一般是1-2K个），并过滤掉大多数的背景样本（background samples）。在二阶段的分类阶段，使用启发式采样、划分foreground-background比例（如1：3）、online hard examnple mining（OHEM），来保持前景与背景的平衡。

与之相比，单阶段检测器必须处理大量的直接从图像采样来的candidate object location。在实践中，这些拥有不同大小，比例，位置的candidate location通常会达到100K个。虽然可以使用启发式采样，但训练效率低下，因为训练会被简单易分样本所主导，这种分类问题通常通过bootstrapping或hard example mining技术来解决。

本文提出了一种新的损失函数，能有效替代这些上述这些处理类不均衡的方法。损失函数是动态规模交叉熵损失(dynamically scaled cross entropy loss)，随着类置信度的增加，规模因子(scaling factor)将退化为0，如图1所示。这些规模因子可以自动降低易分类样本的损失贡献，并使训练快速专注于难样本。实验表明，使用focal loss可以训练出一个高准确率的单阶段检测器，它显著优于当前的基于启发式采样或困难样本挖掘的单阶段检测器。最后，focal loss并不是最重要的，其它实列也可以实现相似的效果。

![image-20201216141617417](https://gitee.com/changyv/md-pic/raw/master/20201216141626.png)

为了证明本文提出的focal loss的有效性，我们设计了一个单阶段检测器RetinaNet，它结合了特征金字塔，并使用了anchor boxes。基于ResNet-101-FPN骨干网的RetinaNet模型，在5fps下，于COCO test-dev上达到了39.1的AP。超过了先前的单阶段检测器，见图2。



![image-20201216141655778](https://gitee.com/changyv/md-pic/raw/master/20201216141657.png)

## 3 Focal Loss

focal loss用来解决单阶段目标检测场景下foreground和background类之间的极度不均衡问题，比如1：1000。首先从二分类问题的CE（cross entropy，交叉熵）开始：
$$
CE(p,y)=
\begin{cases}
-log(p)   &  y=1\\
-log(1-p) &  o.w.
\end{cases}
\tag{1}
$$
上式中$y{\in}\{\pm 1\}$ 表示类的真值(ground-truth)，$p\in[0,1]$ 是模型对于类的label为1的概率的预测值。为了表示简便，我们使用$p_t$ :
$$
p_t=
\begin{cases}
p &y=1\\
1-p &o.w.
\end{cases}
\tag{2}
$$
重写$CE(p,y)=CE(p_t)=-log(p_t)$ 。

CE loss 对应图1中的蓝色曲线，当简单易分类样本的数量足够多时，这些损失值将掩盖稀有类(rare class)。 

### 3.1 Balanced cross entropy

解决类不均衡的普遍方法是引入权重因子$\alpha \in [0,1]$ (对应类别为1)和$1-\alpha$ (对应类别为-1)。我们使用类似方式定义CE Loss：
$$
CE(p_t)=-\alpha log(p_t)
\tag{3}
$$


上式就是focal loss 的baseline。

### 3.2 focal loss definition

分类器训练过程中类别不均衡问题将淹没(overwhelms)交叉熵损失函数，易分负样本占据了大量损失并主导了梯度。尽管$\alpha$ 平衡了正负样本的重要性，但却不能有效区分简单样本和难分样本。于是，作者重构损失函数，降低了简单样本的权重，并使训练专注于难分负样本。

为交叉熵损失函数增加一个调制因子(modulating factor)：$(1- p_t)^{\gamma}$  ，于是focal loss定义为：
$$
FL(p_t)=-(1-p_t)^{\gamma}log(p_t)
\tag{4}
$$
图1显示了不同$\gamma \in [0,5]$ 的foca loss的可视化。可以看出，focal loss具有两个性质：

1. 当样本分类有误，$p_t$很小时，调制因子趋于1且对损失无影响。随着$p_t \rightarrow 1$ ，调制因子趋于0，易分类样本损失权重下降。
2. 参数$\gamma$ 调整易分样本的损失权重。当$\gamma=0$ 时，FL等价于CE，随着$\gamma$ 的增加，调制因子的影响变大，实验中，发现$\gamma=2$ 时效果最好。

调制因子减少了简单样本对损失的贡献，并拓展了对损失贡献小的易分样本的范围。

> 对应图1，拓展易分样本的范围，表现在曲线右端变得更加平滑。

实验中，作者使用了一个平衡变量$\alpha$ ：
$$
FL(p_t)=-\alpha(1-p_t)^\gamma log(p_t)
\tag{5}
$$
在实现中，作者发现，将sigmoid运算和loss计算结合起来的损失层，有更好的数值稳定性。并且，作者说明这种形式不是最重要的，其它实例的focal loss 也有同样的效果。

### 3.3 class imblance and model initialization

二分类模型默认初始化y=-1和y=1为相同概率，如果使用这种初始化方式，会造成训练的不稳定，为了解决这个问题，作者引入一个概念：prior，即训练开始前，模型为稀有类(rare class，foreground)估计一个p值。

作者令$\pi$ 表示prior，并使评估模型稀有样本的p值足够小，比如0.01。我们发现这能提高交叉熵损失和focal loss在严重类不均衡下的训练稳定性。

### 3.4 class imbalance and two-stage detectors

  两阶段检测器通常使用交叉熵损失来训练，并使用以下机制处理类不均衡问题：

1. 一个两级级联(two-stage cascade)
2. 偏差小批量采样(biased minibatch sampling)

第一阶段的级联就是object proposal机制，能将早期可能的目标区域(possible object location)减少到1-2K个。区域选择并不是随机的，而是选择与实际目标区域相关的区域，这将移去大量的易分负样本。第二阶段的偏差采样常用于构造minibatches，比如1：3比例的正负样本，这里的比例就像隐含的$\alpha$ 平衡因子。本文提出的focal loss就是在单阶段检测系统中，直接通过损失函数来解决实现以上这些机制的效果。

## 4 retinaNet detector 

RetinaNet由骨干网和两个子网组成。骨干网负责计算整个输入图像的卷积特征图，是一个off-self的卷积网络。第一个子网进行卷积目标(convolutional object)分类；第二个子网进行边界框(bounding box)回归。如下图3：

![](https://gitee.com/changyv/md-pic/raw/master/20201216201518.png)

> RetinaNet在前馈型ResNet结构(a)之上，使用了一个特征金字塔网络作为骨干网，来构造出丰富的、多尺度的卷积特征金字塔(b)。并引入两个子网络，一个用来分类anchor boxes (c)，一个用来从anchor boxes中回归ground-truth object boxes(d)。 
>
> 金字塔的每个层级可以被用来检测不同尺度的目标。

**Feature Pyramid Network Backbone: **  

RetinaNet骨干网使用了特征金字塔(FPN)，FPN使用一个自顶向下的路径和侧连接，增强了一个标准卷积网，使网络能有效构造出丰富的、多尺度的金字塔特征。金字塔的每个层级可以被用来检测不同尺度的目标。

作者在残差网络(ResNet)上建立了一个特征金字塔网络(FPN)。FPN能够提升全卷积网络(FCN)的多尺度预测，于是，我们在ResNet结构上建立了FPN。我们构造了P3到P7层的金字塔，每层有C=256个通道(channel)。

接下来，就是关于网络的具体结构：Anchors、classification subset、box regression subset，推论和训练。

## 5 Experiments

![image-20201219144159736](https://gitee.com/changyv/md-pic/raw/master/20201219144203.png)

在COCO数据集上评估了RetinaNet模型，并与当前最优的单双阶段检测器比较test-dev结果，如上表2。和单阶段相比，我们的方法增加了5.9的AP。和双阶段方法相比，比Faster-RCNN高了2.3个点。

## 6 Conclusion

本文证明了类不均衡问题是提升单精度检测精度的最大障碍，针对该问题，提出了focal loss，在交叉熵损失上引入调制因子，使专注于难分样本。设计了一个全卷积单阶段检测器，通过实验分析，表明该网络拥有目前最优的精度和速度。

源码：https://github.com/facebookresearch/Detectron

以上。