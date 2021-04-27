---
layout: post
title: Faster R-CNN
date: 2021-04-12
Author: liu_bai 
tags: [论文阅读]
comments: true
toc: false
---

在Fast RCNN的基础上，提出了区域推荐网络RPN，使推荐区域的选择，基于图像本身的信息。通过训练微调，RPN和fast rcnn检测器组成的网络，实现了端到端的训练。

<!--more -->

# Faster RCNN

论文信息：

<img src="https://gitee.com/changyv/md-pic/raw/master/20210412112513.png" alt="image-20210412112503193" style="zoom:50%;" />

[论文链接](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)

发表时间：2015年，于NIPS

## Abstract

提出一种区域推荐网络RPN( region proposal network, RPN)，关于FPN：

+ RPN共享全图的卷积特征图，用于产生推荐区域。
+ RPN是全卷积网络，给出推荐区域的同时，也会给出置信度，
+ RPN使用端到端的方式进行训练。

作者将RPN加入到Fast RCNN网络的特征图共享层——通过注意力机制(attention mechanisms),：RPN告诉检测网络，那些区域应该被注意。

## 1 Introduction

相比于Fast RCNN使用选择性搜索算法产生区域推荐，Faster RCNN利用深度卷积神经网络，通过训练，基于图像本身数据信息，产生出推荐区域。

作者观察发现，fast rcnn中用于目标检测的卷积特征图，可以被用来产生推荐区域。于是设计出平行网络RPN。

相比于使用图像金字塔(pyramids of images)和金字塔过滤器(pyramids of filters)，作者提出"anchor box"。anchor机制使RPN可以预测不同尺寸和纵横比的推荐区域。如下图：

<img src="https://gitee.com/changyv/md-pic/raw/master/20210412143220.png" alt="image-20210412143210986" style="zoom:50%;" />



## 3 Faster R-CNN

Faster RCNN由两个模块组成：第一个模块是深层全卷积网络，产生区域推荐，第二个模块是fast rcnn检测器，使用第一个模块产生的区域推荐。网络结构图如下：

<img src="https://gitee.com/changyv/md-pic/raw/master/20210412143344.png" alt="image-20210412143342595" style="zoom:50%;" />

接下来的3.1和3.2两个小节，作者分别介绍了RPN是如何设计，苏荷使用特征共享训练这两个模块。

**3.1 RPN**

RPN接受图像作为输入，输出矩形框（即推荐区域）及其得分（该分数与目标类别无关）的集合。

为了产生推荐区域，在特征图上滑动窗口：n\*n大小，论文中n=3。滑动窗口得到的特征图将被映射为低维特征图：ZF网络中是256维度，VGG网络中是512维。随后使用RELU。该低维特征将被做为两个平行子网的输入：边界框回归网络reg和边界框分类网络cls。这个mini网络的结构如下图左：

<img src="https://gitee.com/changyv/md-pic/raw/master/20210412143540.png" alt="image-20210412143538304" style="zoom:50%;" />

在每个滑动窗口位置，同时预测多个推荐区域，最多有K个推荐区域。于是reg子网输出4k，cls子网输出2k。这K个同一位置不同尺寸的边界框叫做anchor。见上图右。

论文中，作者使用3个不同的尺寸和3个不同的纵横比，所以每个滑窗的K=9。对于W\*H大小的卷积特征图，一共有W\*H\*K个anchors。

RPN网络参数(VGG)：512\* (2K+4k)个。

**损失函数**

为了训练RPN，作者为每个anchor，设计了二类别标签集（是目标或不是目标，分别对应0和1）。正标签anchor有两类：

​	i. 与ground truth box有高IOU，

​	ii.与任何ground-truth box有0.7以上的IoU。

IoU低于0.3的anchor将被设置为负样本。

针对于一张图像，Faster RCNN的损失函数定义如下：
$$
L(p_i,t_i)=\frac{1}{N_{cls}}\sum_iL_{cls}(p_i,p_i^*) \\+ \lambda \frac{1}{N_{reg}}\sum_i p_i^*L_{reg}(t_i,t_i^*)
\tag{1}
$$

> i为anchor序号；$p_i$ 为第i个anchor的目标置信度；ground-truth $p_i^*$ 取值为0或1（anchor是正样本，则为1）;
>
> $t_i$ 是一个四维向量，描述预测边界框的四个坐标。$t_i^*$ 是正样本anchor的ground truth 边界框。

$L_{cls}$ 是log损失：两类别object or not object;

 $L_{reg}(t_i,t_i^*)=R(t_i-t_i^*)$ ，其中R是smooth L1。 

$ p_i^*L_{reg}$表示回归损失只有当positive anchor的时候才会被激活。

 $\frac{1}{N_{cls}}$和$\frac{1}{N_{reg}}$ 起到正则化的作用：cls使用mini-batch size（$N_{cls}=256$），reg使用anchor location的个数（$N_{reg} \thickapprox 2400$） 。

边界框回归使用的坐标参数，使用以下变换：

<img src="https://gitee.com/changyv/md-pic/raw/master/20210412144238.png" alt="image-20210412144216806" style="zoom:50%;" />

**Training RPNs**

RPN实现了端到端的方式训练：通过反向传播和随机梯度下降。

anchor采样：来自同一图像的mini-batch包含多个正负anchor，为避免负anchor主导损失，一幅图像中随机采样256个anchor，正负占比1:1，如果正anchor数量不够128个，则是哟个负anchor填充。

参数初始化：使用均值为0，方差为0.01的高斯分布。

**3.2 Sharing features for rpn and fast rcnn**

作者使用以下训练方式，使RPN和fast rcnn共享卷积层特征图：

​	1. 训练RPN，使用ImageNet预训练模型进行初始化。RPN用于产生推荐区域。

 2. 使用第1步中产生的推荐区域，训练微调后的fast rcnn。同样使用ImageNet预训练模型进行初始化。

    > 以上两步是独立进行的。即前两步中的网络是单独训练的。

 3. 使用第2步训练后的检测器网络，来初始化RPN的训练。

    > 第3步的过程是迭代式的。但作者发现提升可以忽略不记。
    >
    > 第3步，两个网络共享卷积层输出的特征图。
    
 4. 调整共享卷积层。

3.3 Implementation setails

作者给出的应用细节：

+ re-scale图像，将短边提高到600像素，
+ 多尺度的特征提取可以增加精度，但无法做到与速度的权衡，
+ 无论是ZFNet还是VGG，最后一个卷积层的步长都为16像素。

对于anchors，

+ 使用3种不同尺寸的anchors：128\*128, 256\*256, 512\*512，
+ 和三种不同的纵横比：1:1, 1:2, 2:1.
+ 训练种，忽略掉了cross-boundary, 以为这些boundary对于loss作用不大，

对于proposal，

+ 根据cls的得分，使用NMS
+ 非极大值抑制中IoU阈值为0.7

## 4 Experiments

faster rcnn在当时，在多个数据集上取得了SOTA的mAP。

## Reference

[1]	https://blog.csdn.net/happyday_d/article/details/85870358

[2]	code https://github.com/chenyuntc/simple-faster-rcnn-pytorch

[3]	code https://www.cnblogs.com/kerwins-AC/p/9752679.html



