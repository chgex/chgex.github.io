---
layout: post
title: Batch Normalization
date: 2021-03-25
Author: JIUGE 
tags: [论文阅读]
comments: true
toc: false
---

这篇论文发表于2015年，作者提出了一种批量归一化方法(Batch Normalization)，将其加入到模型中，能显著加速模型训练。
<!-- more -->

# 论文简读：Batch Normalization: Accelerating Deep Network Training by Reducing

论文信息：

![image-20210325204615792](https://gitee.com/changyv/md-pic/raw/master/20210325204617.png)

论文链接：https://arxiv.org/abs/1502.03167

发表时间：2015年，于ICML

> Internal Covariate Shift， 内部协变量偏移。

## Abstract

训练深层神经网络的困难在于：训练过程中，输入数据的分布会随着前一层参数的变化而发生改变。作者将这一现象叫做内部协变量偏移（ internal covariate shift）。因此，一般都会调低学习率，并“小心的”初始化模型参数，尤其在训练饱和线性模型(models with saturating nonlinearties)时。

作者提出了一种批量归一化方法(Batch Normalization)，将其作为模型的一部分，并为每个小批量执行该归一化方法。从而不必担心学习率和参数初始化的问题，甚至可以去掉模型的Dropout层。

作者在当前（2015年）最优的图像分类模型上，加入了该批量归一化方法，使模型在ImageNet上的测试错误率下降到了4.82%。

## 1 Introduction

随机梯度下降(Stochastic gradient
descent , SGD)是一种训练模型的有效方法，它通过以下方式来优化模型参数：
$$
\theta=arg\;\min_{\theta}\frac{1}{N}\sum_{i=1}^{N}\ell(x_i,\theta)
$$

> 其中，$x_{1...N}$ 表示训练集，
>
> $\theta$ 表示网络的参数。

训练过程中，每一次小批量(mini-batch)使用SGD更新一次参数。

作者指出，虽然SGD效果很好，但它容易受模型超参数的影响，尤其是学习率和初始化的模型参数。随着模型的深入，对输入的影响将被放大：因为，当传递到下一层的输入具有不同的分布时，下一层的参数则需要去适应这个新的分布，即发生了协变量偏移(covariate shift)。

作者举了一个子网络的例子来说明：固定子网输入的分布，会产生积极的影响。
$$
\ell=F_2(F_1(u,\theta_1),\theta_2)
$$

> F1和F2为转换函数，
>
> 参数$\theta$ 是由最小化损失 $\ell$ 而学到的。

上式可以写为：
$$
\ell = F_2(x,\theta_2)\\
in\; that:\; x=F_1(u,\theta_1)
$$
于是，梯度下降为：
$$
\theta_2 \leftarrow \theta_2 -\frac{\alpha}{m}\sum_{i=1}^{m}
\frac{\partial F_2(x_i,\theta_2)}{\partial \theta_2}
$$

> 式子中，$\alpha$ 是学习率，$m$为小批量样本个数。

可见，$F_2$ 与输入$x_i$ 具有相关性。因此，维持输入x的分布不变，对模型的泛化能力有一定帮助。

作者还说明，固定输入的分布，对于子网之外的网络层，也有积极影响：

假设外层子网是一个拥有sigmoid激活函数的层，即：
$$
z=g(Wu+b)\\
in\; that: g(x)=\frac{1}{1+exp(-x)}
$$

> u为子网的输入，
>
> W和b分别是权重矩阵和偏差向量，它们通过训练学习得到，

当$|x|$ 很大时，$g'(x)$ 趋向于0，所以在$x=Wu+b$ 中，梯度下降之后，只有绝对值较小的$|u|$ 会被保留，而其它维度可能会梯度消失，同时，这也将导致模型的训练时间变长。

在实践中，对于饱和度问题和梯度消失问题，通常有以下解决方式：

+ 使用ReLU(Rectified Linear Units, 修正线性单元)激活函数，其中$ReLU(x)=max(x,0)$

+ 模型参数初始化，
+ 较小的学习率。

作者指出，如果能使非线性输入的分布在网络训练过程中的保持稳定，则优化器将不太可能陷入饱和状态。于是，作者提出了批量正则化方法。

作者在当时最好的图像分类器上应用了批量正则化方法，将网络训练到相同精确度，所需要的迭代次数，只有先前的7%左右。

## 2  Towards Reducing Internal Covariate Shift

作者将训练过程中，由于网络参数改变导致的分布变化，定义为内部协变量偏移（Internal Covariate Shift）。

接下来，作者就减少内部协变量偏移做了一些推导。详见原文。

## 3 Normalization via Mini-Batch Statistics

作者对批量归一化做了推导。

对于一个接受d维输入$x=(x^{(1)},x^{(2)},...,x^{(d)})$的网络层，通过以下式子对每个维度做归一化：
$$
\hat{x}^{k}=\frac{x^{(k)} -E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}
$$
为了避免输入转换之后，改变其线性范围，（sigmoid的输出是线性的，不能在转换之后，变成非线性的），引入两个变量：$\gamma^{k}$ ，$\beta^{k}$ ，于是变换结果y为：
$$
y^{k}=\gamma^{(k)}\hat{x}^{(k)}+\beta^{(k)}
$$
这两个变量也将通过学习得到。其中：
$$
\gamma^{k}=\sqrt{Var[x^{k}]}\\
\beta^{(k)}=E[x^{(k)}]
$$
假设小批量中共有m个样本，即：
$$
\mathcal{B}=\{x_1,x_2,...,x_m\}
$$
经过正则化处理，得到输变换结果$y_i$：
$$
BN_{\gamma,\beta}:x_1,...,x_m \rightarrow y_1,...,y_m
$$
在小批量中使用BN，过程见下图：

<img src="https://gitee.com/changyv/md-pic/raw/master/20210325195849.png" alt="image-20210325195839473" style="zoom:55%;" />

根据链式求导法则，正则化转换之后，反传梯度，更新参数，计算如下所示：

<img src="https://gitee.com/changyv/md-pic/raw/master/20210325200152.png" alt="image-20210325200150513" style="zoom:55%;" />

使用BN，能加速网络的训练。

在整个网络中，应用BN，过程如下：

<img src="https://gitee.com/changyv/md-pic/raw/master/20210325200549.png" alt="image-20210325200547488" style="zoom:50%;" />

即使学习率较高，训练使用批量正则化的网络，也能有不错的表现。

## 4 Experiments

**Activations over time**

作者为了验证训练中内部协变量偏移的影响，基于MNIST手写数字识别数据集，在LeNet为了做了验证。测试集上的正确率变化如下图：

<img src="https://gitee.com/changyv/md-pic/raw/master/20210325201132.png" alt="image-20210325201130930" style="zoom:50%;" />

**ImageNet classification **

作者在Inception网络（2014年提出）上使用了批量归一化，在ImageNet分类任务数据集上进行训练。

**accelerating BN networks**

随后，作者做了分别以下修改：

+ 增加学习率。对网络训练基本没影响。

+ 去掉网络的dropout层。在验证集上的正确率有所提升。
+ 打乱训练样本。在验证集上的正确率提升了1%。
+ 减少L2权重正则化，该参数被用来控制过拟合。
+ 加速学习率衰减。

**Single-Networks classification**

接下来，作者在LSVRC2012数据集上训练了以下网络：

+ Incaption, 学习率为0.0015
+ BN-baseline, 同Inception网络，但使用了批量正则化。
+ BN-x5, 使用了批量正则化的Inception网络，但学习率为0.0075。
+ BN-x30，同BN-x5，但学习率为0.045。
+ BN-x5-sigmoid，同BN-x5，但使用sigmoid非线性激活函数代替修正线性单元激活函数。

结果如下图所示：

<img src="https://gitee.com/changyv/md-pic/raw/master/20210325203118.png" alt="image-20210325203117367" style="zoom:50%;" />

## 5 Conclusion

+ 提出批量正则化，加速网络训练。
+ 将BN应用到当前最优的图像分类模型中，网络训练速度显著提升。
