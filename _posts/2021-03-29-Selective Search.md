---
layout: post
title: Selective Search for Object Regression
date: 2021-03-29
Author: liu_bai 
tags: [论文阅读]
comments: true
toc: false
---

这篇论文发表于2014年，提出了一种基于分割的选择性搜索算法，用定位出可能的目标区域。

<!--  more -->

# 论文简读：Selective Search for Object Regression

论文信息：

<img src="D:%5Cblog%5Cphoto%5Cimage-20210329220335066.png" alt="image-20210329220335066" style="zoom:80%;" />

发表时间：2014年，于JPR(Journal of Pattern Recognition, JPR)

## Abstract

论文围绕目标识别中生成可能目标的位置（possible object location）问题，提出一种选择性搜索(select search)方法，该方法结合穷举搜索(exhaustive search)和图像分割(segmentation)的优点：

+ 方法目标和穷举搜索一样，在于捕捉所有可能存在目标的位置。
+ 该方法依赖于图像图像结构，进行采样操作。

## 1 Introduction

作者以下列图a为例，说明图片中的物体，是具有层次性的。所以在目标检测任务中，多尺寸的分割是很有必要的。

又以图b,c,d为例，说明仅使用单一策略，是无法做到分割的。

<img src="D:%5Cblog%5Cphoto%5Cimage-20210326152836628.png" alt="image-20210326152836628" style="zoom:75%;" />

图中：

+ 猫可以使用颜色区分，但它们具有相同的纹理（texture）。
+ 变色龙和周围叶子颜色相同，但可以使用纹理区分。
+ 轮胎和车具有不同的颜色和纹理，但它们是整体。

所以，仅使用单个视觉特征，无法解决分割的模糊性。

穷举搜索（exhaustive select）具有以下缺点：

+ 它旨在穷举所有可能的目标位置，但这不具有计算可行型，
+ 需要使用精心设计的网格尺寸和丛横比（aspect ratios），才能降低搜索的空间大小，
+ 这种固定尺寸的搜索方式，将产生非常多的box，其中的大部分box是没用的，

作者提出的选择性搜索，旨在产生独立于类别的、数据驱动的、高质量的目标位置。

选择性方法应用于目标识别领域，作者在Pascal VOC数据集上测试该方法的性能，该数据集有20个类别，使用bounding box的方式进行评价。

## 2 Related Work

**Exhaustive Search**

图像中的目标，可能以任意尺寸出现在任意位置。如果搜索所有位置，将花费巨大的计算成本。因此，一般使用移动滑窗的搜索方式：移动滑窗技术使用粗糙的搜索网格和固定的纵横比。常用于级联分类器的预处理步骤。

与移动滑窗技术相关的是基于局部的目标定位（part-based object localisation）方法。该方法使用线性SVM和HOG特征，来执行穷举搜索。

分支定界技术（branch and bound technique, Lampert et al. ），该方法使用外观模型来指导搜索，在图像中直接搜索出最佳窗口。虽然该方法减少了location的数量，但平均每张图像会产生10,0000个窗口。

不同于上述两个方法，选择性搜索方法使用潜在的图像结构来产生物体的可能位置。

**Segmentation**

分割（segmentation）方法，也可以得到可能的目标位置。该方法生成多个前景/背景分割，通过学习，从前景分割中预测出可能有目标的分割。这些分割方法依赖于区域识别算法：通过随机初始化，获得一系列前景对象位置和背景对象位置。但这种方法得到的目标位置质量不一。

**sampling strategies**

关于选择性搜索，可以归纳为以下几点：

+ 使用segmentation而不是exhaustive search来产生object location。
+ 选择性搜索算法使用多种策略处理候选location,
+ 使用自顶向下的处理方式，生成object location。

## 3 Select Search

选择性搜索算法基于以下方面：

+ 使用层级算法(hierarchical algorithm)，捕获所有大小的location，
+ 使用多种区域分组策略，
+ 更快的计算速度。

**3.1 Hirerarchical grouping**

区域要比像素产生更多的信息，所以作者通过使用基于区域（region-based）的特征，来得到小的初始的区域。算法处理初始区域的过程如下所示：

<img src="https://gitee.com/changyv/md-pic/raw/master/20210329105913.png" alt="image-20210329105904193" style="zoom:50%;" />

+ 该算法是一个贪心算法，
+ 通过计算所有两两相邻区域的相似度和合并度，考虑是否进行合并。
+ 从最小区域开始，直到扩展到整个图片为止。

**3.2 Diversification strategies**

作者使用多种策略来计算两个区域的相似度：

策略一：颜色空间增强。见下图。

<img src="https://gitee.com/changyv/md-pic/raw/master/20210329110821.png" alt="image-20210329110820053" style="zoom:65%;" />



策略二：使用多种区域相似度计算方法，并分别赋予计算方法相应的权重。

1. 计算颜色相似度。

$$
s_{colour}(r_i,r_j)=\sum_{k=1}^{n}min(c_j^k,c_j^k)
\tag{1}
$$

> 首先将色彩空间转为HSV：
>
> 每个颜色通道使用25个bin值，输入是彩色图像，有3个通道，所以一共有75个bin值。
>
> 区域$r_i$ 的颜色直方图$C_i=\{c_i^1,...,c_i^n\}$，其中，n=75。
>
> 颜色向量使用$L_1$ 范数进行正则化处理。

如果区域 $r_i$ 和 $r_j$ 合并为 $r_t$ ,则$r_t$ 的颜色向量直接通过以下方式计算：
$$
C_t=\frac{size(r_i) * C_i+ size(r_j) * C_j}{size(r_i) +size(r_j)}
\tag{2}
$$

> 新合并区域的颜色向量由原来的2个区域向量得到，而不必重新计算区域的bin值，所以具有简化计算的作用。

2. 计算纹理相似度（texture similarity）

$$
s_{texture}(r_i,r_j)=\sum_{k=1}^{n}min(t_i^k,t_j^k)
\tag{3}
$$

> 在一个颜色通道的8个不同方向上，使用$\sigma=1$ 的高斯导数。
>
> 每个方向使用10个bin值，最后得到区域$r_i$ 的纹理向量$T_i={t_i^1,...,t_i^n}$,  n=240。
>
> 同样使用$L_1$ 范数进行正则化处理。 

合并得到的新区域的纹理相似度，计算公式与上一个方法相同。

3. 计算尺寸相似度$s_{size}(r_i,r_j)$ 

$$
s_{size}(r_i,r_j)=1-\frac{size(r_i)+size(r_j)}{size(im)}
\tag{4}
$$

> 其中size(im)表示整个图片像素大小。
>
> $s_{size}$具有以下作用：
>
> + 促使小区域尽早合并，
>
> + 避免单一区域一个接一个的吞并相邻区域。

4. 计算$s_{fill}(r_i,r_j)$，用于衡量区域$r_i,r_j$ 合并的适合度。

$$
s_{fill}(r_i,r_j)=1-\frac{size(BB_{i,j})-size(r_i)-size(r_j)}{size(im)}

\tag{5}
$$

其中，$size(BB_{i,j})$ 表示将区域$r_i,r_j$ 包含在内的bounding box的大小。

作者将以上四种方式计算出来的相似度值，加权求和，得到最终相似度：
$$
s(r_i,r_j)=a_1s_{colour}(r_i,r_j)+a_2s_{texture}(r_i,r_j)+a_3s_{size}(r_i,r_j)+a_4s_{fill}(r_i,r_j)
\tag{6}
$$
**3.3 combining locations **

暂时跳过。

## 4 Object Recognition using selective search

目标识别领域有两种特征类型：定向梯度直方图(histograms of oriented gradients, HOG)和词袋(bag-of-words)。作者基于词袋进行物体识别，使用了更精细的空间金字塔划分。模型中使用SVM（支持向量机）进行分类。

训练过程见下图：

![image-20210329212139332](https://gitee.com/changyv/md-pic/raw/master/20210329212148.png)

初始正样本由所有真值物体窗（ground truth object window）组成。初始负样本由选择性搜素算法所产生的Location组成，这些location与正样本有20%-50%的重叠率。如果两个负样本重叠率超过了70%，则其中一个会被抛弃。

## 5 Evaluation

作者基于以下四个方面对选择性搜索算法进行了评估：

+ 多种相似度计算方法。

+ location的质量，
+ 目标识别。

为了评估选择性搜索算法产生的location的质量，作者定义了两个变量：Average Best Overlap, ADO 和 Mean Average Best Overlap,  MABO，即:

<img src="D:%5Cblog%5Cphoto%5Cimage-20210329213411662.png" alt="image-20210329213411662" style="zoom: 60%;" />

作者基于VOC 2007 测试集，比较了当时不同方法的召回率，MABO，window location数量。

## 6 Conclusion

+ 图像中，目标具有层次性，所以仅仅使用单一的分组算法是无法捕获所有可能的对象位置。

+ 基于分割的选择性搜索算法，使用多个互不相同但具有互补性的分组策略。

+ 实验结果表明，选择性搜索算法可以成功的用于基于词袋的定位和识别。



