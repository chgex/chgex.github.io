---
layout: post
title: hello-world
date: 2020-12-10
Author: JIUGE 
tags: [other, markdown]
comments: true
toc: false
---

本文意在记录第一个博客，并对文本显示情况，latex公式显示情况做测试。

# hello-post

这个站点建立于2002年12月10日。

往后几年，会一直更新，记录学习和日常。

## 1- 文本测试

给出一张图片的候选边界框集合，检测器分类模块从背景对象中识别出前景对象。令$\theta$ 表示分类器，它通过以下优化方式习得：
$$
\min_\theta\sum_{i}^{N}\sum\ell(p_{i,j,k})
\tag{1}
$$
N表示总图片数量；$p_{i,j,k}$表示第i张图片中第j个候选对象属于第k类的概率，是由分类器$\theta$ 预测出的；$\ell(\cdot)$ 为损失函数。大多数分类器都是通过最小化交叉熵损失函数(cross entropy loss)或方差(variants)进行学习。

但式(1)容易受到类间不平衡问题的影响。于是将上式(1)改写为：
$$
\min_\theta\sum_{i}^{N}(\sum_{j_+}^{n+}\ell(p_{i,j_+})+\sum_{j_-}^{n_-}\ell(p_{i,j_-}))
\tag{2}
$$
其中，$j_+，j_-$分别表示第j个正样本和负样本；$n_+，n_-$ 分别表示正负样本的个数。当负样本数远远大于正样本数时，式子中的后一项将占据主导，这将导致负样本对损失的贡献远远超过正样本。常用的解决方式是增大正样本的损失权重。论文提出了一种解决方法：**Rank**

### Ranking

为了减少类间不平衡问题，我们优化正负样本之间的排序。对于一个样本对(正负样本各一个)，理想的模型是使用一个大间隔$\gamma$ 将正样本排在负样本之前：
$$
\forall j_+,j_-,p_{j_+}-p_{j_-}\geq\gamma
$$

> $\gamma$是非负常量，表示间隔。

该排序模型很好的优化了单个正负样本之间的关系。于是对单个图像的排序目标，就可以写成：

$$
\min_\theta\sum_{j_+}^{n_+}\sum_{j_-}^{n_-}\ell(p_{j_-}-p_{j_+}+\gamma)
\tag{3}
$$
其中，损失函数$\ell(\cdot)$ 使用了hinge loss(铰链损失)：
$$
\ell_{hinge}(z)=[z]_+=
\begin{cases}
z, \space z>0    \\
0,\space o.w.
\end{cases}
$$
式(3)进一步等价于：
$$
\frac{1}{n_+n_-}
\sum_{j_+}^{n_+}\sum_{j_-}^{n_-}\ell(p_{j_-}-p_{j_+}+\gamma)\\
=E_{j_+,j_-[\ell(p_{j_-}-p_{j_+}+\gamma)]}
\\
\tag{4}
$$
上式表示最小化分类损失等价于最小化随机采样对排序损失的期望。

排序损失通过比较每个正负样本的排名，解决了类间不平衡问题。

然而，却忽略了：负样本中的难分样本也是不平衡的，除此之外，还有排序问题会引发大量的配对问题。

针对这些问题，引入了Distributional Ranking:

## 3-图片测试

这是一张图片：

![](https://gitee.com/changyv/md-pic/raw/master/logo.png)

## 4-其它

关于本文头信息：

```
layout: post
title: 
date: 
author: 
tags: [sample, document]
comments: true
toc: true
pinned: true
```

几点需要注意：

+ 属性和值之间要有空一格，如`layout: post`；
+ `title`使用英文，不能出现任何符号
+ `tags`中英文都可以；

以上。