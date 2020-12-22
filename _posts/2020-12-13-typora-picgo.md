---
layout: post
title: typora+picgo编辑文本
date: 2020-12-13
Author: JIUGE 
tags: [document]
comments: true
toc: false
---

使用typora+picgo组合件，编辑文本并同步。

<!-- more -->

# typora配置远程图床

使用工具：

+ typora 

+ picgo
+ gitee或者github远程仓库

1-安装 picgo

typora-->偏好设置--->图像--->上传服务设定，选择"下载picgo app"，自动跳转到下载页面。下载安装；

> 选择安装在app文件夹下。

2-配置远程图床

在picgo页面，打开插件安装，选择gitee，配置与gitee的远程仓库的连接，这里需要仓库的token(口令)。

安装插件之前，需要有node.js环境。没有按装的话，会自动跳转到下载页面。

> 选择将node.js安装在app文件夹下。

3-测试

在typora页面进行测试。

**总结：**

虽然避免了本地文件丢失，md文档找不到图片的情况，但md文档中，图片加载将受到网速的影响。如果图片大小超过1M，会出现图片访问不了的情况。好在移动性强，发给别人的原文档，不用担心图片的显示问题。就目前使用场景，还是利大于弊的。

以上。