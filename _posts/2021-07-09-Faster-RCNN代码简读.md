---
layout: post
title: Faster RCNN代码简读
date: 2021-07-08
Author: chge
tags: [论文阅读]
comments: true
toc: false

---

Faster RCNN代码简读。

<!-- more -->

# Faster RCNN代码简读

基于pytorch实现的Faster RCNN。从以下几个方面，记录一下Faster RCNN各个部分的代码实现。

代码用的是这个：[链接](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/faster_rcnn) 

## 1 自定义Dataset类

数据集为PASCAL VOC2012数据集。

自定义一个Dataset类，需要继承`torch.utils.data.Dataset`类，并实现三个方法：

1. `__getitem__`
2. `__len__`
3. `get_height_and_width`，多GPU训练时，会用到该方法。

`my_dataset.py`里自定义的Dataset类如下：

```python
from torch.utils.data import Dataset

class VOC2012DataSet(Dataset):
    """读取解析PASCAL VOC2012数据集"""
    def __init__(self, voc_root, transforms, txt_name: str = "train.txt"):
        self.root = os.path.join(voc_root, "VOCdevkit", "VOC2012")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")
        
        # read train.txt or val.txt file
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines()]

        # check file
        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), "not found '{}' file.".format(xml_path)

        # read class_indict
        json_file = './pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        json_file = open(json_file, 'r')
        self.class_dict = json.load(json_file)
        self.transforms = transforms
```

+ `train.txt`和`test.txt`文件，分别保存训练集和测试集的图片名。这些图片都保存在`JPEGImages`文件夹下。

+ `transform`是图片预处理操作，比如pytorch中的`ToTensor()`，随机水平翻转等。

`class_dict`保存PASCAL VOC数据集20个类的类名。

初始化：

+ 参数：PASCAL VOC数据集图片路径，标注数据路径，是否训练集/测试集。

+ 标注文件为`.xml`格式，一张图片对应一个标注文件。所有标注文件路径保存在`xml_list`中。

```python
	def __len__(self):
        return len(self.xml_list)
```

`len`方法返回训练集或测试集的样本数。

```python
    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
```

`getitem`方法，通过解析单个标注文件，得到图片中包含的目标信息，这些信息包括：

+ 该图片中所有目标的ground truth边界框，
+ 目标对应的label ( 是`class_dict`中对应的索引，不是类别名这个字符串)，
+ 图片id，是否为单目标 (iscrowd)，面积area。

`self.transforms`是图片预处理步骤，处理完原始图片之后，`getitem`方法返回处理之后的图片`image`和图片中的目标信息`target`。

```python
    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width
```

`get_height_and_width`方法返回索引号为`idx`的图片的高，宽信息。

自定义`dataset`类之后，就可以使用`torch.utils.data.DataLoader()`方法，小批量加载数据了：

```python
  train_data_set = VOC2012DataSet(VOC_root, data_transform["train"], "train.txt")
  train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                  batch_size=batch_size,
                                                  shuffle=True)
```

## 2 Faster RCNN大致框架

首先是network_files文件夹，该文件夹下的文件是fast rcnn的网络结构及其主要部件。

下图是Faster RCNN的框架图：（出处）。主要部分有：用于提取特征的backbone，从特征图获得推荐区域 (RoI) 的RPN网络，处理RoI得到固定长度（7×7）特征的RoIPolling，展平层和2个全连接层组成的Two MLPHead，预测类别和坐标偏移的FasterRCNNPredictor，对预测结果做后处理的postprecess。

![fast rcnn](./__md__/fasterRCNN.png) 



先看`faster_rcnn_framework.py`：

### 2.1 FasterRCNNBase

首先是定义了一个基类：` FasterRCNNBase` ，该基类继承了`nn.module`类，以下是该类初始化：

```python
class FasterRCNNBase(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False
```

这个基类定义了Faster RCNN的主要框架。该类需要4个参数：

1. backbone: 特征提取网络；

2. rpn: 区域建议网络；
3. roi_head: 包括RoI Pooling，Two MLPHead (包括1个展平层+2个全连接层 )，Faster RCNNPredictor (包括cls_logists，box_pred)，Postprocess Detection (后处理，如nms...) ；
4. transform: 对图像解进行预处理，在forward中用到。

再来看一下该基类的前向传播：

```python
def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:         # 进一步判断传入的target的boxes参数是否符合规定
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                          boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2  # 防止输入的是个一维向量
            original_image_sizes.append((val[0], val[1]))
        # original_image_sizes = [img.shape[-2:] for img in images]

        images, targets = self.transform(images, targets)  # 对图像进行预处理

        # print(images.tensors.shape)
        features = self.backbone(images.tensors)  # 将图像输入backbone得到特征图
        if isinstance(features, torch.Tensor):  # 若只在一层特征层上预测，将feature放入有序字典中，并编号为‘0’
            features = OrderedDict([('0', features)])  # 若在多层特征层上预测，传入的就是一个有序字典

        # 将特征层以及标注target信息传入rpn中
        # proposals: List[Tensor], Tensor_shape: [num_proposals, 4],
        # 每个proposals是绝对坐标，且为(x1, y1, x2, y2)格式
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 将rpn生成的数据以及标注target信息传入fast rcnn后半部分
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # 对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
```

`forward`包含2个参数：

1. images:   `list[Tensor]`，即一批图片组成的list，列表元素是原始图片经过数据增强之后得到的`Tensor`；
2. target: 和自定义的Dataset中的target是同一个东西，包括真实边界框，id，label，面积，等；

以下为forward的逐部分讲解：

1.首先对images和target进行预处理：

` images, targets = self.transform(images, targets)` 

+ 预处理包括 ：对图像进行标准化处理和`resize`处理（`resize`处理仅仅是对图像的宽高进行限制）；

2.然后将Tensor格式的图像，送入骨干网，提取整张图像的特征图：

` features = self.backbone(images.tensors)`

+ 如果加入了FPN，则会得到许多特征图，将这些特征图放入有序字典，每张特征图都有对应的序号；

3.随后，经过rpn网络，得到推荐区域和推荐区域的rpn loss：

` proposals, proposal_losses = self.rpn(images, features, targets)`

+ 如果是训练模式，则target不能为空，如果是测试模型，则此处的target为空 (None)；

+ 经过RPN网络后，得到一些推荐区域proposals，此处的每个proposals都是绝对坐标，格式为`(x1, y1, x2, y2)`；

4.在得到推荐区域之后，将其输入到roi_head部分，得到最终检测框`detections`和fast rcnn的损失`detector_loss`: 

` detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)`

+ `images.image_sizes`是图像处理后的尺寸，得到的检测框detections与处理后的图像的尺寸相对应，所以还需要将检测框映射回原图尺寸相对应的位置，就是下一步；

5.将检测框映射回原始图像对应的尺寸：

`detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)`

+ `original_image_sizes`保存着图像的原始尺寸，是一个列表，元素就是每张图片的原始尺寸高和宽；

6.最后的一部分就是记录rpn损失和fast rcnn损失（即网络损失）：

```python
losses = {}
losses.update(detector_losses)
losses.update(proposal_losses)
```

以上就是类`FasterRCNNBase`的初始化部分和前向传播部分。

**流程小结：**对于图片和标注，经过骨干网提取出特征图，经过RPN网络得到推荐区域（训练时，在此过程中会计算出rpn损失），经过roi head处理，得到检测结果（训练中，此处计算fast rcnn网络损失），结果后处理，得到与原图尺寸相对应的检测框结果。

`FasterRCNNBase`的流程基本包括了Faster RCNN的主要部分，在该类的基础上，才定义出了完整的Faster RCNN类：`class FasterRCNN(FasterRCNNBase)`，以下部分为class FasterRCNN类的详解。

### 2.2 Faster RCNN类

`FasterRCNN`类继承自`FasterRCNNBase`类，在大体框架上，加入了前后处理的部分。

首先是该类的初始化函数，先来看初始化所需要的参数：

```python
def __init__(self, backbone, num_classes=None,
                 # transform parameter
                 min_size=800, max_size=1333,      # 预处理resize时限制的最小尺寸与最大尺寸
                 image_mean=None, image_std=None,  # 预处理normalize时使用的均值和方差
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,    # rpn中在nms处理前保留的proposal数(根据score)
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # rpn中在nms处理后保留的proposal数
                 rpn_nms_thresh=0.7,  # rpn中进行nms处理时使用的iou阈值
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # rpn计算损失时，采集正负样本设置的阈值
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # rpn计算损失时采样的样本数，以及正样本占总样本的比例
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 # 移除低目标概率      fast rcnn中进行nms处理的阈值   对预测结果根据score排序取前100个目标
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,   # fast rcnn计算误差时，采集正负样本设置的阈值
                 box_batch_size_per_image=512, box_positive_fraction=0.25,  # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例
                 bbox_reg_weights=None)
```

以下是各参数的详解：

#### 2.2.1 参数详解

1.首先是图像预处理参数：

```python
min_size=800, max_size=1333,      # 预处理resize时限制的最小尺寸与最大尺寸
image_mean=None, image_std=None,  # 预处理normalize时使用的均值和方差
```

图像预处理包括resize和标准化。

2.RPN网络的参数：

这个RPN部分就是下图虚线框住的部分，

![image-20210709132055116](__md__/image-20210709132055116.png) 

RPN网络输入：图片、标注和特征图，输出：推荐区域、rpn损失；

```python
rpn_anchor_generator=None, rpn_head=None,
rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,    # rpn中在nms处理前保留的proposal数(根据score)
rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # rpn中在nms处理后保留的proposal数
rpn_nms_thresh=0.7,  # rpn中进行nms处理时使用的iou阈值
rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # rpn计算损失时，采集正负样本设置的阈值
rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # rpn计算损失时采样的样本数，以及正样本占总样本的比例
rpn_score_thresh=0.0,
```

+ rpn_anchor_generator：rpn网络的anchor生成器；
+ rpn_head：由3×3的卷积层（对应论文中3×3的滑动窗口），2个分支：分类 (cls_logists) 和回归 (box_pred) ，所组成；
+ rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000，分别表示在使用非极大值抑制之前，训练和测试时，将rpn网络生成的推荐区域，按照预测的score进行筛选，保留2K和1K个proposal；
+ rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000，表示rpn中在nms处理后保留下来的proposal个数；
+ rpn_nms_thresh=0.7，是在rpn中进行nms处理时使用的iou阈值；
+ rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3，是rpn计算损失时，采集正负样本时所设置的阈值；
+ rpn_batch_size_per_image=256, rpn_positive_fraction=0.5，是rpn网络计算损失时，一张图片采样的样本数，以及正样本占总样本的比例；
+ rpn_score_thresh=0.0，表示nms时，score的阈值；

3.Box参数：

这一部分对应网络部分如下：

![image-20210709132726375](__md__/image-20210709132726375.png)

得到推荐区域之后，经过Two MLPHead和2个分支，得到cls和box，此处计算的损失是fastrcnn loss，即网络损失。以下参数就是这一部分。

```python
box_roi_pool=None, box_head=None, box_predictor=None
box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,   # fast rcnn计算误差时，采集正负样本设置的阈值
box_batch_size_per_image=512, box_positive_fraction=0.25,  # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例
bbox_reg_weights=None
```

与预测边界框（box）及其执行度（score）相关的参数说明：

+ box_roi_pool：就是RoI Pooling层；

+ box_head：就是Two MLPHead，即一个展平层+2个全连接层。对应代码如下：

  ```python
  class TwoMLPHead(nn.Module):
      """
      Standard heads for FPN-based models
      Arguments:
          in_channels (int): number of input channels
          representation_size (int): size of the intermediate representation
      """
      def __init__(self, in_channels, representation_size):
          super(TwoMLPHead, self).__init__()
          self.fc6 = nn.Linear(in_channels, representation_size)
          self.fc7 = nn.Linear(representation_size, representation_size)
  
      def forward(self, x):
          x = x.flatten(start_dim=1)
          x = F.relu(self.fc6(x))
          x = F.relu(self.fc7(x))
          return x
  ```

+ box_predictor：对应2个全连接层的分支，一个分支用于预测类别概率，一个分支用于预测边界框回归参数，对应代码如下：

  ```python
  class FastRCNNPredictor(nn.Module):
      """
      Standard classification + bounding box regression layers
      for Fast R-CNN.
      Arguments:
          in_channels (int): number of input channels
          num_classes (int): number of output classes (including background)
      """
      def __init__(self, in_channels, num_classes):
          super(FastRCNNPredictor, self).__init__()
          self.cls_score = nn.Linear(in_channels, num_classes)
          self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
      def forward(self, x):
          if x.dim() == 4:
              assert list(x.shape[2:]) == [1, 1]
          x = x.flatten(start_dim=1)
          scores = self.cls_score(x)
          bbox_deltas = self.bbox_pred(x)
          return scores, bbox_deltas
  ```

  根据`box_predictor`得到的box和box scores，然后就是后处理阶段，对应图中`postprcess detection`部分。

+ box_score_thresh=0.05，根据该score阈值，对boxes做过滤，即过滤掉很低score的边界框； 

+ box_nms_thresh=0.5, box_detections_per_img=100，过滤掉低score的边界框之后，对剩余的边界框，按照score，进行非极大值抑制处理，对应的代码如下，box_detections_per_img表示一张图片，在nms之后，只保存100个边界框；

  ```python
  def nms(boxes, scores, iou_threshold):
      # type: (Tensor, Tensor, float) -> Tensor
      """
      Performs non-maximum suppression (NMS) on the boxes according
      to their intersection-over-union (IoU).
  
      NMS iteratively removes lower scoring boxes which have an
      IoU greater than iou_threshold with another (higher scoring)
      box.
  
      Parameters
      ----------
      boxes : Tensor[N, 4])
          boxes to perform NMS on. They
          are expected to be in (x1, y1, x2, y2) format
      scores : Tensor[N]
          scores for each one of the boxes
      iou_threshold : float
          discards all overlapping
          boxes with IoU < iou_threshold
  
      Returns
      -------
      keep : Tensor
          int64 tensor with the indices
          of the elements that have been kept
          by NMS, sorted in decreasing order of scores
      """
      return torch.ops.torchvision.nms(boxes, scores, iou_threshold)
  
  ```

与计算fast loss相关的参数的参数说明：

+ box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5，计算fast rcnn误差时，对于正负样本的设定：预测边界框与ground truth的IoU阈值大于box_fg_iou_thresh，就定义为正样本，小于box_bg_iou_thresh就定义为负样本；
+ box_batch_size_per_image=512, box_positive_fraction=0.25,  计算fast rcnn的loss时，采样的样本数，以及正样本占所有样本的比例；
+ bbox_reg_weights=None；

#### 2.2.2 其它部分：

**rpn_anchor_generator**

一共有3种长度，3种纵横比的anchor生成器：

```python
if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
)
```

**RoI Pooling**

```python
if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
                output_size=[7, 7], # Roi Pool 之后得到大小
                sampling_ratio=2) # 缩放到7×7，所以需要采样率
```

+ RoI Pooling层直接使用了`torchvision.ops.MultiScaleRoIAlign`；

+ 由于骨干网络使用了带有RPN网络的resNet50，骨干网输出4个不同尺寸的特征图，所以此处的`featmap_names=['0', '1', '2', '3']`；

RoI Pooling层将每个推荐区域的特征向量长度固定到为7×7，于是该层的输出为 out_channels×7×7 维（其中out_channels是骨干网提取出的特征图的channels数）。

**Two MLPHead**

Two MLPHead由一个展平层和2个全连接层组成。输入：`out_channels×7×7`的推荐区域的特征，输出：1024维的特征向量。对应代码如下。

```python
if box_head is None:
	resolution = box_roi_pool.output_size[0]  # 默认等于7
    representation_size = 1024
    box_head = TwoMLPHead(
    	out_channels * resolution ** 2,
    	representation_size
    )
```

**FastRCNNPredictor**

Two MLPhead层之后，跟随2个分支，这两个分支分别做类别预测和边界框回归。

对应代码如下，输入：1024维特征向量，输出：scores, bbox_deltas。

```python
# 在box_head的输出上预测部分
if box_predictor is None:
	representation_size = 1024
    box_predictor = FastRCNNPredictor(
    representation_size,
    num_classes)
   
```

其中FastRCNNPredictor代码如下

```python
class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas
```

**数据处理**

transform为预数据处理，用于处理输入图片。

```python
 if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        # 对数据进行标准化，缩放，打包成batch等处理部分
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

```

#### 2.2.3 组合成Faster RCNN

**roi_head**

将上述RoI Pooling，bbox_head，box_predictor组合在一起，就构成了roi_head：

```python
 # 将roi pooling, box_head以及box_predictor结合在一起
roi_heads = RoIHeads(
    # box
    box_roi_pool, box_head, box_predictor, # 3个主要部分
    box_fg_iou_thresh, box_bg_iou_thresh,  # 0.5  0.5
    box_batch_size_per_image, box_positive_fraction,  # 512  0.25
    bbox_reg_weights,
    box_score_thresh, box_nms_thresh, box_detections_per_img)  # 0.05  0.5  100
```

于是，整个Faster RCNN网络定义为：

```
super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)
```

+ backbone为骨干网，输出特征图（带有FPN的骨干网络，输出的特征图有多个）；
+ rpn网络，输出推荐区域（边界框坐标），计算rpn loss；
+ roi heads，包括roi pooling、two mlphead、predictor (双分支)、对检测结果做后处理；
+ transform，对输入图片做预处理；

以上就是`faster_rcnn_framework.py`的主要部分，接下来就是各个组件的详解部分了。

## 3 RPN网络

对应文件`rpn_function.py`。

RPN网络结构图：

<img src="./__md__/image-20210709132055116.png" alt="image-20210709132055116" style="zoom:100%;" />

在backbone获得输入图片的特征图之后，使用3×3的滑动窗口，以窗口中心点作为不同尺寸anchor的中心点，加入9中不同类型的anchor，2个分支分别预测anchor类别和坐标偏移。滑动窗口是以3×3的卷积操作来实现的，2个分支是1×1的卷积层（channel不同）。3×3的滑动窗口和2个分支共同构成RPNHead。下图就是滑动窗口和2个分支，（图片来自原论文）。

![img2](./__md__/image-20210708190235308.png)



### 3.1 RPNHead

首先看`RPNHead`类，该类通过滑动窗口，得到不同的anchor，然后预测这些anchor的类别概率（0或1）和边界框偏移。代码如下。

```python
class RPNHead(nn.Module):
    """
    add a RPN head with classification and regression
    通过滑动窗口计算预测目标概率与bbox regression参数

    Arguments:
        in_channels: number of channels of the input feature
        num_anchors: number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 计算预测的目标分数（这里的目标只是指前景或者背景）
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 计算预测的目标bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

```

RPNHead包括：一个3×3的滑动窗口（以大小为3×3的卷积核来实现），一个类别预测分支，一个边界框回归分支。

前向传播过程中，返回预测类别（0或1）和边界框偏移（x,y,w,h）。

### 3.2  AnchorsGenerator

该类继承自`nn.Module`，用于生成anchor。

初始化参数：anchor尺寸和纵横比，对应代码如下。

```python
class AnchorsGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    """
    anchors生成器
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Arguments:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """
    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}
        
        # ...
      
    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        # 获取每个预测特征层的尺寸(height, width)
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])

        # 获取输入图像的height和width
        image_size = image_list.tensors.shape[-2:]

        # 获取变量类型和设备类型
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # one step in feature map equate n pixel stride in origin image
        # 计算特征层上的一步等于原始图像上的步长
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]

        # 根据提供的sizes和aspect_ratios生成anchors模板
        self.set_cell_anchors(dtype, device)

        # 计算/读取所有anchors的坐标信息（这里的anchors信息是映射到原图上的所有anchors信息，不是anchors模板）
        # 得到的是一个list列表，对应每张预测特征图映射回原图的anchors坐标信息
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        # 遍历一个batch中的每张图像
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # 遍历每张预测特征图映射回原图的anchors坐标信息
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        # 将每一张图像的所有预测特征层的anchors坐标信息拼接在一起
        # anchors是个list，每个元素为一张图像的所有anchors信息
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        # Clear the cache in case that memory leaks.
        self._cache.clear()
        return anchors
```

anchor_generator返回的anchors包括一个batch中所有图片的anchor。

### 3.3 RPNHead

RPNHead就是上一节中所写的，包括一个3×3的卷积层（滑动窗口）和2个预测分支，代码如下。

```python
class RPNHead(nn.Module):
    """
    add a RPN head with classification and regression
    通过滑动窗口计算预测目标概率与bbox regression参数

    Arguments:
        in_channels: number of channels of the input feature
        num_anchors: number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 计算预测的目标分数（这里的目标只是指前景或者背景）
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 计算预测的目标bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

```

## 4 总结

简单梳理了一下Fast RCNN的代码架构，细节性的代码部分，暂时跳过。现在一些anchor free和基于中心点的目标检测网络，在精度上已经超过了双阶段检测器，接下来的重点会放在FCOS，PolarMask这些网络上。







