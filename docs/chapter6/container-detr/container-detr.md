---
comments: true
title: 集装箱编号位置定位
---

![object-detection](./imgs/object-detection.png)

## 前言

随着全球贸易的快速发展，集装箱运输已成为国际贸易中不可或缺的一部分。然而，集装箱信息的准确识别与管理依然是物流行业中的一大挑战。

本文将以该命题为依据，使用HuggingFace生态工具完成目标检测任务，开发者在做目标检测任务时将数据集转化为本命题一致的数据格式，快速启动训练。

## 代码

### 导入函数库

```bash
pip install -U datasets transformers[torch] evaluate timm albumentations accelerate
```

```python
import albumentations
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)
```

### 加载数据集

```python
data = load_dataset("moyanxinxu/container")
```

```python title='data'
DatasetDict({
    train: Dataset({
        features: ['image', 'image_id', 'width', 'height', 'objects'],
        num_rows: 1049
    })
    test: Dataset({
        features: ['image', 'image_id', 'width', 'height', 'objects'],
        num_rows: 263
    })
})
```

```python title='data["train"][0]'
{
    'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1120x1080>,
    'image_id': 475,
    'width': 1120,
    'height': 1080,
    'objects':
    {
        'area': [23562.0],
        'bbox': [[415.0, 592.0, 374.0, 63.0]],
        'category': ['container'],
        'id': [0]
    }
}
```

字段说明：

| 字段               | 说明                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------ |
| `image`            | 应为`PIL`格式的图片                                                                        |
| `image_id`         | 图片ID                                                                                     |
| `width`            | 图片宽度，单位像素                                                                         |
| `height`           | 图片高度，单位像素                                                                         |
| `objects`          | 包含图像中所有对象的字典                                                                   |
| `objects.area`     | 目标面积列表，存储图片上所有目标的面积                                                     |
| `objects.bbox`     | 目标锚框列表，存储图片上所有目标的锚框位置，每个边界框格式为 `[xmin, ymin, width, height]` |
| `objects.category` | 目标类别列表，存储图片上所有目标的字符标签                                                 |
| `objects.id`       | 目标ID列表，存储图片上所有目标的数字标签                                                   |

下面为部分数据集：

<iframe
  src="https://huggingface.co/datasets/moyanxinxu/container/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

### 加载处理器

```python
preprocessor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
```

### 定义图像增广

```python
aug = albumentations.Compose(
    transforms=[
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)
```

!!! Note
    `albumentations`函数库提供了各种各样的图像增强技术。开发者可以在这片代码中自定义图像增强方法。

参数`transforms`指定图像的变化方式，参数`bbox_params`指定标签的变化方式。

| 代码                                                   | 功能                                                             |
| ------------------------------------------------------ | ---------------------------------------------------------------- |
| `Resize(480, 480)`                                     | 将图像尺寸调整为 `480x480`                                       |
| `HorizontalFlip(p=1)`                                  | 以概率`p`水平翻转图像                                            |
| `albumentations.RandomBrightnessContrast(p=1.0)`       | 以概率`p`随机调整图像的亮度和对比度                              |
| `BboxParams(format="coco", label_fields=["category"])` | 指明边界框格式采用`COCO`数据集格式，并且标签类别字段为`category` |



### 定义数据格式转化函数

```python
def datapipe(data):
    images, bboxes, areas, categories, targets = [], [], [], [], []

    image_ids = data["image_id"]

    for image, objects in zip(data["image"], data["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = aug(image=image, bboxes=objects["bbox"], category=objects["id"])

        areas.append(objects["area"])

        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    for image_id, category, area, box in zip(image_ids, categories, areas, bboxes):
        annotations = []

        for _category, _area, _box in zip(category, area, box):
            new_ann = {
                "image_id": image_id,
                "category_id": _category,
                "isCrowd": 0,
                "area": _area,
                "bbox": list(_box),
            }
            annotations.append(new_ann)
        targets.append({"image_id": image_id, "annotations": annotations})
    return preprocessor(images=images, annotations=targets, return_tensors="pt")
```

上述代码对原始数据集格式进行转化。转化后的数据集格式见`train_data[0]`。

```python
train_data = data["train"].with_transform(datapipe)
# val_data = data["validation"].with_transform(datapipe)
test_data = data["test"].with_transform(datapipe)
```

```python title='train_data[0]'
{
'pixel_values':
tensor([[[-0.3198, -0.3712, -0.4568,  ...,  0.0569,  0.0398,  0.0398],
         [-0.3883, -0.4397, -0.5424,  ...,  0.0569,  0.0398,  0.0398],
         [-0.4739, -0.5424, -0.6623,  ...,  0.0398,  0.0227,  0.0227],
         ...,
         [ 0.3138,  0.2796,  0.2282,  ...,  0.2282,  0.2624,  0.2796],
         [ 0.2453,  0.2453,  0.2624,  ...,  0.2624,  0.2796,  0.2624],
         [ 0.1939,  0.2282,  0.2967,  ...,  0.2967,  0.2796,  0.2624]],

        [[-0.1275, -0.2150, -0.3375,  ...,  0.1001,  0.0826,  0.0826],
         [-0.1975, -0.3025, -0.4251,  ...,  0.1001,  0.0826,  0.0826],
         [-0.3200, -0.4251, -0.5651,  ...,  0.0826,  0.0651,  0.0651],
         ...,
         [ 0.5028,  0.4678,  0.4328,  ...,  0.1527,  0.2052,  0.2402],
         [ 0.4328,  0.4328,  0.4678,  ...,  0.1702,  0.1702,  0.1702],
         [ 0.3803,  0.4153,  0.4853,  ...,  0.1702,  0.1527,  0.1352]],

        [[ 0.0953,  0.0082, -0.1138,  ...,  0.2348,  0.2173,  0.1999],
         [ 0.0082, -0.0964, -0.2184,  ...,  0.2348,  0.2173,  0.1999],
         [-0.1312, -0.2358, -0.3753,  ...,  0.2173,  0.1999,  0.1825],
         ...,
         [ 0.7925,  0.7576,  0.7054,  ...,  0.3568,  0.4265,  0.4788],
         [ 0.7054,  0.7228,  0.7402,  ...,  0.3393,  0.3916,  0.4091],
         [ 0.6531,  0.7054,  0.7751,  ...,  0.3393,  0.3568,  0.3568]]]),

'pixel_mask':
tensor([[1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        ...,
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1]]),

'labels':
{
    'size': tensor([800, 800]),
    'image_id': tensor([475]),
    'class_labels': tensor([0]),
    'boxes': tensor([[0.4625, 0.5773, 0.3339, 0.0583]]),
    'area': tensor([65449.9961]),
    'iscrowd': tensor([0]),
    'orig_size': tensor([480, 480])
}

}
```

### 定义整理函数

```python
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    output = preprocessor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]

    ret = {}

    ret["pixel_values"] = output["pixel_values"]
    ret["pixel_mask"] = output["pixel_mask"]
    ret["labels"] = labels

    return ret
```

### 加载模型

```python
id2label = {0: "container-id"}
label2id = {"container-id": 0}


model = AutoModelForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
```


### 定义超参

```python
training_args = TrainingArguments(
    output_dir="detr-resnet-50-container-finetuned",
    per_device_train_batch_size=10,
    num_train_epochs=20,
    eval_strategy="epoch",
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
)
```

### 定义训练器

```python
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=preprocessor,
)
```

### 训练

```python
tainer.train()
```

### 推理

```python
from transformers import pipeline

obj_detector = pipeline("object-detection", model="detr-resnet-50-container-finetuned")

results = obj_detector(train_dataset[0]["image"])
```

## 参考资料

<div class="grid cards" markdown>

- HuggingFace社区教程

    ---

    [走进计算机视觉](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome)

- HuggingFace目标检测代码案例

    ---

    [使用DETR完成目标检测](# https://huggingface.co/learn/computer-vision-course/unit3/vision-transformers/vision-transformer-for-objection-detection)

</div>