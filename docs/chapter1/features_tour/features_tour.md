---
comments: true
title: Features介绍
---

!!! quote "翻译自[HuggingFace Dataset Features](https://huggingface.co/docs/datasets/main/en/about_dataset_features)"

## `Features`介绍

`Features`定义了数据集的内部结构。它用于指定底层序列化格式。更重要的是，`Features`包含从列名称和类型到`ClassLabel`的所有内容的高级信息。你可以将`Features`视为数据集的核心。

| 特性                  | 描述                                                                               |
| --------------------- | ---------------------------------------------------------------------------------- |
| 定义数据集结构        | 就像数据集的蓝图，告诉我们数据集包含哪些列（或特征），以及每列数据的类型。         |
| 指定序列化格式        | 指定数据是如何存储和表示的，例如，使用 CSV、JSON 还是其他格式。                    |
| 包含高级信息          | 除了列名和类型，还可能包含其他信息，例如：                                         |
| `ClassLabel`          | 如果数据集用于分类任务，包含类别标签的信息，例如，"猫"、"狗" 或 "鸟"。             |
| `Feature Description` | 对每个特征的描述，解释其含义和用途。                                               |
| `Feature Importance`  | 对于某些机器学习模型，包含特征重要性的信息，表明哪些特征对模型的预测结果影响更大。 |

`Features`格式很简单：`dict[column_name, column_type]`。它是一个字典，其中包含列名称和列类型对。有很多代表数据类型的列类型。

让我们来看看`GLUE`基准中的`MRPC`数据集的`features`：

```python
from datasets import load_dataset

dataset = load_dataset("glue", "mrpc", split="train")
```

```python title="dataset.feature"
{
    "sentence1": Value(dtype="string", id=None),
    "sentence2": Value(dtype="string", id=None),
    "label": ClassLabel(names=["not_equivalent", "equivalent"], id=None),
    "idx": Value(dtype="int32", id=None),
}
```

<iframe
  src="https://huggingface.co/datasets/nyu-mll/glue/embed/viewer/mrpc/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

有关受支持数据类型的完整列表，可以查阅[HuggingFace关于`Value`的文档](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Value)。

以下是整理出来的数据类型及其描述。

<div class="grid" markdown>

| 数据类型                   | 描述           |
| -------------------------- | -------------- |
| `null`                     | 表示值不存在   |
| `bool`                     | 布尔值         |
| `int8`                     | 8位有符号整数  |
| `int16`                    | 16位有符号整数 |
| `int32`                    | 32位有符号整数 |
| `int64`                    | 64位有符号整数 |
| `uint8`                    | 8位无符号整数  |
| `uint16`                   | 16位无符号整数 |
| `uint32`                   | 32位无符号整数 |
| `uint64`                   | 64位无符号整数 |
| `float16`                  | 16位浮点数     |
| `float32` (alias `float`)  | 32位浮点数     |
| `float64` (alias `double`) | 64位浮点数     |

| 数据类型                                 | 描述                               |
| ---------------------------------------- | ---------------------------------- |
| `time32[(s/ms)]`                         | 时间点，精度为秒或毫秒             |
| `time64[(us/ns)]`                        | 时间点，精度为微秒或纳秒           |
| `timestamp[(s/ms/us/ns)]`                | 时间戳，精度为秒、毫秒、微秒或纳秒 |
| `timestamp[(s/ms/us/ns), tz=(tzstring)]` | 时间戳，包含时区偏移               |
| `date32`                                 | 日期，不包含时间信息               |
| `date64`                                 | 日期，包含时间信息                 |
| `duration[(s/ms/us/ns)]`                 | 时间间隔或持续时间                 |
| `decimal128(precision, scale)`           | 十进制数，固定精度和比例           |
| `decimal256(precision, scale)`           | 十进制数，精度和比例更高           |
| `binary`                                 | 原始二进制数据                     |
| `large_binary`                           | 大型二进制数据                     |
| `string`                                 | 文本数据                           |
| `large_string`                           | 大型文本数据                       |

</div>

## 未完待续
