---
comments: true
title: Features介绍
---

## 前言

`Features`类是一种用来定义数据集结构的特殊字典，该字典期望的格式为`dict[str, FieldType]`，其中键对应列名，值对应相应的数据类型。

有关受支持的`FieldType`类型可以查阅[HuggingFace关于`FieldType`的文档](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Features)，以下是受支持的数据类型及其描述。

| `FieldType`                                   | 描述                                                                                                                       |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `Value`                                       | 指定单一数据类型，例如`int64`或`string`。                                                                                  |
| `ClassLabel`                                  | 指定一组预定义的类别，这些类别可以具有与之关联的标签，并且将作为整数存储在数据集中。比如`['bad', 'ok', 'good']`            |
| `dict`                                        | 指定一个复合特征，其中包含子字段到子特征的映射。子字段可以任意方式嵌套。                                                   |
| `list`, `LargeList`, `Sequence`               | 指定一个复合特征，其中包含一个子特征序列，所有子特征的类型相同。                                                           |
| `Array2D`, `Array3D`, `Array4D`, `Array5D`    | 用于存储多维数组。                                                                                                         |
| `Audio`                                       | 用于存储指向音频文件的绝对路径或一个字典，其中包含指向音频文件的相对路径和字节内容。                                       |
| `Image`                                       | 用于存储指向图像文件的绝对路径、一个`NumPy`数组对象、一个`PIL`对象或一个`dict`，其中包含指向图像文件的相对路径和字节内容。 |
| `Translation`, `TranslationVariableLanguages` | 特定于机器翻译。                                                                                                           |

## `Features`属性介绍

### 简单数据集定义

```python
from datasets import Features, Value, ClassLabel
features = Features(
    {
        "text": Value(dtype="string"),
        "label": ClassLabel(num_classes=3, names=["negative", "positive"]),
    }
)
```

该例子定义了一个包含两个特征的简单数据集结构。

- `text`：字符串类型，用于存储文本数据。
- `label`：类别标签类型，用于存储情感类别标签，取值为`negative`或`positive`。

```python title="features"
{
    "text": Value(dtype="string", id=None),
    "label": ClassLabel(names=["negative", "positive"], id=None),
}
```

### 复合数据集定义

```python
from datasets import Features, Value, ClassLabel, Sequence

features = Features(
    {
        "text": Value(dtype="string"),
        "entities": Sequence(
            {
                "start": Value(dtype="int64"),
                "end": Value(dtype="int64"),
                "label": ClassLabel(num_classes=3, names=["PERSON", "ORG", "LOC"]),
            }
        ),
    }
)
```

该例子定义了一个包含`entities`复合特征的数据集结构。

- `text`: 字符串类型，用于存储文本数据。
- `entities`: 序列类型，用于存储文本中的实体信息。每个实体包含三个特征。
    - `start`: 整数类型，表示实体在文本中的起始位置。
    - `end`: 整数类型，表示实体在文本中的结束位置。
    - `label`: 类别标签类型，表示实体的类别，可以是`PERSON`, `ORG`或`LOC`。

```python title="features"
{
    "text": Value(dtype="string", id=None),
    "entities": Sequence(
        feature={
            "start": Value(dtype="int64", id=None),
            "end": Value(dtype="int64", id=None),
            "label": ClassLabel(names=["PERSON", "ORG", "LOC"], id=None),
        },
        length=-1,
        id=None,
    ),
}
```

### 多维数组

```python
from datasets import Features, Array2D, Value

features = Features(
    {
        "image": Array2D(shape=(224, 224, 3), dtype="float32"),
        "label": Value("int64"),
    }
)
```

该例子定义了一个包含`image`特征的数据集结构。

- `image`: 多维数组类型，用于存储图像数据，形状为`(224, 224, 3)`，数据类型为`float32`。
- `label`: 整数类型，用于存储图像的类别标签。

```python title="features"
{
    "image": Array2D(shape=(224, 224, 3), dtype="float32", id=None),
    "label": Value(dtype="int64", id=None),
}
```

### 音频数据

```python
from datasets import Features, Audio, ClassLabel

features = Features(
    {
        "audio": Audio(sampling_rate=44100),
        "label": ClassLabel(num_classes=2, names=["negative", "positive"]),
    }
)
```

该例子定义了一个包含`audio`和`label`特征的数据集结构。

- `audio`: 音频类型，用于存储音频数据，采样率为`44100 Hz`。
- `label`: 整数类型，用于存储音频情感类别标签。

```python title="features"
{
    "audio": Audio(sampling_rate=44100, mono=True, decode=True, id=None),
    "label": ClassLabel(names=["negative", "positive"], id=None),
}
```

### 机器翻译

```python
from datasets import Features, Translation, Value

features = Features(
    {
        "source_text": Value(dtype="string"),
        "target_text": Translation(languages=["en", "fr"]),
    }
)
```

该例子定义了一个包含`source_text`和`target_text`特征的数据集结构。

- `source_text`: 字符串类型，用于存储源语言文本数据。
- `target_text`: 翻译类型，用于存储目标语言文本数据，支持英语和法语两种语言。

```python title="features"
{
    "source_text": Value(dtype="string", id=None),
    "target_text": Translation(languages=["en", "fr"], id=None),
}
```

### 其他

有关受支持的`Value`数据类型的完整列表，可以查阅[HuggingFace关于`Value`的文档](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Value)，以下是整理出来的常用的数据类型及其描述。

| 数据类型 | 描述           | 数据类型  | 描述               |
| -------- | -------------- | --------- | ------------------ |
| `null`   | 表示值不存在   | `float32` | 32位浮点数         |
| `bool`   | 布尔值         | `float64` | 64位浮点数         |
| `int32`  | 32位有符号整数 | `date64`  | 日期，包含时间信息 |
| `int64`  | 64位有符号整数 | `string`  | 文本数据           |
| $\cdots$ | $\cdots$       | $\cdots$  | $\cdots$           |

下面是数据集`mrpc`的数据集主页，可以看到网页根据`Features`在数据集卡片上正确显示了列名及其数据类型。

<iframe
  src="https://huggingface.co/datasets/nyu-mll/glue/embed/viewer/mrpc/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

## `Features`方法介绍

有关受支持的`Features`方法的完整列表，可以查阅[HuggingFace关于`Features`方法的文档](https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/main_classes#datasets.Features)，以下是整理出来的常用方法及其描述。

| 方法                | 说明                                                   |
| ------------------- | ------------------------------------------------------ |
| `from_dict`         | 从字典构建`Features`。                                 |
| `to_dict`           | 返回特征的字典表示。                                   |
| `copy`              | `Features`对象的深复制。                               |
| `reorder_fields_as` | 重新排序字段以匹配另一个`Features`对象的顺序。         |
| `flatten`           | 通过删除嵌套字典并创建具有连接名称的新列来扁平化特征。 |
| $\cdots$            | $\cdots$                                               |

### `from_dict`方法

```python
from datasets import Features

Features.from_dict({"text": { "_type": "Value", "dtype": "string", "id": None}})
```

该方法使用从`from_dict`方法从字典创建`Features`对象。

```python title='Features.from_dict({"text": {"_type": "Value", "dtype": "string", "id": None,}})'
{"text": Value(dtype="string", id=None)}
```

### `to_dict`方法

```python
from datasets import Features, Value

features = Features(
    {
        "text": Value(dtype="string"),
        "label": Value(dtype="int64"),
    }
)
```

该例子首先创建了`features`，然后利用`to_dict`方法返回了字典格式的`features`。

```python title="features.to_dict()"
{
    "text": {"dtype": "string", "_type": "Value"},
    "label": {"dtype": "int64", "_type": "Value"},
}
```

### `reorder_fields_as`方法

```python
from datasets import Features, Value, ClassLabel

features = Features(
    {
        "text": Value("string"),
        "label": ClassLabel(names=["positive", "negative"]),
    }
)

other_features = Features(
    {
        "label": ClassLabel(names=["positive", "negative"]),
        "text": Value("string"),
    }
)
reordered_features = features.reorder_fields_as(other_features)
```

该例子创建字段顺序不同的两个`Features`对象，然后利用`reorder_fields_as`重新排序`features`字段以匹配`other_features`字段的顺序。

```python title="reordered_features"
{
    "label": ClassLabel(names=["positive", "negative"], id=None),
    "text": Value(dtype="string", id=None),
}
```

### `flatten`方法

```python
from datasets import Features, Value

nested_features = Features(
    {
        "a": Value("string"),
        "b": {
            "c": Value("int32"),
            "d": Value("float32"),
        },
    }
)

flattened_features = nested_features.flatten()
```

```python title="nested_features"
{
    "a": Value(dtype="string", id=None),
    "b": {"c": Value(dtype="int32", id=None), "d": Value(dtype="float32", id=None)},
}
```

该例子利用`flatten`方法删除嵌套字典并创建具有连接名称的新列来扁平化特征。

```python title="flattened_features"
{
    "a": Value(dtype="string", id=None),
    "b.c": Value(dtype="int32", id=None),
    "b.d": Value(dtype="float32", id=None),
}
```

## 参考资料

<div class="grid cards" markdown>

1. [HuggingFace关于`FieldType`/`Features`的文档](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Features)
2. [HuggingFace关于`Value`的文档](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Value)

</div>
