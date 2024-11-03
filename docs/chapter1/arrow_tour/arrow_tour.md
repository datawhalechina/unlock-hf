---
comments: true
title: Arrow介绍
---

!!! quote "翻译自[HuggingFace Arrow](https://huggingface.co/docs/datasets/main/en/about_arrow)"

## `Arrow`是什么？

`Arrow`是一种数据格式，可以快速处理和移动大量数据。它使用列式内存布局存储数据，它的**标准格式**具有以下优点：

| 特征       | 描述                                                                             |
| ---------- | -------------------------------------------------------------------------------- |
| 读取方式   | 支持**零拷贝**读取，从而消除了几乎所有序列化开销。                               |
| 跨语言支持 | 支持多种编程语言。                                                                 |
| 存储方式   | 面向列的存储，在查询和处理数据切片或列时速度更快。                               |
| 兼容性     | 数据可以无缝传递给主流机器学习工具，如`NumPy`、`Pandas`、`PyTorch`和`TensorFlow`。 |
| 列类型     | 支持多种列类型，甚至支持嵌套列类型。                                               |

## 内存映射

`Datasets`使用`Arrow`作为其本地缓存系统。它允许数据集由磁盘缓存作为后盾，该缓存被内存映射以实现快速查找。

这种架构允许在设备内存较小的机器上使用大型数据集。

例如，加载完整的英文维基百科数据集只需要几兆字节的内存：

```python
import os
import psutil
import timeit
from datasets import load_dataset

mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
wiki = load_dataset("wikipedia", "20220301.en", split="train")
mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

print(f"RAM memory used: {(mem_after - mem_before)} MB")
```

之所以能做到这一点，是因为`Arrow`数据实际上是从磁盘内存映射的，而不是直接加载到内存中的。内存映射允许访问磁盘上的数据，并利用虚拟内存功能进行快速查找。

## 性能

使用Arrow在内存映射数据集中进行迭合是快速的。在笔记本电脑上遍用维基百科，速度为1-3 Gbit/s。

```python
s = """batch_size = 1000
for batch in wiki.iter(batch_size):
    ...
"""

elapsed_time = timeit.timeit(stmt=s, number=1, globals=globals())
print(
    f"Time to iterate over the {wiki.dataset_size >> 30} GB dataset: {elapsed_time:.1f} sec, "
    f"ie. {float(wiki.dataset_size >> 27)/elapsed_time:.1f} Gb/s"
)
```

```python
Time to iterate over the 18 GB dataset: 31.8 sec, ie. 4.8 Gb/s
```
