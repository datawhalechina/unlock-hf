---
comments: true
title: Trainer工具介绍
---

![trainer.png](./imgs/trainer.png)

## 前言

`HuggingFace`的`Trainer`是一个强大的工具，它简化了深度学习模型的训练过程，提供统一的接口，可以轻松地训练各种模型。

传统上，训练模型包含以下基本步骤：

- 定义损失函数
- 设置训练模式
- 迭代数据集
- 计算损失
- 执行反向传播
- 输出训练日志

然而，随着训练技巧需求的增加，例如以下需求：

- 权重衰减
- 分布式训练
- 动态步长
- 动态学习率
- $\cdots$

开发者需要编写大量的代码，这会让工程变得非常复杂。

!!! Success
    `Trainer` 的优势在于其封装了这些复杂的操作，使开发者能够专注于模型本身，而无需担心底层实现细节。所以抛开繁杂的训练代码，使用`Trainer`是一件多么省时省力的事情，所以投入`Trainer`的怀抱吧。

## Trainer API

`Trainer`的属性和方法都可以在[Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer)进行查阅。

| 参数名称                        | 数据类型                                                          | 注释                                                                                                                        |
| ------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `model`                         | `PreTrainedModel`                                                 | 模型实例                                                                                                                    |
|                                 | `nn.Module`                                                       |                                                                                                                             |
| `args`                          | `TrainingArguments`                                               | 训练时的超参数                                                                                                              |
| `data_collator`                 | `DataCollator`                                                    | 数据整理器                                                                                                                  |
| `train_dataset`                 | `torch.utils.data.Dataset`                                        | 训练集。如果接收的是`datasets.Dataset`，训练阶段数据集中不被使用的列都会被删除                                              |
|                                 | `torch.utils.data.IterableDataset`                                |                                                                                                                             |
|                                 | `datasets.Dataset`                                                |                                                                                                                             |
| `eval_dataset`                  | `torch.utils.data.Dataset`                                        | 评估集。如果接收的是`datasets.Dataset`，评估阶段数据集中不被使用的列都会被删除                                              |
|                                 | `datasets.Dataset`                                                |                                                                                                                             |
| `tokenizer`                     | `PreTrainedTokenizerBase`                                         | 分词器实例。在对数据进行分批次时，分词器会将数据填充到最大长度；在保存模型时，分词器文件也将被同时保存                      |
| `model_init`                    | `Callable`                                                        | 用于初始化模型的函数                                                                                                        |
| `compute_metrics`               | `Callable[[EvalPrediction], Dict]`                                | 用于计算评估指标的函数。该函数输入为`EvalPrediction`类，返回结果必须为包含指标及其具体数值的字典                            |
| `callbacks`                     | `List[TrainerCallback]`                                           | 回调函数列表                                                                                                                |
| `optimizers`                    | `Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]` | 用于训练的优化器和学习率调度器。默认为(`None`, `None`)，即对应（`AdamW`优化器实例，线性学习率调度器）                       |
| `preprocess_logits_for_metrics` | `Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`            | 用于在评估步骤中对模型输出的置信度进行预处理的函数。该函数最后返回运算过后的置信度，并将置信度和标签传递给`compute_metrics` |

## 使用案例

待补充

## 参考资料

<div class="grid cards" markdown>

1. [HuggingFace API Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
2. [回调函数](../callbacks/callbacks.md)

</div>
