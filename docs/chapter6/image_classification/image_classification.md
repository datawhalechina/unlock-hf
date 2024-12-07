---
comments: true
title: 菜肴图像分类
---

![image_classification](./imgs/image_classification.png)

## 前言

## 代码

```python
model_checkpoint = "google/vit-base-patch16-224-in21k"
```

### 导入函数库

```python
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)
```

### 读取数据集

```python
dataset = load_dataset("food101", split="train[:5000]")

labels = dataset.features["label"].names

label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
```

下面是数据集`food101`的数据集主页。

<iframe
  src="https://huggingface.co/datasets/ethz/food101/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

### 加载模型

```python
model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    ignore_mismatched_sizes=True,
)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model, config)
```

使用参数高效微调后打印可训练参数如下：

```python title="model.print_trainable_parameters()"
trainable params: 667,493 || all params: 86,543,818 || trainable%: 0.7713
```

### 加载预处理器

```python
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
```

### 定义数据转换

```python
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [
        val_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch
```

### 数据预处理

```python
splits = dataset.train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)
```

### 定义评价指标

```python
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
```

### 定义动态数据整理

```python
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
```

### 定义训练参数

```python
args = TrainingArguments(
    "vit-finetuned-lora-food101",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=5e-3,
    per_device_train_batch_size=128,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=128,
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    label_names=["labels"],
    use_cpu=False,
)
```

### 定义训练器

```python
trainer = Trainer(
    lora_model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
```

### 训练与评估

```python
trainer.train()
trainer.evaluate(val_ds)
```

下面是训练时的过程结果。

| 轮次 | 评估损失 | 评估准确率 |
| ---- | -------- | ---------- |
| 0.8  | 4.0372   | 0.80       |
| 1.6  | 3.5086   | 0.876      |
| 2.4  | 3.0289   | 0.896      |
| 4.0  | 2.4545   | 0.908      |

## 参考资料

待补充
