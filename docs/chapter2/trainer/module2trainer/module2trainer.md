---
comments: true
title: Trainer训练自定义模型
---

![module2trainer.jpg](./imgs/module2trainer.jpg)

## 前言

理论上来讲，基于`torch`或`tensorflow`实现的模型，并且**正确**重载对应方法（`forward`方法和`from_pretrained`方法），那么就可以借助`Trainer`这个强大的工具避开繁琐的代码编写，提高工作效率。

下面将借用动手学深度学习内的线性回归的简洁实现来封装基于`nn.Module`实现的模型，以进一步使用`Trainer`。将重点讲解封装步骤，`d2l`部分不再展开。

## 代码

### 引入函数库

```python
import torch
from torch import nn
from d2l import torch as d2l
from transformers import Trainer, TrainingArguments

true_w = torch.tensor([2, -3.4])
true_b = 4.2
```

在这里人为设定线性回归的真实权重为`2`和`-3.4`，偏置为`4.2`，接下来的目标是让神经网络无限逼近这个权重。

### 定义数据集

```python
class CustDatasetForRegression(torch.utils.data.Dataset):
    def __init__(self, true_w, true_b, num_samples):
        self.true_w = true_w
        self.true_b = true_b
        self.num_samples = num_samples

        self.features, self.labels = d2l.synthetic_data(true_w, true_b, num_samples)

    def __getitem__(self, idx):
        item = {"inputs": self.features[idx], "labels": self.labels[idx]}
        return item

    def __len__(self):
        return len(self.features)
```

定义数据集时，确保重载后的`__getitem__`方法返回的字典的`key`对应于模型重载后的`forward`方法里面的形参名称。

```python
data = CustDatasetForRegression(true_w, true_b, 1000)
```

### 定义模型

```python
class CustomModelForRegression(nn.Module):
    def __init__(self):
        super(CustomModelForRegression, self).__init__()
        self.net = nn.Sequential(nn.Linear(2, 1))

    def forward(self, inputs, labels=None):
        logits = self.net(inputs)

        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels)
            return {"logits": logits, "loss": loss}
        else:
            return {"logits": logits}

    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = cls(*model_args, **kwargs)
        state_dict = torch.load(pretrained_model_name_or_path)
        model.load_state_dict(state_dict)
        return model
```

自定义模型的步骤：

1. 重载前向传播逻辑。需要注意的是，应当有是否传入`labels`的判断，无论返回的是`ModelOutput`类，还是`Python`的字典，都应包含`logits`，当有标签传入时，还应有`loss`字段。
2. 重载加载预训练模型逻辑。

### 创建模型

```python
model = CustomModelForRegression()
```

```python title="model"
CustomModelForRegression(
  (net): Sequential(
    (0): Linear(in_features=2, out_features=1, bias=True)
  )
)
```

### 创建训练器

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

```python
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=100,
    logging_strategy="epoch",
    per_device_train_batch_size=512,
)

trainer = Trainer(
    model=model,  # 自定义的模型实例
    args=training_args,  # 训练参数
    train_dataset=data,
    optimizers=(optimizer, None),  # 传递优化器
)

trainer.train()  # 开始训练
```

## 训练结果表格

| Step     | Training Loss |
| -------- | ------------- |
| 2        | 32.637400     |
| 4        | 30.398200     |
| 6        | 28.260700     |
| 8        | 26.220500     |
| 10       | 24.275400     |
| $\cdots$ | $\cdots$      |
| 194      | 0.000100      |
| 196      | 0.000100      |
| 198      | 0.000100      |
| 200      | 0.000100      |

### 推理

```python title='print(model.net[0].weight, "\n", model.net[0].bias)'
Parameter containing:
tensor([[ 2.0000, -3.3999]], requires_grad=True)
 Parameter containing:
tensor([4.2001], requires_grad=True)
```

可以看到基本和答案吻合了。

## 参考资料

<div class="grid cards" markdown>

1. [动手学深度学习（线性回归的简单实现）](https://zh.d2l.ai/chapter_linear-networks/linear-regression-concise.html)
2. [昇腾社区（如何封装nn.Module并送入Trainer）](https://www.hiascend.com/forum/thread-0231151069966226074-1-1.html)
3. [HuggingFace遇到的坑](https://www.cnblogs.com/zhangxuegold/p/17534627.html)

</div>
