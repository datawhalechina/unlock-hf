---
comments: true
title: Trainer训练自定义模型
---

![module2trainer.jpg](./imgs/module2trainer.jpg)

## 前言

理论上来讲，基于`torch`或`tensorflow`实现的模型，并且**正确**重载对应方法（`forward`、`from_pretrained`等方法），那么就可以借助`Trainer`这个强大的工具避开繁琐的代码编写，提高工作效率。

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

```python hl_lines="6 16 17 23 27"
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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = cls(*model_args, **kwargs)
        state_dict = torch.load(f"{pretrained_model_name_or_path}/model.bin")
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        self.to(torch.device("cpu"))
        torch.save(self.state_dict(), f"{save_directory}/model.bin")

    def predict(self, inputs, device):
        device = device or self.device
        with torch.no_grad():
            inputs = inputs.to(device)
            out = self(inputs, None)
            return out["logits"].flatten()
```

自定义模型的步骤：

1. 重载前向传播逻辑。需要注意的是，应当有是否传入`labels`的判断，无论返回的是`ModelOutput`类，还是`Python`的字典，都应包含`logits`，当有标签传入时，还应有`loss`字段。
2. 重载加载预训练模型的方法。
3. 重载保存模型的方法。
4. 重载用于预测的方法。

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
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
```

```python
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    logging_strategy="epoch",
    per_device_train_batch_size=512,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    optimizers=(optimizer, None),
)

trainer.train()
```

| Step | Training Loss | Step | Training Loss | Step | Training Loss | Step | Training Loss |
| ---- | ------------- | ---- | ------------- | ---- | ------------- | ---- | ------------- |
| 2    | 31.782100     | 12   | 1.636300      | 22   | 0.000100      | 32   | 0.000100      |
| 4    | 21.786800     | 14   | 0.325800      | 24   | 0.000100      | 34   | 0.000100      |
| 6    | 14.083900     | 16   | 0.008600      | 26   | 0.000100      | 36   | 0.000100      |
| 8    | 8.247200      | 18   | 0.000300      | 28   | 0.000100      | 38   | 0.000100      |
| 10   | 4.171000      | 20   | 0.000100      | 30   | 0.000100      | 40   | 0.000100      |

### 保存与加载

```python
model.save_pretrained("./model/")

model.from_pretrained("./model/")
```

### 推理

```python title='print(model.net[0].weight, "\n", model.net[0].bias)'
Parameter containing:
tensor([[ 1.9996, -3.4005]], requires_grad=True)
 Parameter containing:
tensor([4.2004], requires_grad=True)
```

```python
model.predict(torch.tensor([[2.0, 3.0], [6.0, 7.0]]), torch.device("cpu"))
```

```python
tensor([-2.0019, -7.6055])
```

可以看到基本和答案吻合了。

## 参考资料

<div class="grid cards" markdown>

- 动手学深度学习

    ---
[线性回归的简单实现](https://zh.d2l.ai/chapter_linear-networks/linear-regression-concise.html)

- 昇腾社区

    ---
[封装nn.Module并送入Trainer](https://www.hiascend.com/forum/thread-0231151069966226074-1-1.html)

- 博客园

    ---
[HuggingFace遇到的坑](https://www.cnblogs.com/zhangxuegold/p/17534627.html)

- DBGPT

    ---
[财务报表商务分析机器人](https://github.com/eosphoros-ai/dbgpts/blob/main/workflow/financial-robot-app/financial_robot_app/model.py)

</div>
