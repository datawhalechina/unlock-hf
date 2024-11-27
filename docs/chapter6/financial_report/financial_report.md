---
comments: true
title: 基金年报问答意图识别
---

## 前言

!!! quote "引用"
    该篇代码摘抄自蚂蚁团队的`DBGPT`项目中的金融商业分析案例。原始开源地址可以查看[:material-github:DB-GPT-Hub](https://github.com/eosphoros-ai/DB-GPT-Hub/blob/main/src/dbgpt-hub-nlu/dbgpt_hub_nlu/intent.py)。

大模型在财务报表分析中的应用正成为垂直领域的热门方向:fire:。

尽管大模型能够高效理解复杂财务规则并生成合理的分析结果，但由于财务报表信息庞大且复杂，对数据分析的精准度要求很高，传统的通用`RAG`和`Agent`的解决方案往往难以满足需求。

例如，在处理具体查询（如季度营业净利润）时，传统方法通过向量检索召回相关文本，但财报中多处信息可能引发误判。

此外，涉及财务指标计算时（如毛利率、净利率），需要综合多方面数据进行分析，这进一步增加了复杂性。

为解决这些问题，可以结合财务领域的专业知识，添加**专门的外部模块**进行功能增强。

本文的所做的文本分类任务旨在解决大模型在意图识别任务中存在的模糊性问题，提升其对具体意图的精准识别能力。

!!! note "小结"
    大模型在大多数领域确实是能够取得不错的效果，但是在特定领域，仍然需要结合传统的方法，进行功能增强，比如：

    - AI+导航：如果单纯使用大模型进行地名的提取，在粗粒度的情况下，识别国家、省份或城市等地名，大模型能够取得极好的效果，但是在细粒度的情况下，如从这句话`MSC HAMBURG号将于11月25日停靠上海洋山港三期码头D1泊位，卸货至6号仓库。`提取`POI`点，大模型往往错误提取到`海洋山港三期码头D1泊位`,或者`6号仓库`，但是仅仅将这些传递给导航系统，往往无法找到正确的位置。
    - AI+问答：大语言模型是擅长做问答的，但是从实际业务场景来看，用户的输入往往是非标准的，带有大量的噪声，甚至是缺失，错误的信息，因此往往需要结合**查询改写**，**重排序**等功能，将问题标准化。
    - $\cdots$

## 代码

### 导入函数库

```python
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from peft import PeftConfig, PeftModel, LoraConfig, TaskType, get_peft_model
from transformers import (
    Qwen2ForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
```

### 读取数据集

```python
df = pd.read_json(
    "./data/financial_report.jsonl",
    lines=True,
    orient="records",
)
```

部分样本数据如下：

| question                                                                                               | intent           |
| :----------------------------------------------------------------------------------------------------- | :--------------- |
| 能否根据2020年金宇生物技术股份有限公司的年报，给我简要介绍一下报告期内公司的社会责任工作情况？         | 报告解读分析     |
| 请根据江化微2019年的年报，简要介绍报告期内公司主要销售客户的客户集中度情况，并结合同行业情况进行分析。 | 报告解读分析     |
| 2019年四方科技电子信箱是什么?                                                                          | 年报基础信息问答 |
| 研发费用对公司的技术创新和竞争优势有何影响？                                                           | 专业名称解释     |
| 康希诺生物股份公司在2020年的资产负债比率具体是多少，需要保留至小数点后两位？                           | 财务指标计算     |
| $\dots$                                                                                                | $\dots$          |

```python title="label2id"
{
    "报告解读分析": 0,
    "年报基础信息问答": 1,
    "专业名称解释": 2,
    "财务指标计算": 3,
    "统计对比分析": 4,
    "其他问题": 5,
}
```

```python title="id2label"
{
    0: "报告解读分析",
    1: "年报基础信息问答",
    2: "专业名称解释",
    3: "财务指标计算",
    4: "统计对比分析",
    5: "其他问题",
}
```

```python
df["labels"] = df["intent"].map(labels2id)
```

利用 `pandas` 的映射函数，将 `intent` 列文本数据转化为对应的数字标签，并设置列名为 `labels` 。转换后的样本数据案例为：

| question                                                                                               | intent           |  labels |
| :----------------------------------------------------------------------------------------------------- | :--------------- | ------: |
| 能否根据2020年金宇生物技术股份有限公司的年报，给我简要介绍一下报告期内公司的社会责任工作情况？         | 报告解读分析     |       0 |
| 请根据江化微2019年的年报，简要介绍报告期内公司主要销售客户的客户集中度情况，并结合同行业情况进行分析。 | 报告解读分析     |       0 |
| 2019年四方科技电子信箱是什么?                                                                          | 年报基础信息问答 |       1 |
| 研发费用对公司的技术创新和竞争优势有何影响？                                                           | 专业名称解释     |       2 |
| 康希诺生物股份公司在2020年的资产负债比率具体是多少，需要保留至小数点后两位？                           | 财务指标计算     |       3 |
| $\dots$                                                                                                | $\dots$          | $\dots$ |

将数据集转化为`datasets.Dataset`对象：

```python
ds = Dataset.from_pandas(df)

ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=2024)
```

### 加载模型

```python
model = Qwen2ForSequenceClassification.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    num_labels=len(id2labels),
    id2label=id2labels,
    label2id=labels2id,
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, peft_config)
```

使用参数高效微调后打印可训练参数如下：

```python title="model.print_trainable_parameters()"
trainable params: 4,404,480 || all params: 498,442,624 || trainable%: 0.8836
```

### 数据预处理

```python
def tokenize_and_align_labels(examples, tokenizer, max_length):
    tokenized_inputs = tokenizer(
        examples["question"],
        padding=True,
        max_length=max_length,
        truncation=True,
    )
    return tokenized_inputs
```

该函数用于数据编码。

### 加载分词器

```python
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    pad_token="<|endoftext|>",
    trust_remote_code=True,
)

tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
model.config.pad_token_id = tokenizer.pad_token_id
```

许多语言模型，尤其是生成模型，通常使用特殊的标记（例如 `<|endoftext|>`）表示文本的结束，但未必专门定义了 `pad_token`。

### 定义评价指标

```python
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    accuracy = (preds == labels).mean()
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
```

### 数据集处理

```python
tokenized_ds = ds.map(lambda x: tokenize_and_align_labels(x, tokenizer, None), batched=True)
```

- 利用分词器对文本进行批量编码。

### 定义动态数据整理

```python
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
```

### 定义训练参数

```python
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=2,
    per_device_train_batch_size=32,
    learning_rate=1e-4,
    weight_decay=0.01,
    do_train=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

### 定义训练器

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```

### 训练

```python
trainer.train()
```

下面是训练时的过程结果。

| 轮次 | 评估损失 | 准确率 | 精确率 | 召回率 | F1值   |
| ---- | -------- | ------ | ------ | ------ | ------ |
| 1    | 0.0151   | 0.9985 | 0.9985 | 0.9985 | 0.9985 |
| 2    | 0.0156   | 0.9985 | 0.9985 | 0.9985 | 0.9985 |

### 推理

```python
adapter_path = './output'
peft_config = PeftConfig.from_pretrained(adapter_path)

model = Qwen2ForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path,
    num_labels=len(label2id)
)

tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, adapter_path)
# merge and unload is necessary for inference
model = model.merge_and_unload()

model.config.pad_token_id = tokenizer.pad_token_id
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

def infer(question):
    inputs = tokenizer(
        question,
        padding="longest",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.cpu().numpy()[0]

question = 'xx股份公司在2024年的资产负债比率具体是多少'
prediction = infer(question)
intent_label = {v: k for k, v in label2id.items()}[prediction]
```

| 问题                                       | 预测             |
| ------------------------------------------ | ---------------- |
| xx股份公司在2024年的资产负债比率具体是多少 | 财务指标计算     |
| 2019年四方科技电子信箱是什么               | 年报基础信息问答 |
| $\cdots$               | $\cdots$ |

## 参考资料

<div class="grid cards" markdown>

- 开源的AI原生数据应用开发框架

    ---

    [:material-github:DB-GPT](https://github.com/eosphoros-ai/DB-GPT)

- 商业数据分析案例

    ---

    [:bird:基于DB-GPT的财报分析助手](https://www.yuque.com/eosphoros/dbgpt-docs/cmogrzbtmqf057oe)

</div>
