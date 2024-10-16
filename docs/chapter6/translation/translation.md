---
comments: true
title: "文本翻译"
---

![translation](./imgs/translation.png)

## 前言

## 代码

### 加载数据集

```python
from datasets import load_dataset

raw_datasets = load_dataset("wmt/wmt19", "zh-en", trust_remote_code=True)
```

在这里选择的配置"zh-en"进行中英文文本翻译，下面为全部数据集详细信息：

<iframe
  src="https://huggingface.co/datasets/wmt/wmt19/embed/viewer/zh-en/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

```json title="raw_datasets"
DatasetDict({
    'train': Dataset({
        'features': ['translation'],
        'num_rows': 25984574
    }),
    'validation': Dataset({
        'features': ['translation'],
        'num_rows': 3981
    })
})
```

```json title='raw_datasets["train"][0]'
{'translation': {'en': '1929 or 1989?', 'zh': '1929年还是1989年?'}}
```

### 加载分词器


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
```

查看分词器原始语言与目标语言

```python title="tokenizer.source_lang, tokenizer.target_lang"
('zho', 'eng')
```

### 定义分词函数

```python
def tokenize_fn(examples):
    inputs = [example['zh'] for example in examples['translation']]
    labels = [example['en'] for example in examples['translation']]

    model_inputs = tokenizer(
        inputs,
        text_target=labels,
        max_length = 128)

    return model_inputs
```

### 数据集转化

```python
tokenized_datasets = raw_datasets.map(
    tokenize_fn,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)
```

### 加载模型

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
```

### 定义整理函数

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

### 定义评估函数

```python
# pip install sacrebleu
import evaluate

metric = evaluate.load("sacrebleu")
```

```python
import numpy as np


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}
```

### 定义超参数

```python
from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    f"finetuned-zh-to-en",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
)
```

### 定义训练器

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

### 训练

```python
trainer.train()
```