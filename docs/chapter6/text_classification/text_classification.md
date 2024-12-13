---
comments: true
title: 基金年报问答意图识别
---

![qwen2](./imgs/qwen2.png)

## 前言

!!! quote "引用"
    1. 该篇所用数据集来自蚂蚁团队的`DBGPT`项目中的金融商业分析案例。原始开源地址可以查看[:material-github:DB-GPT-Hub](https://github.com/eosphoros-ai/DB-GPT-Hub/blob/main/src/dbgpt-hub-nlu/datasets/financial_report/data/financial_report.jsonl)。
    2. ==**该篇所用代码引用自DataWhale团队倾心打造的开源大模型食用指南self-llm，该项目是一个围绕开源大模型、针对国内初学者、基于Linux平台的中国宝宝专属大模型教程，针对各类开源大模型提供包括环境配置、本地部署、高效微调等技能在内的全流程指导，简化开源大模型的部署、使用和应用流程，让更多的普通学生、研究者更好地使用开源大模型，帮助开源、自由的大模型更快融入到普通学习者的生活中。原始开源地址可以查看[:material-github:self-llm](https://github.com/datawhalechina/self-llm)**==

大模型在财务报表分析中的应用正成为垂直领域的热门方向:fire:。

尽管大模型能够高效理解复杂财务规则并生成合理的分析结果，但由于财务报表信息庞大且复杂，对数据分析的精准度要求很高，传统的通用`RAG`和`Agent`的解决方案往往难以满足需求。

例如，在处理具体查询（如季度营业净利润）时，传统方法通过向量检索召回相关文本，但财报中多处信息可能引发误判。

此外，涉及财务指标计算时（如毛利率、净利率），需要综合多方面数据进行分析，这进一步增加了复杂性。

为解决这些问题，可以结合财务领域的专业知识，添加**专门的外部模块**进行功能增强。

本文的所做的文本分类任务旨在解决大模型在意图识别任务中存在的模糊性问题，提升其对具体意图的精准识别能力。

## 代码

### 导入函数库

```python
import os

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, LoraModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
```

### 读取数据集

```python
report = pd.read_json("./data/financial_report.jsonl", lines=True, orient="records")

report.columns = ["input", "output"]
ds = Dataset.from_pandas(report)
ds = ds.train_test_split(test_size=0.2, seed=2025) # 固定种子划分数据集
```

利用`pandas`将格式为`jsonl`的数据集读取为`DataFrame`格式，然后使用`from_pandas`方法将其转换为`DatasetDict`格式。

部分样本数据如下：

| input                                                                                                  | output           |
| :----------------------------------------------------------------------------------------------------- | :--------------- |
| 能否根据2020年金宇生物技术股份有限公司的年报，给我简要介绍一下报告期内公司的社会责任工作情况？         | 报告解读分析     |
| 请根据江化微2019年的年报，简要介绍报告期内公司主要销售客户的客户集中度情况，并结合同行业情况进行分析。 | 报告解读分析     |
| 2019年四方科技电子信箱是什么?                                                                          | 年报基础信息问答 |
| 研发费用对公司的技术创新和竞争优势有何影响？                                                           | 专业名称解释     |
| 康希诺生物股份公司在2020年的资产负债比率具体是多少，需要保留至小数点后两位？                           | 财务指标计算     |
| $\dots$                                                                                                | $\dots$          |

### 加载模型

```python
model_name_or_checkpoint_path = "Qwen/Qwen2-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_checkpoint_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model.enable_input_require_grads()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)


model = get_peft_model(model=model, peft_config=peft_config)
```

本次指令微调使用到的模型为阿里开源的`Qwen2-7B-Instruct`，相比于基础模型`Qwen2-7B`，`Qwen2-7B-Instruct`经过专门的指令调优，显著增强了对指令任务的理解和执行能力。

### 加载分词器

```python
tokenizer = AutoTokenizer.from_pretrained(model_name_or_checkpoint_path)
pad_token_id = tokenizer.pad_token_id
max_length = 384
```

### 指令

```python title="指令"
PROMPT = """你是一个财报领域的意图识别的专家，你的任务是根据给定的财报文本，判断该文本的意图，可供你选择的意图有以下几种：
1. 报告解读分析
2. 年报基础信息问答
3. 专业名称解释
4. 财务指标计算
5. 统计对比分析
6. 其他问题

注意：输出的只能为用户输入文本所对应的单个类别，其他的文本都被视为错误的输出。
"""
```

### 数据预处理

```python hl_lines="3 4 5"
def datapipe(data):
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{data['input']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False,
    )

    response = tokenizer(data["output"], add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"]) + response["input_ids"] + [pad_token_id]
    )
    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    labels = labels[:max_length]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

ds = ds.map(datapipe, remove_columns=ds["train"].column_names)

train_ds = ds["train"]
test_ds = ds["test"]
```

| 步骤 | 代码解释                                                                                                                                               |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | 使用分词器对指令（提示词）进行编码。设置`add_special_tokens=False`，避免自动添加特殊标记，确保符合模型的输入格式。                                     |
| 2    | 使用分词器对标签进行编码。同样设置`add_special_tokens=False`，保持一致性。                                                                             |
| 3    | 将指令和标签编码后的`input_ids`进行拼接，并在末尾添加`pad_token_id`。   确保后续使用`DataCollatorForSeq2Seq`时能够根据`pad_token_id`实现序列长度对齐。 |
| 4    | 将指令和标签编码后的`attention_mask`进行拼接，并在末尾添加值为1的标记。 确保模型训练时能够忽略`pad_token_id`对应的位置。                               |
| 5    | 将拼接后的`input_ids`设置为`labels`，并将非标签部分的位置值替换为$-100$。 保证模型在训练时忽略非标签部分的损失计算。                                   |
| 6    | 对序列进行常规的截断操作。 确保序列长度符合模型的输入要求。                                                                                            |

!!! success "小贴士"
    1. `pytorch`中的交叉熵损失函数在计算损失函数的时候默认忽略`-100`对应的位置产生的损失。
    2. 因为本文专注于意图识别任务，所以指令只有一个，但是在实际任务中，指令可能有多个，一般的做法是，在预处理数据集的时候，将其规整为三列数据，第一列为指令，第二列为输入文本，第三列为输出文本，而不是像本文，将指令单独处理。

### 定义动态数据整理

```python
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
```

### 定义训练参数

```python
args = TrainingArguments(
    output_dir="./output/Qwen7B-Instruct-fintuned-financial-report",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=1,
    save_strategy="epoch",
    learning_rate=1e-4,
    gradient_checkpointing=True,
    save_total_limit=2,
)
```

### 定义训练器

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=collator,
)
```

### 训练

```python
trainer.train()
```

下面是训练时的过程结果。

| Epoch | Loss   | Epoch | Loss   |
| ----- | ------ | ----- | ------ |
| 0.06  | 0.7576 | 0.54  | 0.0001 |
| 0.12  | 0.0146 | 0.60  | 0.0071 |
| 0.18  | 0.0655 | 0.65  | 0.0005 |
| 0.24  | 0.0076 | 0.71  | 0.0002 |
| 0.30  | 0.0014 | 0.77  | 0.0000 |
| 0.36  | 0.0061 | 0.83  | 0.0000 |
| 0.42  | 0.0004 | 0.89  | 0.0001 |
| 0.48  | 0.0004 | 0.95  | 0.0000 |

### 推理

#### 加载模型和分词器

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = """你是一个财报领域的意图识别的专家，你的任务是根据给定的财报文本，判断该文本的意图，可供你选择的意图有以下几种：
1. 报告解读分析
2. 年报基础信息问答
3. 专业名称解释
4. 财务指标计算
5. 统计对比分析
6. 其他问题

注意：输出的只能为用户输入文本所对应的单个类别，其他的文本都被视为错误的输出。
"""

model_name_or_checkpoint_path = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_checkpoint_path, use_fast=False, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_checkpoint_path, device_map="auto", torch_dtype=torch.bfloat16
)

model.requires_grad = False

lora_model = PeftModel.from_pretrained(
    model,
    model_id='./output/Qwen7B-Instruct-fintuned-financial-report',
)

```

#### 定义预测函数

```python
def predict(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids, max_new_tokens=512, do_sample=False
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
```

#### 预测

```python
messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {
        "role": "user",
        "content": "xx股份公司在2024年的资产负债比率具体是多少",
    },
]

with torch.no_grad():
    response = predict(messages, model, tokenizer)
```

| 问题                                       | 预测             |
| ------------------------------------------ | ---------------- |
| xx股份公司在2024年的资产负债比率具体是多少 | 财务指标计算     |
| 2019年四方科技电子信箱是什么               | 年报基础信息问答 |
| $\cdots$                                   | $\cdots$         |

## 参考资料

<div class="grid cards" markdown>

- 开源大模型食用指南

    ---

    [:whale:datawhalechina/self-llm](https://github.com/datawhalechina/self-llm/tree/master)

- 商业数据分析案例

    ---

    [:bird:基于DB-GPT的财报分析助手](https://www.yuque.com/eosphoros/dbgpt-docs/cmogrzbtmqf057oe)

</div>
