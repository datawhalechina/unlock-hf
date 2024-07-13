---
comments: true
title: 自动模型
---
![models](imgs/models.png)

## 前言

无论是 CV 领域还是 NLP 领域，模型迭代正以惊人的速度进行着。从 CNN、RNN 到 Transformer，新的模型架构层出不穷，
面对**快速迭代的模型**和**分散的代码库**，开发者需要花费大量时间和精力在模型选择和代码调试上，而不是专注于任务本身。所以一个统一的平台或工具将这些代码都集中在一起，使用相同的标准是再好不过了，所以本文的主角 `AutoModel` 应运而生。

| `AutoModel` 特性 | 描述                                                          |
| :------------- | :---------------------------------------------------------- |
| 模型整合           | 整合来自不同领域和不同来源的模型。提供统一的平台，方便开发者调用各种模型。                       |
| 简化模型使用         | 提供简洁易用的 `API`，隐藏模型的复杂性。开发者只需几行代码即可完成模型的加载、训练和推理，无需关注底层实现细节。 |
| 自动化模型选择        | 根据任务类型和数据特点，自动推荐合适的模型。更进一步，自动搜索最优模型和超参数，无需开发者手动尝试。          |
| 代码复用和可扩展性      | 采用模块化设计，方便开发者复用代码。支持添加新的模型、自定义训练流程等，提高工具的灵活性和可扩展性。          |

## `AutoModel`

`Transformers` 提供了一种简单统一的方式来加载预训练模型。开发者可以使用 `AutoModel` 类加载预训练模型，就像使用 `AutoTokenizer` 加载分词器一样。关键区别在于，对于不同的任务，需要加载不同类型的 `AutoModel`。例如，对于文本分类任务，应该加载 `AutoModelForSequenceClassification`。以下是在自然语言处理领域实际使用中面向不同任务常用的 `AutoModel` 类别介绍。

| 类别                                 | 功能                   | 任务示例        |
| ---------------------------------- | -------------------- | ----------- |
| AutoModelForCausalLM               | 因果语言建模，预测下一个词        | 文本生成，对话系统   |
| AutoModelForMaskedLM               | 掩码语言建模，预测被掩盖的词       | 完形填空，词义消歧   |
| AutoModelForMaskGeneration         | 掩码生成，例如用于 BART 和 T 5 | 文本摘要，机器翻译   |
| AutoModelForSeq2SeqLM              | 序列到序列语言建模            | 机器翻译，文本摘要   |
| AutoModelForSequenceClassification | 序列分类                 | 情感分析，主题分类   |
| AutoModelForMultipleChoice         | 多项选择                 | 阅读理解        |
| AutoModelForNextSentencePrediction | 下一句预测                | 句子相似度       |
| AutoModelForTokenClassification    | 词级别分类                | 命名实体识别，词性标注 |
| AutoModelForQuestionAnswering      | 问答                   | 抽取式问答       |
| AutoModelForTextEncoding           | 文本编码，生成文本表示          | 文本相似度，语义搜索  |

!!! note
	在 HuggingFace 文档中 [AutoModel Class API](https://huggingface.co/docs/transformers/main/en/model_doc/auto) 可以查阅所有支持的 `AutoModel` 类别。不仅包含面向自然语言处理领域，还包含面向语音，计算机视觉等方向的 `AutoModel` 供开发者使用ヾ(≧▽≦*) o。

## 使用 `AutoModel`

### 加载预训练的 `AutoModel`

在自然语言处理领域有很多任务，这里选择使用预训练模型 `uer/gpt2-chinese-poem` 来体验古诗词生成任务，使用 `from_pretrained ` 方法并指定模型名称或本地路径即可加载预训练模型。

方式一：使用 `pipeline` 简化流程

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "uer/gpt2-chinese-poem"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)


text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
result = text_generator("[CLS]梅 山 如 积 翠 ，", max_length=50, do_sample=True)
```

```python title='result[0]["generated_text"]'
'[CLS]梅 山 如 积 翠 ， 历 乱 入 云 齐 。 欲 识 停 车 处 ， 烟 霞 锁 一 溪 。 涧 道 何 纡 回 ， 山 色 忽 如 赭 。 幽 人 隐 重 林 ， 松 萝 自 关 锁 。 独'
```

方式二：精细地控制文本生成过程

```python title='方式二'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "uer/gpt2-chinese-poem"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 设置参数
max_length = 50
do_sample = True

# 输入文本
text = "明月几时有"

# 将文本转换为token ID
input_ids = tokenizer(text, return_tensors="pt").input_ids

# 循环生成文本
for _ in range(max_length):
    # 获取模型输出
    outputs = model(input_ids=input_ids)

    # 获取最后一个token的预测概率分布
    next_token_logits = outputs.logits[:, -1, :]

    # 如果 do_sample 为 True，则进行采样
    if do_sample:
        next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
    else:
        # 否则选择概率最高的token
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

    # 将新生成的token ID添加到输入序列中
    input_ids = torch.cat([input_ids, next_token_id], dim=-1)

# 将生成的token ID解码成文本
generated_text = tokenizer.decode(input_ids[0])
```

```python title='generated_text'
[CLS] 明 月 几 时 有 [SEP] 缘 ， 不 露 尘 中 亦 自 圆 。 借 问 钟 声 何 处 起 ， 碧 云 相 对 卧 僧 禅 。 [SEP] 言 立 志 下 朝 阳 ， 长 学 其 人 峻 法 场 。 忠 愿 得 来 相 富 贵 ，
```

!!! note
	值得注意的是在上面的 `AutoModel` 类别表中，不同的任务模型可能需要传入不同的参数，比如 `AutoModelForSequenceClassification` 在使用 `from_pretrained` 加载模型的时候需要设置参数 `num_labels`，还有很多，当然不必全部记住，当开发者想使用某个任务模型的时候只需要到[官方文档](https://huggingface.co/docs/transformers/main/en/model_doc/auto)查询参数即可。


# 内容还在构建 ing