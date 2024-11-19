---
comments: true
title: modeling
---
## 代码

### `BertEmbeddings`

#### 初始化

```python
class BertEmbeddings(nn.Module):
"""Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

```

该段代码根据配置文件：

1. 创建一个用于处理词嵌入的嵌入层。
    - 嵌入词典的大小为 `vocab_size`。
    - 词向量的维度为 `hidden_size`。
    - 填充词元为 `pad_token_id`。
2. 创建位置编码层。
    - 嵌入词典的大小为 `max_position_embedding`。
    - 词向量的维度为 `hidden_size`。
3. 创建段落嵌入层。
    - 嵌入词典的大小为 `type_vocab_size`。
    - 词向量的维度为 `hidden_size`。
4. 创建层归一化层。
    - 归一化维度为 `hidden_size`。
    - $\epsilon$ 大小为 `layer_norm_eps`。
5. 创建随机丢弃层。
    - 随机丢弃的概率为`hidden_dropout_prob`。
6. 获取位置编码类型，默认为绝对位置编码。
7. 设置大小为`(1, max_position_embeddings)`的寄存器变量`position_ids`。
8. 设置大小同`position_ids`相同，数据为`0`的寄存器变量`token_type_ids`。

???+ note "层归一化"
    层归一化数学表达式为：

    $$
    y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta \tag{1}
    $$

    其中：

    | 符号              | 描述                                        | 具体作用                                              |
    | ----------------- | ------------------------------------------- | ----------------------------------------------------- |
    | $x$               | 当前层输入特征向量                          | 需要进行归一化的特征。                                |
    | $\mathrm{E}[x]$   | 输入特征的均值                              | 用于去中心化，保证数据均值为 0。                      |
    | $\mathrm{Var}[x]$ | 输入特征的方差                              | 用于标准化，保证数据标准差为 1。                      |
    | $\epsilon$        | 防止除零的小常数，默认值 $1 \times 10^{-5}$ | 提高数值稳定性，避免除以零。                          |
    | $\gamma$          | 缩放参数（可学习）                          | 用于调整输出的范围。                                  |
    | $\beta$           | 偏移参数（可学习）                          | 用于调整输出的分布中心。                              |
    | $y$               | 归一化后的输出                              | 具有可调整分布（均值和方差受 $\gamma, \beta$ 控制）。 |

#### 前向传播

```python
class BertEmbeddings(nn.Module):
    def __init__():
        ...

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape,
                    dtype=torch.long,
                    device=self.position_ids.device,
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
```

前向传播的入参及其说明：

| 参数                     | 默认值 | 说明                               |
| ------------------------ | ------ | ---------------------------------- |
| `input_ids`              |        | 输入序列的词汇表索引。             |
| `token_type_ids`         |        | 输入序列的段落类型索引。           |
| `position_ids`           |        | 输入序列的位置索引。               |
| `inputs_embeds`          |        | 输入序列的嵌入表示。               |
| `past_key_values_length` | 0      | 过去键值对的长度，用于处理长序列。 |

前向传播的流程为：

1. 获取输入序列的维度大小`(batch_size, seq_length)`，并获取语句序列长度`(seq_length)`。
2. 根据过去键值对的长度和语句序列长度，获取位置索引。
3. 如果输入序列的向量表示为空，则使用词嵌入层获取输入序列的向量表示。
4. 使用段落嵌入层获取段落向量。
5. 将词向量、段落向量和位置编码求和。
6. 将求和结果进行层归一化$\rightarrow$随机丢弃$\rightarrow$返回结果。
