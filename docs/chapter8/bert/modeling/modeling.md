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

### `BertSelfAttention`

#### 初始化

```python
class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores():
        ...
```

该段代码：

1. 首先判断隐藏层大小能否被自注意力头数整除，同时是否满足没有`embedding_size`属性，如果不满足条件则抛出异常。
2. 初始化自注意力头个数`num_attention_heads`。
3. 初始化每个自注意力头的维度`attention_head_size`。
4. 初始化所有自注意力头的大小`all_head_size`。
5. 初始化`q`、`k`、`v`的线性层。维度为`hidden_size`到`all_head_size`。
6. 初始化随机丢弃层。丢弃概率为`attention_probs_dropout_prob`。
7. 获取位置编码类型，默认为绝对位置编码。
8. 如果位置编码类型为相对位置编码，则初始化距离编码层。
9. 初始化是否为解码器。

#### transpose_for_scores

```python
class BertSelfAttention(nn.Module):
    def __init__():
        ...
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
```

该段代码：

1. 在实际运算中，`x`代表经过`q`,`k`,`v`之一处理过后的张量，其维度为`(batch_size, seq_length, hidden_size)`。
2. 通过索引`x`的前两个维度，再结合自注意力头数和每个自注意力头的维度，将`new_x_shape`设置为`(batch_size, seq_length, num_attention_heads, attention_head_size)`。
3. 将`x`按照`new_x_shape`的维度进行数据变化，变化后的维度为`(batch_size, seq_length, num_attention_heads, attention_head_size)`。
4. 通过`permute`函数，将`x`的维度变化为`(batch_size, num_attention_heads, seq_length, attention_head_size)`。

#### forward

```python
class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        ...
    def transpose_for_scores():
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
```
