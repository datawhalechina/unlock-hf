---
comments: true
title: configuration
---
## 前言

## 代码解读

### 加载词典函数

```python
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab
```

| 步骤 | 操作说明                                                                          |
| ---- | --------------------------------------------------------------------------------- |
| 1    | 函数接收词表文件路径 `vocab_file`。                                               |
| 2    | 创建有序字典 `vocab`，以保证词表的顺序能够保留。                                  |
| 3    | 按行读取文件内容，得到列表 `tokens`。                                             |
| 4    | 遍历列表 `tokens`，删除每行末尾的换行符，并将词与索引以键值对的形式存入 `vocab`。 |
| 5    | 返回构建完成的字典 `vocab`。                                                      |

### 空格清理和分词函数

```python
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens
```

| 步骤 | 操作说明                                              |
| ---- | ----------------------------------------------------- |
| 1    | 使用`strip()` 去掉字符串 `text` 的首尾空白字符。      |
| 2    | 如果清理后的 `text` 是空字符串，直接返回空列表 `[]`。 |
| 3    | 按照空格分割 `text` ，存入列表 `tokens`，并最终返回。 |

### 空格分词器

#### 初始化方法

```python
class WordpieceTokenizer:
"""Runs WordPiece tokenization."""
    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
```

入参说明：

| 参数                       | 说明               |
| -------------------------- | ------------------ |
| `vocab`                    | 词表文件路径       |
| `unk_token`                | 未知标记           |
| `max_input_chars_per_word` | 单个单词最大字符数 |

#### 分词方法

```python
class WordpieceTokenizer:
"""Runs WordPiece tokenization."""
    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        ...

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
```
