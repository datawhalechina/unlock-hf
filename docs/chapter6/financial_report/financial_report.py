import pandas as pd
from datasets import Dataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from transformers import (
    Qwen2ForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model


# def get_device():
#     return torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
#
# device = get_device()

df = pd.read_json(
    "./data/financial_report.jsonl",
    lines=True,
    orient="records",
)

id2labels = dict(zip(pd.factorize(df["intent"])[0].tolist(), df["intent"]))
labels2id = {v: k for k, v in id2labels.items()}

df["labels"] = df["intent"].map(labels2id)

ds = Dataset.from_pandas(df)

ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=2024)

model = Qwen2ForSequenceClassification.from_pretrained(
    "/data/czq/tjx/unlock-hf/model",
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

model.print_trainable_parameters()


def tokenize_and_align_labels(examples, tokenizer, max_length):
    tokenized_inputs = tokenizer(
        examples["question"],
        padding=True,
        max_length=max_length,
        truncation=True,
    )
    return tokenized_inputs


tokenizer = AutoTokenizer.from_pretrained(
    "/data/czq/tjx/unlock-hf/model",
    pad_token="<|endoftext|>",
    trust_remote_code=True,
)

tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

intent_dict = id2labels


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    accuracy = (preds == labels).mean()
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


model.config.pad_token_id = tokenizer.pad_token_id

tokenized_ds = ds.map(
    lambda x: tokenize_and_align_labels(x, tokenizer, None),
    batched=True,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
