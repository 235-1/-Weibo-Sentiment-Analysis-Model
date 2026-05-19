from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
import evaluate
import torch


# 1. GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# 2. 模型（3分类）

model_name = "hfl/chinese-roberta-wwm-ext"

tokenizer = BertTokenizer.from_pretrained(model_name)

model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3   # 三分类
).to(device)


# 3. 数据集（Weibo 3分类）

dataset = load_dataset("deepeye/weibo-emotion-classification")


print(dataset)
print("columns:", dataset["train"].column_names)


# 4. label 处理（安全转换）
def convert_label(example):
    label_map = {
        "负面": 0,
        "中性": 1,
        "正面": 2
    }
    return {"labels": label_map[example["label"]]}
dataset = dataset.map(convert_label)
# 2. 切分
dataset = dataset["train"].train_test_split(test_size=0.1)

print(dataset)

# 5. tokenizer（自动适配 text）

def preprocess(example):
    return tokenizer(
        example["text"],   # 这个数据集是 text
        truncation=True,
        max_length=128
    )

dataset = dataset.map(preprocess)

# 删除无用列保留训练需要字段
dataset = dataset.remove_columns(
    [col for col in dataset["train"].column_names if col not in ["input_ids", "attention_mask", "labels"]]
)

dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)


# 6. metrics F1

f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": (preds == labels).mean(),
        "f1_macro": f1.compute(
            predictions=preds,
            references=labels,
            average="macro"
        )["f1"]
    }

# =========================
# 7. 动态padding（推荐）
# =========================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# 8. 训练参数
# =========================
training_args = TrainingArguments(
    output_dir="./output",

    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    num_train_epochs=3,
    weight_decay=0.01,

    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,

    fp16=torch.cuda.is_available(),

    load_best_model_at_end=True,
    metric_for_best_model="f1_macro"
)

# =========================
# 9. Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# 10. train
trainer.train()

# 11. save
save_path = "./models/weibo-sentiment-3class"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print("训练完成！模型已保存到:", save_path)