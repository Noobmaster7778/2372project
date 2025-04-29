#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import random
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import accuracy_score
import numpy as np

##############################################################################
# 0. 初始化模型名称和 tokenizer
##############################################################################
model_name = "guwenner"  # 或者 "bert-base-chinese" 等
tokenizer = AutoTokenizer.from_pretrained(model_name)

##############################################################################
# 1. 数据增强函数（仅对非实体字符进行替换）
##############################################################################
def augment_example(example, aug_probability=0.3):
    """
    对输入 example 进行数据增强：
    对非实体部分的字符，以一定概率替换成随机常用汉字。
    实体部分保持不变，保证标签不受影响。
    """
    text = example["sentence"]
    entities = example["entities"]
    # 构建实体 mask
    entity_mask = [False] * len(text)
    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        for i in range(start, end):
            if 0 <= i < len(text):
                entity_mask[i] = True

    # 常用汉字列表（可根据需要扩充）
    common_chars = list("的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经之进着等部度家电力如水高自理起小物加量各天")
    new_text_chars = list(text)
    for i, ch in enumerate(new_text_chars):
        if not entity_mask[i]:
            if random.random() < aug_probability:
                new_text_chars[i] = random.choice(common_chars)
    new_text = "".join(new_text_chars)
    return {"sentence": new_text, "entities": entities}

##############################################################################
# 2. 读取 NER 数据
##############################################################################
with open("data/ner_1.json", "r", encoding="utf-8") as f:
    ner_data = json.load(f)
print(f"总样本数: {len(ner_data)}")

##############################################################################
# 3. 构建标签集与 label2id
##############################################################################
unique_labels = set(["O"])
for item in ner_data:
    for ent in item["entities"]:
        etype = ent["type"]
        unique_labels.add("B-" + etype)
        unique_labels.add("I-" + etype)
unique_labels = sorted(list(unique_labels))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)
print("标签数:", num_labels)
print("标签集:", unique_labels)

##############################################################################
# 4. 编码函数：字符级对齐 & BIO
##############################################################################
def encode_char_level(example):
    text = example["sentence"]
    entities = example["entities"]
    tokens = list(text)
    n = len(tokens)
    labels = ["O"] * n
    for ent in entities:
        etype = ent["type"]
        start = ent["start"]
        end = ent["end"]
        if start < 0 or end <= start or end > n:
            continue
        labels[start] = f"B-{etype}"
        for i in range(start + 1, end):
            labels[i] = f"I-{etype}"
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=256,
    )
    word_ids = encoding.word_ids()
    aligned_labels = []
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        else:
            aligned_labels.append(label2id[labels[word_id]])
    encoding["labels"] = aligned_labels
    return encoding

##############################################################################
# 5. 划分训练/验证数据并进行数据增强（仅对训练数据增强）
##############################################################################
random.shuffle(ner_data)
train_ratio = 0.8
train_size = int(train_ratio * len(ner_data))
train_ner = ner_data[:train_size]
val_ner = ner_data[train_size:]
print("训练样本数（原始）:", len(train_ner))
print("验证样本数:", len(val_ner))

augmented_train_ner = [augment_example(example, aug_probability=0.3) for example in train_ner]
combined_train_ner = train_ner + augmented_train_ner
print("合并后训练样本数:", len(combined_train_ner))

processed_train_data = [encode_char_level(item) for item in combined_train_ner]
processed_val_data = [encode_char_level(item) for item in val_ner]

class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
    def __len__(self):
        return len(self.encodings["input_ids"])

train_encodings = {k: [dic[k] for dic in processed_train_data] for k in processed_train_data[0].keys()}
val_encodings = {k: [dic[k] for dic in processed_val_data] for k in processed_val_data[0].keys()}

train_dataset = NERDataset(train_encodings)
val_dataset = NERDataset(val_encodings)

##############################################################################
# 6. 加载模型及配置
##############################################################################
config = AutoConfig.from_pretrained(model_name)
config.num_labels = num_labels
config.label2id = label2id
config.id2label = id2label
# 这里可以再次初始化 tokenizer（可选，已在上面定义过）
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True,
)
data_collator = DataCollatorForTokenClassification(tokenizer)

##############################################################################
# 7. 定义准确率评估函数 compute_metrics
##############################################################################
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_labels.append(l)
                pred_labels.append(p)
    acc = accuracy_score(true_labels, pred_labels)
    return {"accuracy": acc}

##############################################################################
# 8. 训练参数设置
##############################################################################
training_args = TrainingArguments(
    output_dir="./guwen_ner_char",
    num_train_epochs=9,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./ner_logs_char",
    learning_rate=1e-4,
    save_total_limit=2,
)

##############################################################################
# 9. 初始化 Trainer 并训练
##############################################################################
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

##############################################################################
# 10. 评估 & 保存最终模型
##############################################################################
eval_results = trainer.evaluate()
print("最终评估结果:", eval_results)
trainer.save_model("./guwen_ner_char")
print("✅ 模型已保存到 ./guwen_ner_char")

##############################################################################
# 11. 推理函数：字符级预测及 BIO 合并
##############################################################################
def ner_inference_char(text):
    tokens = list(text)
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    enc = enc.to(device)
    with torch.no_grad():
        outputs = model(**enc)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    results = []
    for i, w_id in enumerate(word_ids):
        if w_id is None:
            continue
        label_id = preds[i]
        label_str = id2label[label_id]
        char = tokens[w_id]
        results.append((char, label_str))
    return results

def merge_bio_results(char_label_list):
    merged = []
    current_word = []
    current_label = None
    for char, label in char_label_list:
        if label.startswith("B-"):
            if current_word:
                merged.append(("".join(current_word), current_label))
                current_word = []
            current_label = label[2:]
            current_word = [char]
        elif label.startswith("I-"):
            if current_label == label[2:]:
                current_word.append(char)
            else:
                if current_word:
                    merged.append(("".join(current_word), current_label))
                current_label = label[2:]
                current_word = [char]
        else:
            if current_word:
                merged.append(("".join(current_word), current_label))
                current_word = []
            merged.append((char, "O"))
            current_label = None
    if current_word:
        merged.append(("".join(current_word), current_label))
    return merged

test_sentence = "三月，公及邾仪父盟于蔑。"
inference_res = ner_inference_char(test_sentence)
print("\n推理结果（字符级）:", inference_res)
merged_output = merge_bio_results(inference_res)
print("推理结果（合并后）:", merged_output)
