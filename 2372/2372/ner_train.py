# import json
# from transformers import AutoTokenizer
# from transformers import AutoConfig
# from transformers import DataCollatorForTokenClassification
# # # 读取 NER 数据
# with open("data/ner_1.json", "r", encoding="utf-8") as f:
#     ner_data = json.load(f)
# import json

# # 自动统计所有出现的标签
# unique_labels = {"O"}
# for item in ner_data:
#     for entity in item["entities"]:
#         entity_type = entity["type"]
#         unique_labels.add(f"B-{entity_type}")
#         unique_labels.add(f"I-{entity_type}")

# # 生成 label2id
# id2label = {i: label for i, label in enumerate(sorted(unique_labels))}
# label2id = {label: i for i, label in id2label.items()}
# num_labels = len(id2label)
# print("生成的标签集：", id2label)
# #加载 tokenizer

# model_name = "guwenner"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# def process_example(example):
#     text = example["sentence"]
#     entities = example["entities"]

#     tokens = list(text)  # 按字切分
#     labels = ["O"] * len(tokens)  # 初始化 labels，与 tokens 对齐

#     for entity in entities:
#         etype = entity["type"]
#         start = min(entity["start"], len(labels) - 1)
#         end = min(entity["end"], len(labels))

#         labels[start] = f"B-{etype}"
#         for i in range(start + 1, end):
#             labels[i] = f"I-{etype}"

#     # 用 tokenizer 编码
#     encoding = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=128)
#     word_ids = encoding.word_ids()  # 获取 tokenizer 的 word id 映射

#     aligned_labels = []
#     for word_id in word_ids:
#         if word_id is None:
#             aligned_labels.append(-100)  # [CLS], [SEP]
#         else:
#             aligned_labels.append(label2id[labels[word_id]])

#     encoding["labels"] = aligned_labels  # 这里返回的是 dict
#     return encoding  # ✅ 确保返回的是 dict，而不是 tuple


# # 处理所有数据
# processed_data = [process_example(item) for item in ner_data]

# # 划分训练集和验证集
# train_size = int(0.8 * len(processed_data))
# train_encodings = {k: [dic[k] for dic in processed_data[:train_size]] for k in processed_data[0].keys()}
# val_encodings = {k: [dic[k] for dic in processed_data[train_size:]] for k in processed_data[0].keys()}
# import torch
# from torch.utils.data import Dataset

# class NERDataset(Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         return item
#     def __len__(self):
#         return len(self.encodings["input_ids"])

# train_dataset = NERDataset(train_encodings)
# val_dataset = NERDataset(val_encodings)
# from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

# # 加载模型并修改 num_labels
# config = AutoConfig.from_pretrained(model_name)
# config.num_labels = num_labels
# config.id2label = id2label
# config.label2id = label2id


# model = AutoModelForTokenClassification.from_pretrained(
#     model_name, config=config, ignore_mismatched_sizes=True
# )

# # 定义训练参数
# training_args = TrainingArguments(
#     output_dir="./guwen_ner_finetuned",
#     num_train_epochs=5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     evaluation_strategy="epoch",
#     logging_dir="./ner_logs",
#     learning_rate=5e-5,
# )

# # 训练模型
# data_collator = DataCollatorForTokenClassification(tokenizer)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     data_collator=data_collator  # ✅ 添加 data_collator
# )

# trainer.train()

# output_dir = "guwenbert_finetuned"

# # 训练完成后保存
# model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)


# from transformers import AutoTokenizer, AutoModelForTokenClassification
# import torch

# # **加载 GuwenBERT**
# model_name = "guwenbert_finetuned"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name)

# # **读取已分词文本**
# segmented_file_path = "segmented_text.txt"
# with open(segmented_file_path, "r", encoding="utf-8") as f:
#     segmented_lines = [line.strip() for line in f.readlines() if line.strip()]

# # **确保 tokenizer 不会拆分你的词**
# all_words = set()
# for line in segmented_lines:
#     all_words.update(line.split())

# # ✅ **添加所有词到 tokenizer**
# tokenizer.add_tokens(list(all_words))
# model.resize_token_embeddings(len(tokenizer))

# # **设备选择**
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# def ner_inference(sentence):
#     inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    
#     # ✅ 确保 `inputs` 在相同设备
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     inputs = {k: v.to(device) for k, v in inputs.items()}  # 送到 GPU 或 CPU
    
#     # ✅ 也确保模型在同样的设备
#     model.to(device)
    
#     with torch.no_grad():
#         outputs = model(**inputs)

#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
#     tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())[1:-1]
#     pred_labels = predictions[1:-1]

#     results = []
#     for token, label_id in zip(tokens, pred_labels):
#         label_name = id2label[label_id]
#         results.append((token, label_name))

#     return results

# sentence = "三月，公及邾仪父盟于蔑。"
# print(ner_inference(sentence))
import json
from transformers import AutoTokenizer
from transformers import AutoConfig

# 读取 NER 数据
with open("data/ner_1.json", "r", encoding="utf-8") as f:
    ner_data = json.load(f)

# 加载 tokenizer
model_name = "ethanyt/guwen-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
import json



# 自动统计所有出现的标签
unique_labels = {"O"}
for item in ner_data:
    for entity in item["entities"]:
        entity_type = entity["type"]
        unique_labels.add(f"B-{entity_type}")
        unique_labels.add(f"I-{entity_type}")

# 生成 label2id
id2label = {i: label for i, label in enumerate(sorted(unique_labels))}
label2id = {label: i for i, label in id2label.items()}
num_labels = len(id2label)

print("生成的标签集：", id2label)
# 预处理数据
def process_example(example):
    text = example["sentence"]
    entities = example["entities"]

    tokens = list(text)  # 按字切分
    labels = ["O"] * len(tokens)  # 初始化 labels，与 tokens 对齐

    for entity in entities:
        etype = entity["type"]
        start = min(entity["start"], len(labels) - 1)
        end = min(entity["end"], len(labels))

        labels[start] = f"B-{etype}"
        for i in range(start + 1, end):
            labels[i] = f"I-{etype}"

    # 用 tokenizer 编码
    encoding = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=128)
    word_ids = encoding.word_ids()  # 获取 tokenizer 的 word id 映射

    aligned_labels = []
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)  # [CLS], [SEP]
        else:
            aligned_labels.append(label2id[labels[word_id]])

    encoding["labels"] = aligned_labels  # 这里返回的是 dict
    return encoding  # ✅ 确保返回的是 dict，而不是 tuple

# 处理所有数据
processed_data = [process_example(item) for item in ner_data]

# 划分训练集和验证集
train_size = int(0.8 * len(processed_data))
train_encodings = {k: [dic[k] for dic in processed_data[:train_size]] for k in processed_data[0].keys()}
val_encodings = {k: [dic[k] for dic in processed_data[train_size:]] for k in processed_data[0].keys()}
import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = NERDataset(train_encodings)
val_dataset = NERDataset(val_encodings)
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoConfig, AutoModelForTokenClassification

# 模型名称
model_name = "ethanyt/guwen-ner"

# 2. 加载原模型配置，并修改 num_labels
config = AutoConfig.from_pretrained(model_name)
config.num_labels = num_labels  # 确保 num_labels 变量已经定义

# 3. 重新加载模型，并忽略大小不匹配的权重
model = AutoModelForTokenClassification.from_pretrained(
    model_name, config=config, ignore_mismatched_sizes=True
)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./guwen_ner_finetuned",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./ner_logs",
    learning_rate=5e-5,
)

from transformers import DataCollatorForTokenClassification

# 定义数据整理器
data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator  # ✅ 添加 data_collator
)

trainer.train()

trainer.train()
def ner_inference(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    
    # ✅ 确保 `inputs` 在相同设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 送到 GPU 或 CPU
    
    # ✅ 也确保模型在同样的设备
    model.to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())[1:-1]
    pred_labels = predictions[1:-1]

    results = []
    for token, label_id in zip(tokens, pred_labels):
        label_name = id2label[label_id]
        results.append((token, label_name))

    return results


sentence = "三月，公及邾仪父盟于蔑。"
print(ner_inference(sentence))
