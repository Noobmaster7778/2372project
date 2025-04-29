#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
import os

# ======================
# 定义 RelationDataProcessor（与训练时一致）
# ======================
class RelationDataProcessor:
    def __init__(self, max_seq_length=100):
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        self.rel2idx = {"<PAD>": 0, "无关系": 1}
        self.max_seq_length = max_seq_length
        self.pos_bins = 20  # 位置编码分桶数

    def process_data(self, raw_data):
        # 此处仅用于训练，不在推理阶段使用
        pass

    def encode_samples(self, samples):
        """编码样本数据，返回输入字典和标签数组（标签在推理时不使用）"""
        X_char, X_subj_pos, X_obj_pos, y = [], [], [], []
        for sample in samples:
            # 字符编码：每个 token 转换为索引
            char_ids = [self.char2idx.get(c, 1) for c in sample["tokens"]]
            # 位置编码：使用实体在 token 序列中的位置（subj 和 obj）
            subj_pos = self._position_bucket(sample["subj"]["start"], len(sample["tokens"]))
            obj_pos = self._position_bucket(sample["obj"]["start"], len(sample["tokens"]))
            # 填充序列
            char_ids = pad_sequences([char_ids], maxlen=self.max_seq_length, padding="post")[0]
            subj_pos = pad_sequences([subj_pos], maxlen=self.max_seq_length, padding="post")[0]
            obj_pos = pad_sequences([obj_pos], maxlen=self.max_seq_length, padding="post")[0]
            X_char.append(char_ids)
            X_subj_pos.append(subj_pos)
            X_obj_pos.append(obj_pos)
            y.append(self.rel2idx[sample["relation"]])
        return {
            "char_input": np.array(X_char),
            "subj_pos_input": np.array(X_subj_pos),
            "obj_pos_input": np.array(X_obj_pos)
        }, np.array(y)

    def _position_bucket(self, entity_pos, seq_length):
        positions = []
        for i in range(seq_length):
            distance = i - entity_pos
            if distance < -10:
                bucket = 0
            elif distance > 10:
                bucket = self.pos_bins - 1
            else:
                bucket = (distance + 10) // 2 + 1
            positions.append(bucket)
        return positions

# ======================
# 定义并注册自定义层 EnhancedRelationModel
# ======================
@register_keras_serializable(package="Custom", name="EnhancedRelationModel")
class EnhancedRelationModel(tf.keras.Model):
    def __init__(self, char_vocab_size, pos_vocab_size, num_relations, **kwargs):
        super(EnhancedRelationModel, self).__init__(**kwargs)
        self.char_vocab_size = char_vocab_size
        self.pos_vocab_size = pos_vocab_size
        self.num_relations = num_relations

        self.char_embed = tf.keras.layers.Embedding(char_vocab_size, 128, mask_zero=True)
        self.subj_pos_embed = tf.keras.layers.Embedding(pos_vocab_size, 32)
        self.obj_pos_embed = tf.keras.layers.Embedding(pos_vocab_size, 32)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2
        ))
        self.attention = tf.keras.layers.Attention(use_scale=True)
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.output_layer = tf.keras.layers.Dense(num_relations, activation='softmax')

    def call(self, inputs):
        char_input = inputs["char_input"]
        subj_pos = inputs["subj_pos_input"]
        obj_pos = inputs["obj_pos_input"]
        char_emb = self.char_embed(char_input)
        subj_emb = self.subj_pos_embed(subj_pos)
        obj_emb = self.obj_pos_embed(obj_pos)
        combined = tf.concat([char_emb, subj_emb, obj_emb], axis=-1)
        lstm_out = self.bilstm(combined)
        attn_out = self.attention([lstm_out, lstm_out])
        pooled = tf.reduce_mean(attn_out, axis=1)
        x = self.dense1(pooled)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "char_vocab_size": self.char_vocab_size,
            "pos_vocab_size": self.pos_vocab_size,
            "num_relations": self.num_relations
        })
        return config

    @classmethod
    def from_config(cls, config):
        # 如果配置中缺少参数，则设定默认值（请根据训练时参数调整）
        if "char_vocab_size" not in config:
            config["char_vocab_size"] = 75
        if "pos_vocab_size" not in config:
            config["pos_vocab_size"] = 20
        if "num_relations" not in config:
            config["num_relations"] = 17
        return cls(**config)

# ---------------------------
# 1. 加载关系识别模型（.keras 格式）
# ---------------------------
relation_model = load_model("optimized_relation_model.keras",
                              custom_objects={"EnhancedRelationModel": EnhancedRelationModel})
print("✅ 关系识别模型加载成功！")

# ---------------------------
# 2. 读取并解析 ner_output.txt，同时提取头部年份信息
# ---------------------------
input_ner_file = "ner_output.txt"
with open(input_ner_file, "r", encoding="utf-8") as f:
    ner_text = f.read()

# 文件格式假设：
# 原始句子：<句子>。
# 识别的实体：<实体列表>
# 样本之间空行分隔
pattern = re.compile(r"原始句子：(.*?)。\n识别的实体：(.*?)\n\n", re.DOTALL)
matches = pattern.findall(ner_text)

sentences_data = []
for sentence, entities_str in matches:
    # 检查是否以【...】开头（提取年份信息）
    header_match = re.match(r"【(.*?)】", sentence)
    if header_match:
        year_group = header_match.group(1)
        sentence_body = sentence[len(header_match.group(0)):].strip()
    else:
        year_group = "未分组"
        sentence_body = sentence.strip()
    try:
        entities = eval(entities_str)
    except Exception as e:
        print(f"解析实体出错：{e}\n实体字符串：{entities_str}")
        entities = []
    # 过滤掉标签为 "O" 的项
    filtered_entities = [e for e in entities if e[1] != "O"]
    # 将每个实体从元组转换为字典，并为其添加“start”属性（用在 tokens 中的索引）
    entities_d = []
    for idx, (text, etype) in enumerate(filtered_entities):
        entities_d.append({"text": text, "start": idx, "type": etype})
    sentences_data.append({
        "year_group": year_group,
        "sentence": sentence.strip(),      # 包含头部年份信息
        "sentence_body": sentence_body,      # 去除头部后的句子，用于特征构造
        "entities": entities_d
    })

print(f"共解析出 {len(sentences_data)} 个句子。")

# ---------------------------
# 3. 生成候选实体对（仅使用实体列表，不包含年份信息）
# ---------------------------
def generate_entity_pairs(entities):
    pairs = []
    n = len(entities)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((entities[i], entities[j]))
    return pairs

# ---------------------------
# 4. 利用 RelationDataProcessor 对候选样本进行编码
# ---------------------------
processor = RelationDataProcessor(max_seq_length=100)
# 注意：processor.char2idx 和 processor.rel2idx 是在训练时动态构建的，
# 这里我们直接使用训练时的配置（默认：词表大小约75，关系类别17）
# 若需要更精确，请从训练时保存的配置加载。

# ---------------------------
# 5. 构造候选样本，并进行关系预测
# ---------------------------
relation_results = []
for item in sentences_data:
    sentence = item["sentence"]
    sentence_body = item["sentence_body"]
    year_group = item["year_group"]
    entities = item["entities"]
    if len(entities) < 2:
        continue
    pairs = generate_entity_pairs(entities)
    # 构造 tokens 列表：训练时 tokens 是所有实体的文本组成的列表
    tokens = [e["text"] for e in entities]
    for pair in pairs:
        sample = {
            "tokens": tokens,
            "subj": pair[0],
            "obj": pair[1],
            "relation": "无关系"  # 此处关系不影响编码
        }
        # 编码样本（encode_samples 返回 (inputs_dict, y)）
        features, _ = processor.encode_samples([sample])
        # features 中包含 "char_input", "subj_pos_input", "obj_pos_input"
        pred = relation_model.predict(features)
        rel_label_idx = np.argmax(pred, axis=-1)[0]
        # 关系映射（更新后统一处理）
        relation_mapping = {
            0: "无关系",
            1: "Dispatch",
            2: "致赠关系",
            3: "Alliance",            # 将“盟约地点”统一为 Alliance
            4: "入侵关系",
            5: "军事行动",
            6: "婚姻关系",
            7: "War",
            8: "Occupation",
            9: "军事同盟",
            10: "Crowning",
            11: "天气事件",
            12: "Death",
            13: "Meeting",
            14: "Alliance",           # Alliance
            15: "Diplomatic Visit"
        }
        rel_label = relation_mapping.get(rel_label_idx, "Unknown")
        # 统一处理：如果预测结果为“盟约地点”或“Alliance”，输出统一为 Alliance
        if rel_label in ["盟约地点", "Alliance"]:
            rel_label = "Alliance"
        relation_results.append({
            "year_group": year_group,
            "sentence": sentence,
            "entity_pair": (pair[0]["text"], pair[1]["text"]),
            "relation": rel_label
        })

# ---------------------------
# 6. 按年份分组并写入输出文件
# ---------------------------
grouped_results = defaultdict(list)
for res in relation_results:
    grouped_results[res["year_group"]].append(res)

output_relation_file = "relation_output.txt"
with open(output_relation_file, "w", encoding="utf-8") as f_out:
    for group, results in grouped_results.items():
        f_out.write(f"【{group}】:\n")
        for res in results:
            f_out.write(f"句子：{res['sentence']}\n")
            f_out.write(f"实体对：{res['entity_pair']}\n")
            f_out.write(f"关系：{res['relation']}\n\n")
        f_out.write("\n")
        
print(f"✅ 关系识别结果已保存至 {output_relation_file}")
