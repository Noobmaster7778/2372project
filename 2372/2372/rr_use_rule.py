#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import json

# ---------------------------
# 1. 定义规则
# ---------------------------
global_exclude_types = {"Time", "Weather"}

RELATION_TYPE_RULES = {
    "Marriage": {("Person", "Person")},
    "War": {("Person", "Person"), ("Person", "Location"), ("Location", "Person"), ("Location", "Location")},
    "Alliance": {("Person", "Person"), ("Person", "Location"), ("Location", "Person")},
    "Diplomatic Visit": {("Person", "Person"), ("Person", "Location")},
    "Death": {("Person", "Event"), ("Event", "Person")},
    "Crowning": {("Person", "Person")},
    "Dispatch": {("Person", "Person"), ("Person", "Location")},
    "Gift": {("Person", "Person")},
    "Occupation": {("Person", "Location"), ("Location", "Person")},
    "Meeting": {("Person", "Person")},
    "Move": {("Person", "Location")},
    "Natural Disaster": {("Event", "Location"), ("Location", "Event")},
    "Extra": {("Any", "Any")},
    "无关系": {("Any", "Any")}
}

TRIGGER_WORDS = {
    "Marriage": {"婚", "嫁", "娶"},
    "War": {"伐", "战", "侵", "败", "杀", "灭", "攻"},
    "Alliance": {"盟", "结盟", "会盟"},
    "Diplomatic Visit": {"使", "来聘", "朝", "访"},
    "Death": {"卒", "崩", "薨", "死", "去世"},
    "Crowning": {"即位", "加冕", "立位", "封"},
    "Dispatch": {"使", "派", "遣"},
    "Gift": {"赠", "馈赠"},
    "Occupation": {"占领", "侵占"},
    "Meeting": {"会见", "见面", "会", "会合"},
    "Move": {"入", "迁", "逃", "奔", "降"},
    "Natural Disaster": {"洪水", "地震", "旱灾", "虫灾", "霜冻", "冰雹", "大灾", "大水"},
    "Extra": set()  # 可留空兜底
}

def is_valid_pair(rel, e1, e2, sentence):
    if e1["type"] in global_exclude_types or e2["type"] in global_exclude_types:
        return False
    if rel == "Marriage" and e1["text"] == e2["text"]:
        return False
    types = (e1["type"], e2["type"])
    if types not in RELATION_TYPE_RULES.get(rel, set()) and (types[::-1] not in RELATION_TYPE_RULES.get(rel, set())):
        return False
    if TRIGGER_WORDS.get(rel):
        if not any(t in sentence for t in TRIGGER_WORDS[rel]):
            return False
    return True

# ---------------------------
# 2. 读取 NER 输出
# ---------------------------
input_ner_file = "ner_output.txt"
with open(input_ner_file, "r", encoding="utf-8") as f:
    ner_text = f.read()

pattern = re.compile(r"原始句子：(.*?)\n识别的实体：(.*?)\n\n", re.DOTALL)
matches = pattern.findall(ner_text)

samples = []
for sentence_raw, entities_str in matches:
    try:
        entities_raw = eval(entities_str)
    except:
        continue
    sentence_body = re.sub(r"^【.*?】", "", sentence_raw.strip())
    entities = []
    for (text, tag) in entities_raw:
        if tag == "O": continue
        start = sentence_body.find(text)
        if start == -1: continue
        entities.append({"text": text, "type": tag, "start": start, "end": start+len(text)})
    if len(entities) >= 2:
        samples.append({"sentence": sentence_body, "entities": entities})

# ---------------------------
# 3. 识别关系
# ---------------------------
def generate_entity_pairs(entities):
    return [(entities[i], entities[j]) for i in range(len(entities)) for j in range(i+1, len(entities)) if entities[i]["start"] != entities[j]["start"]]

relation_results = []
for sample in samples:
    sentence = sample["sentence"]
    entities = sample["entities"]
    pairs = generate_entity_pairs(entities)
    for e1, e2 in pairs:
        matched = False
        for rel in TRIGGER_WORDS:
            if is_valid_pair(rel, e1, e2, sentence):
                relation_results.append({
                    "sentence": sentence,
                    "entity_pair": (e1, e2),
                    "relation": rel,
                    "score": 1.0
                })
                matched = True
                break
        if not matched:
            relation_results.append({
                "sentence": sentence,
                "entity_pair": (e1, e2),
                "relation": "无关系",
                "score": 0.0
            })

# ---------------------------
# 4. 保存结果
# ---------------------------
with open("relation_output_rule.txt", "w", encoding="utf-8") as f_out:
    for res in relation_results:
        f_out.write(f"句子：{res['sentence']}\n")
        f_out.write(f"实体对：{res['entity_pair']}\n")
        f_out.write(f"关系：{res['relation']} (规则识别)\n\n")

print(f"✅ 规则法识别完成，共输出 {len(relation_results)} 条关系。结果保存在 relation_output_rule.txt")
