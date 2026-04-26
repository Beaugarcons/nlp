import streamlit as st
import pandas as pd
import spacy
import re
from pyvis.network import Network
import streamlit.components.v1 as components


st.set_page_config(page_title="信息抽取与知识图谱系统", layout="wide")

# -----------------------------
# 缓存模型（避免重复加载）
# -----------------------------
@st.cache_resource
def load_model():
    import spacy
    return spacy.load("en_core_web_sm")

nlp = load_model()

# -----------------------------
# 模块1：NER（改良版）
# -----------------------------
def extract_entities(text):
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })

    return entities

# -----------------------------
# BIO标注
# -----------------------------
def to_bio(text, entities):
    tokens = text.split()
    bio = ["O"] * len(tokens)

    for ent in entities:
        ent_tokens = ent["text"].split()
        for i in range(len(tokens)):
            if tokens[i:i+len(ent_tokens)] == ent_tokens:
                bio[i] = "B-" + ent["label"]
                for j in range(1, len(ent_tokens)):
                    if i+j < len(tokens):
                        bio[i+j] = "I-" + ent["label"]

    return list(zip(tokens, bio))

# -----------------------------
# 高亮显示
# -----------------------------
def highlight_text(text, entities):
    colors = {
        "PERSON": "#ffd54f",
        "ORG": "#81c784",
        "GPE": "#64b5f6",
        "LOC": "#64b5f6"
    }

    for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
        color = colors.get(ent["label"], "#e57373")
        span = f"<span style='background-color:{color};padding:2px;border-radius:4px'>{ent['text']}({ent['label']})</span>"
        text = text[:ent["start"]] + span + text[ent["end"]:]

    return text

# -----------------------------
# 模块2：改良版关系抽取（弱监督+规则）
# -----------------------------
def extract_relations_pipeline(text, entities):
    relations = []

    relation_rules = [
        ("FOUNDER_OF", ["PERSON"], ["ORG"], ["founded", "co-founded"]),
        ("CEO_OF", ["PERSON"], ["ORG"], ["ceo", "chief executive"]),
        ("LOCATED_IN", ["ORG"], ["GPE", "LOC"], ["in", "based in", "located in"]),
        ("BORN_IN", ["PERSON"], ["GPE", "LOC"], ["born in"])
    ]

    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j:
                continue

            e1 = entities[i]
            e2 = entities[j]

            # 限制距离（关键）
            if abs(e1["start"] - e2["start"]) > 80:
                continue

            # 只看中间文本
            if e1["start"] < e2["start"]:
                middle_text = text[e1["end"]:e2["start"]].lower()
            else:
                middle_text = text[e2["end"]:e1["start"]].lower()

            for rel, subj_types, obj_types, keywords in relation_rules:
                if e1["label"] in subj_types and e2["label"] in obj_types:
                    if any(k in middle_text for k in keywords):
                        relations.append((e1["text"], e2["text"], rel))

    # 👉 去重
    relations = list(set(relations))

    return [
        {"source": s, "target": t, "relation": r}
        for s, t, r in relations
    ]

# -----------------------------
# 模块2（伪Joint版本）
# -----------------------------
def extract_joint_like_fixed(text, entities):
    relations = []

    # 关系定义（带类型约束）
    relation_rules = [
        ("FOUNDER_OF", ["PERSON"], ["ORG"], ["founded", "co-founded"]),
        ("CEO_OF", ["PERSON"], ["ORG"], ["ceo", "chief executive"]),
        ("LOCATED_IN", ["ORG"], ["GPE", "LOC"], ["in", "based in", "located in"]),
        ("BORN_IN", ["PERSON"], ["GPE", "LOC"], ["born in"])
    ]

    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j:
                continue

            e1 = entities[i]
            e2 = entities[j]

            # 👉 限制距离（关键！）
            if abs(e1["start"] - e2["start"]) > 80:
                continue

            # 👉 只看中间文本（核心）
            if e1["start"] < e2["start"]:
                middle_text = text[e1["end"]:e2["start"]]
            else:
                middle_text = text[e2["end"]:e1["start"]]

            middle_text = middle_text.lower()

            # 👉 匹配规则
            for rel, subj_types, obj_types, keywords in relation_rules:
                if e1["label"] in subj_types and e2["label"] in obj_types:
                    if any(k in middle_text for k in keywords):
                        relations.append({
                            "source": e1["text"],
                            "target": e2["text"],
                            "relation": rel
                        })

    relations = list({
        (r["source"], r["target"], r["relation"])
        for r in relations
    })

    return [
        {"source": s, "target": t, "relation": r}
        for s, t, r in relations
    ]

# -----------------------------
# 模块3：知识图谱
# -----------------------------
def show_graph(entities, relations):
    net = Network(height="500px", width="100%", directed=True)

    color_map = {
        "PERSON": "#ffb74d",
        "ORG": "#81c784",
        "GPE": "#64b5f6",
        "LOC": "#64b5f6"
    }

    # 节点
    for ent in entities:
        net.add_node(
            ent["text"],
            label=ent["text"],
            color=color_map.get(ent["label"], "#e57373")
        )

    # 边
    for rel in relations:
        net.add_edge(
        rel["source"],
        rel["target"],
        title=rel["relation"],
        label=rel["relation"]   # 👈 加这个
    )

    net.save_graph("graph.html")

    with open("graph.html", "r", encoding="utf-8") as f:
        components.html(f.read(), height=550)

# -----------------------------
# 页面
# -----------------------------
st.title("信息抽取与知识图谱构建系统")

text = st.text_area("请输入文本", "Steve Jobs founded Apple in California. Tim Cook is the CEO of Apple.")

mode = st.selectbox("选择抽取模式", ["Pipeline", "Joint-like（推荐）"])

show_bio = st.checkbox("显示BIO标注")

# -----------------------------
# 抽取逻辑
# -----------------------------
if mode == "Pipeline":
    entities = extract_entities(text)
    relations = extract_relations_pipeline(text, entities)
else:
    entities = extract_entities(text)
    relations = extract_joint_like_fixed(text, entities)

# -----------------------------
# 模块1展示
# -----------------------------
st.subheader("命名实体识别")

if show_bio:
    st.write(to_bio(text, entities))
else:
    st.markdown(highlight_text(text, entities), unsafe_allow_html=True)

# -----------------------------
# 模块2展示
# -----------------------------
st.subheader("关系抽取")

if relations:
    df = pd.DataFrame(relations)
    st.dataframe(df)
else:
    st.write("未抽取到关系")

# -----------------------------
# 模块3展示
# -----------------------------
st.subheader("知识图谱")
show_graph(entities, relations)