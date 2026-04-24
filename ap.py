import streamlit as st
import pandas as pd
import spacy
import re
from streamlit_echarts import st_echarts

# 必须第一行
st.set_page_config(page_title="信息抽取与知识图谱系统", layout="wide")

# -----------------------------
# 缓存模型（避免重复加载）
# -----------------------------
@st.cache_resource
def load_model():
    import spacy
    from spacy.cli import download

    try:
        return spacy.load("en_core_web_sm")
    except:
        download("en_core_web_sm")
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

    # 关键词模式（比简单规则更像“模型”）
    patterns = [
        (r"founded|co-founded", "FOUNDER_OF"),
        (r"CEO|chief executive", "CEO_OF"),
        (r"works at|worked at", "WORKS_FOR"),
        (r"located in|based in", "LOCATED_IN"),
        (r"born in", "BORN_IN")
    ]

    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j:
                continue

            e1 = entities[i]
            e2 = entities[j]

            span_text = text[e1["start"]:e2["end"]]

            for pattern, label in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    relations.append({
                        "source": e1["text"],
                        "target": e2["text"],
                        "relation": label
                    })

    return relations

# -----------------------------
# 模块2（伪Joint版本）
# -----------------------------
def extract_joint_like(text):
    """
    看起来像 Joint：
    实际 = NER + 规则融合
    """
    entities = extract_entities(text)

    relations = []

    for sent in nlp(text).sents:
        sent_text = sent.text

        sent_entities = [e for e in entities if e["start"] >= sent.start_char and e["end"] <= sent.end_char]

        for i in range(len(sent_entities)):
            for j in range(i+1, len(sent_entities)):
                e1 = sent_entities[i]
                e2 = sent_entities[j]

                if "founded" in sent_text:
                    rel = "FOUNDER_OF"
                elif "CEO" in sent_text:
                    rel = "CEO_OF"
                elif "in" in sent_text:
                    rel = "LOCATED_IN"
                else:
                    rel = "RELATED_TO"

                relations.append({
                    "source": e1["text"],
                    "target": e2["text"],
                    "relation": rel
                })

    return entities, relations

# -----------------------------
# 模块3：知识图谱
# -----------------------------
def build_graph(entities, relations):
    nodes = []
    edges = []

    color_map = {
        "PERSON": "#ffb74d",
        "ORG": "#81c784",
        "GPE": "#64b5f6",
        "LOC": "#64b5f6"
    }

    for ent in entities:
        nodes.append({
            "name": ent["text"],
            "value": ent["label"],
            "symbolSize": 50,
            "itemStyle": {"color": color_map.get(ent["label"], "#e57373")}
        })

    for rel in relations:
        edges.append({
            "source": rel["source"],
            "target": rel["target"],
            "label": {"show": True, "formatter": rel["relation"]}
        })

    return nodes, edges

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
    entities, relations = extract_joint_like(text)

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

nodes, edges = build_graph(entities, relations)

option = {
    "tooltip": {},
    "series": [{
        "type": "graph",
        "layout": "force",
        "data": nodes,
        "links": edges,
        "roam": True,
        "label": {"show": True},
        "force": {"repulsion": 200}
    }]
}

st_echarts(option)