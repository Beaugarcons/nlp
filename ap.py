import streamlit as st
import pandas as pd
import spacy
from spacy import displacy
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="信息抽取与知识图谱系统", layout="wide")

# -----------------------------
# 缓存模型（避免重复加载，并适配 Streamlit Cloud 自动下载）
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # 如果在云端未找到模型，自动下载
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_model()

# -----------------------------
# 模块1：NER与BIO标注
# -----------------------------
def extract_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    return entities

def to_bio(doc):
    # 直接利用 spaCy 底层的 token 属性，百分百精准，不会因标点符号错位
    bio_data = []
    for token in doc:
        if token.is_space:
            continue
        tag = f"{token.ent_iob_}-{token.ent_type_}" if token.ent_iob_ != "O" else "O"
        bio_data.append({"Token": token.text, "BIO Tag": tag})
    return pd.DataFrame(bio_data)

def highlight_text(doc):
    # 使用 spaCy 官方可视化工具，解决手动拼接字符串导致的索引错乱问题
    colors = {"PERSON": "#ffd54f", "ORG": "#81c784", "GPE": "#64b5f6", "LOC": "#64b5f6"}
    options = {"ents": ["PERSON", "ORG", "GPE", "LOC"], "colors": colors}
    html = displacy.render(doc, style="ent", options=options, page=False)
    # 添加简单的 CSS 使得在 Streamlit 黑暗模式下也能看清文字
    return f"<div style='background-color:#ffffff; padding:15px; border-radius:8px; color:black; line-height:2.0;'>{html}</div>"

# -----------------------------
# -----------------------------
# 模块2：关系抽取（改良增强版）
# -----------------------------
def extract_relations(doc, entities):
    relations = []
    
    # 关系定义（关系名称，主体类型，客体类型，触发词）
    # 适当扩充了触发词库，提高召回率
    relation_rules = [
        ("FOUNDER_OF", ["PERSON"], ["ORG"], ["founded", "co-founded", "creator", "founder"]),
        ("CEO_OF", ["PERSON"], ["ORG"], ["ceo", "chief executive", "head"]),
        ("LOCATED_IN", ["ORG"], ["GPE", "LOC"], ["in", "based in", "located in", "headquartered"]),
        ("BORN_IN", ["PERSON"], ["GPE", "LOC"], ["born in", "from"])
    ]

    # ✅ 1. 遍历 doc.sents 实现“多句子互不干扰”
    for sent in doc.sents:
        # 获取当前这句话里的所有实体
        sent_entities = [e for e in entities if e["start"] >= sent.start_char and e["end"] <= sent.end_char]

        # ✅ 2. 同一句内任意两个实体进行组合（全面覆盖单句内的多个关系）
        for i in range(len(sent_entities)):
            for j in range(len(sent_entities)):
                if i == j:
                    continue

                e1 = sent_entities[i]
                e2 = sent_entities[j]

                # ❌ 移除了原有的 distance > 80 的限制，允许跨越长句子提取关系

                # ✅ 3. 重构上下文窗口截取逻辑
                # 获取两个实体在句子中最靠左的起点和最靠右的终点
                left_bound = min(e1["start"], e2["start"])
                right_bound = max(e1["end"], e2["end"])
                
                # 向左向右各延伸20个字符的窗口，防止关系词跑到了实体的外围
                span_start = max(sent.start_char, left_bound - 20)
                span_end = min(sent.end_char, right_bound + 20)
                
                # 截取这段上下文作为判断依据
                context_text = doc.text[span_start:span_end].lower()

                # ✅ 4. 匹配规则
                for rel, subj_types, obj_types, keywords in relation_rules:
                    # 先判断实体类型是否匹配主客体约束
                    if e1["label"] in subj_types and e2["label"] in obj_types:
                        # 只要触发词存在于我们的上下文窗口内，就判定关系成立
                        if any(k in context_text for k in keywords):
                            relations.append((e1["text"], e2["text"], rel))

    # 去重处理，防止同一对实体关系因为多个关键词被重复添加
    relations = list(set(relations))
    
    return [{"source": s, "target": t, "relation": r} for s, t, r in relations]

# -----------------------------
# 模块3：知识图谱渲染
# -----------------------------
def show_graph(entities, relations):
    net = Network(height="500px", width="100%", directed=True, bgcolor="#222222", font_color="white")
    
    # 调整物理引擎参数，使拖拽和排版更平滑
    net.force_atlas_2based()

    color_map = {
        "PERSON": "#ffd54f",
        "ORG": "#81c784",
        "GPE": "#64b5f6",
        "LOC": "#64b5f6"
    }

    # 提取有关系的实体名称，过滤掉孤立节点（如果需要展示所有实体可去掉此逻辑）
    connected_nodes = set()
    for rel in relations:
        connected_nodes.add(rel["source"])
        connected_nodes.add(rel["target"])

    # 添加节点
    for ent in entities:
        if ent["text"] in connected_nodes:
            net.add_node(
                ent["text"],
                label=ent["text"],
                title=f"Type: {ent['label']}", # 鼠标悬浮提示
                color=color_map.get(ent["label"], "#e57373")
            )

    # 添加边
    for rel in relations:
        net.add_edge(
            rel["source"],
            rel["target"],
            title=rel["relation"],
            label=rel["relation"],
            color="#ffffff"
        )

    # 保存并读取 HTML
    try:
        net.save_graph("graph.html")
        with open("graph.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=550)
    except Exception as e:
        st.error(f"图谱渲染出错: {e}")

# -----------------------------
# 页面 UI 布局
# -----------------------------
st.title("🕸️ 信息抽取与知识图谱构建系统")

default_text = "Steve Jobs founded Apple in California. Tim Cook is the CEO of Apple. Bill Gates was born in Seattle."
text = st.text_area("请输入要分析的文本 (英文)", default_text, height=100)

if text:
    doc = nlp(text)
    entities = extract_entities(doc)
    relations = extract_relations(doc, entities)

    st.markdown("---")
    
    # -----------------------------
    # 模块1展示
    # -----------------------------
    st.subheader("📍 模块 1: 命名实体识别 (NER)")
    show_bio = st.checkbox("查看底层 BIO 标注序列")
    
    if show_bio:
        st.dataframe(to_bio(doc), use_container_width=True)
    else:
        if entities:
            st.markdown(highlight_text(doc), unsafe_allow_html=True)
        else:
            st.info("未识别到指定的实体类型 (PERSON, ORG, GPE, LOC)。")

    st.markdown("---")
    
    # -----------------------------
    # 模块2展示
    # -----------------------------
    st.subheader("🔗 模块 2: 实体关系抽取")
    if relations:
        df = pd.DataFrame(relations)
        # 调整列表头大写显示更专业
        df.columns = ["主体 (Subject)", "客体 (Object)", "关系词 (Predicate)"]
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("暂未提取到符合预设规则的关系。请尝试包含 'founded', 'CEO', 'based in', 'born in' 等关键词的文本。")

    st.markdown("---")
    
    # -----------------------------
    # 模块3展示
    # -----------------------------
    st.subheader("🌐 模块 3: 知识图谱交互可视化")
    if relations:
        show_graph(entities, relations)
    else:
        st.info("需要提取到关系后才能构建知识图谱。")