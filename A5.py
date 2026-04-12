import streamlit as st
import spacy
import requests
import time
import nltk
import spacy.cli
spacy.cli.download("en_core_web_sm")
from fastcoref import spacy_component

# 页面配置
st.set_page_config(
    page_title="篇章分析综合平台",
    layout="wide",
    page_icon="📊"
)

# =============================
# ✅ 模型加载（已修复）
# =============================
@st.cache_resource
def load_models():
    nltk.download('punkt', quiet=True)
    import spacy.cli
    spacy.cli.download("en_core_web_sm")

    nlp = spacy.load("en_core_web_sm")

    # ✅ 使用 HuggingFace 模型（云端可用）
    nlp.add_pipe(
        "fastcoref",
        config={'model_name_or_path': 'biu-nlp/f-coref'}
    )

    return nlp

nlp = load_models()

# =============================
# 工具函数
# =============================
def get_rst_data():
    url = "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/train.txt"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            lines = [line for line in res.text.split('\n') if line.strip()][:3]
            return lines
    except:
        pass
    return [
        "Although the products are good <S> they are expensive . <S>",
        "The company reported earnings <S> that exceeded expectations . <S>"
    ]

def render_coref_html(text, clusters):
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    highlights = []

    for cluster_idx, cluster in enumerate(clusters):
        color = colors[cluster_idx % len(colors)]
        for start, end in cluster:
            highlights.append((start, end, color, cluster_idx))

    highlights.sort(key=lambda x: x[0], reverse=True)
    html_text = text

    for start, end, color, cluster_idx in highlights:
        html_text = (
            html_text[:start]
            + f'<span style="background:{color}30;padding:3px;border-radius:4px;">'
            + html_text[start:end]
            + f'<sup>[{cluster_idx+1}]</sup></span>'
            + html_text[end:]
        )

    return html_text

# =============================
# UI
# =============================
st.title("📊 篇章分析综合平台")

tab1, tab2, tab3 = st.tabs([
    "🔪 话语分割",
    "🔗 篇章关系",
    "🔍 指代消解"
])

# =============================
# 模块1：话语分割
# =============================
with tab1:
    st.subheader("EDU 分割")

    samples = get_rst_data()
    text = " ".join([s.replace("<S>", "") for s in samples])

    doc = nlp(text)

    current = []
    for token in doc:
        current.append(token.text)
        if token.is_punct or token.dep_ == "advcl":
            st.write(" ".join(current))
            current = []

# =============================
# 模块2：浅层篇章分析
# =============================
with tab2:
    st.subheader("篇章连接词分析")

    user_input = st.text_area(
        "输入文本",
        "I like apples although they are expensive."
    )

    if st.button("分析"):
        connectors = ["although", "because", "however", "since"]

        found = False
        for conn in connectors:
            if conn in user_input.lower():
                found = True
                parts = user_input.split(conn)

                st.success(f"检测到连接词: {conn}")
                st.write("Arg1:", parts[0])
                st.write("Arg2:", parts[1])
                break

        if not found:
            st.warning("未检测到连接词")

# =============================
# 模块3：指代消解（已修复）
# =============================
with tab3:
    st.subheader("指代消解")

    text = st.text_area(
        "输入文本",
        "Barack Obama visited Cairo. He gave a speech."
    )

    if st.button("运行指代消解"):
        with st.spinner("处理中..."):
            doc = nlp(text)

            # ✅ 正确调用
            clusters = doc._.coref_clusters

            st.success(f"检测到 {len(clusters)} 个簇")

            html = render_coref_html(text, clusters)
            st.markdown(html, unsafe_allow_html=True)

            for i, cluster in enumerate(clusters):
                mentions = [text[s:e] for s, e in cluster]
                st.write(f"簇{i+1}:", mentions)

# 页脚
st.markdown("---")
st.caption("NLP Discourse App • Streamlit + SpaCy + FastCoref")