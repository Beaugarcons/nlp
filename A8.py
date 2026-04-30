import streamlit as st
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

st.set_page_config(page_title="机器翻译对比与评测系统", layout="wide")

# -----------------------------
# 缓存模型（避免重复加载）
# -----------------------------

@st.cache_resource
def load_nmt():
    return pipeline(
        task="translation",
        model="Helsinki-NLP/opus-mt-en-zh"
    )

# -----------------------------
# 简单规则翻译（逐词直译）
# -----------------------------

from deep_translator import GoogleTranslator

def rule_based_translate(text):
    words = text.split()
    translated = []

    for w in words:
        try:
            zh = GoogleTranslator(source='en', target='zh-CN').translate(w)
        except:
            zh = w
        translated.append(zh)

    return " ".join(translated)

# -----------------------------
# 页面结构
# -----------------------------
st.title("机器翻译机制对比与评测平台")

tab1, tab2, tab3 = st.tabs([
    "NMT翻译引擎",
    "规则翻译 vs NMT",
    "BLEU评测"
])

# =============================
# 模块1：NMT翻译
# =============================
with tab1:
    st.subheader("神经机器翻译（NMT）")

    # 示例按钮
    if st.button("使用示例句"):
        st.session_state["text1"] = "It rains cats and dogs."

    text = st.text_area("请输入英文句子", key="text1")

    if st.button("开始翻译", key="nmt"):
        if text.strip() != "":
            with st.spinner("模型翻译中..."):
                result = translator(text)
                translation = result[0]['translation_text']

            st.success("翻译结果：")
            st.write(translation)

# =============================
# 模块2：规则 vs NMT
# =============================
with tab2:
    st.subheader("规则翻译 vs 神经翻译")

    # 示例按钮
    if st.button("使用示例句（对比）"):
        st.session_state["text2"] = "It rains cats and dogs."

    text2 = st.text_area("请输入英文句子", key="text2")

    if st.button("对比翻译"):
        if text2.strip() != "":
            with st.spinner("生成中..."):
                nmt_result = translator(text2)[0]['translation_text']
                rule_result = rule_based_translate(text2)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("规则直译")
                st.write(rule_result)

            with col2:
                st.markdown("神经翻译（NMT）")
                st.write(nmt_result)

# =============================
# 模块3：BLEU评测
# =============================
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt')

with tab3:
    st.subheader("BLEU自动评测")

    # -----------------------------
    # 示例按钮（关键）
    # -----------------------------
    if st.button("使用示例句（It rains cats and dogs.）"):
        st.session_state["src"] = "It rains cats and dogs."
        st.session_state["ref1"] = "倾盆大雨"
        st.session_state["ref2"] = "雨 大 倾盆"
        st.session_state["ref3"] = "下着大雨"

    # -----------------------------
    # 输入区域（可手动改）
    # -----------------------------
    source = st.text_area("英文原文", key="src")
    reference1 = st.text_area("参考译文1（标准）", key="ref1")
    reference2 = st.text_area("参考译文2（语序错误）", key="ref2")
    reference3 = st.text_area("参考译文3（同义表达）", key="ref3")

    # Candidate（可自动生成 or 手动改）
    if st.button("生成机器译文"):
        if source.strip():
            with st.spinner("生成中..."):
                st.session_state["cand"] = translator(source)[0]['translation_text']

    candidate = st.text_area("机器译文（Candidate）", key="cand")

    # -----------------------------
    # BLEU计算
    # -----------------------------
    def compute_bleu(ref, cand):
        if not ref or not cand:
            return 0

        ref_tokens = list(ref)
        cand_tokens = list(cand)

        smoothie = SmoothingFunction().method1

        score = sentence_bleu(
            [ref_tokens],
            cand_tokens,
            smoothing_function=smoothie
        )
        return round(score, 4)

    if st.button("计算BLEU"):
        score1 = compute_bleu(reference1, candidate)
        score2 = compute_bleu(reference2, candidate)
        score3 = compute_bleu(reference3, candidate)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("标准参考")
            st.write(reference1)
            st.write(f"BLEU: {score1}")

        with col2:
            st.markdown("语序错误")
            st.write(reference2)
            st.write(f"BLEU: {score2}")

        with col3:
            st.markdown("同义表达")
            st.write(reference3)
            st.write(f"BLEU: {score3}")

        # -----------------------------
        # 自动解释（动态）
        # -----------------------------
        st.markdown("结果分析")

        st.markdown(f"""
标准参考 BLEU = {score1}  
语序错误 BLEU = {score2}  
同义表达 BLEU = {score3}  

可以观察：

1. 如果语序错误但词汇相同，BLEU 仍可能较高  
2. 如果语义正确但表达不同（同义句），BLEU 可能较低  

说明 BLEU 的特点：

- 优点：自动化、可量化  
- 缺点：不理解语义，只基于 n-gram 匹配  
        """)