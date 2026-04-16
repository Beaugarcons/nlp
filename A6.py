import streamlit as st
import nltk
from nltk.util import ngrams
from collections import Counter
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

# --- 1. 页面配置与资源加载 ---
st.set_page_config(page_title="LM 训练与分析平台", layout="wide")

@st.cache_resource
def init_resources():
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except:
        pass
    # 加载蒸馏版模型，确保在 Streamlit Cloud 稳定运行
    bert = pipeline("fill-mask", model="distilbert-base-uncased")
    gpt2_pipe = pipeline("text-generation", model="distilgpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    gpt2_tok = GPT2Tokenizer.from_pretrained("distilgpt2")
    return bert, gpt2_pipe, gpt2_model, gpt2_tok

bert_pipe, gpt2_gen, gpt2_model, gpt2_tok = init_resources()

# --- 2. 自定义样式 ---
st.markdown(r"""
<style>
    .theory-box {
        background-color: #f0f7ff;
        border-left: 5px solid #007bff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .stat-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .stat-val { font-size: 24px; font-weight: bold; color: #007bff; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 语言模型 (LM) 训练与对比分析平台")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 n-gram 统计", "🧠 RNN 序列记忆", "🤖 模型架构对比", "📉 PPL 流畅度评价"])

# --- Tab 1: n-gram ---
with tab1:
    st.subheader("1. 统计学接龙：n-gram 模型")
    st.markdown("""
    <div class="theory-box">
    <b>💡 原理：</b> 就像玩“成语接龙”，模型通过统计过去词语出现的规律来预测下一个词。<br>
    <b>🎯 目的：</b> 训练电脑形成“常识”。例如统计出“吃”后面常跟“饭”而不是“书”。<br>
    <b>📈 结果：</b> 联合概率越高，说明这个句子在给定的知识库中越“合理”。
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        corpus_options = {
            "科技生活": "artificial intelligence is the key to the future. computers can learn from data.",
            "日常对话": "how are you today? i am fine thank you. the weather is very nice.",
            "文学名句": "to be or not to be that is the question. stay hungry stay foolish."
        }
        selected = st.selectbox("👉 选择预设参考语料（知识库）", options=list(corpus_options.keys()))
        corpus_text = st.text_area("知识库内容", corpus_options[selected], height=100)
    
    with c2:
        test_options = ["artificial intelligence", "the weather is nice", "stay hungry"]
        test_sent = st.selectbox("👉 选择或输入测试句", options=test_options, index=0)
        use_smooth = st.checkbox("开启平滑（防止遇到生词时概率变0）", value=True)

    # 计算逻辑
    tokens = nltk.word_tokenize(corpus_text.lower())
    v_size = len(set(tokens))
    bi_counts = Counter(ngrams(tokens, 2))
    tri_counts = Counter(ngrams(tokens, 3))
    
    test_tokens = nltk.word_tokenize(test_sent.lower())
    test_tri = list(ngrams(test_tokens, 3))
    
    prob, details = 1.0, []
    for tri in test_tri:
        c_tri = tri_counts.get(tri, 0)
        c_bi = bi_counts.get(tri[:2], 0)
        p = (c_tri + 1)/(c_bi + v_size) if use_smooth else (c_tri/c_bi if c_bi > 0 else 0)
        prob *= p
        details.append({"片段": " ".join(tri), "出现次数": c_tri, "条件概率": f"{p:.4f}"})
    
    st.markdown(f'<div class="stat-card"><div class="stat-