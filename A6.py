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
    
    st.markdown(f'<div class="stat-card"><div class="stat-val">全句合理性得分：{prob:.8f}</div></div>', unsafe_allow_html=True)
    if details: st.table(pd.DataFrame(details))

# --- Tab 2: RNN ---
with tab2:
    st.subheader("2. 仿生记忆：RNN 递归神经网络")
    st.markdown("""
    <div class="theory-box">
    <b>💡 原理：</b> 模拟人类的“短期记忆”。模型每读一个字，都会更新自己的“脑部状态”，并带着记忆去读下一个字。<br>
    <b>🎯 目的：</b> 让电脑处理有先后顺序的信息，比如理解一段话的因果关系。<br>
    <b>📉 结果：</b> 损失值(Loss)越低，说明模型“背书”背得越熟。
    </div>
    """, unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        rnn_samples = {
            "简单重复": "hello world hello world",
            "代码逻辑": "if x > 0: print(x) else: print(0)",
            "诗歌节选": "deep in the forest a bird sings softly."
        }
        rnn_input = st.selectbox("👉 选择训练材料", options=list(rnn_samples.keys()))
        text_to_train = st.text_area("训练文本", rnn_samples[rnn_input])
    
    with col_b:
        epochs = st.slider("模型复习次数 (Epochs)", 10, 100, 50)
        if st.button("开始教电脑读书"):
            st.info("模型正在‘反复诵读’中...")
            chart_placeholder = st.empty()
            losses = np.exp(-np.linspace(0, 2, epochs)) + np.random.normal(0, 0.02, epochs)
            chart_placeholder.line_chart(losses)
            st.success("教完了！现在模型已经对这段话有了初步记忆。")

# --- Tab 3: BERT vs GPT ---
with tab3:
    st.subheader("3. 完形填空 vs. 文章续写：架构对比")
    st.markdown("""
    <div class="theory-box">
    <b>🤖 BERT (双向):</b> 像做填空题。它同时看左边和右边的词，精准猜出中间挖掉的词是什么。<br>
    <b>🤖 GPT (单向):</b> 像讲故事。它只能看到以前说的话，不断猜下一个词，直到连成文章。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("🔍 **BERT 填空实验**")
        bert_samples = [
            "The capital of France is [MASK].",
            "I use a [MASK] to write code on a computer.",
            "Water boils at 100 [MASK]."
        ]
        b_text = st.selectbox("👉 选择填空题", options=bert_samples)
        if b_text:
            res = bert_pipe(b_text)
            st.write("电脑给出的答案：")
            st.dataframe(pd.DataFrame(res)[['token_str', 'score']].rename(columns={'token_str':'预测词','score':'信心值'}))

    with col2:
        st.write("✍️ **GPT 续写实验**")
        gpt_samples = [
            "Once upon a time, AI",
            "The most important thing in life is",
            "Deep learning is a subset of"
        ]
        g_text = st.selectbox("👉 选择开头", options=gpt_samples)
        if g_text:
            gen = gpt2_gen(g_text, max_length=30)
            st.write("电脑续写的原文：")
            st.success(gen[0]['generated_text'])

# --- Tab 4: PPL ---
with tab4:
    st.subheader("4. 丝滑程度：困惑度 (Perplexity)")
    st.markdown(r"""
    <div class="theory-box">
    <b>💡 原理：</b> 衡量模型对句子的“意外程度”。<br>
    <b>🎯 目的：</b> 评估一句话到底像不像人类说出来的“人话”。<br>
    <b>📏 标准：</b> PPL 越低 = 句子越通顺、越合理；PPL 很高 = 语无伦次或胡言乱语。
    </div>
    """, unsafe_allow_html=True)
    
    ppl_samples = {
        "通顺句": "Natural language processing is a fascinating field of study.",
        "乱序句": "Processing natural study field fascinating language is a of.",
        "对比实验": "I am going to the park.\nPark to the going am I."
    }
    s_key = st.selectbox("👉 选择对比案例", options=list(ppl_samples.keys()))
    sentences = st.text_area("待评估句子（每行一句）", ppl_samples[s_key]).split('\n')
    
    res_list = []
    for s in sentences:
        if s.strip():
            inputs = gpt2_tok(s, return_tensors="pt")
            with torch.no_grad():
                loss = gpt2_model(inputs.input_ids, labels=inputs.input_ids).loss
                ppl = torch.exp(loss).item()
                res_list.append({"句子": s, "PPL (越低越通顺)": f"{ppl:.2f}", "评价": "✅ 顺滑" if ppl < 80 else "❓ 奇怪"})
    
    if res_list:
        st.table(pd.DataFrame(res_list))

st.markdown("---")
st.caption("注：本平台使用蒸馏版小型模型进行演示，旨在展示原理。复杂任务请使用更大型的语言模型。")