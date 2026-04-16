import streamlit as st
import nltk
from nltk.util import ngrams
from collections import Counter
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="语言模型训练与对比分析平台", layout="wide")

@st.cache_resource
def init_resources():
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except:
        pass
    # 加载小型模型以适配 Cloud 环境
    bert = pipeline("fill-mask", model="distilbert-base-uncased")
    gpt2_pipe = pipeline("text-generation", model="distilgpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    gpt2_tok = GPT2Tokenizer.from_pretrained("distilgpt2")
    return bert, gpt2_pipe, gpt2_model, gpt2_tok

bert_pipe, gpt2_gen, gpt2_model, gpt2_tok = init_resources()

# 自定义样式
st.markdown(r"""
<style>
    .theory-box {
        background-color: #f8fafc;
        border-left: 5px solid #3b82f6;
        padding: 1.2rem;
        margin-bottom: 20px;
        border-radius: 8px;
        color: #1e293b;
    }
    .step-label { font-weight: bold; color: #3b82f6; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 语言模型训练与对比分析平台")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 n-gram 统计", "🧠 RNN 序列记忆", "🤖 架构对比", "📉 PPL 评价"])

# --- 模块 1: n-gram ---
with tab1:
    st.subheader("1. n-gram 语言模型与平滑")
    st.markdown("""
    <div class="theory-box">
    <b>做什么：</b> 模型通过拆解文本片段（如每3个词一组），统计它们在库中出现的频率。<br>
    <b>为什么：</b> 电脑通过“数数”来学习经验。如果它发现“AI is good”出现了10次，而“AI is bad”只出现了1次，它就会认为前者更合理。<br>
    <b>结果意义：</b> 联合概率越大，说明这句话越符合你提供的“知识库”规律。
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('<div class="step-label">Step 1: 提供知识库</div>', unsafe_allow_html=True)
        ngram_corpus = {
            "科技新闻": "artificial intelligence is transforming the world. machines can learn from big data.",
            "日常口语": "how are you doing today? i am doing great and the weather is beautiful.",
            "经典名言": "to be or not to be that is the question. stay hungry stay foolish."
        }
        selected_ngram = st.selectbox("选择参考语料", options=list(ngram_corpus.keys()))
        corpus_input = st.text_area("知识库编辑器", ngram_corpus[selected_ngram], height=100)
    
    with c2:
        st.markdown('<div class="step-label">Step 2: 输入测试句</div>', unsafe_allow_html=True)
        test_sent = st.text_input("测试这句话的合理性", "artificial intelligence is")
        use_smooth = st.checkbox("开启平滑 (防止遇到生词时概率直接归零)", value=True)

    # 计算逻辑
    tokens = nltk.word_tokenize(corpus_input.lower())
    v_size = len(set(tokens))
    bi_counts = Counter(ngrams(tokens, 2))
    tri_counts = Counter(ngrams(tokens, 3))
    
    if test_sent:
        test_tokens = nltk.word_tokenize(test_sent.lower())
        test_tri = list(ngrams(test_tokens, 3))
        prob, details = 1.0, []
        for tri in test_tri:
            c_tri = tri_counts.get(tri, 0)
            c_bi = bi_counts.get(tri[:2], 0)
            p = (c_tri + 1)/(c_bi + v_size) if use_smooth else (c_tri/c_bi if c_bi > 0 else 0)
            prob *= p
            details.append({"三元片段": " ".join(tri), "概率": f"{p:.4f}"})
        st.metric("全句生成联合概率", f"{prob:.8f}")
        if details: st.table(pd.DataFrame(details))

# --- 模块 2: RNN 训练 ---
with tab2:
    st.subheader("2. 字符级 RNN 训练")
    st.markdown("""
    <div class="theory-box">
    <b>做什么：</b> 我们在训练一个“循环神经网络”。它会逐个读取字符，并在读取下一个字符时保留上一个字符的“残余记忆”。<br>
    <b>为什么：</b> 语言是有先后顺序的。RNN 通过“隐藏状态”记录历史信息，从而学会像人类一样处理序列数据。<br>
    <b>结果意义：</b> Loss（损失值）越低，说明模型对这段话的记忆越精准。
    </div>
    """, unsafe_allow_html=True)

    col_rnn1, col_rnn2 = st.columns([1, 2])
    with col_rnn1:
        st.markdown('<div class="step-label">Step 1: 训练设置</div>', unsafe_allow_html=True)
        rnn_samples = {
            "单词记忆": "sequence modeling is powerful.",
            "逻辑学习": "if a=1 then b=2; if a=1 then b=2;",
            "简单重复": "ai is cool ai is cool ai is cool"
        }
        s_rnn = st.selectbox("选择训练语料", options=list(rnn_samples.keys()))
        raw_text = st.text_area("训练语料内容", rnn_samples[s_rnn], height=80)
        h_size = st.slider("记忆维度 (Hidden Size)", 16, 128, 64)
        epochs = st.slider("训练轮数 (Epochs)", 10, 300, 100)
        start_train = st.button("开始训练模型", use_container_width=True)

    with col_rnn2:
        st.markdown('<div class="step-label">Step 2: 训练状态</div>', unsafe_allow_html=True)
        if start_train:
            chars = sorted(list(set(raw_text)))
            char_to_ix = {ch: i for i, ch in enumerate(chars)}
            ix_to_char = {i: ch for i, ch in enumerate(chars)}
            
            # 维度修正逻辑
            inputs = torch.LongTensor([char_to_ix[ch] for ch in raw_text[:-1]]).view(1, -1) 
            targets = torch.LongTensor([char_to_ix[ch] for ch in raw_text[1:]]).view(-1)

            class RNNModel(nn.Module):
                def __init__(self, v_size, h_size):
                    super().__init__()
                    self.embed = nn.Embedding(v_size, h_size)
                    self.rnn = nn.RNN(h_size, h_size, batch_first=True)
                    self.fc = nn.Linear(h_size, v_size)
                def forward(self, x, h):
                    x = self.embed(x) # x: [1, seq_len, h_size]
                    out, h = self.rnn(x, h)
                    return self.fc(out), h

            model = RNNModel(len(chars), h_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            loss_hist = []
            chart = st.line_chart()
            for epoch in range(epochs):
                h0 = torch.zeros(1, 1, h_size)
                logits, _ = model(inputs, h0)
                loss = criterion(logits.view(-1, len(chars)), targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_hist.append(loss.item())
                if epoch % 10 == 0: chart.line_chart(loss_hist)
            
            st.success("训练成功！模型已将这段话存入参数中。")
            st.markdown('<div class="step-label">Step 3: 尝试续写结果</div>', unsafe_allow_html=True)
            with torch.no_grad():
                res, cur = raw_text[0], char_to_ix[raw_text[0]]
                h = torch.zeros(1, 1, h_size)
                for _ in range(min(len(raw_text), 20)):
                    inp = torch.LongTensor([[cur]])
                    out, h = model(inp, h)
                    cur = torch.argmax(out).item()
                    res += ix_to_char[cur]
                st.code(res)

# --- 模块 3: 架构对比 ---
with tab3:
    st.subheader("3. BERT (双向) vs GPT-2 (自回归)")
    st.markdown("""
    <div class="theory-box">
    <b>BERT (双向):</b> 像是一个经验丰富的编辑。它同时看左边和右边的上下文，通过“周围环境”来推断中间缺失的信息。<br>
    <b>GPT (单向):</b> 像是一个预言家。它只看过去发生的事情，不断推测未来下一个词是什么，从而实现流畅续写。
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="step-label">BERT: 完形填空</div>', unsafe_allow_html=True)
        bert_inputs = [
            "The capital of UK is [MASK].",
            "Artificial intelligence will change the [MASK].",
            "I love eating [MASK] for dinner."
        ]
        b_input = st.selectbox("选择填空题目", options=bert_inputs)
        if b_input:
            res = bert_pipe(b_input)
            st.dataframe(pd.DataFrame(res)[['token_str', 'score']].rename(columns={'token_str':'建议词','score':'可信度'}))

    with col2:
        st.markdown('<div class="step-label">GPT: 自动续写</div>', unsafe_allow_html=True)
        gpt_inputs = [
            "In a world where robots",
            "The secret to a happy life is",
            "To train a large language model,"
        ]
        g_input = st.selectbox("选择续写开头", options=gpt_inputs)
        if g_input:
            gen = gpt2_gen(g_input, max_length=40)
            st.success(gen[0]['generated_text'])

# --- 模块 4: PPL ---
with tab4:
    st.subheader("4. 通顺度评价：困惑度 (Perplexity)")
    st.markdown(r"""
    <div class="theory-box">
    <b>做什么：</b> 计算模型对句子的“意外程度”。公式为 $PPL = \exp(Loss)$。<br>
    <b>为什么：</b> 如果一个句子语序混乱，模型会感到非常“困惑”，得分就会很高。如果句子很地道，得分就很低。<br>
    <b>结果意义：</b> PPL 越低，说明句子越像“人话”。
    </div>
    """, unsafe_allow_html=True)

    ppl_samples = {
        "对比：通顺 vs 混乱": "I am a student at the university.\nStudent am university I at a.",
        "对比：逻辑相似度": "The cat is sleeping on the mat.\nThe mat is sleeping on the cat.",
        "自定义": ""
    }
    s_ppl = st.selectbox("选择对比案例", options=list(ppl_samples.keys()))
    ppl_input = st.text_area("输入多行文本进行对比", ppl_samples[s_ppl], height=100)
    
    if ppl_input:
        lines = ppl_input.split('\n')
        results = []
        for line in lines:
            if line.strip():
                tokens = gpt2_tok(line, return_tensors="pt")
                with torch.no_grad():
                    loss = gpt2_model(tokens.input_ids, labels=tokens.input_ids).loss
                    ppl = torch.exp(loss).item()
                    results.append({"句子": line, "PPL 得分": f"{ppl:.2f}", "结论": "✅ 通顺" if ppl < 100 else "❌ 别扭"})
        st.table(pd.DataFrame(results))