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
    st.subheader("1. n-gram 语言模型与平滑")
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
        corpus_text = st.text_area("基础语料 (English)", corpus_options[selected], height=100)
    
    with c2:
        test_options = ["artificial intelligence", "the weather is nice", "stay hungry"]
        test_sent = st.selectbox("待测句子", options=test_options, index=0)
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
    
    st.markdown(f'<div class="stat-card"><div class="stat-val">联合概率：{prob:.8f}</div></div>', unsafe_allow_html=True)
    if details: st.table(pd.DataFrame(details))

# --- 模块 2: RNN 训练 ---
with tab2:
    st.subheader("2. 字符级 RNN 极简训练器")
    
    st.markdown("""
    <div class="theory-box">
    <b>🎯 目的：</b> 我们正在构建一个简单的递归神经网络（RNN）。它不看整个词，而是逐个字符地“阅读”文本。<br>
    <b>💡 原理：</b> RNN 的核心在于它拥有“隐藏状态”（Hidden State），就像人类的短期记忆。它每读一个字母，都会更新记忆，从而学会预测下一个字母。<br>
    <b>📈 结果：</b> 随着训练轮数（Epochs）增加，损失值（Loss）会下降。这代表模型背后的数学矩阵正在逐步逼近你输入的文本规律。<br>
    <b>🔨作用：</b> 这是现代 AI 处理序列数据（如文字、语音、股票走势）的祖先模型。
    </div>
    """, unsafe_allow_html=True)

    col_p1, col_p2 = st.columns([1, 2])
    
    with col_p1:
        st.markdown('<p class="step-header">Step 1: 准备训练数据</p>', unsafe_allow_html=True)
        rnn_corpus = {
            "简单重复": "hello world hello world",
            "代码模式": "for i in range(10): print(i)",
            "科学定义": "deep learning is a subset of machine learning."
        }
        selected_rnn = st.selectbox("选择参考语料", options=list(rnn_corpus.keys()))
        raw_text = st.text_area("训练语料编辑器", rnn_corpus[selected_rnn], height=100)
        
        st.markdown('<p class="step-header">Step 2: 设置超参数</p>', unsafe_allow_html=True)
        h_size = st.slider("记忆维度 (Hidden Size)", 16, 128, 64, help="数值越大，模型“记忆力”越强，但也更容易过拟合。")
        epochs = st.slider("复习轮数 (Epochs)", 10, 500, 100, help="模型对这段话重复阅读的次数。")
        lr = st.number_input("学习率 (Learning Rate)", 0.001, 0.1, 0.01, format="%.3f")
        start_train = st.button("开始教电脑读书", use_container_width=True)

    with col_p2:
        st.markdown('<p class="step-header">Step 3: 观察学习过程</p>', unsafe_allow_html=True)
        if start_train:
            # 数据预处理
            chars = sorted(list(set(raw_text)))
            char_to_ix = {ch: i for i, ch in enumerate(chars)}
            ix_to_char = {i: ch for i, ch in enumerate(chars)}
            vocab_size = len(chars)
            
            # 构造输入输出 (预测下一个字符)
            inputs = [char_to_ix[ch] for ch in raw_text[:-1]]
            targets = [char_to_ix[ch] for ch in raw_text[1:]]
            
            # 转换格式：(序列长度, Batch, 特征维度)
            X = torch.LongTensor(inputs).view(-1, 1) 
            Y = torch.LongTensor(targets)

            # 简单的编码模型
            class SimpleRNN(nn.Module):
                def __init__(self, v_size, h_size):
                    super().__init__()
                    self.embed = nn.Embedding(v_size, h_size)
                    self.rnn = nn.RNN(h_size, h_size, batch_first=True)
                    self.fc = nn.Linear(h_size, v_size)
                
                def forward(self, x, h):
                    x = self.embed(x)
                    out, h = self.rnn(x, h)
                    out = self.fc(out)
                    return out, h

            model = SimpleRNN(vocab_size, h_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            loss_hist = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            chart = st.line_chart()

            # 训练循环
            for epoch in range(epochs):
                hidden = torch.zeros(1, 1, h_size)
                output, hidden = model(X.unsqueeze(0), hidden)
                
                loss = criterion(output.squeeze(0), Y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_hist.append(loss.item())
                if epoch % 10 == 0 or epoch == epochs-1:
                    chart.line_chart(loss_hist)
                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(f"当前训练轮数: {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
            st.success("训练完成！模型已尝试记住该序列。")
            
            # 模型推理演示
            st.markdown('<p class="step-header">Step 4: 测试生成能力</p>', unsafe_allow_html=True)
            with torch.no_grad():
                test_char = raw_text[0]
                result = test_char
                input_eval = torch.LongTensor([char_to_ix[test_char]]).view(1, 1)
                hidden = torch.zeros(1, 1, h_size)
                
                for _ in range(min(len(raw_text), 30)):
                    out, hidden = model(input_eval, hidden)
                    idx = torch.argmax(out).item()
                    result += ix_to_char[idx]
                    input_eval = torch.LongTensor([idx]).view(1, 1)
                
                st.write("🤖 模型尝试续写结果：")
                st.code(result)
                st.caption("提示：由于模型极其微小且语料极少，生成结果可能在几个字符后陷入循环。")

# --- Tab 3: BERT vs GPT ---
with tab3:
    st.subheader("3. BERT (双向) vs GPT-2 (自回归)")
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
    st.subheader("4. 基于 GPT-2 的困惑度计算(Perplexity)")
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