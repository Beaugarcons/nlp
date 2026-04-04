import streamlit as st
import nltk
from nltk.wsd import lesk
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import spacy
from spacy import displacy
import pandas as pd
import time
from streamlit.components.v1 import html

# --- 页面配置 ---
st.set_page_config(
    page_title="NLP Semantic Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 极简主义CSS样式 ---
st.markdown("""
<style>
/* ===== 全局重置与基础样式 ===== */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* 全局字体 */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ===== 主容器样式 ===== */
.main {
    background-color: #FFFFFF;
    color: #1A1A1A;
}

/* 移除默认容器内边距 */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1200px;
}

/* ===== 标题样式 ===== */
h1, h2, h3, h4 {
    font-weight: 600;
    letter-spacing: -0.025em;
    color: #111827;
    margin-bottom: 0.75rem;
}

h1 {
    font-size: 2.5rem;
    line-height: 1.1;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2 {
    font-size: 1.5rem;
    border-bottom: 2px solid #F3F4F6;
    padding-bottom: 0.5rem;
    margin-top: 2rem;
}

h3 {
    font-size: 1.25rem;
    color: #374151;
}

/* ===== 标签页样式 ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 1px solid #E5E7EB;
    padding: 0;
    margin-bottom: 2rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6B7280;
    font-weight: 500;
    padding: 0.75rem 1.5rem;
    border-radius: 8px 8px 0 0;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: none;
    position: relative;
    overflow: hidden;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #F9FAFB;
    color: #4B5563;
    transform: translateY(-1px);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    border: none;
}

.stTabs [data-baseweb="tab"][aria-selected="true"]::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: white;
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from { transform: translateX(-100%); }
    to { transform: translateX(0); }
}

/* ===== 卡片容器样式 ===== */
.css-1r6slb0 {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    border: 1px solid #F3F4F6;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 1rem;
}

.css-1r6slb0:hover {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    transform: translateY(-2px);
}

/* ===== 输入框样式 ===== */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border-radius: 8px;
    border: 2px solid #E5E7EB;
    padding: 0.75rem 1rem;
    font-size: 0.95rem;
    color: #1F2937;
    background: white;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #6366F1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    outline: none;
}

.stTextInput > div > div > input:hover,
.stTextArea > div > div > textarea:hover {
    border-color: #D1D5DB;
}

/* 输入框标签 */
.stTextInput > label,
.stTextArea > label {
    font-weight: 500;
    color: #374151;
    margin-bottom: 0.5rem;
    display: block;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ===== 按钮样式 ===== */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    width: 100%;
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: 0.5s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:active {
    transform: translateY(0);
}

.stButton > button:disabled {
    background: linear-gradient(135deg, #9CA3AF 0%, #6B7280 100%);
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
}

/* ===== 指标卡样式 ===== */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #E2E8F0;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    border-color: #6366F1;
}

[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 700;
    color: #111827;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

[data-testid="stMetricLabel"] {
    font-size: 0.9rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

/* ===== 表格样式 ===== */
.stTable {
    background: white;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #F3F4F6;
    margin: 1rem 0;
    transition: all 0.3s ease;
}

.stTable:hover {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.stTable table {
    width: 100%;
    border-collapse: collapse;
}

.stTable thead th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 600;
    padding: 1rem 1.5rem;
    text-align: left;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.85rem;
}

.stTable tbody tr {
    border-bottom: 1px solid #F3F4F6;
    transition: background-color 0.2s ease;
}

.stTable tbody tr:hover {
    background-color: #F9FAFB;
}

.stTable tbody td {
    padding: 1rem 1.5rem;
    color: #374151;
    font-size: 0.95rem;
}

.stTable tbody tr:last-child {
    border-bottom: none;
}

/* ===== 分隔线样式 ===== */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #E5E7EB, transparent);
    margin: 2rem 0;
}

/* ===== 警告/成功/错误消息样式 ===== */
.stAlert {
    border-radius: 8px;
    border: none;
    animation: fadeIn 0.5s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 成功样式 */
div[data-testid="stSuccess"] > div {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 1rem;
}

/* 警告样式 */
div[data-testid="stWarning"] > div {
    background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 1rem;
}

/* 错误样式 */
div[data-testid="stError"] > div {
    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 1rem;
}

/* 信息样式 */
div[data-testid="stInfo"] > div {
    background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 1rem;
}

/* ===== 加载动画 ===== */
.stSpinner > div {
    border-top-color: #6366F1;
    border-left-color: #6366F1;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* ===== 工具提示 ===== */
[data-testid="stTooltip"] {
    animation: tooltipFade 0.2s ease-out;
}

@keyframes tooltipFade {
    from { opacity: 0; transform: translateY(-5px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ===== 依存图容器 ===== */
[data-testid="stIFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #E5E7EB;
    transition: all 0.3s ease;
}

[data-testid="stIFrame"]:hover {
    border-color: #6366F1;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1);
}

/* ===== 列间距 ===== */
.stColumn {
    padding: 0 0.5rem;
}

/* ===== 响应式调整 ===== */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    h1 {
        font-size: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
    }
}

/* ===== 滚动条样式 ===== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #F3F4F6;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5A6FD8 0%, #6A3D8E 100%);
}
</style>
""", unsafe_allow_html=True)

# --- 资源加载与缓存 ---
@st.cache_resource
def load_models():
    """加载NLP模型（带缓存）"""
    # 显示加载状态
    with st.spinner("🔄 加载NLP模型中..."):
        # NLTK 数据
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        # BERT
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        # SpaCy
        nlp = spacy.load("en_core_web_sm")
    return tokenizer, bert_model, nlp

# 加载模型
tokenizer, bert_model, nlp = load_models()

# --- 辅助函数 ---
def get_bert_embedding(text, target_word):
    """获取BERT词向量"""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    embeddings = outputs.last_hidden_state[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # 查找目标词索引
    target_idxs = [i for i, t in enumerate(tokens) 
                   if target_word.lower() in t.lower().replace('##', '')]
    
    if not target_idxs:
        return None
    return torch.mean(embeddings[target_idxs], dim=0).numpy()

# --- 主界面 ---
# 顶部标题和描述
st.markdown("""
<div style="text-align: center; margin-bottom: 3rem;">
    <h1 style="margin-bottom: 0.5rem;">🔍 NLP 语义分析工具</h1>
    <p style="color: #6B7280; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
        NLP 语义分析工具
    </p>
</div>
""", unsafe_allow_html=True)

# 标签页
tab1, tab2 = st.tabs(["📊 词义消歧对比", "🎯 语义角色标注"])

# --- 标签页1: 词义消歧 ---
with tab1:
    st.header("词义消歧 (Word Sense Disambiguation)")
    st.markdown("""
    <p style="color: #6B7280; margin-bottom: 2rem;">
        比较同一词语在不同上下文中的语义差异，结合传统NLP方法与现代深度学习模型
    </p>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.subheader("上下文输入")
            s1 = st.text_input(
                "句子 1",
                "I went to the bank to deposit my money.",
                help="包含目标多义词的第一个句子"
            )
            s2 = st.text_input(
                "句子 2", 
                "I sat by the river bank.",
                help="包含目标多义词的第二个句子"
            )
            target = st.text_input(
                "目标词语",
                "bank",
                help="需要在两个句子中分析的多义词"
            )
            
            analyze_btn = st.button(
                "🔍 开始分析",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            st.subheader("分析说明")
            st.info("""
            **工作原理：**
            1. **Lesk算法** - 基于词典覆盖的传统消歧方法
            2. **BERT向量** - 基于上下文的深度语义表示
            3. **余弦相似度** - 量化语义相似性
            
            **结果解读：**
            - 相似度接近 1.0：语义高度相似
            - 相似度接近 0.0：语义差异显著
            """)
    
    if analyze_btn and s1 and s2 and target:
        with st.spinner("正在进行语义分析..."):
            time.sleep(0.5)  # 模拟加载动画
            
            st.divider()
            
            # 结果显示区域
            results_col1, results_col2 = st.columns(2)
            
            with results_col1:
                st.markdown("### 📖 Lesk 算法结果")
                
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown("**句子 1 分析**")
                    synset1 = lesk(nltk.word_tokenize(s1), target)
                    if synset1:
                        st.success(f"**词义：** {synset1.name()}")
                        st.markdown(f"**定义：** {synset1.definition()}")
                        if synset1.examples():
                            st.markdown(f"**示例：** {synset1.examples()[0]}")
                    else:
                        st.warning("未找到匹配的词义")
                
                with col_s2:
                    st.markdown("**句子 2 分析**")
                    synset2 = lesk(nltk.word_tokenize(s2), target)
                    if synset2:
                        st.success(f"**词义：** {synset2.name()}")
                        st.markdown(f"**定义：** {synset2.definition()}")
                        if synset2.examples():
                            st.markdown(f"**示例：** {synset2.examples()[0]}")
                    else:
                        st.warning("未找到匹配的词义")
            
            with results_col2:
                st.markdown("### 🧠 BERT 向量分析")
                v1 = get_bert_embedding(s1, target)
                v2 = get_bert_embedding(s2, target)
                
                if v1 is not None and v2 is not None:
                    similarity = 1 - cosine(v1, v2)
                    
                    # 动态颜色
                    if similarity > 0.8:
                        color = "#10B981"
                        interpretation = "语义高度相似"
                    elif similarity > 0.5:
                        color = "#F59E0B"
                        interpretation = "语义部分相似"
                    else:
                        color = "#EF4444"
                        interpretation = "语义差异显著"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem;">
                            <div style="font-size: 0.9rem; color: #6B7280; margin-bottom: 0.5rem;">余弦相似度</div>
                            <div style="font-size: 3rem; font-weight: 700; color: {color};">
                                {similarity:.4f}
                            </div>
                            <div style="font-size: 1rem; color: {color}; font-weight: 600; margin-top: 0.5rem;">
                                {interpretation}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 进度条可视化
                    st.markdown("**语义相似度**")
                    st.progress(float(similarity))
                    
                    st.info("""
                    **解读说明：**
                    - BERT向量捕捉了词语在特定上下文中的深度语义表示
                    - 相似度计算基于词向量在768维空间中的余弦距离
                    - 该结果反映了模型对词语上下文意义的理解
                    """)
                else:
                    st.error("⚠️ 未能从BERT中定位目标词语，请检查拼写是否正确")

# --- 标签页2: 语义角色标注 ---
with tab2:
    st.header("语义角色标注 (Semantic Role Labeling)")
    st.markdown("""
    <p style="color: #6B7280; margin-bottom: 2rem;">
        分析句子的语义结构，识别核心论元和修饰成分
    </p>
    """, unsafe_allow_html=True)
    
    with st.container():
        col_input, col_guide = st.columns([2, 1], gap="large")
        
        with col_input:
            srl_input = st.text_area(
                "输入分析句子",
                "Apple is manufacturing new smartphones in China this year.",
                height=100,
                help="输入需要分析语义角色的英文句子"
            )
            
            analyze_srl_btn = st.button(
                "🎯 提取语义角色",
                type="primary",
                use_container_width=True
            )
        
        with col_guide:
            st.markdown("""
            **语义角色说明：**
            
            - **A0 (施事者)** - 动作的执行者
            - **谓词** - 句子的核心动作
            - **A1 (受事者)** - 动作的接受者
            - **AM-LOC (地点)** - 动作发生的地点
            - **AM-TMP (时间)** - 动作发生的时间
            
            *基于启发式规则和依存句法分析*
            """)
    
    if analyze_srl_btn and srl_input:
        with st.spinner("正在分析句子结构..."):
            time.sleep(0.3)
            
            # 处理句子
            doc = nlp(srl_input)
            
            # 语义角色数据初始化
            srl_data = {
                "语义角色": ["A0 (施事者)", "谓词 (Predicate)", "A1 (受事者)", "AM-LOC (地点)", "AM-TMP (时间)"],
                "内容": ["", "", "", "", ""],
                "状态": ["未识别", "未识别", "未识别", "未识别", "未识别"]
            }
            
            # 启发式SRL提取
            for token in doc:
                # 提取谓词
                if token.dep_ == "ROOT" or (token.pos_ == "VERB" and token.dep_ != "aux"):
                    srl_data["内容"][1] = token.lemma_
                    srl_data["状态"][1] = "已识别"
                
                # 提取主语
                if token.dep_ in ("nsubj", "nsubjpass"):
                    srl_data["内容"][0] = token.text
                    srl_data["状态"][0] = "已识别"
                
                # 提取直接宾语
                if token.dep_ == "dobj":
                    srl_data["内容"][2] = token.text
                    srl_data["状态"][2] = "已识别"
                
                # 提取地点
                if token.dep_ == "prep" and token.text.lower() in ["in", "at", "on"]:
                    pobj = [child for child in token.children if child.dep_ == "pobj"]
                    if pobj:
                        srl_data["内容"][3] = f"{token.text} {pobj[0].text}"
                        srl_data["状态"][3] = "已识别"
                
                # 提取时间
                if token.ent_type_ in ("DATE", "TIME"):
                    srl_data["内容"][4] = token.text
                    srl_data["状态"][4] = "已识别"
            
            st.divider()
            
            # 结果显示
            st.markdown("### 📊 语义角色提取结果")
            
            # 创建结果表格
            df = pd.DataFrame(srl_data)
            
            # 添加状态颜色
            def color_status(val):
                if val == "已识别":
                    return 'background-color: #D1FAE5; color: #065F46; font-weight: 600;'
                else:
                    return 'background-color: #FEF3C7; color: #92400E;'
            
            styled_df = df.style.applymap(color_status, subset=['状态'])
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            st.divider()
            
            # 依存句法图
            st.markdown("### 🌳 依存句法分析")
            
            with st.expander("查看依存关系图", expanded=True):
                html = displacy.render(doc, style="dep", options={
                    "distance": 120,
                    "compact": True,
                    "color": "#6366F1",
                    "bg": "#FFFFFF"
                })
                
                # 添加容器样式
                html_with_style = f"""
                <div style="
                    background: white;
                    border-radius: 12px;
                    padding: 1.5rem;
                    border: 1px solid #E5E7EB;
                    overflow: auto;
                ">
                    {html}
                </div>
                """
                
                st.components.v1.html(html_with_style, height=400, scrolling=True)
            
            # 句法树文本表示
            with st.expander("查看详细句法分析"):
                st.code("\n".join([f"{token.text:<15} {token.pos_:<10} {token.dep_:<15} {token.head.text}" 
                                 for token in doc]), language="text")

# --- 页脚 ---
st.markdown("""
<hr style="margin: 3rem 0; border: none; border-top: 1px solid #E5E7EB;" />

<div style="text-align: center; color: #9CA3AF; font-size: 0.875rem; padding: 1rem;">
    <p>NLP Semantic Analyzer | Built with Streamlit, SpaCy, BERT & NLTK</p>
    <p style="margin-top: 0.5rem; opacity: 0.7;">
        基于 Transformer 架构的语义分析工具
    </p>
</div>
""", unsafe_allow_html=True)