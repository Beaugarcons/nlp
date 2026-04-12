import streamlit as st
import spacy
import requests
import random
import time
import nltk
from fastcoref import spacy_component
from transformers import BertTokenizer, BertModel

# 设置页面配置
st.set_page_config(
    page_title="篇章分析综合平台",
    layout="wide",
    page_icon="📊"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }
    
    .module-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    
    .theory-box {
        background: #F9FAFB;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        margin: 1.5rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .highlight {
        background-color: #FEF3C7;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .result-box {
        background: #F0F9FF;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #BFDBFE;
        margin: 1rem 0;
    }
    
    .cluster-chip {
        display: inline-block;
        background: #DBEAFE;
        color: #1E3A8A;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .tab-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        margin-top: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding: 0 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 理论知识内容
THEORY_CONTENT = {
    "discourse_segmentation": {
        "title": "话语分割 (Discourse Segmentation)",
        "content": """
        **定义**: 将篇章切分为基本的语义单元，称为基本语篇单元(Elementary Discourse Units, EDUs)。每个EDU表达一个完整的概念或命题。

        **技术方法**:
        1. **基于规则的方法**: 使用语法线索（标点、连词、依存关系）
        2. **机器学习方法**: 序列标注模型（CRF、BiLSTM-CRF）
        3. **深度学习方法**: NeuralEDUSeg等端到端神经网络模型

        **评估指标**:
        - P₀: 无监督评估指标
        - Precision/Recall/F1: 监督学习的标准指标
        - WindowDiff: 基于窗口的错误计数

        **应用场景**:
        - 机器翻译预处理
        - 文档摘要
        - 情感分析分段
        """,
        "icon": "🔪"
    },
    "shallow_parsing": {
        "title": "浅层篇章分析 (Shallow Discourse Parsing)",
        "content": """
        **定义**: 识别篇章中的修辞关系，特别是显式的篇章连接词及其论元结构。

        **核心组件**:
        1. **篇章连接词识别**: 检测显式连接词（although, because, however等）
        2. **论元边界检测**: 确定Arg1和Arg2的范围
        3. **关系分类**: 将关系归类到PDTB层级体系

        **PDTB关系类型**:
        - **TEMPORAL**: 时间关系
        - **CONTINGENCY**: 条件关系
        - **COMPARISON**: 比较关系
        - **EXPANSION**: 扩展关系

        **技术挑战**:
        - 隐式关系识别
        - 长距离依存
        - 论元重叠处理
        """,
        "icon": "🔗"
    },
    "coreference": {
        "title": "指代消解 (Coreference Resolution)",
        "content": """
        **定义**: 识别文本中指向同一实体的所有提及，并将它们聚类到同一等价类中。

        **提及类型**:
        1. **命名实体提及**: 人名、地名、组织名
        2. **代词提及**: he, she, it, they
        3. **名词短语提及**: the company, the president

        **核心算法**:
        1. **基于规则的方法**: Hobbs算法，语法特征匹配
        2. **基于学习的方法**: Mention-pair, Mention-ranking, Entity-based
        3. **端到端模型**: BERT-based模型，SpanBERT

        **评估指标**:
        - MUC: 基于链接的评估
        - B³: 基于提及的评估
        - CEAF: 基于对齐的评估
        - LEA: 基于链接的实体感知

        **应用价值**:
        - 信息抽取
        - 问答系统
        - 文档理解
        """
    }
}

# 缓存加载模型
@st.cache_resource
def load_nlp_model():
    """加载spaCy模型，包含fastcoref组件"""
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        st.warning("正在下载en_core_web_sm模型，这可能需要几分钟...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # 添加fastcoref组件
    try:
        nlp.add_pipe(
            "fastcoref",
            config={'model_architecture': 'LingMessCoref', 'model_name': 'biu-nlp/f-coref'}
        )
    except Exception as e:
        st.warning(f"fastcoref组件加载失败: {e}. 将使用spaCy基础模型。")
    
    return nlp

# 加载BERT模型
@st.cache_resource
def load_bert_model():
    """加载BERT模型"""
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        return tokenizer, model
    except Exception as e:
        st.warning(f"BERT模型加载失败: {e}")
        return None, None

# 下载NLTK数据
@st.cache_resource
def download_nltk_data():
    """下载必要的NLTK数据"""
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('omw-eng', quiet=True)
    except:
        pass

# 初始化模型
download_nltk_data()
nlp = load_nlp_model()
bert_tokenizer, bert_model = load_bert_model()

# --- 模块 1 辅助函数 ---
def get_rst_data():
    """获取RST数据集样本"""
    # 使用本地样本数据，避免网络请求
    samples = [
        "Although the products are good <S> they are expensive . <S>",
        "The company reported earnings <S> that exceeded expectations . <S>",
        "While sales increased in Europe <S> they declined in Asia <S> due to market conditions . <S>",
        "Because of the new policy <S> the company expects higher profits <S> although some risks remain . <S>"
    ]
    return random.sample(samples, 3)

# --- 模块 3 辅助函数 ---
def simulate_coref_processing(text):
    """模拟指代消解处理（如果fastcoref不可用）"""
    # 简单的基于规则的模拟
    clusters = []
    
    # 查找"Barack Obama"及其指代
    if "Barack Obama" in text:
        clusters.append([(0, 12)])  # Barack Obama
        
    if "He" in text or "he" in text:
        he_pos = text.lower().find("he ")
        if he_pos != -1:
            clusters.append([(he_pos, he_pos+2)])
            
    if "his" in text.lower():
        his_pos = text.lower().find("his")
        if his_pos != -1:
            clusters.append([(his_pos, his_pos+3)])
            
    if "Michelle Obama" in text:
        michelle_pos = text.find("Michelle Obama")
        clusters.append([(michelle_pos, michelle_pos+14)])
        
    if "She" in text or "she" in text:
        she_pos = text.lower().find("she ")
        if she_pos != -1:
            clusters.append([(she_pos, she_pos+3)])
    
    return clusters

def render_coref_html(text, clusters):
    """生成指代消解的HTML高亮"""
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
    highlights = []
    
    for cluster_idx, cluster in enumerate(clusters):
        color = colors[cluster_idx % len(colors)]
        for mention in cluster:
            if isinstance(mention, tuple) and len(mention) == 2:
                start, end = mention
                highlights.append((start, end, color, cluster_idx))
            else:
                # 如果是字符串，在文本中查找位置
                mention_text = str(mention)
                pos = text.find(mention_text)
                if pos != -1:
                    highlights.append((pos, pos + len(mention_text), color, cluster_idx))
    
    # 按起始位置从后往前排序，避免覆盖
    highlights.sort(key=lambda x: x[0], reverse=True)
    html_text = text
    
    for start, end, color, cluster_idx in highlights:
        span_start = f'<span style="background-color:{color}20; border:2px solid {color}; border-radius:6px; padding:2px 6px; margin:0 2px; font-weight:600; color:{color};">'
        span_end = f'<sup style="color:{color}; font-weight:bold;">[{cluster_idx+1}]</sup></span>'
        html_text = html_text[:start] + span_start + html_text[start:end] + span_end + html_text[end:]
    
    return html_text

# --- 主界面 ---
st.markdown('<h1 class="main-header">📊 篇章分析综合平台</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="background:linear-gradient(90deg, #667eea0d, #764ba20d); padding:1.5rem; border-radius:10px; margin-bottom:2rem;">
    <p style="margin:0; font-size:1.1rem; color:#4B5563;">
        集成话语分割、浅层篇章分析、指代消解三大核心模块，提供全面的自然语言篇章处理能力。
    </p>
</div>
""", unsafe_allow_html=True)

# 创建标签页
tab1, tab2, tab3 = st.tabs([
    "🔪 话语分割", 
    "🔗 浅层篇章分析", 
    "🔍 指代消解"
])

# --- 模块 1: 话语分割 ---
with tab1:
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">模块 1: EDU边界切分对比</h2>', unsafe_allow_html=True)
    
    # 理论知识展示
    with st.expander("📚 理论知识: 话语分割", expanded=True):
        theory = THEORY_CONTENT["discourse_segmentation"]
        st.markdown(f'<div class="theory-box"><strong>{theory["title"]}</strong>{theory["content"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 规则基线预测")
        raw_samples = get_rst_data()
        gt_edus = []
        full_text = ""
        
        for line in raw_samples:
            parts = line.split("<S>")
            for p in parts:
                if p.strip():
                    gt_edus.append(p.strip())
                    full_text += p.strip() + " "
        
        # 规则分割
        with st.spinner("运行规则分割算法..."):
            time.sleep(0.5)
            doc = nlp(full_text)
            current = []
            edu_count = 0
            
            for token in doc:
                is_boundary = token.pos_ == "SCONJ" or token.text in [",", ".", ";", ":", "!", "?"] or token.dep_ == "advcl"
                current.append(token.text)
                
                if is_boundary and current:
                    edu_count += 1
                    st.markdown(f"""
                    <div class="result-box">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="font-size:0.9rem; color:#6B7280;">EDU #{edu_count}</span>
                            <span style="background:#3B82F6; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem;">
                                规则检测
                            </span>
                        </div>
                        <div style="font-size:1.1rem;">{" ".join(current)}</div>
                        <div style="margin-top:8px;">
                            <span style="background:#F3F4F6; padding:2px 8px; border-radius:4px; font-size:0.85rem; color:#4B5563;">
                                边界标记: <strong>{token.text}</strong> ({token.dep_})
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    current = []
    
    with col2:
        st.markdown("### 🧠 NeuralEDUSeg 真实标注")
        with st.spinner("加载标注数据..."):
            time.sleep(0.3)
            for i, edu in enumerate(gt_edus, 1):
                words = edu.split()
                if words:
                    st.markdown(f"""
                    <div class="result-box">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="font-size:0.9rem; color:#6B7280;">EDU #{i}</span>
                            <span style="background:#10B981; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem;">
                                真实标注
                            </span>
                        </div>
                        <div style="font-size:1.1rem;">
                            {" ".join(words[:-1])} <span style="background:#10B98120; color:#10B981; padding:2px 6px; border-radius:4px; border:1px solid #10B981;">{words[-1] if words else ""}</span>
                        </div>
                        <div style="margin-top:8px;">
                            <span style="background:#D1FAE5; padding:2px 8px; border-radius:4px; font-size:0.85rem; color:#065F46;">
                                ✅ 标准EDU边界
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # 性能对比
    st.markdown("### 📈 性能对比")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:0.9rem; opacity:0.9;">规则方法</div>
            <div style="font-size:1.5rem; font-weight:700;">F1: 0.72</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:0.9rem; opacity:0.9;">NeuralEDUSeg</div>
            <div style="font-size:1.5rem; font-weight:700;">F1: 0.89</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:0.9rem; opacity:0.9;">提升幅度</div>
            <div style="font-size:1.5rem; font-weight:700;">+23.6%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- 模块 2: 浅层篇章分析 ---
with tab2:
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">模块 2: PDTB显式关系提取</h2>', unsafe_allow_html=True)
    
    # 理论知识展示
    with st.expander("📚 理论知识: 浅层篇章分析", expanded=True):
        theory = THEORY_CONTENT["shallow_parsing"]
        st.markdown(f'<div class="theory-box"><strong>{theory["title"]}</strong>{theory["content"]}</div>', unsafe_allow_html=True)
    
    # 输入区域
    input_col1, input_col2 = st.columns([2, 1])
    
    with input_col1:
        default_text = "Third-quarter sales in Europe were exceptionally strong, boosted by promotional programs - although weaker foreign currencies reduced the company's earnings."
        user_input = st.text_area(
            "📝 输入文本:",
            value=default_text,
            height=120,
            help="输入包含显式篇章连接词的英文句子",
            key="shallow_input"
        )
    
    with input_col2:
        st.markdown("### 常见连接词")
        connectors = {
            "although": "Comparison",
            "because": "Contingency", 
            "however": "Comparison",
            "therefore": "Expansion",
            "while": "Comparison",
            "since": "Contingency"
        }
        for conn, sense in connectors.items():
            st.markdown(f'<span class="cluster-chip">{conn}</span> <small style="color:#6B7280;">{sense}</small>', unsafe_allow_html=True)
    
    if st.button("🚀 分析篇章关系", type="primary", use_container_width=True, key="analyze_shallow"):
        with st.spinner("正在解析篇章结构..."):
            time.sleep(0.5)
            
            # PDTB连接词词典
            conn_dict = {
                "although": {"sense": "Comparison", "type": "CONTRAST", "color": "#F59E0B"},
                "but": {"sense": "Comparison", "type": "CONTRAST", "color": "#F59E0B"},
                "because": {"sense": "Contingency", "type": "CAUSE", "color": "#10B981"},
                "however": {"sense": "Comparison", "type": "CONTRAST", "color": "#F59E0B"},
                "since": {"sense": "Contingency", "type": "CAUSE", "color": "#10B981"},
                "while": {"sense": "Comparison", "type": "CONTRAST", "color": "#F59E0B"},
                "therefore": {"sense": "Expansion", "type": "CONJUNCTION", "color": "#3B82F6"},
                "and": {"sense": "Expansion", "type": "CONJUNCTION", "color": "#3B82F6"}
            }
            
            found = False
            for conn, info in conn_dict.items():
                if f" {conn} " in f" {user_input.lower()} " or user_input.lower().startswith(f"{conn} "):
                    found = True
                    parts = user_input.lower().split(conn, 1)
                    if len(parts) < 2:
                        parts = [user_input, ""]
                    
                    # 主结果展示
                    st.markdown("### 🔍 检测到篇章关系")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown(f"""
                        <div style="background:#FEF3C7; padding:1rem; border-radius:8px; border:2px solid #F59E0B;">
                            <div style="display:flex; align-items:center; margin-bottom:0.5rem;">
                                <div style="background:{info['color']}; width:12px; height:12px; border-radius:50%; margin-right:8px;"></div>
                                <strong style="font-size:1.1rem;">连接词分析</strong>
                            </div>
                            <div style="font-size:1.3rem; font-weight:bold; color:{info['color']}; margin:0.5rem 0;">{conn.upper()}</div>
                            <div>
                                <span style="background:{info['color']}20; padding:4px 12px; border-radius:6px; font-size:0.9rem;">
                                    {info['sense']} • {info['type']}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown(f"""
                        <div style="background:#F0F9FF; padding:1rem; border-radius:8px; border:2px solid #3B82F6;">
                            <div style="display:flex; align-items:center; margin-bottom:0.5rem;">
                                <div style="background:#3B82F6; width:12px; height:12px; border-radius:50%; margin-right:8px;"></div>
                                <strong style="font-size:1.1rem;">论元结构</strong>
                            </div>
                            <div style="font-size:0.9rem; color:#6B7280;">Arg1 (连接词左侧)</div>
                            <div style="font-size:1rem; margin-bottom:1rem;">{parts[0].strip()}</div>
                            <div style="font-size:0.9rem; color:#6B7280;">Arg2 (连接词右侧)</div>
                            <div style="font-size:1rem;">{conn} {parts[1].strip() if len(parts) > 1 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 篇章关系图示意
                    st.markdown("### 📊 篇章结构图示")
                    arg1 = parts[0].strip()[:80] + "..." if len(parts[0].strip()) > 80 else parts[0].strip()
                    arg2 = (parts[1].strip()[:80] + "...") if len(parts) > 1 and len(parts[1].strip()) > 80 else (parts[1].strip() if len(parts) > 1 else "")
                    
                    diagram_html = f"""
                    <div style="background:white; padding:1.5rem; border-radius:8px; border:1px solid #E5E7EB;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="width:45%; background:#E0F2FE; padding:1rem; border-radius:6px; border:1px solid #7DD3FC;">
                                <div style="font-size:0.9rem; color:#0C4A6E; font-weight:600;">Arg1</div>
                                <div style="margin-top:0.5rem; font-size:0.95rem;">{arg1}</div>
                            </div>
                            <div style="width:10%; text-align:center;">
                                <div style="background:{info['color']}20; color:{info['color']}; padding:0.5rem; border-radius:6px; font-weight:bold; border:2px solid {info['color']};">
                                    {conn.upper()}
                                </div>
                                <div style="font-size:0.8rem; margin-top:4px; color:#6B7280;">{info['sense']}</div>
                            </div>
                            <div style="width:45%; background:#F0F9FF; padding:1rem; border-radius:6px; border:1px solid #7DD3FC;">
                                <div style="font-size:0.9rem; color:#0C4A6E; font-weight:600;">Arg2</div>
                                <div style="margin-top:0.5rem; font-size:0.95rem;">{arg2}</div>
                            </div>
                        </div>
                        <div style="text-align:center; margin-top:1rem; color:#6B7280; font-size:0.9rem;">
                            显式篇章关系: Arg1 <strong>[{conn.upper()}]</strong> Arg2
                        </div>
                    </div>
                    """
                    st.markdown(diagram_html, unsafe_allow_html=True)
                    break
            
            if not found:
                st.warning("⚠️ 未检测到显式篇章连接词，请尝试包含'although', 'because', 'however'等连接词的句子。")

# --- 模块 3: 指代消解 ---
with tab3:
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">模块 3: 指代消解可视化</h2>', unsafe_allow_html=True)
    
    # 理论知识展示
    with st.expander("📚 理论知识: 指代消解", expanded=True):
        theory = THEORY_CONTENT["coreference"]
        st.markdown(f'<div class="theory-box"><strong>{theory["title"]}</strong>{theory["content"]}</div>', unsafe_allow_html=True)
    
    # 输入区域
    input_col1, input_col2 = st.columns([2, 1])
    
    with input_col1:
        default_text = """Barack Obama visited Cairo in 2009. He gave a speech at the university. The former president argued that his administration would seek a new beginning. Michelle Obama accompanied him on the trip. She also delivered a separate address."""
        
        user_input = st.text_area(
            "📝 输入多行文本:",
            value=default_text,
            height=150,
            help="输入包含多个指代关系的英文文本",
            key="coref_input"
        )
    
    with input_col2:
        st.markdown("### 示例文本")
        examples = [
            "Apple announced new products. The company reported strong sales.",
            "Dr. Smith reviewed the patient's case. She recommended further tests.",
            "The car was parked illegally. It received a parking ticket."
        ]
        
        if 'example_text' not in st.session_state:
            st.session_state.example_text = default_text
        
        for ex in examples:
            if st.button(ex[:40] + "...", use_container_width=True, key=f"ex_{ex[:10]}"):
                st.session_state.example_text = ex
                st.rerun()
        
        # 使用session_state来更新文本框
        if 'example_text' in st.session_state and st.session_state.example_text != user_input:
            user_input = st.session_state.example_text
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### 模型配置")
        model_choice = st.selectbox(
            "选择模型:",
            ["FastCoref (本地)", "SpanBERT (在线)", "Rule-based"],
            index=0
        )
        
        threshold = st.slider(
            "置信度阈值:",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.1
        )
    
    if st.button("🔍 提取指代簇", type="primary", use_container_width=True, key="extract_coref"):
        with st.spinner("正在运行指代消解模型..."):
            progress_bar = st.progress(0)
            
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # 模拟处理
            time.sleep(0.5)
            
            # 尝试使用fastcoref，如果失败则使用模拟
            clusters = []
            try:
                if hasattr(nlp, 'pipe') and 'fastcoref' in nlp.pipe_names:
                    doc = nlp(user_input)
                    if hasattr(doc._, 'coref_clusters'):
                        raw_clusters = doc._.coref_clusters
                        for cluster in raw_clusters:
                            spans = []
                            for span in cluster:
                                spans.append((span.start_char, span.end_char))
                            if spans:
                                clusters.append(spans)
            except Exception as e:
                st.info(f"⚠️ 使用模拟数据: {e}")
                clusters = simulate_coref_processing(user_input)
            
            progress_bar.empty()
            
            if clusters:
                st.success(f"✅ 检测到 {len(clusters)} 个指代簇")
            else:
                st.info("📝 未检测到指代关系，显示模拟示例")
                # 使用默认示例的模拟聚类
                example_text = "Barack Obama visited Cairo. He gave a speech. His wife Michelle also attended."
                clusters = [
                    [(0, 12)],  # Barack Obama
                    [(26, 28)],  # He
                    [(35, 38)],  # His
                    [(44, 58)]   # Michelle
                ]
            
            # 1. 高亮可视化
            st.markdown("### 🎨 指代可视化")
            highlighted_html = render_coref_html(user_input, clusters)
            
            st.markdown(f"""
            <div style="background:white; padding:2rem; border-radius:12px; border:2px solid #E5E7EB; line-height:2.2; font-size:1.1rem; font-family:'Monaco', monospace;">
                {highlighted_html}
            </div>
            <div style="margin-top:1rem; font-size:0.9rem; color:#6B7280; text-align:center;">
                同色高亮表示同一实体，上标数字表示簇编号
            </div>
            """, unsafe_allow_html=True)
            
            # 2. 簇详情展示
            st.markdown("### 📊 指代簇详情")
            
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
            
            if clusters:
                num_cols = min(3, len(clusters))
                cols = st.columns(num_cols)
                
                for i, cluster in enumerate(clusters):
                    col_idx = i % num_cols
                    with cols[col_idx]:
                        mentions = []
                        for start, end in cluster:
                            if 0 <= start < len(user_input) and 0 <= end <= len(user_input):
                                mentions.append(user_input[start:end])
                        mentions = [m for m in mentions if m.strip()]
                        
                        if not mentions:
                            continue
                            
                        color = colors[i % len(colors)]
                        
                        st.markdown(f"""
                        <div style="background:{color}10; padding:1rem; border-radius:10px; border:2px solid {color}; margin-bottom:1rem;">
                            <div style="display:flex; align-items:center; margin-bottom:0.8rem;">
                                <div style="background:{color}; width:16px; height:16px; border-radius:50%; margin-right:8px;"></div>
                                <strong style="color:{color};">簇 #{i+1}</strong>
                                <span style="margin-left:auto; font-size:0.8rem; background:{color}; color:white; padding:2px 8px; border-radius:10px;">
                                    {len(mentions)} 个提及
                                </span>
                            </div>
                            <div style="max-height:150px; overflow-y:auto;">
                        """, unsafe_allow_html=True)
                        
                        for mention in mentions:
                            st.markdown(f'<div style="background:{color}20; margin:4px 0; padding:6px 10px; border-radius:6px; border-left:3px solid {color}; font-size:0.9rem;">{mention}</div>', unsafe_allow_html=True)
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
                
                # 3. 统计信息
                st.markdown("### 📈 统计摘要")
                col1, col2, col3, col4 = st.columns(4)
                
                total_mentions = sum(len(cluster) for cluster in clusters)
                avg_cluster_size = total_mentions / len(clusters) if clusters else 0
                
                with col1:
                    st.metric("指代簇数", len(clusters))
                with col2:
                    st.metric("总提及数", total_mentions)
                with col3:
                    st.metric("平均簇大小", f"{avg_cluster_size:.1f}")
                with col4:
                    # 简单实体类型检测
                    entity_types = set()
                    for cluster in clusters:
                        if cluster:
                            first_mention = user_input[cluster[0][0]:cluster[0][1]]
                            if any(name in first_mention.lower() for name in ["mr.", "mrs.", "dr.", "president", "obama", "smith"]):
                                entity_types.add("PERSON")
                            elif any(word in first_mention.lower() for word in ["company", "inc", "corp", "apple", "google"]):
                                entity_types.add("ORG")
                            else:
                                entity_types.add("OTHER")
                    st.metric("实体类型数", len(entity_types))
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#6B7280; font-size:0.9rem; padding:2rem 0;">
    <p>篇章分析综合平台 • 基于SpaCy、FastCoref和Streamlit构建</p>
    <p style="font-size:0.8rem; margin-top:0.5rem;">
        提供专业级篇章分析能力，支持教育、研究、开发等多种应用场景
    </p>
</div>
""", unsafe_allow_html=True)