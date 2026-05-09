import streamlit as st
import random
import plotly.graph_objects as go
from transformers import pipeline

st.set_page_config(page_title="细粒度情感分析与舆情监测平台", layout="wide")

# ======================
# 加载模型（缓存避免重复加载）
# ======================
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

model = load_model()

# ======================
# 工具函数
# ======================
def analyze_sentiment(text):
    result = model(text)[0]
    label = result['label']
    score = result['score']
    return label, score


def draw_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={'text': "Confidence Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'thickness': 0.3}
        }
    ))
    return fig


# ======================
# 页面结构
# ======================
tab1, tab2, tab3 = st.tabs([
    "模块1：情感分类",
    "模块2：显式 vs 隐式",
    "模块3：舆情仪表盘"
])


# =====================================================
# 🟢 模块1：基础情感分类 + 仪表盘
# =====================================================
with tab1:
    st.subheader("单文本情感分析")

    # 示例数据
    examples = [
        "这个产品真的很好用，性价比很高",
        "质量太差了，用两天就坏了",
        "一般般，没有特别惊艳",
        "外观很好看，但是性能一般",
        "客服态度很好，物流也很快"
    ]

    selected_example = st.selectbox("选择示例（可选）", [""] + examples)

    # 输入框（自动填充）
    text = st.text_area("请输入评论", value=selected_example)

    if st.button("开始分析"):
        if text:
            label, score = analyze_sentiment(text)

            st.write("情感极性：", label)
            st.plotly_chart(draw_gauge(score), use_container_width=True)


# =====================================================
# 🟢 模块2：显式 vs 隐式情感
# =====================================================
with tab2:
    st.subheader("显式情感 vs 隐式情感")

    st.markdown("""
    显式情感：包含明显情绪词（如“太棒了”、“很差”）  
    隐式情感：不包含情绪词，但通过事实表达情绪（如“用半小时就没电了”）
    """)

    # 示例
    explicit_examples = [
        "这屏幕画质太垃圾了",
        "真的非常好用，太棒了",
        "质量差得离谱"
    ]

    implicit_examples = [
        "在太阳底下根本看不清屏幕",
        "手机玩游戏半小时就没电了",
        "用了几天开始卡顿"
    ]

    col1, col2 = st.columns(2)

    # ======================
    # 显式情感
    # ======================
    with col1:
        selected_explicit = st.selectbox(
            "显式情感示例",
            [""] + explicit_examples,
            key="exp"
        )

        explicit_text = st.text_area(
            "显式情感评价",
            value=selected_explicit
        )

        if st.button("分析显式情感"):
            if explicit_text:
                label, score = analyze_sentiment(explicit_text)
                st.write("结果：", label)
                st.write("置信度：", score)

    # ======================
    # 隐式情感
    # ======================
    with col2:
        selected_implicit = st.selectbox(
            "隐式情感示例",
            [""] + implicit_examples,
            key="imp"
        )

        implicit_text = st.text_area(
            "隐式客观描述",
            value=selected_implicit
        )

        if st.button("分析隐式情感"):
            if implicit_text:
                label, score = analyze_sentiment(implicit_text)
                st.write("结果：", label)
                st.write("置信度：", score)


# =====================================================
# 🟢 模块3：舆情分析仪表盘（升级版）
# =====================================================
with tab3:
    st.subheader("舆情挖掘与可视化仪表盘")

    # 模拟语料库（按情感分类）
    positive_samples = [
        "这个产品真的很好用",
        "性价比很高",
        "非常满意",
        "体验很好",
        "值得推荐",
        "质量不错",
        "物流很快",
        "客服态度很好"
    ]

    neutral_samples = [
        "一般般",
        "还可以",
        "没有特别突出",
        "中规中矩",
        "还行吧"
    ]

    negative_samples = [
        "质量太差了",
        "用了两天就坏了",
        "体验很差",
        "不值这个价格",
        "非常失望",
        "包装破损",
        "发热严重"
    ]

    if st.button("生成测试舆情数据"):

        # ✅ 随机生成10-15条评论
        num_samples = random.randint(10, 15)

        comments = []
        for _ in range(num_samples):
            category = random.choice(["pos", "neu", "neg"])

            if category == "pos":
                comments.append(random.choice(positive_samples))
            elif category == "neu":
                comments.append(random.choice(neutral_samples))
            else:
                comments.append(random.choice(negative_samples))

        st.write("生成的评论数据：")
        st.write(comments)

        # ======================
        # 批量情感分析
        # ======================
        results = {"Positive": 0, "Neutral": 0, "Negative": 0}

        for c in comments:
            label, _ = analyze_sentiment(c)

            if "positive" in label.lower():
                results["Positive"] += 1
            elif "negative" in label.lower():
                results["Negative"] += 1
            else:
                results["Neutral"] += 1

        # ======================
        # KPI 指标（大屏风格）
        # ======================
        col1, col2, col3 = st.columns(3)

        col1.metric("Positive", results["Positive"])
        col2.metric("Neutral", results["Neutral"])
        col3.metric("Negative", results["Negative"])

        # ======================
        # 环形饼图（更现代）
        # ======================
        fig = go.Figure(data=[go.Pie(
            labels=list(results.keys()),
            values=list(results.values()),
            hole=0.5
        )])

        fig.update_layout(
            title="舆情情感分布",
            template="plotly_dark"  # ⭐ 科技感关键
        )

        st.plotly_chart(fig, use_container_width=True)