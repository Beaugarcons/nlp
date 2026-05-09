import streamlit as st
import random
import plotly.graph_objects as go
from transformers import pipeline

# ======================
# 页面配置（必须放最前）
# ======================
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

    text = st.text_area("请输入一段商品评论")

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

    col1, col2 = st.columns(2)

    with col1:
        explicit_text = st.text_area("显式情感评价")

        if st.button("分析显式情感"):
            if explicit_text:
                label, score = analyze_sentiment(explicit_text)
                st.write("结果：", label)
                st.write("置信度：", score)

    with col2:
        implicit_text = st.text_area("隐式客观描述")

        if st.button("分析隐式情感"):
            if implicit_text:
                label, score = analyze_sentiment(implicit_text)
                st.write("结果：", label)
                st.write("置信度：", score)


# =====================================================
# 🟢 模块3：舆情分析仪表盘
# =====================================================
with tab3:
    st.subheader("舆情挖掘与可视化")

    if st.button("生成测试舆情数据"):

        comments = [
            "这个产品真的很好用",
            "质量太差了",
            "一般般，没有特别好",
            "性价比很高",
            "用了两天就坏了",
            "还可以",
            "物流很快",
            "体验很差",
            "挺满意的",
            "包装破损",
            "非常推荐",
            "不值这个价格",
            "客服态度很好",
            "手机发热严重",
            "屏幕清晰度不错"
        ]

        results = {
            "Positive": 0,
            "Neutral": 0,
            "Negative": 0
        }

        # 批量分析
        for c in comments:
            label, _ = analyze_sentiment(c)

            if "positive" in label.lower():
                results["Positive"] += 1
            elif "negative" in label.lower():
                results["Negative"] += 1
            else:
                results["Neutral"] += 1

        # 绘制饼图
        fig = go.Figure(data=[go.Pie(
            labels=list(results.keys()),
            values=list(results.values()),
            hole=0.3
        )])

        st.plotly_chart(fig, use_container_width=True)