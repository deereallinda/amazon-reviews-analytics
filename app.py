# ================================================================
# AMAZON REVIEWS – MODERN ANALYTICS DASHBOARD (CLEAN UI VERSION)
# Author: Linda Mthembu
# ================================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from textblob import TextBlob
from pathlib import Path

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Amazon Analytics Dashboard",
    page_icon="📦",
    layout="wide",
)

# ================================================================
# GLOBAL CSS (CLEAN UI, DARK MODE, NEUMORPHIC CARDS)
# ================================================================
st.markdown("""
<style>

body {
    background-color: #111 !important;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3, h4, h5, h6, label, p, div {
    color: #f0f0f0 !important;
    font-family: 'Segoe UI', sans-serif;
}

[data-testid="stMetricValue"] {
    font-size: 32px;
}

.metric-icon {
    font-size: 24px;
    margin-left: 6px;
}

.insight-title {
    font-size: 26px;
    font-weight: bold;
    color: #f1c40f;
    margin-bottom: 10px;
}

.insight-card {
    background: linear-gradient(145deg, #1c1c1c, #141414);
    padding: 14px 20px;
    border-radius: 10px;
    border: 1px solid #2a2a2a;
    margin-bottom: 12px;
    color: #e6e6e6;
    box-shadow: 4px 4px 8px #0a0a0a, -4px -4px 8px #1e1e1e;
    transition: 0.25s ease;
}

.insight-card:hover {
    transform: translateY(-2px);
    box-shadow: 6px 6px 12px #000, -6px -6px 12px #222;
}

</style>
""", unsafe_allow_html=True)

# ================================================================
# LOAD DATA
# ================================================================
@st.cache_data
def load_data(rel_path: str = "data/amazon_reviews_clean.csv") -> pd.DataFrame:
    try:
        csv_path = Path(__file__).parent / rel_path
        return pd.read_csv(csv_path, parse_dates=["Time"])
    except FileNotFoundError:
        st.error("❌ Data file not found.")
        return pd.DataFrame()

df_raw = load_data()
if df_raw.empty:
    st.stop()

# ================================================================
# SIDEBAR FILTERS
# ================================================================
st.sidebar.title("🛠 Filters")
rating_options = [1, 2, 3, 4, 5]

selected_ratings = st.sidebar.multiselect(
    "Select Ratings",
    options=rating_options,
    default=rating_options
)

df = df_raw[df_raw["Score"].isin(selected_ratings)].copy()

if df.empty:
    st.warning("No reviews found for selected ratings.")
    st.stop()

# ================================================================
# PAGE HEADER
# ================================================================
st.markdown("## 📦 Amazon Product Analytics Dashboard")
st.markdown("Insights into customer sentiment, CLV-style segments, and product performance.")

# ================================================================
# KPI SECTION
# ================================================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_reviews = len(df)
avg_rating = df["Score"].mean()
rating_delta_pct = ((avg_rating - 3) / 3) * 100
unique_products = df["ProductId"].nunique()
unique_users = df["UserId"].nunique()

kpi1.metric("Total Reviews", f"{total_reviews:,}")
kpi2.metric("Average Rating", f"{avg_rating:.2f} ⭐", f"{rating_delta_pct:.1f}% vs neutral")
kpi3.metric("Unique Products", f"{unique_products:,}")
kpi4.metric("Active Users", f"{unique_users:,}")

# Divider
st.markdown("---")

# ================================================================
# KEY INSIGHTS SUMMARY
# ================================================================
st.markdown('<div class="insight-title">⭐ Key Insights Summary</div>', unsafe_allow_html=True)

# 1: Sentiment stats
high_pct = len(df[df["Score"] >= 4]) / len(df) * 100
low_pct = len(df[df["Score"] <= 2]) / len(df) * 100

# 2: CLV calculation
def clv_segment_from_count(n):
    if n >= 100: return "Power Buyer (VIP)"
    if n >= 30: return "Loyal"
    if n >= 10: return "Regular"
    return "Occasional"

user_summary = df.groupby("UserId").agg(
    purchase_count=("ProductId", "count")
).reset_index()

user_summary["segment"] = user_summary["purchase_count"].apply(clv_segment_from_count)

top_segment = user_summary["segment"].value_counts().idxmax()

# Insight Cards
st.markdown(f"""
<div class="insight-card">
<b>1. Customer Sentiment:</b> Average rating is <b>{avg_rating:.2f}⭐</b>,  
about <b>{rating_delta_pct:.1f}%</b> {'above' if rating_delta_pct > 0 else 'below'} the neutral baseline.  
Positive reviews: <b>{high_pct:.1f}%</b> • Negative reviews: <b>{low_pct:.1f}%</b>.
</div>

<div class="insight-card">
<b>2. Customer Segments (CLV-Style):</b> The dominant behavioural group is  
<b>{top_segment}</b>, revealing where long-term value and loyalty opportunities lie.
</div>

<div class="insight-card">
<b>3. Product Performance:</b> Leading products show strong rating patterns, helping identify  
which items to promote and which require quality or delivery improvements.
</div>
""", unsafe_allow_html=True)

# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3 = st.tabs(["💬 Sentiment Analysis", "👥 Customer Segments", "📊 Product Performance"])

# ================================================================
# TAB 1 — SENTIMENT
# ================================================================
with tab1:
    st.subheader("Customer Sentiment Overview")

    @st.cache_data
    def compute_polarity(df):
        df_temp = df.copy()
        df_temp["polarity"] = df_temp["Summary"].astype(str).apply(lambda t: TextBlob(t).sentiment.polarity)
        return df_temp

    df_sent = compute_polarity(df)
    df_sent["emotion"] = df_sent["polarity"].apply(
        lambda p: "Joy" if p >= 0.4 else "Anger/Sad" if p <= -0.4 else "Neutral"
    )

    colA, colB = st.columns([1, 1.4])

    with colA:
        counts = df_sent["emotion"].value_counts().reset_index()
        counts.columns = ["Emotion", "Count"]
        fig = px.pie(
            counts,
            names="Emotion",
            values="Count",
            hole=0.55,
            color="Emotion",
            color_discrete_map={
                "Joy": "#2ecc71",
                "Neutral": "#95a5a6",
                "Anger/Sad": "#e74c3c",
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.markdown("### 📝 Voice of the Customer")

        tab_pos, tab_neg = st.tabs(["✅ Positive Highlights", "⚠️ Negative Feedback"])

        with tab_pos:
            positives = df_sent[df_sent["polarity"] > 0.2]["Summary"].head(5)
            for p in positives:
                st.success("• " + str(p))

        with tab_neg:
            negatives = df_sent[df_sent["polarity"] < -0.2]["Summary"].head(5)
            for n in negatives:
                st.error("• " + str(n))

    st.markdown("""
#### 🧠 Interpretation from Data
- Sentiment polarity provides a quick snapshot of emotional customer response.
- Joy indicates strong satisfaction; Anger/Sad highlights recurring product or delivery issues.
- This helps product teams prioritise fixes and helps marketing amplify strengths.
""")

# ================================================================
# TAB 2 — CUSTOMER SEGMENTS (CLV)
# ================================================================
with tab2:
    st.subheader("Customer Value Segmentation (CLV-Style)")

    st.markdown("""
**CLV (Customer Lifetime Value)** measures how valuable a customer is based on long-term purchasing.  
CLV-style segmentation helps identify high-value buyers, growth opportunities, and at-risk customers.
""")

    seg_counts = user_summary["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Users"]

    fig_seg = px.bar(
        seg_counts,
        x="Segment",
        y="Users",
        color="Segment",
        text="Users",
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig_seg.update_traces(textposition="outside")
    st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("""
#### 🧠 Interpretation from Data
- Power Buyers drive disproportionate engagement and should receive loyalty perks.
- Regular users can be nudged upward with targeted retention strategies.
- Occasional users require activation campaigns.
""")

# ================================================================
# TAB 3 — PRODUCT PERFORMANCE
# ================================================================
with tab3:
    st.subheader("Product Rating Distribution")

    top_n = st.slider("Show Top N Products", 5, 50, 10)
    min_reviews = st.number_input("Minimum Reviews", 1, 100, 5)

    prod_stats = df.groupby("ProductId").agg(
        count=("Score", "count"),
        avg=("Score", "mean")
    ).reset_index()

    prod_stats = prod_stats[prod_stats["count"] >= min_reviews]
    top_products = prod_stats.sort_values(by="count", ascending=False).head(top_n)

    subset = df[df["ProductId"].isin(top_products["ProductId"])]

    chart_data = subset.groupby(["ProductId", "Score"]).size().reset_index(name="Count")
    chart_data["Score"] = chart_data["Score"].astype(str)

    fig_prod = px.bar(
        chart_data,
        x="Count",
        y="ProductId",
        color="Score",
        orientation="h",
        title="Rating Distribution for Top Products",
        category_orders={"Score": ["5", "4", "3", "2", "1"]}
    )
    fig_prod.update_layout(height=600)
    st.plotly_chart(fig_prod, use_container_width=True)

    st.markdown("""
#### 🧠 Interpretation from Data
- Products with strong high-rating dominance perform well and should be promoted.
- Products with many low ratings require quality checks or listing corrections.
""")
