# ================================================================
# AMAZON REVIEWS – SENTIMENT, SEGMENTS & PRODUCT ANALYTICS
# Author: Linda Mthembu (Enhanced UX Edition)
# ================================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from textblob import TextBlob
from pathlib import Path

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Amazon Reviews Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------
# CSS STYLING
# ------------------------------------------------
st.markdown(
    """
<style>
    .block-container {padding-top: 1.2rem;}
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #d6d6d6;
        padding: 10px;
        border-radius: 5px;
        color: #262730;
    }
    .insight-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #eef2f7;
        border-left: 6px solid #4a90e2;
        margin-bottom: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ================================================================
# LOAD DATA
# ================================================================
@st.cache_data
def load_data(rel_path: str = "data/amazon_reviews_clean.csv") -> pd.DataFrame:
    try:
        app_dir = Path(__file__).parent
        csv_path = app_dir / rel_path
        df = pd.read_csv(csv_path, parse_dates=["Time"])
        return df
    except FileNotFoundError:
        st.error(f"❌ File not found: {rel_path}")
        return pd.DataFrame()

data = load_data()
if data.empty:
    st.stop()

# ================================================================
# SIDEBAR — FILTERS
# ================================================================
st.sidebar.title("🛠️ Controls")
st.sidebar.divider()

rating_options = [1, 2, 3, 4, 5]

selected_ratings = st.sidebar.multiselect(
    "Select Star Ratings:",
    options=rating_options,
    default=rating_options,
    format_func=lambda x: f"{x} Stars"
)

df = data[data["Score"].isin(selected_ratings)].copy()

st.sidebar.divider()
st.sidebar.info(f"📊 **{len(df):,}** reviews in current view")

if df.empty:
    st.warning("No reviews match the selected filters.")
    st.stop()

# ================================================================
# HEADER & KPI SECTION
# ================================================================
st.title("📦 Amazon Product Analytics Dashboard")
st.markdown("Insights into customer sentiment, CLV-style segments, and product performance.")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(label="Total Reviews", value=f"{len(df):,}")

with kpi2:
    avg_score = df["Score"].mean()
    st.metric(label="Average Rating", value=f"{avg_score:.2f} ⭐")

with kpi3:
    st.metric(label="Unique Products", value=f"{df['ProductId'].nunique():,}")

with kpi4:
    st.metric(label="Active Users", value=f"{df['UserId'].nunique():,}")

st.markdown("---")

# ================================================================
# 🔥 KEY INSIGHTS SUMMARY
# ================================================================
st.markdown("### ⭐ Key Insights Summary")

st.markdown(
    f"""
<div class='insight-box'>
<strong>1. Customer Sentiment:</strong>  
Average rating is <strong>{avg_score:.2f}⭐</strong>, indicating overall customer satisfaction with a balanced mix of positive and neutral experiences.
</div>

<div class='insight-box'>
<strong>2. Customer Segments (CLV-Style):</strong>  
A small group of <strong>Power Buyers</strong> contributes disproportionately to product interactions — critical for retention and loyalty strategies.
</div>

<div class='insight-box'>
<strong>3. Product Performance:</strong>  
Top-reviewed products show clear rating patterns, highlighting which items dominate the market and which need quality improvements.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3 = st.tabs([
    "💬 Sentiment Analysis",
    "👥 Customer Segments (CLV-Style)",
    "📊 Product Performance"
])

# ================================================================
# TAB 1 — SENTIMENT ANALYSIS
# ================================================================
with tab1:
    st.subheader("Customer Emotions & Feedback")

    @st.cache_data
    def compute_polarity(df_input):
        df_temp = df_input.copy()
        df_temp["polarity"] = df_temp["Summary"].astype(str).apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        return df_temp

    df_sent = compute_polarity(df)

    def emotion_label(p):
        if p >= 0.4: return "Joy"
        if p <= -0.4: return "Anger/Sad"
        return "Neutral"

    df_sent["emotion"] = df_sent["polarity"].apply(emotion_label)

    col_chart, col_text = st.columns([1, 1.5])

    with col_chart:
        emotion_counts = df_sent["emotion"].value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]

        fig_donut = px.pie(
            emotion_counts,
            names="Emotion",
            values="Count",
            hole=0.55,
            color="Emotion",
            color_discrete_map={
                "Joy": "#2ecc71",
                "Neutral": "#95a5a6",
                "Anger/Sad": "#e74c3c",
            },
            title="Emotional Distribution",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_text:
        st.markdown("##### 📝 Voice of the Customer")

        sub_tab_pos, sub_tab_neg = st.tabs(["✅ Positive Highlights", "⚠️ Negative Feedback"])

        pos_reviews = df_sent[df_sent["polarity"] > 0.2]["Summary"].head(5).tolist()
        neg_reviews = df_sent[df_sent["polarity"] < -0.2]["Summary"].head(5).tolist()

        with sub_tab_pos:
            if pos_reviews:
                for rev in pos_reviews:
                    st.success(f"“{rev}”")
            else:
                st.info("No strongly positive reviews found.")

        with sub_tab_neg:
            if neg_reviews:
                for rev in neg_reviews:
                    st.error(f"“{rev}”")
            else:
                st.info("No strongly negative reviews found.")

    joy_pct = len(df_sent[df_sent["emotion"] == "Joy"]) / len(df_sent) * 100
    anger_pct = len(df_sent[df_sent["emotion"] == "Anger/Sad"]) / len(df_sent) * 100

    with st.expander("🧠 Interpretation from Data"):
        st.markdown(
            f"""
### What the data is showing  

- The sentiment distribution provides a quick emotional health check of customer experience.
- **Joy:** ~{joy_pct:.1f}% — strong satisfaction and product alignment with expectations.  
- **Anger/Sad:** ~{anger_pct:.1f}% — indicators of friction points requiring attention.

### Why this matters  

- Positive sentiment highlights what should be leveraged in branding and marketing.
- Negative sentiment helps identify product defects, delivery issues, or listing problems.
"""
        )

# ================================================================
# TAB 2 — CUSTOMER VALUE SEGMENTATION (CLV)
# ================================================================
with tab2:
    st.subheader("Customer Value Segmentation (CLV-Style)")

    st.markdown("""
**CLV (Customer Lifetime Value)** estimates how valuable a customer is over time based  
on their product interactions and behaviour.

**CLV-Style Segmentation** groups users into categories that represent different  
levels of value contribution — essential for retention and marketing strategies.
""")

    user_agg = (
        df.groupby("UserId")
        .agg(
            Number_of_summaries=("Summary", "count"),
            No_of_prods_purchased=("ProductId", "count"),
        )
        .reset_index()
    )

    def clv_segment(row):
        if row["No_of_prods_purchased"] >= 100: return "Power Buyer (VIP)"
        if row["No_of_prods_purchased"] >= 30: return "Loyal"
        if row["No_of_prods_purchased"] >= 10: return "Regular"
        return "Occasional"

    user_agg["clv_segment"] = user_agg.apply(clv_segment, axis=1)
    seg_counts = user_agg["clv_segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "User Count"]

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_bar = px.bar(
            seg_counts,
            x="Segment",
            y="User Count",
            color="Segment",
            text="User Count",
            title="User Count by Segment",
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("### 💡 Strategy Recommendations")
        st.info("**Power Buyers:** Reward with loyalty perks and early access.")
        st.success("**Loyal:** Great candidates for cross-sell and up-sell.")
        st.warning("**Regular:** Nurture to move them toward loyalty.")
        st.info("**Occasional:** Promote low-risk introductory offers.")

    with st.expander("🧠 Interpretation from Data"):
        st.markdown("""
### What the data is showing

Different customers contribute differently to overall product engagement.  
Understanding these patterns directs marketing efforts more efficiently.

### Segment Insights  
- **Power Buyers:** High-value, revenue-driving users.  
- **Loyal:** Consistent customers with strong retention potential.  
- **Regular:** Middle-tier users who can be nurtured upward.  
- **Occasional:** Broad but low-value audience.

### Why this matters  
CLV segmentation helps Amazon:
- Prioritize retention efforts  
- Allocate marketing spend strategically  
- Protect and grow the highest-value customers  
""")

# ================================================================
# TAB 3 — PRODUCT PERFORMANCE
# ================================================================
with tab3:
    st.subheader("Product Performance Matrix")

    col_controls, col_viz = st.columns([1, 3])

    with col_controls:
        top_n = st.slider("Number of Top Products:", 5, 50, 10)
        min_reviews = st.number_input("Minimum reviews per product:", min_value=1, value=5)

    prod_stats = (
        df.groupby("ProductId")
        .agg(
            review_count=("Score", "count"),
            avg_score=("Score", "mean"),
        )
        .reset_index()
    )

    prod_stats = prod_stats[prod_stats["review_count"] >= min_reviews]
    top_products = prod_stats.sort_values(by="review_count", ascending=False).head(top_n)
    subset = df[df["ProductId"].isin(top_products["ProductId"])]

    chart_data = (
        subset.groupby(["ProductId", "Score"])
        .size()
        .reset_index(name="Count")
    )

    chart_data["Score"] = chart_data["Score"].astype(str)

    fig_stack = px.bar(
        chart_data,
        x="Count",
        y="ProductId",
        color="Score",
        orientation="h",
        title=f"Rating Distribution for Top {top_n} Products",
        category_orders={"Score": ["5", "4", "3", "2", "1"]},
        color_discrete_sequence=px.colors.sequential.Viridis,
    )

    with col_viz:
        st.plotly_chart(fig_stack, use_container_width=True)

    with st.expander("🧠 Interpretation from Data"):
        st.markdown("""
### What the data is showing  

- Products with strong 5★ and 4★ distribution indicate high customer satisfaction.  
- Products with high 1★ and 2★ reviews likely have defects, misleading descriptions, or supply issues.

### Why this matters  

- Helps identify which products to promote aggressively.  
- Flags underperforming products needing quality review.  
- Supports inventory and pricing decisions.  
""")

