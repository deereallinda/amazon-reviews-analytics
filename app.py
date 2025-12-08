# ================================================================
# AMAZON REVIEWS – SENTIMENT, SEGMENTS & PRODUCT ANALYTICS
# Author: Linda Mthembu (Upgraded UX Version)
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
# BASIC CSS TWEAKS
# ------------------------------------------------
st.markdown(
    """
<style>
    .block-container {
        padding-top: 1rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #d6d6d6;
        padding: 10px;
        border-radius: 5px;
        color: #262730;
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
    """
    Load the cleaned Amazon reviews CSV relative to this app.
    """
    try:
        app_dir = Path(__file__).parent
        csv_path = app_dir / rel_path
        df = pd.read_csv(csv_path, parse_dates=["Time"])
        return df
    except FileNotFoundError:
        st.error(f"❌ File not found: {rel_path}. Please ensure the data file exists.")
        return pd.DataFrame()


data = load_data()

# Stop if data is missing
if data.empty:
    st.stop()

# ================================================================
# SIDEBAR – FILTERS
# ================================================================
st.sidebar.title("🛠️ Controls")
st.sidebar.divider()

st.sidebar.subheader("Filter Reviews by Rating")

rating_options = [1, 2, 3, 4, 5]

selected_ratings = st.sidebar.multiselect(
    "Select Star Ratings:",
    options=rating_options,
    default=rating_options,
    format_func=lambda x: f"{x} Stars",
)

if not selected_ratings:
    st.sidebar.warning("Please select at least one rating to display data.")
    st.stop()

df = data[data["Score"].isin(selected_ratings)].copy()

st.sidebar.divider()
st.sidebar.markdown("**Data Scope:**")
st.sidebar.info(f"📊 **{len(df):,}** reviews in current view")

# If filtering kills all data
if df.empty:
    st.warning("No reviews match the selected filters. Adjust the ratings on the left.")
    st.stop()

# ================================================================
# PAGE HEADER & KPI ROW
# ================================================================
st.title("📦 Amazon Product Analytics")
st.markdown(
    """
Insights into customer **sentiment**, **lifetime value segments**, and **product performance**  
based on Amazon product review data.
"""
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(label="Total Reviews", value=f"{len(df):,}")

with kpi2:
    avg_score = df["Score"].mean()
    st.metric(
        label="Average Rating",
        value=f"{avg_score:.2f} ⭐",
        delta=f"{avg_score - 3:.2f} vs neutral (3⭐)",
    )

with kpi3:
    unique_products = df["ProductId"].nunique()
    st.metric(label="Unique Products", value=f"{unique_products:,}")

with kpi4:
    unique_users = df["UserId"].nunique()
    st.metric(label="Active Users", value=f"{unique_users:,}")

st.markdown("---")

# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3 = st.tabs(
    [
        "💬 Sentiment Analysis",
        "👥 Customer Segments",
        "📊 Product Performance",
    ]
)

# ================================================================
# TAB 1 – SENTIMENT OVERVIEW
# ================================================================
with tab1:
    st.subheader("Customer Emotions & Feedback")

    # 1. Compute polarity (cached)
    @st.cache_data
    def compute_polarity(df_input: pd.DataFrame) -> pd.DataFrame:
        df_temp = df_input.copy()

        def polarity(text):
            try:
                return TextBlob(str(text)).sentiment.polarity
            except Exception:
                return 0.0

        df_temp["polarity"] = df_temp["Summary"].astype(str).apply(polarity)
        return df_temp

    df_sent = compute_polarity(df)

    def emotion_label(p: float) -> str:
        if p >= 0.4:
            return "Joy"
        if p <= -0.4:
            return "Anger/Sad"
        return "Neutral"

    df_sent["emotion"] = df_sent["polarity"].apply(emotion_label)

    # 2. Layout: donut + voice of customer
    col_chart, col_text = st.columns([1, 1.5], gap="large")

    with col_chart:
        emotion_counts = (
            df_sent["emotion"].value_counts().reset_index()
        )  # columns: index, emotion
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
        fig_donut.update_layout(showlegend=True)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_text:
        st.markdown("##### 📝 Voice of the Customer")

        sub_tab_neg, sub_tab_pos = st.tabs(
            ["⚠️ Negative Feedback", "✅ Positive Highlights"]
        )

        with sub_tab_neg:
            neg_reviews = (
                df_sent[df_sent["polarity"] < -0.2]["Summary"].dropna().head(5).tolist()
            )
            if neg_reviews:
                for rev in neg_reviews:
                    st.error(f"“{rev}”")
            else:
                st.info("No strongly negative summaries found for this selection.")

        with sub_tab_pos:
            pos_reviews = (
                df_sent[df_sent["polarity"] > 0.2]["Summary"].dropna().head(5).tolist()
            )
            if pos_reviews:
                for rev in pos_reviews:
                    st.success(f"“{rev}”")
            else:
                st.info("No strongly positive summaries found for this selection.")

    # 3. Executive interpretation
    joy_pct = (
        len(df_sent[df_sent["emotion"] == "Joy"]) / len(df_sent) * 100
        if len(df_sent) > 0
        else 0
    )
    anger_pct = (
        len(df_sent[df_sent["emotion"] == "Anger/Sad"]) / len(df_sent) * 100
        if len(df_sent) > 0
        else 0
    )

    with st.expander("🧠 Executive Interpretation (Click to expand)"):
        st.markdown(
            f"""
**What this shows**

- The donut chart summarizes how customers *feel* about their purchases (Joy, Neutral, Anger/Sad).
- The example summaries provide concrete, real-world customer statements.

**Current emotional mix**

- **Joy:** ~{joy_pct:.1f}% of summaries  
- **Anger/Sad:** ~{anger_pct:.1f}% of summaries  

**How to use this**

- A high **Joy** share means marketing can confidently highlight customer satisfaction.
- A visible **Anger/Sad** share flags recurring product, delivery, or expectation issues.
- Negative summaries point to problems to fix; positive ones highlight strengths to amplify.

**Business value**

- Product teams can prioritise fixes based on recurring complaints.
- Support teams can prepare for common pain points.
- Marketing can reuse natural customer language that resonates.
- Executives get a quick emotional health check of the customer base.
"""
        )

# ================================================================
# TAB 2 – CUSTOMER SEGMENTS (CLV-STYLE)
# ================================================================
with tab2:
    st.subheader("Customer Value Segmentation (CLV-Style)")

    # 1. Aggregate by user
    user_agg = (
        df.groupby("UserId")
        .agg(
            Number_of_summaries=("Summary", "count"),
            No_of_prods_purchased=("ProductId", "count"),
            avg_score=("Score", "mean"),
        )
        .reset_index()
    )

    def clv_segment(row) -> str:
        if row["No_of_prods_purchased"] >= 100:
            return "Power Buyer (VIP)"
        if row["No_of_prods_purchased"] >= 30:
            return "Loyal"
        if row["No_of_prods_purchased"] >= 10:
            return "Regular"
        return "Occasional"

    user_agg["clv_segment"] = user_agg.apply(clv_segment, axis=1)

    seg_counts = (
        user_agg["clv_segment"].value_counts().reset_index()
    )  # Segment, User Count
    seg_counts.columns = ["Segment", "User Count"]

    col_seg_chart, col_seg_text = st.columns([2, 1])

    with col_seg_chart:
        fig_seg = px.bar(
            seg_counts,
            x="Segment",
            y="User Count",
            color="Segment",
            text="User Count",
            title="User Count by CLV Segment",
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig_seg.update_traces(textposition="outside")
        fig_seg.update_layout(xaxis_title="", yaxis_title="Number of Users")
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_seg_text:
        st.markdown("### 💡 Strategic Pointers")
        st.info(
            "**Power Buyers (VIP):**\nHigh-value customers, ideal for loyalty perks, early access, and special offers."
        )
        st.success(
            "**Loyal:**\nStrong cross-sell and upsell candidates – recommend related or premium products."
        )
        st.warning(
            "**Regular:**\nCan be nudged into Loyalty via 'Buy Again', bundle deals, and reminders."
        )
        st.info(
            "**Occasional:**\nGood segment for broad promotions, introductory offers, and remarketing."
        )

    with st.expander("🧠 Executive Interpretation (Click to expand)"):
        st.markdown(
            """
**What this shows**

- Customers are grouped into segments based on how many products they’ve interacted with (reviewed).
- Segments approximate **Customer Lifetime Value (CLV)** patterns.

**Segments**

- **Power Buyer (VIP):** very high engagement (100+ products).  
- **Loyal:** consistent, repeated engagement (30–99 products).  
- **Regular:** moderate engagement (10–29 products).  
- **Occasional:** low engagement (<10 products).

**How to use this**

- Protect and delight **Power Buyers** – they drive a disproportionate share of revenue.
- Use targeted, personalised campaigns for **Loyal** and **Regular** customers to increase basket size.
- Allocate marketing budget wisely: prioritise converting **Regular → Loyal** over chasing one-off Occasional buyers.

**Business value**

- Guides **retention**, **loyalty**, and **CRM** strategies.
- Helps executives understand whether the business is driven by a few heavy users or a broad base.
- Links directly to revenue planning and marketing spend allocation.
"""
        )

# ================================================================
# TAB 3 – PRODUCT POPULARITY & RATINGS
# ================================================================
with tab3:
    st.subheader("Product Performance Matrix")

    col_controls, col_viz = st.columns([1, 3])

    with col_controls:
        st.markdown("#### Display Settings")
        top_n = st.slider("Number of Top Products:", 5, 50, 10)
        min_reviews = st.number_input(
            "Minimum reviews per product:", min_value=1, value=5
        )

    # 1. Product-level stats
    prod_stats = (
        df.groupby("ProductId")
        .agg(
            review_count=("Score", "count"),
            avg_score=("Score", "mean"),
        )
        .reset_index()
    )

    prod_stats = prod_stats[prod_stats["review_count"] >= min_reviews]

    if prod_stats.empty:
        st.warning(
            "No products meet the minimum review threshold. Lower the 'Minimum reviews' setting."
        )
    else:
        top_products = prod_stats.sort_values(
            by="review_count", ascending=False
        ).head(top_n)

        subset = df[df["ProductId"].isin(top_products["ProductId"])]

        # Group for stacked bar: count by ProductId & Score
        chart_data = (
            subset.groupby(["ProductId", "Score"])
            .size()
            .reset_index(name="Count")
        )

        # Treat Score as string for clearer color handling
        chart_data["Score"] = chart_data["Score"].astype(str)

        fig_stack = px.bar(
            chart_data,
            x="Count",
            y="ProductId",
            color="Score",
            orientation="h",
            title=f"Rating Distribution for Top {top_n} Products",
            labels={"ProductId": "Product ID", "Score": "Star Rating"},
            category_orders={"Score": ["5", "4", "3", "2", "1"]},
            color_discrete_sequence=px.colors.sequential.Viridis,
        )

        # Sort by total review count
        fig_stack.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=550,
        )

        with col_viz:
            st.plotly_chart(fig_stack, use_container_width=True)

    with st.expander("🧠 Executive Interpretation (Click to expand)"):
        st.markdown(
            """
**What this shows**

- The most-reviewed products (by volume).
- For each product, how many 5★, 4★, 3★, 2★, and 1★ ratings it receives.

**How to read it**

- Long bars dominated by high star ratings = **strong, healthy products**.
- Long bars with many low-star ratings = **problem products**: poor quality, misleading descriptions, or delivery issues.
- Products with many reviews and mixed ratings may require closer investigation.

**How to use this**

- Promote high-volume, high-rating products in campaigns and on landing pages.
- Investigate high-volume, low-rating products for potential fixes, supplier changes, or listing updates.
- Use this view alongside sentiment analysis to pinpoint specific failure reasons.

**Business value**

- Directly informs **product strategy**, **promotions**, and **quality control**.
- Helps avoid pushing products that damage customer trust.
- Highlights products that can be scaled aggressively because they are both popular and well-reviewed.
"""
        )
