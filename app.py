# ================================================================
# AMAZON REVIEWS – SENTIMENT, SEGMENTS & PRODUCT ANALYTICS
# Author: Linda Mthembu (Upgraded Version)
# ================================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from pathlib import Path

# SET PAGE CONFIGURATION FIRST
st.set_page_config(
    page_title="Amazon Reviews Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS FOR METRICS AND HEADERS
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #d6d6d6;
        padding: 10px;
        border-radius: 5px;
        color: #262730;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# LOAD DATA
# ================================================================
@st.cache_data
def load_data(rel_path: str = "data/amazon_reviews_clean.csv") -> pd.DataFrame:
    # Simulating data creation if file doesn't exist for demo purposes
    # In production, ensure the CSV exists or handle the error
    try:
        app_dir = Path(__file__).parent
        csv_path = app_dir / rel_path
        df = pd.read_csv(csv_path, parse_dates=["Time"])
        return df
    except FileNotFoundError:
        st.error(f"File not found: {rel_path}. Please ensure data exists.")
        return pd.DataFrame()

data = load_data()

# Stop execution if data is empty
if data.empty:
    st.stop()

# ================================================================
# SIDEBAR — FILTERS
# ================================================================
st.sidebar.title("🛠️ Controls")
st.sidebar.divider()

st.sidebar.subheader("Filter Reviews")
rating_options = [1, 2, 3, 4, 5]

# Added a "Select All" logic implicitly by defaulting to all
selected_ratings = st.sidebar.multiselect(
    "Select Star Ratings:",
    options=rating_options,
    default=rating_options,
    format_func=lambda x: f"{x} Stars"
)

# Apply Filter
df = data[data["Score"].isin(selected_ratings)].copy()

st.sidebar.divider()
st.sidebar.markdown(f"**Data Scope:**")
st.sidebar.info(f"📊 **{len(df):,}** reviews loaded")

# ================================================================
# MAIN PAGE HEADER & KPI ROW
# ================================================================
st.title("📦 Amazon Product Analytics")
st.markdown("Insights into customer sentiment, lifetime value segments, and product performance.")

# KPI Row - Instant value for the user
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(label="Total Reviews", value=f"{len(df):,}")

with kpi2:
    avg_score = df['Score'].mean()
    delta_color = "normal" if avg_score > 3 else "inverse"
    st.metric(label="Average Rating", value=f"{avg_score:.2f} ⭐", delta=f"{avg_score-3:.2f} vs Neutral")

with kpi3:
    unique_products = df['ProductId'].nunique()
    st.metric(label="Unique Products", value=f"{unique_products:,}")

with kpi4:
    unique_users = df['UserId'].nunique()
    st.metric(label="Active Users", value=f"{unique_users:,}")

st.markdown("---")

# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3 = st.tabs([
    "💬 Sentiment Analysis",
    "👥 Customer Segments",
    "📊 Product Performance"
])

# ================================================================
# TAB 1 — SENTIMENT OVERVIEW
# ================================================================
with tab1:
    st.subheader("Customer Emotions & Feedback")
    
    # 1. Compute Polarity (Cached)
    @st.cache_data
    def compute_polarity_cached(df_input):
        def polarity(text):
            try:
                return TextBlob(str(text)).sentiment.polarity
            except:
                return 0.0
        
        # Working on a copy to prevent SettingWithCopy warnings
        df_temp = df_input.copy()
        df_temp["polarity"] = df_temp["Summary"].astype(str).apply(polarity)
        return df_temp

    df = compute_polarity_cached(df)

    def emotion_label(p):
        if p >= 0.4: return "Joy"
        if p <= -0.4: return "Anger/Sad"
        return "Neutral"

    df["emotion"] = df["polarity"].apply(emotion_label)

    # 2. Visuals
    col_chart, col_text = st.columns([1, 1.5], gap="large")

    with col_chart:
        emotion_counts = df["emotion"].value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]
        
        # UX Improvement: Plotly Donut Chart
        fig_donut = px.pie(
            emotion_counts, 
            names="Emotion", 
            values="Count", 
            hole=0.5,
            color="Emotion",
            color_discrete_map={"Joy": "#2ecc71", "Neutral": "#95a5a6", "Anger/Sad": "#e74c3c"},
            title="Emotional Distribution"
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_text:
        st.markdown("##### 📝 Voice of the Customer")
        
        # Tabs inside the column for better organization
        sub_tab_neg, sub_tab_pos = st.tabs(["⚠️ Negative Feedback", "✅ Positive Highlights"])
        
        with sub_tab_neg:
            neg_reviews = df[df["polarity"] < -0.2]["Summary"].head(5).tolist()
            if neg_reviews:
                for rev in neg_reviews:
                    st.error(f"\"{rev}\"")
            else:
                st.info("No strongly negative reviews found in this selection.")

        with sub_tab_pos:
            pos_reviews = df[df["polarity"] > 0.2]["Summary"].head(5).tolist()
            if pos_reviews:
                for rev in pos_reviews:
                    st.success(f"\"{rev}\"")
            else:
                st.info("No strongly positive reviews found in this selection.")

    # 3. Executive Interpretation (Hidden in Expander)
    with st.expander("🧠 Executive Interpretation (Click to Expand)"):
        st.markdown("""
        **Business Value:**
        * **Joy ({:.1f}%)**: Indicates features to double-down on in marketing.
        * **Anger ({:.1f}%)**: Indicates urgent friction points. If this grows, check supplier quality immediately.
        """.format(
            (len(df[df['emotion']=='Joy'])/len(df)*100) if len(df)>0 else 0,
            (len(df[df['emotion']=='Anger/Sad'])/len(df)*100) if len(df)>0 else 0
        ))

# ================================================================
# TAB 2 — CUSTOMER SEGMENTS (CLV)
# ================================================================
with tab2:
    st.subheader("Customer Value Segmentation")
    
    # 1. Data Processing
    user_agg = (
        df.groupby("UserId")
        .agg(
            Number_of_summaries=("Summary", "count"),
            No_of_prods_purchased=("ProductId", "count"),
        )
    )

    def clv_segment(row):
        if row["No_of_prods_purchased"] >= 100: return "Power Buyer (VIP)"
        if row["No_of_prods_purchased"] >= 30: return "Loyal"
        if row["No_of_prods_purchased"] >= 10: return "Regular"
        return "Occasional"

    user_agg["clv_segment"] = user_agg.apply(clv_segment, axis=1)
    seg_counts = user_agg["clv_segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "User Count"]

    # 2. Visuals
    col1, col2 = st.columns([2, 1])

    with col1:
        # UX Improvement: Plotly Bar Chart with Hover
        fig_bar = px.bar(
            seg_counts, 
            x="Segment", 
            y="User Count",
            color="Segment",
            text="User Count",
            title="User Count by Segment",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("### 💡 Strategy")
        
        # Dynamic advice based on selection
        st.info("**Power Buyers:**\nTarget with exclusive loyalty programs/early access.")
        st.warning("**Regulars:**\nTarget with 'Buy Again' notifications and cross-selling.")

    with st.expander("🧠 Executive Interpretation (Click to Expand)"):
        st.markdown("""
        * **Power Buyers** are your revenue engine. Even if small in number, they drive volume.
        * **Occasional Buyers** are mostly one-off traffic. Focus marketing spend on converting **Regulars -> Loyal**.
        """)

# ================================================================
# TAB 3 — PRODUCT POPULARITY & RATINGS
# ================================================================
with tab3:
    st.subheader("Product Performance Matrix")
    
    col_controls, col_viz = st.columns([1, 3])

    with col_controls:
        st.markdown("#### Settings")
        top_n = st.slider("Number of Top Products:", 5, 50, 10)
        min_reviews = st.number_input("Min Reviews Required:", min_value=1, value=5)

    # 1. Data Processing
    # Filter products with at least 'min_reviews'
    prod_stats = df.groupby("ProductId").agg(
        review_count=("Score", "count"),
        avg_score=("Score", "mean")
    ).reset_index()
    
    prod_stats = prod_stats[prod_stats['review_count'] >= min_reviews]
    top_products = prod_stats.sort_values(by="review_count", ascending=False).head(top_n)

    # Merge back to get individual scores for distribution
    subset = df[df["ProductId"].isin(top_products["ProductId"])]

    # 2. Visuals
    with col_viz:
        # UX Improvement: Stacked Bar Chart for Rating Breakdown
        # We need to aggregate counts per score per product
        chart_data = subset.groupby(['ProductId', 'Score']).size().reset_index(name='Count')
        
        fig_stacked = px.bar(
            chart_data, 
            x="Count", 
            y="ProductId", 
            color="Score", 
            orientation='h',
            title=f"Rating Distribution for Top {top_n} Products",
            labels={"ProductId": "Product ID", "Score": "Star Rating"},
            category_orders={"Score": [5, 4, 3, 2, 1]}, # Ensure 5 stars is distinct
            color_continuous_scale=px.colors.sequential.Viridis
        )
        # Sort y-axis by total count
        fig_stacked.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
        st.plotly_chart(fig_stacked, use_container_width=True)

    with st.expander("🧠 Executive Interpretation (Click to Expand)"):
        st.markdown("""
        * **Long Bars with Yellow/Green (High Scores):** Winners. Maintain inventory.
        * **Long Bars with Purple/Blue (Low Scores):** High visibility but bad reputation. **Action:** Investigate quality immediately.
        """)