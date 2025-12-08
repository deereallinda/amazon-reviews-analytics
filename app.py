# =============================================================================
# AMAZON REVIEWS ANALYTICS APP
# Author: Linda Mthembu (Upgraded Version)
# =============================================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from pathlib import Path

# =============================================================================
# CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="Amazon Reviews Analytics", 
    page_icon="📦",
    layout="wide"
)

# Custom CSS for metric cards and headers
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #d6d6d6;
        padding: 10px;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f9f9f9;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data(rel_path: str = "data/amazon_reviews_clean.csv") -> pd.DataFrame:
    try:
        app_dir = Path(__file__).parent
        csv_path = app_dir / rel_path
        # Create dummy data if file is missing for demo purposes
        if not csv_path.exists():
             st.error(f"File not found at {csv_path}. Please ensure data exists.")
             return pd.DataFrame()
             
        df = pd.read_csv(csv_path, parse_dates=["Time"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

data = load_data()

if data.empty:
    st.stop()

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================
st.sidebar.title("🛠️ Controls")
st.sidebar.divider()

st.sidebar.subheader("Filter Reviews by Rating")
rating_options = [1, 2, 3, 4, 5]

selected_ratings = st.sidebar.multiselect(
    "Select Star Ratings:",
    options=rating_options,
    default=rating_options,
    format_func=lambda x: f"{x} Stars"
)

# Apply Filter
df = data[data["Score"].isin(selected_ratings)].copy()

st.sidebar.divider()
st.sidebar.markdown("**Data Scope:**")
st.sidebar.info(f"📊 **{len(df):,}** reviews loaded")

# =============================================================================
# MAIN PAGE HEADER & METRICS
# =============================================================================
st.title("📦 Amazon Reviews – Analytics Dashboard")

# KPI Row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Reviews", f"{len(df):,}")
k2.metric("Avg Rating", f"{df['Score'].mean():.2f} ⭐")
k3.metric("Unique Products", f"{df['ProductId'].nunique():,}")
k4.metric("Active Users", f"{df['UserId'].nunique():,}")

st.markdown("---")

# =============================================================================
# DATA PROCESSING FOR TABS
# =============================================================================

# 1. User Aggregates
user_agg = (
    df.groupby("UserId")
    .agg(
        Number_of_summaries=("Summary", "count"),
        num_text=("Text", "count"),
        avg_score=("Score", "mean"),
        No_of_prods_purchased=("ProductId", "count"),
    )
)

# 2. CLV Segment Logic
def clv_segment(row):
    if row["No_of_prods_purchased"] >= 100: return "Power Buyer (VIP)"
    if row["No_of_prods_purchased"] >= 30: return "Loyal"
    if row["No_of_prods_purchased"] >= 10: return "Regular"
    return "Occasional"

user_agg["clv_segment"] = user_agg.apply(clv_segment, axis=1)

# 3. Helpfulness Ratio
# Avoid division by zero
mask_den = df["HelpfulnessDenominator"] > 0
df.loc[mask_den, "helpfulness_ratio"] = (
    df.loc[mask_den, "HelpfulnessNumerator"] / df.loc[mask_den, "HelpfulnessDenominator"]
)
df["helpfulness_ratio"] = df["helpfulness_ratio"].fillna(0.0)

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👥 Customer Segments",
    "📊 Products & Ratings",
    "👍 Helpfulness vs Rating",
    "📝 Review Verbosity",
    "💬 Sentiment Overview",
])

# -----------------------------------------------------------------------------
# TAB 1 – CUSTOMER SEGMENTS
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("👥 Customer Segmentation (CLV-Style)")
    st.caption("CLV = Customer Lifetime Value. This groups users by their purchasing volume.")

    col1, col2 = st.columns([1, 2])

    seg_counts = user_agg["clv_segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    with col1:
        st.dataframe(seg_counts, use_container_width=True)

    with col2:
        fig_seg = px.bar(
            seg_counts, 
            x="Segment", 
            y="Count", 
            color="Segment",
            text="Count",
            title="User Count by CLV Segment",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_seg.update_traces(textposition='outside')
        st.plotly_chart(fig_seg, use_container_width=True)

    with st.expander("🧠 Interpretation from data (Click to open)", expanded=True):
        
        st.markdown("""
        **What is CLV-Style Segmentation?**
        We are grouping users based on frequency to estimate their value.
        
        * **Power Buyers (VIP):** High purchase volume. Even if they are few, they drive significant revenue.
        * **Loyal:** Consistent repeat purchasers.
        * **Occasional:** The "long tail" of customers who buy once or twice.
        
        **Strategy:** * *Power Buyers:* Reward with exclusivity.
        * *Occasional:* Re-engage with discounts.
        """)

# -----------------------------------------------------------------------------
# TAB 2 – PRODUCT POPULARITY & RATINGS
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("📊 Product Popularity & Rating Distribution")

    top_n = st.slider("Number of top products to show", 5, 50, 10)

    # Calculate top products
    prod_counts = df["ProductId"].value_counts().head(top_n)
    top_prod_ids = prod_counts.index
    
    subset_prod = df[df["ProductId"].isin(top_prod_ids)]

    col_chart, col_data = st.columns([2, 1])

    with col_chart:
        # Stacked bar for distribution
        chart_data = subset_prod.groupby(['ProductId', 'Score']).size().reset_index(name='Count')
        fig_prod = px.bar(
            chart_data, 
            x="Count", 
            y="ProductId", 
            color="Score", 
            orientation='h',
            title=f"Rating Distribution (Top {top_n} Products)",
            labels={"ProductId": "Product ID", "Score": "Stars"},
            category_orders={"Score": [5, 4, 3, 2, 1]},
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig_prod.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_prod, use_container_width=True)

    with col_data:
        st.markdown("#### Review Counts")
        st.dataframe(prod_counts.rename("Reviews"), use_container_width=True)

    with st.expander("🧠 Interpretation from data"):
        st.markdown("""
        * **High Volume + High Rating (Yellow/Green):** Market leaders. Safe to promote.
        * **High Volume + Low Rating (Purple/Blue):** "Review Bombs." These products sell well but disappoint customers. **Action:** Check for defect batches or misleading descriptions.
        """)

# -----------------------------------------------------------------------------
# TAB 3 – HELPFULNESS VS RATING
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("👍 Helpfulness vs Rating Behaviour")

    sample_size = st.slider("Sample size for scatter plot (improves speed)", 1000, 20000, 5000)
    sample = df.sample(min(sample_size, len(df)), random_state=42)

    col_scatter, col_bar = st.columns(2)

    with col_scatter:
        # Scatter Plot
        fig_scat = px.scatter(
            sample, 
            x="helpfulness_ratio", 
            y="Score", 
            opacity=0.3,
            title="Helpfulness Ratio vs Rating (Sampled)",
            labels={"helpfulness_ratio": "Helpfulness Ratio (0 to 1)"}
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    with col_bar:
        # Bar Plot (Buckets)
        sample["bucket"] = pd.cut(
            sample["helpfulness_ratio"],
            bins=[-0.01, 0, 0.25, 0.5, 0.75, 1.0],
            labels=["0", "Low (0–0.25)", "Med (0.25–0.5)", "High (0.5–0.75)", "Very High (0.75–1)"]
        )
        bucket_agg = sample.groupby("bucket")["Score"].mean().reset_index()
        
        fig_buck = px.bar(
            bucket_agg, 
            x="bucket", 
            y="Score",
            title="Avg Rating per Helpfulness Bucket",
            color="Score",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_buck, use_container_width=True)

    with st.expander("🧠 Interpretation from data"):
        st.markdown("""
        * **Correlation Check:** If the "Very High" helpfulness bucket has a lower average score, it means **negative reviews are being voted as most helpful**.
        * **Bias Check:** If almost all helpful reviews are 5-star, the ecosystem may be biased or incentivised.
        """)

# -----------------------------------------------------------------------------
# TAB 4 – REVIEW VERBOSITY
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("📝 Review Length Analysis")
    
    # Generate Viewer Type on the fly if not in CSV to ensure code works
    if "viewer_type" not in df.columns:
        # Calculate frequency per user
        user_counts = df['UserId'].map(df['UserId'].value_counts())
        df['viewer_type'] = np.where(user_counts > 5, 'Frequent', 'Occasional')

    # Filter outliers for better visualization
    df_len = df[df["Text_length"] < 500]

    fig_box = px.box(
        df_len, 
        x="Score", 
        y="Text_length", 
        color="viewer_type",
        title="Review Length Distribution by Score & User Type",
        labels={"Text_length": "Word Count", "viewer_type": "User Type"}
    )
    st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("🧠 Interpretation from data"):
        st.markdown("""
        * **Box Height:** Taller boxes mean more variation in how much people write.
        * **Insight:** Frequent reviewers (Red/Blue distinction) often write longer, more nuanced reviews. 
        * **Negative vs Positive:** Typically, angry customers (1-2 stars) write longer text (essays) explaining *why* they are angry, whereas 5-star reviews can be short ("Great!").
        """)

# -----------------------------------------------------------------------------
# TAB 5 – SENTIMENT OVERVIEW
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("💬 Sentiment Analysis of Summaries")

    # Polarity Calculation
    @st.cache_data
    def compute_polarity(series: pd.Series) -> pd.Series:
        return series.astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)

    df["polarity"] = compute_polarity(df["Summary"])

    def label_emotion(p):
        if p >= 0.4: return "Joy"
        if p <= -0.4: return "Anger/Sad"
        return "Neutral"

    df["emotion"] = df["polarity"].apply(label_emotion)

    col_pie, col_voice = st.columns([1, 2])

    with col_pie:
        emotion_counts = df["emotion"].value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]
        
        fig_donut = px.pie(
            emotion_counts, 
            names="Emotion", 
            values="Count", 
            hole=0.4,
            title="Emotional Distribution",
            color="Emotion",
            color_discrete_map={"Joy": "#2ecc71", "Neutral": "#95a5a6", "Anger/Sad": "#e74c3c"}
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_voice:
        st.markdown("#### 🗣️ Voice of the Customer")
        
        # Positive First as requested
        pos_examples = df[df["polarity"] > 0.2]["Summary"].head(5).tolist()
        neg_examples = df[df["polarity"] < -0.2]["Summary"].head(5).tolist()
        
        tab_pos, tab_neg = st.tabs(["✅ Positive Highlights", "⚠️ Negative Feedback"])
        
        with tab_pos:
            if pos_examples:
                for s in pos_examples: st.success(f"\"{s}\"")
            else:
                st.info("No positive summaries in selection.")
                
        with tab_neg:
            if neg_examples:
                for s in neg_examples: st.error(f"\"{s}\"")
            else:
                st.info("No negative summaries in selection.")

    with st.expander("🧠 Interpretation from data"):
        st.markdown("""
        * **Joy Dominance:** If Green (Joy) > 60%, customer satisfaction is healthy.
        * **Anger Spikes:** If Red (Anger) exceeds 15%, investigate the specific negative keywords immediately.
        """)