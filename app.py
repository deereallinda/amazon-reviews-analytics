# =============================================================================
# AMAZON REVIEWS ANALYTICS APP
# Author: Linda Mthembu
# =============================================================================

import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from pathlib import Path

# Set style for seaborn
sns.set(style="whitegrid")
st.set_page_config(page_title="Amazon Reviews Analytics", layout="wide")


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
@st.cache_data
def load_data(rel_path: str = "data/amazon_reviews_clean.csv") -> pd.DataFrame:
    try:
        app_dir = Path(__file__).parent
        csv_path = app_dir / rel_path
        
        # Check if file exists to prevent crash in demo
        if not csv_path.exists():
            st.error(f"Data file not found: {csv_path}")
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
# STEP 2: SIDEBAR (RATINGS FILTER)
# =============================================================================
st.sidebar.header("⭐ Filter Reviews")

rating_options = [1, 2, 3, 4, 5]

selected_ratings = st.sidebar.multiselect(
    "Select rating scores to include:",
    options=rating_options,
    default=rating_options
)

# Filter data based on selection
data_filtered = data[data["Score"].isin(selected_ratings)].copy()

st.sidebar.markdown(f"🔎 **{len(data_filtered):,} reviews included**")


# =============================================================================
# MAIN PAGE
# =============================================================================
st.title("📦 Amazon Reviews – Analytics Dashboard")
st.markdown(
    """
    Explore key insights from Amazon product reviews:
    - **Customer Segments**
    - **Product Popularity**
    - **Sentiment & Verbosity**
    """
)


# =============================================================================
# PRE-CALCULATIONS
# =============================================================================
# 1. User Aggregates for CLV
user_agg = (
    data_filtered.groupby("UserId")
    .agg(
        Number_of_summaries=("Summary", "count"),
        num_text=("Text", "count"),
        avg_score=("Score", "mean"),
        No_of_prods_purchased=("ProductId", "count"),
    )
)

def clv_segment(row):
    if row["No_of_prods_purchased"] >= 100: return "Power Buyer"
    if row["No_of_prods_purchased"] >= 30: return "Loyal"
    if row["No_of_prods_purchased"] >= 10: return "Regular"
    return "Occasional"

user_agg["clv_segment"] = user_agg.apply(clv_segment, axis=1)

# 2. Helpfulness Ratio
mask_den = data_filtered["HelpfulnessDenominator"] > 0
data_filtered.loc[mask_den, "helpfulness_ratio"] = (
    data_filtered.loc[mask_den, "HelpfulnessNumerator"]
    / data_filtered.loc[mask_den, "HelpfulnessDenominator"]
)
data_filtered["helpfulness_ratio"] = data_filtered["helpfulness_ratio"].fillna(0.0)


# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👥 Customer Segments",
    "📊 Products & Ratings",
    "👍 Helpfulness",
    "📝 Verbosity",
    "💬 Sentiment"
])


# -----------------------------------------------------------------------------
# TAB 1 – CUSTOMER SEGMENTS
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("👥 Customer Value Segmentation (CLV-Style)")
    
    # CLV Explanation
    st.info(
        "**What does CLV-Style mean?**\n\n"
        "**CLV** stands for **Customer Lifetime Value**. In this context, we segment users based on their "
        "purchase frequency (Volume) to identify who brings the most value to the business.\n"
        "- **Power Buyers:** High volume, high value.\n"
        "- **Occasional:** Low volume, lower immediate value."
    )

    seg_counts = user_agg["clv_segment"].value_counts()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Segment Counts")
        st.dataframe(seg_counts.rename("Users").to_frame())

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        seg_counts.plot(kind="bar", ax=ax, color="#4c72b0")
        ax.set_title("Users per Segment")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🧠 Interpretation from data")
    st.markdown(
        """
        - **Power Buyers** contribute disproportionately to revenue.
        - **Loyal customers** are strong candidates for cross-selling.
        - **Occasional customers** may need discounts to reactivate.
        """
    )


# -----------------------------------------------------------------------------
# TAB 2 – PRODUCT POPULARITY
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("📊 Product Popularity & Rating Distribution")

    top_n = st.slider("Select number of top products", 5, 30, 10)

    prod_counts = data_filtered["ProductId"].value_counts().head(top_n).rename("review_count")

    st.markdown("#### Top Products by Review Count")
    st.dataframe(prod_counts.to_frame())

    top_prod_ids = prod_counts.index
    subset = data_filtered[data_filtered["ProductId"].isin(top_prod_ids)]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(
        data=subset,
        y="ProductId",
        hue="Score",
        order=top_prod_ids,
        ax=ax,
        palette="viridis"
    )
    ax.set_title("Rating Distribution for Top Products")
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Product ID")
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🧠 Interpretation from data")
    st.markdown(
        """
        - Products with high review volume **AND** high ratings are strong performers.
        - A spike in low ratings (orange/purple bars) signals quality issues or misleading descriptions.
        """
    )


# -----------------------------------------------------------------------------
# TAB 3 – HELPFULNESS VS RATING
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("👍 Helpfulness vs Rating Behaviour")

    sample_size = st.slider("Sample size for scatter plot", 1000, 20000, 5000)
    sample = data_filtered.sample(min(sample_size, len(data_filtered)), random_state=42)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=sample,
        x="helpfulness_ratio",
        y="Score",
        alpha=0.3,
        ax=ax
    )
    ax.set_title("Helpfulness Ratio vs Rating")
    ax.set_xlabel("Helpfulness Ratio (0–1)")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🧠 Interpretation from data")
    st.markdown(
        """
        - If the most helpful reviews have balanced scores, the system is healthy.
        - If only 5-star reviews are marked helpful, there may be bias.
        """
    )


# -----------------------------------------------------------------------------
# TAB 4 – REVIEW VERBOSITY
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("📝 Review Verbosity Analysis")

    # Create viewer_type if not exists
    if "viewer_type" not in data_filtered.columns:
         # Rough approximation for display
         user_counts = data_filtered['UserId'].map(data_filtered['UserId'].value_counts())
         data_filtered['viewer_type'] = np.where(user_counts > 5, 'Frequent', 'Occasional')

    df_len = data_filtered[data_filtered["Text_length"] < 500]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df_len,
        x="Score",
        y="Text_length",
        hue="viewer_type",
        ax=ax
    )
    ax.set_title("Review Length by Rating")
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🧠 Interpretation from data")
    st.markdown(
        """
        - **Frequent reviewers** often write longer, more detailed reviews.
        - Long negative reviews are often the most valuable for Product Managers to read.
        """
    )


# -----------------------------------------------------------------------------
# TAB 5 – SENTIMENT OVERVIEW
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("💬 Sentiment Overview of Review Summaries")

    @st.cache_data
    def compute_polarity(series: pd.Series) -> pd.Series:
        def polarity(text):
            try:
                return TextBlob(str(text)).sentiment.polarity
            except:
                return 0.0
        return series.astype(str).apply(polarity)

    data_filtered["polarity"] = compute_polarity(data_filtered["Summary"])

    def emotion_label(p):
        if p >= 0.4: return "Joy"
        if p <= -0.4: return "Anger/Sad"
        return "Neutral"

    data_filtered["emotion"] = data_filtered["polarity"].apply(emotion_label)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Emotion Distribution")
        emotion_counts = data_filtered["emotion"].value_counts()
        
        fig, ax = plt.subplots(figsize=(4, 4))
        emotion_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax, colors=['#66b3ff','#99ff99','#ff9999'])
        ax.set_ylabel("")
        st.pyplot(fig)

    with col2:
        st.markdown("### Voice of the Customer")
        
        # 1. POSITIVE FIRST
        st.write("✅ **Positive Highlights:**")
        pos_examples = data_filtered[data_filtered["polarity"] > 0.2]["Summary"].head(5)
        for s in pos_examples:
            st.write(f"- {s}")

        st.write("---")

        # 2. NEGATIVE SECOND
        st.write("⚠️ **Negative Feedback:**")
        neg_examples = data_filtered[data_filtered["polarity"] < -0.2]["Summary"].head(5)
        for s in neg_examples:
            st.write(f"- {s}")

    st.markdown("---")
    st.markdown("### 🧠 Interpretation from data")
    st.markdown(
        """
        - A high percentage of **Joy** indicates strong customer satisfaction.
        - **Anger/Sad** highlights recurring problems or customer frustration.
        """
    )