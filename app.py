# ================================================================
# AMAZON REVIEWS – SENTIMENT, SEGMENTS & PRODUCT ANALYTICS
# Author: Linda Mthembu
# ================================================================

import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from pathlib import Path

sns.set(style="whitegrid")
st.set_page_config(page_title="Amazon Reviews Dashboard", layout="wide")


# ================================================================
# LOAD DATA
# ================================================================
@st.cache_data
def load_data(rel_path: str = "data/amazon_reviews_clean.csv") -> pd.DataFrame:
    """
    Loads the cleaned Amazon reviews CSV relative to the app directory.
    """
    app_dir = Path(__file__).parent
    csv_path = app_dir / rel_path
    df = pd.read_csv(csv_path, parse_dates=["Time"])
    return df


data = load_data()


# ================================================================
# SIDEBAR — RATING FILTER ONLY
# ================================================================
st.sidebar.header("⭐ Filter Reviews by Rating")

rating_options = [1, 2, 3, 4, 5]

selected_ratings = st.sidebar.multiselect(
    "Select rating scores to include:",
    options=rating_options,
    default=rating_options   # show all by default
)

df = data[data["Score"].isin(selected_ratings)].copy()

st.sidebar.markdown(f"🔎 **{len(df):,} reviews included**")


# ================================================================
# PAGE HEADER
# ================================================================
st.title("📦 Amazon Reviews – Insights Dashboard")
st.markdown(
    """
    Explore key insights from Amazon product reviews:

    **1️⃣ Sentiment Overview — customer emotions**  
    **2️⃣ Customer Segments (CLV-Style) — who matters most**  
    **3️⃣ Product Popularity & Ratings — which products win and which fail**  

    Use the **rating filter** on the left to drill deeper into specific review types  
    (e.g., only negative reviews, only 5-star reviews, etc.)
    """
)


# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3 = st.tabs(
    [
        "💬 Sentiment Overview",
        "👥 Customer Segments",
        "📊 Product Popularity",
    ]
)


# ================================================================
# TAB 1 — SENTIMENT OVERVIEW
# ================================================================
with tab1:
    st.subheader("💬 Sentiment Overview of Review Summaries")
    st.markdown("Sentiment is calculated using TextBlob polarity (-1 to +1).")

    @st.cache_data
    def compute_polarity(series: pd.Series) -> pd.Series:
        def polarity(text):
            try:
                return TextBlob(str(text)).sentiment.polarity
            except:
                return 0.0
        return series.astype(str).apply(polarity)

    df["polarity"] = compute_polarity(df["Summary"])

    def emotion_label(p):
        if p >= 0.4: return "Joy"
        if p <= -0.4: return "Anger/Sad"
        return "Neutral"

    df["emotion"] = df["polarity"].apply(emotion_label)

    col1, col2 = st.columns([1, 2])

    # Pie Chart + Emotion Distribution
    with col1:
        st.markdown("### Emotion Distribution")
        emotion_counts = df["emotion"].value_counts()

        fig, ax = plt.subplots(figsize=(4, 4))
        emotion_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    # Positive + Negative Example Lists
    with col2:
        st.markdown("### Example Summaries")
        
        st.write("**Negative Summaries:**")
        for s in df[df["polarity"] < -0.2]["Summary"].head(6):
            st.write(f"- {s}")

        st.write("---")
        st.write("**Positive Summaries:**")
        for s in df[df["polarity"] > 0.2]["Summary"].head(6):
            st.write(f"- {s}")

    st.markdown(
        """
        **💡 Insight:**  
        The sentiment distribution gives a quick snapshot of customer mood.  
        - Negative summaries reveal pain points.  
        - Positive summaries highlight strengths and opportunities.
        """
    )


# ================================================================
# TAB 2 — CUSTOMER SEGMENTS (CLV)
# ================================================================
with tab2:
    st.subheader("👥 Customer Segmentation (CLV-Style)")
    st.markdown("Segment customers based on number of products reviewed (proxy for purchases).")

    user_agg = (
        df.groupby("UserId")
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

    seg_counts = user_agg["clv_segment"].value_counts()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Segment Counts")
        st.dataframe(seg_counts.rename("Users"))

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        seg_counts.plot(kind="bar", ax=ax, color=["#3b8eea", "#60a5fa", "#93c5fd", "#bfdbfe"])
        ax.set_title("Users per Segment")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.markdown(
        """
        **💡 Insight:**  
        - **Power Buyers** generate the highest lifetime value → retain & reward.  
        - **Loyal customers** are strong upsell/cross-sell candidates.  
        - **Regular & Occasional customers** may need promotions or reminders.
        """
    )


# ================================================================
# TAB 3 — PRODUCT POPULARITY & RATING DISTRIBUTION
# ================================================================
with tab3:
    st.subheader("📊 Product Popularity & Rating Distribution")
    st.markdown("Top products ordered by number of reviews.")

    top_n = st.slider("Select number of top products", 5, 30, 10)

    prod_counts = df["ProductId"].value_counts().head(top_n).rename("review_count")

    st.markdown("### Top Products by Review Count")
    st.dataframe(prod_counts.to_frame())

    top_prod_ids = prod_counts.index
    subset = df[df["ProductId"].isin(top_prod_ids)]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(
        data=subset,
        y="ProductId",
        hue="Score",
        order=top_prod_ids,
        ax=ax,
    )
    ax.set_title("Rating Distribution for Top Products")
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Product ID")
    st.pyplot(fig)

    st.markdown(
        """
        **💡 Insight:**  
        - High-volume AND high-rating products = ideal for promotion.  
        - High-volume BUT low-rating products = quality issues or misleading descriptions.  
        - The rating filter (left panel) helps drill into product-level sentiment.
        """
    )
