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
    app_dir = Path(__file__).parent
    csv_path = app_dir / rel_path
    df = pd.read_csv(csv_path, parse_dates=["Time"])
    return df


data = load_data()


# ================================================================
# SIDEBAR — RATING FILTER
# ================================================================
st.sidebar.header("⭐ Filter Reviews by Rating")

rating_options = [1, 2, 3, 4, 5]

selected_ratings = st.sidebar.multiselect(
    "Select rating scores to include:",
    options=rating_options,
    default=rating_options
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

Use the ⭐ rating filter on the left to drill into negative-only,
positive-only, or mixed review subsets.
"""
)


# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3 = st.tabs([
    "💬 Sentiment Overview",
    "👥 Customer Segments",
    "📊 Product Popularity"
])


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

    with col1:
        st.markdown("### Emotion Distribution")
        emotion_counts = df["emotion"].value_counts()

        fig, ax = plt.subplots(figsize=(4, 4))
        emotion_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    with col2:
        st.markdown("### Example Positive & Negative Summaries")

        st.write("**Negative Summaries:**")
        for s in df[df["polarity"] < -0.2]["Summary"].head(6):
            st.write(f"- {s}")

        st.write("---")
        st.write("**Positive Summaries:**")
        for s in df[df["polarity"] > 0.2]["Summary"].head(6):
            st.write(f"- {s}")

    # EXECUTIVE EXPLANATION
    st.markdown("""
    ---
    ## 🧠 Executive Interpretation

    This section helps us understand **how customers feel** about their purchases.

    **What this chart shows:**  
    - A breakdown of emotions (Joy, Neutral, Anger/Sad) extracted from review summaries.  
    - Real examples of positive and negative customer remarks.

    **What we can derive:**  
    - A high percentage of **Joy** indicates strong customer satisfaction.  
    - A noticeable share of **Anger/Sad** highlights recurring problems or customer frustration.  
    - Example summaries reveal concrete issues (quality, shipping, expectations) or strengths.

    **Business value:**  
    - Product teams can prioritise issues based on negative sentiment.  
    - Support teams can prepare for common complaints.  
    - Marketing can incorporate phrases customers love into campaigns.  
    - Executives get a fast "emotional health check" of the product ecosystem.
    """)
    

# ================================================================
# TAB 2 — CUSTOMER SEGMENTS (CLV)
# ================================================================
with tab2:

    st.subheader("👥 Customer Segmentation (CLV-Style)")
    st.markdown("Customers are segmented based on the number of products they have reviewed.")

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

    # EXECUTIVE EXPLANATION
    st.markdown("""
    ---
    ## 🧠 Executive Interpretation

    This segmentation identifies the **value of different customer groups**.

    **What this chart shows:**  
    - *Power Buyers* review/purchase 100+ products → extremely valuable customers  
    - *Loyal* customers make regular purchases  
    - *Regular* customers buy occasionally  
    - *Occasional* buyers show minimal engagement

    **What we can derive:**  
    - Power Buyers contribute disproportionately to revenue.  
    - Loyal customers are strong cross-sell and upsell candidates.  
    - Occasional customers may need discounts or recommendations to reactivate.

    **Business value:**  
    - Retention teams can protect Power Buyers with perks and benefits.  
    - Marketing can target Loyal customers with bundles or high-margin items.  
    - CRM teams can design campaigns for Regular and Occasional buyers.  
    - Executives can understand customer base health at a glance.
    """)


# ================================================================
# TAB 3 — PRODUCT POPULARITY & RATINGS
# ================================================================
with tab3:

    st.subheader("📊 Product Popularity & Rating Distribution")
    st.markdown("Shows the most-reviewed products and how customers rate them.")

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

    # EXECUTIVE EXPLANATION
    st.markdown("""
    ---
    ## 🧠 Executive Interpretation

    This visualisation shows **which products attract the most customer attention**  
    and how satisfied customers are with them.

    **What this chart shows:**  
    - The most-reviewed (most popular) products  
    - The breakdown of 1–5 star ratings for each  
    - Easy comparison between products

    **What we can derive:**  
    - Products with many reviews AND high ratings are strong performers.  
    - Products with high review volume BUT low ratings may have defects or misleading descriptions.  
    - A spike in low ratings signals quality issues, supplier problems, or incorrect product listings.

    **Business value:**  
    - Marketing can promote top-rated high-volume products.  
    - Product teams can investigate low-rated but high-volume items.  
    - Executives can track product portfolio health and identify revenue drivers.
    """)
