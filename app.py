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
# SIDEBAR
# ================================================================
st.sidebar.header("📅 Filter Reviews by Date")

min_date = data["Time"].min().date()
max_date = data["Time"].max().date()

date_range = st.sidebar.date_input(
    "Select review date range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

start_date, end_date = date_range
mask = (data["Time"].dt.date >= start_date) & (data["Time"].dt.date <= end_date)
df = data[mask].copy()

st.sidebar.markdown(f"🔎 {len(df):,} reviews selected")


# ================================================================
# PAGE TITLE
# ================================================================
st.title("📦 Amazon Reviews – Insights Dashboard")
st.markdown(
    """
    This dashboard highlights three key insights derived from Amazon product review data:

    **1️⃣ Sentiment Overview — What emotions dominate customer feedback?**  
    **2️⃣ Customer Segments (CLV-Style) — Who are our most valuable customers?**  
    **3️⃣ Product Popularity & Rating Distribution — Which products perform best?**  

    Use the date filter on the left to explore trends for specific time periods.
    """
)


# ================================================================
# TAB SETUP
# ================================================================
tab1, tab2, tab3 = st.tabs(
    [
        "💬 Sentiment Overview",
        "👥 Customer Segments",
        "📊 Product Popularity",
    ]
)


# ================================================================
# TAB 1 – SENTIMENT OVERVIEW
# ================================================================
with tab1:
    st.subheader("💬 Sentiment Overview of Review Summaries")
    st.markdown("Sentiment is computed using TextBlob polarity scores (range: -1 to +1).")

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
        The sentiment mix reveals customer mood at a glance.  
        Negative summaries highlight pain points, while positive ones show what customers love.
        """
    )


# ================================================================
# TAB 2 – CUSTOMER SEGMENTS (CLV)
# ================================================================
with tab2:
    st.subheader("👥 Customer Segmentation (CLV-Style)")
    st.markdown("Segment customers by number of products reviewed (proxy for purchases).")

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
        st.write("### Segment Counts")
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
        - **Power Buyers** are high-value — target with loyalty perks & early access.  
        - **Loyal customers** respond well to bundle deals and cross-sell strategies.  
        - **Occasional users** may need promotional nudges.
        """
    )


# ================================================================
# TAB 3 – PRODUCT POPULARITY & RATINGS
# ================================================================
with tab3:
    st.subheader("📊 Product Popularity & Rating Distribution")

    st.write("Top products are ordered by number of reviews.")

    top_n = st.slider("Select number of top products", 5, 30, 10)

    prod_counts = df["ProductId"].value_counts().head(top_n).rename("review_count")
    st.write("### Top Products by Review Count")
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
        - High-review + high-rating products are strong promotion candidates.  
        - High-review + low-rating products may have defects or misleading descriptions.  
        - This helps product teams prioritise quality checks and marketing focus.
        """
    )
