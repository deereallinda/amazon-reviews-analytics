# =============================================================================
# AMAZON REVIEWS ANALYTICS APP
# Author: Linda Mthembu
#
# I built this Streamlit app to showcase key insights from the Amazon reviews
# dataset: customer segments, product popularity, helpfulness vs rating,
# review verbosity, and sentiment highlights.
# =============================================================================

import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path         


from textblob import TextBlob

sns.set(style="whitegrid")
st.set_page_config(page_title="Amazon Reviews Analytics", layout="wide")


# =============================================================================
# STEP 1: LOAD CLEANED DATA
# =============================================================================
@st.cache_data
def load_data(rel_path: str = "data/amazon_reviews_clean.csv") -> pd.DataFrame:
    """
    Load the cleaned CSV relative to this app.py file, so it works both
    locally and on Streamlit Cloud.
    """
    app_dir = Path(__file__).parent          # folder where app.py lives
    csv_path = app_dir / rel_path            # e.g. <repo>/data/amazon_reviews_clean.csv

    df = pd.read_csv(csv_path, parse_dates=["Time"])
    return df


data = load_data()   # use default path



st.title("📦 Amazon Reviews – Analytics Dashboard")
st.markdown(
    """
    This dashboard is built on a cleaned Amazon reviews dataset.

    **What I analyse here:**
    - Customer segments and top buyers
    - Product popularity and rating distribution
    - Helpfulness vs. rating behaviour
    - Review verbosity (how much people write)
    - Quick sentiment overview of review summaries
    """
)


# =============================================================================
# STEP 2: SIDEBAR FILTERS
# =============================================================================
st.sidebar.header("Filters")

# Date range filter
min_date = data["Time"].min()
max_date = data["Time"].max()
date_range = st.sidebar.date_input(
    "Review date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

start_date, end_date = date_range
mask_date = (data["Time"].dt.date >= start_date) & (data["Time"].dt.date <= end_date)
data_filtered = data[mask_date].copy()

st.sidebar.write(f"Filtered rows: {len(data_filtered):,}")


# =============================================================================
# STEP 3: CREATE SOME REUSABLE FEATURES
# =============================================================================

# I recompute year-month so I can use it in multiple charts.
data_filtered["year_month"] = data_filtered["Time"].dt.to_period("M").astype(str)

# I recompute user-level aggregates to drive customer segmentation.
user_agg = (
    data_filtered.groupby("UserId")
    .agg(
        Number_of_summaries=("Summary", "count"),
        num_text=("Text", "count"),
        avg_score=("Score", "mean"),
        No_of_prods_purchased=("ProductId", "count"),
    )
)

# Simple CLV-style segment
def clv_segment(row):
    if row["No_of_prods_purchased"] >= 100:
        return "Power Buyer"
    if row["No_of_prods_purchased"] >= 30:
        return "Loyal"
    if row["No_of_prods_purchased"] >= 10:
        return "Regular"
    return "Occasional"


user_agg["clv_segment"] = user_agg.apply(clv_segment, axis=1)

# Helpfulness ratio
mask_den = data_filtered["HelpfulnessDenominator"] > 0
data_filtered.loc[mask_den, "helpfulness_ratio"] = (
    data_filtered.loc[mask_den, "HelpfulnessNumerator"]
    / data_filtered.loc[mask_den, "HelpfulnessDenominator"]
)
data_filtered["helpfulness_ratio"] = data_filtered["helpfulness_ratio"].fillna(0.0)


# =============================================================================
# STEP 4: TABS FOR DIFFERENT INSIGHTS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "👥 Customer Segments",
        "📊 Products & Ratings",
        "👍 Helpfulness vs Rating",
        "📝 Review Verbosity",
        "💬 Sentiment Overview",
    ]
)


# -----------------------------------------------------------------------------
# TAB 1 – CUSTOMER SEGMENTS
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("👥 Customer Segmentation (CLV-style)")

    seg_counts = user_agg["clv_segment"].value_counts()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Segment counts")
        st.dataframe(seg_counts.rename("users").to_frame())

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        seg_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Segment")
        ax.set_ylabel("Number of users")
        ax.set_title("Users per CLV Segment")
        st.pyplot(fig)

    st.markdown(
        """
        **How I interpret this:**

        - *Power Buyers* and *Loyal* customers are key targets for cross-sell,
          upsell and loyalty rewards.
        - *Occasional* buyers might need stronger nudges such as promotions or
          personalised recommendations.
        """
    )


# -----------------------------------------------------------------------------
# TAB 2 – PRODUCT POPULARITY & RATINGS
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("📊 Product Popularity & Rating Distribution")

    top_n = st.slider("Number of top products to show", 5, 30, 10)

    prod_counts = (
        data_filtered["ProductId"].value_counts().head(top_n).rename("review_count")
    )

    st.markdown("#### Top products by review count")
    st.dataframe(prod_counts.to_frame())

    # Rating distribution for top products
    top_prod_ids = prod_counts.index
    subset = data_filtered[data_filtered["ProductId"].isin(top_prod_ids)]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=subset,
        y="ProductId",
        hue="Score",
        order=top_prod_ids,
        ax=ax,
    )
    ax.set_title("Rating Distribution for Top Products")
    ax.set_xlabel("Number of reviews")
    ax.set_ylabel("ProductId")
    st.pyplot(fig)

    st.markdown(
        """
        **My business view:**

        - Products with both **high volume** and **high average rating** are safe
          candidates to promote.
        - Products with many reviews but **low scores** deserve deeper
          investigation (returns, defects, misleading descriptions, etc.).
        """
    )


# -----------------------------------------------------------------------------
# TAB 3 – HELPFULNESS VS RATING
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("👍 Helpfulness vs Rating Behaviour")

    sample_size = st.slider("Sample size for scatter plot", 5_000, 50_000, 20_000)
    sample = data_filtered.sample(
        min(sample_size, len(data_filtered)), random_state=42
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=sample,
        x="helpfulness_ratio",
        y="Score",
        alpha=0.3,
        ax=ax,
    )
    ax.set_xlabel("Helpfulness Ratio (0–1)")
    ax.set_ylabel("Rating (Score)")
    ax.set_title("Helpfulness Ratio vs Rating")
    st.pyplot(fig)

    bucketed = pd.cut(
        data_filtered["helpfulness_ratio"],
        bins=[-0.01, 0, 0.25, 0.5, 0.75, 1.0],
        labels=["0", "0–0.25", "0.25–0.5", "0.5–0.75", "0.75–1"],
    )

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.barplot(x=bucketed, y=data_filtered["Score"], estimator=np.mean, ci=None, ax=ax2)
    ax2.set_xlabel("Helpfulness Ratio Bucket")
    ax2.set_ylabel("Average Rating")
    ax2.set_title("Average Rating by Helpfulness Bucket")
    st.pyplot(fig2)

    st.markdown(
        """
        **How I explain this:**

        - If the most helpful reviews have balanced scores (not only 5-stars),
          the review system looks healthy.
        - If almost all highly-helpful reviews are 5-star, there might be
          selection bias or incentivised reviewing.
        """
    )


# -----------------------------------------------------------------------------
# TAB 4 – REVIEW VERBOSITY
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("📝 How Verbose Are Reviewers?")

    if "viewer_type" not in data_filtered.columns:
        st.warning(
            "Column `viewer_type` is missing in this CSV. "
            "You can add it in the cleaning notebook (Frequent vs Not Frequent)."
        )
    else:
        # Cap text length to remove huge outliers
        df_len = data_filtered[data_filtered["Text_length"] < 500]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=df_len,
            x="Score",
            y="Text_length",
            hue="viewer_type",
            ax=ax,
        )
        ax.set_title("Review Length by Rating and Viewer Type")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Text length (words)")
        st.pyplot(fig)

        st.markdown(
            """
            **My insight:**

            - Frequent reviewers who write longer reviews often provide the most
              useful feedback.
            - Long negative reviews highlight detailed pain points; they are
              valuable for product and UX teams.
            """
        )


# -----------------------------------------------------------------------------
# TAB 5 – SENTIMENT OVERVIEW (SUMMARIES)
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("💬 Sentiment Overview of Review Summaries")

    st.markdown(
        "Here I use TextBlob to get a quick polarity score for each summary."
    )

    @st.cache_data
    def compute_polarity(series: pd.Series) -> pd.Series:
        # I wrapped this in a function so Streamlit can cache it.
        def get_p(text):
            try:
                return TextBlob(str(text)).sentiment.polarity
            except Exception:
                return 0.0

        return series.astype(str).apply(get_p)

    data_filtered["polarity"] = compute_polarity(data_filtered["Summary"])

    def label_emotion(p):
        if p >= 0.4:
            return "Joy"
        if p <= -0.4:
            return "Anger/Sad"
        return "Neutral"

    data_filtered["emotion"] = data_filtered["polarity"].apply(label_emotion)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Emotion counts")
        st.dataframe(data_filtered["emotion"].value_counts().to_frame("count"))

        fig, ax = plt.subplots(figsize=(4, 4))
        data_filtered["emotion"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    with col2:
        st.markdown("#### Example positive & negative summaries")

        neg_examples = data_filtered[data_filtered["polarity"] < -0.2]["Summary"].head(10)
        pos_examples = data_filtered[data_filtered["polarity"] > 0.2]["Summary"].head(10)

        st.markdown("**Negative examples:**")
        for s in neg_examples:
            st.write(f"• {s}")

        st.markdown("---")
        st.markdown("**Positive examples:**")
        for s in pos_examples:
            st.write(f"• {s}")

    st.markdown(
        """
        **How I would use this:**

        - The emotion mix shows overall customer mood.
        - The example summaries provide concrete, human-readable phrases that
          product and marketing teams can react to.
        """
    )
