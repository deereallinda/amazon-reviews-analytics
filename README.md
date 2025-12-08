# 📦 Amazon Reviews Analytics Dashboard  
### End-to-End NLP, Customer Segmentation, and Product Insights  
**Author:** Linda Mthembu  

---

## 🚀 Project Overview
The **Amazon Reviews Analytics Dashboard** is an interactive Streamlit application that transforms raw Amazon product review data into actionable business insights.  
It combines **NLP sentiment analysis**, **Customer Lifetime Value (CLV-style) segmentation**, and **product rating analytics** in one clean, executive-friendly interface.

🔗 **Live Dashboard:** https://amazon-reviews-analytics-de8wxthfqorydgya6f4js3.streamlit.app/  
🔗 **Dataset Source:** Amazon Fine Food Reviews (Kaggle)  
🔗 **Developer:** [@deereallinda](https://github.com/deereallinda)

---

## 📊 Dataset Information
This project uses a **cleaned 50,000-row sample** extracted from the original Amazon Fine Food Reviews dataset:

| Attribute | Value |
|----------|-------|
| **Original dataset size** | 568,454 reviews |
| **Working sample size** | 50,000 rows |
| **Year range** | 1999–2012 |
| **Fields** | ProductId, UserId, Score, Summary, Text, Helpfulness, Time |

The sample ensures fast performance inside Streamlit while preserving the core behavioral patterns of the full dataset.

---

## 🧠 Key Features of the Dashboard

### 💬 **1. Sentiment Analysis (NLP)**
- Uses TextBlob to calculate polarity scores for review summaries  
- Categorizes emotions into **Joy**, **Neutral**, and **Anger/Sad**  
- Presents top positive and negative customer comments  
- Includes executive interpretations for decision-makers  

### 👥 **2. Customer Segmentation (CLV-Style)**
Segments customers based on number of products reviewed:
- **Power Buyers (VIP)**
- **Loyal**
- **Regular**
- **Occasional**

This helps identify high-value users and guide retention strategies.

### 📊 **3. Product Performance Analytics**
- Identifies top products by review volume  
- Shows 5★ → 1★ rating distributions  
- Helps reveal product quality issues and promotional opportunities  

---

## 🏗️ Tech Stack
| Component | Technology |
|----------|------------|
| **Frontend UI** | Streamlit |
| **Visualizations** | Plotly Express, Plotly Graph Objects |
| **Backend / Processing** | Python, Pandas, NumPy |
| **NLP** | TextBlob |
| **Deployment** | Streamlit Cloud |
| **Version Control** | Git + GitHub |

---

## 📁 Project Structure
