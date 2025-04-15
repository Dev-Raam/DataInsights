import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from textblob import TextBlob
from tqdm import tqdm

# Enable progress bar for pandas operations
tqdm.pandas()

# Ensure the directory for saving charts exists
chart_dir = "static/charts"
os.makedirs(chart_dir, exist_ok=True)

# Load the unified dataset
file_path = "data/unified_customer_data.csv"
print("📥 Loading dataset...")
df = pd.read_csv(file_path)
print("✅ Dataset loaded successfully.")

# Basic Data Summary
print("\n📊 Basic Data Info:")
print(df.info())
print("\n📈 Summary Statistics:")
print(df.describe())

# Handle missing values
print("🛠 Handling missing values...")
df.fillna({
    "Total Spending": 0,
    "Purchase Frequency": 0,
    "Return Rate": 0,
    "Churn": 0,
    "Customer Age": df["Customer Age"].median() if "Customer Age" in df.columns else 30,
    "Gender": "Unknown",
}, inplace=True)

# Customer Segmentation
print("📌 Segmenting customers...")
bins = [0, 100, 500, 1000, 5000, df['Total Spending'].max()]
labels = ['Low', 'Medium', 'High', 'Premium', 'VIP']
df['Spending Category'] = pd.cut(df['Total Spending'], bins=bins, labels=labels)
print("\n✅ Customer Segmentation Complete:")
print(df['Spending Category'].value_counts())

# Sales Trends Over Time
print("📈 Processing sales trends...")
df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], errors='coerce')
df_monthly_sales = df.groupby(df["Purchase Date"].dt.to_period("M")).agg({"Total Purchase Amount": "sum"})

# Plot and Save Sales Trends Chart
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_monthly_sales, x=df_monthly_sales.index.astype(str), y="Total Purchase Amount")
plt.xticks(rotation=45)
plt.title("Monthly Sales Trends")
plt.xlabel("Month")
plt.ylabel("Total Sales")

sales_chart_path = f"{chart_dir}/monthly_sales_trends.png"
plt.savefig(sales_chart_path)
plt.close()
print(f"📊 Saved Sales Trends Chart at {sales_chart_path}")

# Review Sentiment Analysis
if "Review Text" in df.columns and df["Review Text"].notna().sum() > 0:
    print("📝 Performing sentiment analysis on reviews...")

    def get_sentiment(text):
        return TextBlob(str(text)).sentiment.polarity

    # Apply sentiment analysis with a progress bar
    df["Sentiment Score"] = df["Review Text"].progress_apply(get_sentiment)

    print("\n📊 Average Review Sentiment Score:", df["Sentiment Score"].mean())

    # Sentiment Distribution Plot
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Sentiment Score"], bins=30, kde=True)
    plt.title("Review Sentiment Distribution")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Count")

    sentiment_chart_path = f"{chart_dir}/sentiment_distribution.png"
    plt.savefig(sentiment_chart_path)
    plt.close()
    print(f"📊 Saved Sentiment Distribution Chart at {sentiment_chart_path}")

print("✅ Data Analysis Complete! Check the visualizations for insights.")
