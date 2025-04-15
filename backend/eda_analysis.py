import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# Ensure the directory for saving charts exists
chart_dir = "static/charts"
os.makedirs(chart_dir, exist_ok=True)

# Load dataset
file_path = "data/unified_customer_data.csv"
print("📥 Loading dataset...")
df = pd.read_csv(file_path, low_memory=False)
print("✅ Dataset loaded successfully.\n")

# ---- PROGRESS BAR ----
steps = [
    "Data Overview", "Handling Missing Values", "Converting Data Types",
    "Plotting Spending Distribution", "Churn vs. Spending",
    "Customer Age Distribution", "Feature Correlation Heatmap"
]

progress_bar = tqdm(total=len(steps), desc="EDA Progress", ncols=100, position=0, leave=True)

# ---- DATA OVERVIEW ----
progress_bar.set_description("🔹 Data Overview")
print(f"\nTotal Records: {df.shape[0]}, Total Features: {df.shape[1]}")
print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")
print(f"Duplicate Rows: {df.duplicated().sum()}")
progress_bar.update(1)

# ---- HANDLE MISSING VALUES ----
progress_bar.set_description("📌 Handling Missing Values")
df.fillna({
    "Total Spending": 0,
    "Purchase Frequency": 0,
    "Return Rate": 0,
    "Churn": 0,
    "Customer Age": df["Customer Age"].median(),
    "Gender": "Unknown",
}, inplace=True)
progress_bar.update(1)

# ---- CONVERT DATA TYPES ----
progress_bar.set_description("🔄 Converting Data Types")
num_cols = ["Total Spending", "Purchase Frequency", "Return Rate", "Product Price", "Quantity"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, force errors to NaN
df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], errors='coerce')  # Convert date properly
progress_bar.update(1)

# ---- SPENDING DISTRIBUTION ----
progress_bar.set_description("📊 Plotting Spending Distribution")
plt.figure(figsize=(8, 5))
sns.histplot(df["Total Spending"].dropna(), bins=30, kde=True)
plt.title("Distribution of Customer Spending")
plt.xlabel("Total Spending")
plt.ylabel("Count")

spending_chart_path = f"{chart_dir}/spending_distribution.png"
plt.savefig(spending_chart_path)
plt.close()
print(f"📊 Saved Spending Distribution Chart at {spending_chart_path}")
progress_bar.update(1)

# ---- CHURN VS SPENDING ----
progress_bar.set_description("📊 Churn vs. Spending")
plt.figure(figsize=(8, 5))
sns.boxplot(x="Churn", y="Total Spending", data=df)
plt.title("Churn vs. Total Spending")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Total Spending")

churn_chart_path = f"{chart_dir}/churn_vs_spending.png"
plt.savefig(churn_chart_path)
plt.close()
print(f"📊 Saved Churn vs. Spending Chart at {churn_chart_path}")
progress_bar.update(1)

# ---- CUSTOMER AGE DISTRIBUTION ----
progress_bar.set_description("📊 Customer Age Distribution")
plt.figure(figsize=(8, 5))
sns.histplot(df["Customer Age"].dropna(), bins=20, kde=True)
plt.title("Distribution of Customer Ages")
plt.xlabel("Age")
plt.ylabel("Count")

age_chart_path = f"{chart_dir}/customer_age_distribution.png"
plt.savefig(age_chart_path)
plt.close()
print(f"📊 Saved Customer Age Distribution Chart at {age_chart_path}")
progress_bar.update(1)

# ---- CORRELATION HEATMAP ----
progress_bar.set_description("🔎 Feature Correlation Heatmap")
plt.figure(figsize=(10, 6))
corr_matrix = df.select_dtypes(include=['number']).corr()  # Select only numeric columns
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")

heatmap_chart_path = f"{chart_dir}/feature_correlation_heatmap.png"
plt.savefig(heatmap_chart_path)
plt.close()
print(f"📊 Saved Feature Correlation Heatmap at {heatmap_chart_path}")
progress_bar.update(1)

progress_bar.close()
print("✅ EDA Complete! Check the visualizations for insights.")
