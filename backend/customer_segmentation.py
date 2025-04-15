import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer

# Ensure the directory for saving charts exists
chart_dir = "static/charts"
os.makedirs(chart_dir, exist_ok=True)

# Load the dataset
file_path = "data/unified_customer_data.csv"
print("📥 Loading dataset...")
df = pd.read_csv(file_path)
print("✅ Dataset loaded successfully.")

# Select relevant features for clustering
features = ["Total Spending", "Purchase Frequency", "Return Rate"]
df_selected = df[features].copy()

# Handle missing values
print("🛠 Handling missing values...")
df_selected.fillna(df_selected.median(), inplace=True)

# Normalize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Finding the optimal number of clusters using Elbow Method
print("🔍 Finding optimal number of clusters...")
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,10))
visualizer.fit(df_scaled)

# Save the Elbow Method chart
elbow_chart_path = f"{chart_dir}/customer_elbow_method.png"
visualizer.show(outpath=elbow_chart_path)
print(f"📊 Saved Elbow Method chart at {elbow_chart_path}")

# Apply K-Means clustering with optimal clusters
optimal_clusters = visualizer.elbow_value_
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df["Cluster"] = kmeans.fit_predict(df_scaled)

# Visualize the clusters
print("📈 Visualizing customer clusters...")
plt.figure(figsize=(10,6))
sns.scatterplot(x=df["Total Spending"], y=df["Purchase Frequency"], hue=df["Cluster"], palette="viridis")
plt.title("Customer Segments Based on Spending and Frequency")
plt.xlabel("Total Spending")
plt.ylabel("Purchase Frequency")
plt.legend(title="Cluster")

# Save the customer segmentation chart
segmentation_chart_path = f"{chart_dir}/customer_segmentation.png"
plt.savefig(segmentation_chart_path)
plt.close()
print(f"📊 Saved customer segmentation chart at {segmentation_chart_path}")

print("✅ Customer Segmentation Complete! Check the visualizations for insights.")
