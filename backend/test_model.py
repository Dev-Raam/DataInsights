import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# 📥 Load processed data
print("📥 Loading processed dataset for testing...")
df = pd.read_csv("data/processed_customer_data.csv")
print("✅ Data loaded successfully!")

# ---- Load Models ----
print("🔄 Loading trained models...")
churn_model = joblib.load("models/churn_rf.pkl")  # Load Churn Prediction Model
segmentation_model = joblib.load("models/customer_segmentation.pkl")  # Load Customer Segmentation Model
recommendation_model = joblib.load("models/product_svd.pkl")  # Load Product Recommendation Model
print("✅ Models loaded successfully!")

# ---- Test Churn Prediction ----
print("\n🔍 Testing Churn Prediction Model...")
sample_customer = df.sample(1)  # Pick a random customer for testing

# Prepare features
churn_features = ['Total Spending', 'Purchase Frequency', 'Return Rate']
X_sample = sample_customer[churn_features]

# Standardize using the same scaler as training
scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample)

# Predict churn probability
churn_prediction = churn_model.predict(X_sample_scaled)
churn_prob = churn_model.predict_proba(X_sample_scaled)[:, 1]  # Get probability of churn

print(f"🎯 Churn Prediction: {'Churn' if churn_prediction[0] == 1 else 'No Churn'} (Probability: {churn_prob[0]:.2f})")

# Analyze churn probability distribution
print("\n📊 Analyzing churn probability distribution...")
print(df['Churn'].value_counts(normalize=True))

# ---- Test Customer Segmentation ----
print("\n🔍 Testing Customer Segmentation Model...")
segmentation_features = ['Recency', 'Frequency', 'Monetary']
X_segment = sample_customer[segmentation_features].fillna(df[segmentation_features].median())
segment = segmentation_model.predict(X_segment)
print(f"🎯 Predicted Customer Segment: {segment[0]}")

# Analyze segment characteristics
print("\n📊 Analyzing segment characteristics...")
# Compute mean only for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns  # Select numeric columns only
print(df.groupby('Segment')[numeric_cols].mean())

# ---- Test Product Recommendation ----
print("\n🔍 Testing Product Recommendation System...")
if pd.isna(sample_customer["Customer ID"].values[0]):
    print("🚨 No valid Customer ID found for recommendation. Skipping...")
else:
    customer_id = int(sample_customer["Customer ID"].values[0])

    # Load user-item matrix
    user_item_matrix = df.pivot_table(index="Customer ID", columns="Product Category", values="Total Purchase Amount", aggfunc="sum").fillna(0)
    user_item_sparse = csr_matrix(user_item_matrix.values)

    # Compute similarity with other customers
    customer_index = user_item_matrix.index.get_loc(customer_id)
    similarities = cosine_similarity(user_item_sparse[customer_index], user_item_sparse).flatten()
    top_similar_customers = np.argsort(similarities)[::-1][1:6]  # Get top 5 similar customers

    # Recommend products from similar customers
    recommended_products = set()
    for similar_customer in top_similar_customers:
        similar_cust_id = user_item_matrix.index[similar_customer]
        top_products = df[df["Customer ID"] == similar_cust_id]["Product Category"].unique()
        recommended_products.update(top_products)

    print(f"🎯 Recommended Products for Customer {customer_id}: {recommended_products}")

# Analyze recommendation effectiveness
print("\n📊 Analyzing recommendation effectiveness...")
print(df.groupby("Product Category")["Total Purchase Amount"].sum().sort_values(ascending=False))

print("\n✅ Model Testing Complete! 🚀")