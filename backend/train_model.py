import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# 📥 Load and preprocess data explicitly
print("📥 Loading and preprocessing data...")
file_path = "data/unified_customer_data.csv"  # Ensure this file exists
df = pd.read_csv(file_path)
print("✅ Data loaded successfully!")

# ---- Step 1: Train Churn Prediction Model ----
print("🚀 Training Churn Prediction Model...")
if 'Churn' in df.columns:
    df['Churn'].fillna(0, inplace=True)  
else:
    raise ValueError("🚨 'Churn' column not found in dataset!")

churn_features = ['Total Spending', 'Purchase Frequency', 'Return Rate']
X = df[churn_features]
y = df['Churn']

# Ensure y contains no NaN values
print(f"🔍 Churn column - Unique values: {y.unique()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Save models
joblib.dump(rf_model, "models/churn_rf.pkl")
joblib.dump(xgb_model, "models/churn_xgb.pkl")
print("✅ Churn Prediction Models Trained and Saved!")

# ---- Step 2: Train Customer Segmentation Model ----
print("🚀 Training Customer Segmentation Model...")

# Check if Purchase Date exists
if "Purchase Date" not in df.columns:
    raise ValueError("🚨 'Purchase Date' column is missing, unable to compute Recency and Frequency!")

# Convert Purchase Date to datetime
df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], errors="coerce")

# Compute Recency (days since last purchase)
df["Recency"] = (df["Purchase Date"].max() - df["Purchase Date"]).dt.days

# Compute Frequency (number of purchases per customer)
df["Frequency"] = df.groupby("Customer ID")["Purchase Date"].transform("count")

# Compute Monetary Value (total spending per customer)
df["Monetary"] = df.groupby("Customer ID")["Total Purchase Amount"].transform("sum")

# Ensure all segmentation features exist
segmentation_features = ['Recency', 'Frequency', 'Monetary']
for col in segmentation_features:
    if col not in df.columns:
        raise ValueError(f"🚨 Missing feature: '{col}' not found in dataset!")

# Handle missing values (fill NaN with median)
df[segmentation_features] = df[segmentation_features].fillna(df[segmentation_features].median())

# Prepare data for clustering
X_segmentation = df[segmentation_features]

# Train K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Segment'] = kmeans.fit_predict(X_segmentation)

# Save model
joblib.dump(kmeans, "models/customer_segmentation.pkl")
print("✅ Customer Segmentation Model Trained and Saved!")


# ---- Step 3: Train Product Recommendation Model ----
print("🚀 Training Product Recommendation Model...")
user_item_matrix = df.pivot_table(index="Customer ID", columns="Product Category", values="Total Purchase Amount", aggfunc="sum").fillna(0)
user_item_sparse = csr_matrix(user_item_matrix.values)
svd = TruncatedSVD(n_components=min(4, user_item_matrix.shape[1]))
latent_matrix = svd.fit_transform(user_item_sparse)
joblib.dump(svd, "models/product_svd.pkl")
print("✅ Product Recommendation Model Trained and Saved!")

# Save Processed Data
df.to_csv("data/processed_customer_data.csv", index=False)
print("✅ All models trained and data saved! 🚀")
