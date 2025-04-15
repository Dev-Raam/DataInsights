import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure the directory for saving charts exists
chart_dir = "static/charts"
os.makedirs(chart_dir, exist_ok=True)

# Load dataset
file_path = "data/unified_customer_data.csv"
print("📥 Loading dataset...")
df = pd.read_csv(file_path, low_memory=False)  
print("✅ Dataset loaded successfully.")

# Handle missing values
print("🛠 Handling missing values...")
df.fillna({
    "Total Spending": 0,
    "Purchase Frequency": 0,
    "Return Rate": 0,
    "Churn": 0,  
    "Customer Age": df["Customer Age"].median(),
    "Gender": "Unknown",
}, inplace=True)

# Convert date columns to numeric timestamps
date_columns = ["Purchase Date", "Review Date", "InvoiceDate", "Date of Experience"]
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce').astype("int64") // 10**9  

# Drop irrelevant columns
drop_columns = ["Customer ID", "Customer Name", "Review Text", "Review Title", "Profile Link", "Reviewer Name"]
df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

# Encode categorical features
print("🔄 Encoding categorical variables...")
categorical_columns = ["Gender", "Product Category", "Payment Method", "Country"]
for col in categorical_columns:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Define features and target
target = "Churn"
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# Ensure all features are numeric before scaling
numeric_features = X.select_dtypes(include=[np.number]).columns
X = X[numeric_features]

# Standardize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data
print("✂️ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
print("🌲 Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train XGBoost model
print("🚀 Training XGBoost model...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, model_name, filename):
    y_pred = model.predict(X_test)
    print(f"\n🔍 {model_name} Performance:")
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"🎯 Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"📢 Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"⭐ F1 Score: {f1_score(y_test, y_pred):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} - Confusion Matrix")
    
    # Save the confusion matrix
    plt.savefig(f"{chart_dir}/{filename}.png")
    plt.close()
    print(f"📊 Saved {model_name} confusion matrix at {chart_dir}/{filename}.png")

# Run evaluation
evaluate_model(rf_model, "Random Forest", "churn_confusion_matrix_rf")
evaluate_model(xgb_model, "XGBoost", "churn_confusion_matrix_xgb")

# Feature Importance (Random Forest)
feature_importance = rf_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance[sorted_idx][:10], y=np.array(features)[sorted_idx][:10], palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Important Features for Churn Prediction")

# Save the feature importance chart
plt.savefig(f"{chart_dir}/churn_feature_importance.png")
plt.close()
print(f"📊 Saved feature importance chart at {chart_dir}/churn_feature_importance.png")

print("✅ Churn Prediction Complete! Check the model performance and charts.")
