import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors  # Efficient nearest neighbor search
from fuzzywuzzy import process  # Added for fuzzy matching

# Load dataset
file_path = "data/unified_customer_data.csv"
print("📥 Loading dataset...")
df = pd.read_csv(file_path, low_memory=False)
print("✅ Dataset loaded successfully.")

# Initialize progress bar
progress = tqdm(total=6, desc="Processing", bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} ✅")

# Handling missing values with explicit type conversion
print("🛠 Handling missing values...")
df.fillna("", inplace=True)
df = df.infer_objects()  # Convert object types properly
progress.update(1)

# Collaborative Filtering (User-Item Matrix)
print("📊 Building user-item interaction matrix...")
user_item_matrix = df.pivot_table(index="Customer ID", columns="Product Category", values="Total Purchase Amount", aggfunc="sum").fillna(0)

# Convert to numeric to avoid dtype issues
user_item_matrix = user_item_matrix.apply(pd.to_numeric, errors="coerce").fillna(0)
user_item_sparse = csr_matrix(user_item_matrix)
progress.update(1)

# Determine SVD components dynamically
n_features = user_item_matrix.shape[1]
n_components = min(50, max(1, n_features - 1))  # Ensure at least 1 component

# Apply SVD for dimensionality reduction
print(f"🔄 Applying SVD with {n_components} components...")
svd = TruncatedSVD(n_components=n_components, random_state=42)
latent_matrix = svd.fit_transform(user_item_sparse)
progress.update(1)

# Use Nearest Neighbors for similarity search (instead of full cosine similarity matrix)
print("🔎 Finding similar customers using Nearest Neighbors...")
nn_model = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")  # Finds top 5 similar customers
nn_model.fit(latent_matrix)
progress.update(1)

# Function to recommend products
def recommend_products(customer_id, top_n=5):
    if customer_id not in user_item_matrix.index:
        print(f"❌ Customer ID {customer_id} not found!")
        return []
    
    customer_idx = user_item_matrix.index.get_loc(customer_id)
    
    # Get nearest customers (excluding the customer themselves)
    distances, indices = nn_model.kneighbors([latent_matrix[customer_idx]], n_neighbors=top_n + 1)
    similar_customers = indices[0][1:]  # Ignore the first one (it’s the same customer)

    recommended_categories = set()
    for similar_customer in similar_customers:
        similar_cust_id = user_item_matrix.index[similar_customer]
        top_products = df[df["Customer ID"] == similar_cust_id]["Product Category"].unique()
        recommended_categories.update(top_products)

    print(f"🎯 Recommended Categories for Customer {customer_id}: {recommended_categories}")
    return recommended_categories

# Content-Based Filtering (TF-IDF on product descriptions)
print("📖 Applying content-based filtering...")
if "Description" in df.columns and df["Description"].nunique() > 1:  # Avoids empty or single-value errors
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)  # 🔹 Limit features to save memory
    tfidf_matrix = vectorizer.fit_transform(df["Description"])

    # 🔹 Reduce dimensionality of TF-IDF matrix to save memory
    svd_tfidf = TruncatedSVD(n_components=100)  # 🔹 Reduce TF-IDF dimensions to 100
    reduced_tfidf_matrix = svd_tfidf.fit_transform(tfidf_matrix)

    # 🔹 Use Nearest Neighbors instead of full cosine similarity
    nn_model_products = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
    nn_model_products.fit(reduced_tfidf_matrix)

    def recommend_similar_products(product_name, top_n=5):
        # Use fuzzy matching to find the closest product if an exact match isn't found
        if product_name not in df["Description"].values:
            closest_match = process.extractOne(product_name, df["Description"].dropna().unique())
            if closest_match and closest_match[1] > 80:  # Only accept close matches with a confidence > 80%
                print(f"⚠️ Product '{product_name}' not found. Did you mean: '{closest_match[0]}'?")
                product_name = closest_match[0]
            else:
                print(f"❌ Product '{product_name}' not found!")
                return []

        product_idx = df[df["Description"] == product_name].index[0]
        distances, indices = nn_model_products.kneighbors([reduced_tfidf_matrix[product_idx]], n_neighbors=top_n + 1)
        similar_products = indices[0][1:]  # Ignore the first one (same product)

        recommended_products = df.iloc[similar_products]["Description"].values
        print(f"🔍 Recommended Products similar to '{product_name}': {recommended_products}")
        return recommended_products
else:
    print("⚠️ Not enough unique product descriptions for content-based filtering!")

progress.update(1)

# Run sample recommendation
print("\n🔹 Sample Recommendations:")
recommend_products(customer_id=12345)

if "Description" in df.columns and df["Description"].nunique() > 1:
    recommend_similar_products(product_name="Wireless Bluetooth Speaker")

progress.update(1)
progress.close()

print("✅ Product Recommendation Complete! Check recommendations above.")
