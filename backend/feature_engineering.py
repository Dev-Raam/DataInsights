import pandas as pd
import numpy as np

# Load the original cleaned datasets from Step 1 (Preprocessing)
amazon_reviews = pd.read_csv("data/Amazon_Reviews.csv")
ecommerce_data_custom = pd.read_csv("data/ecommerce_customer_data_custom_ratios.csv")
ecommerce_data_large = pd.read_csv("data/ecommerce_customer_data_large.csv")
online_retail = pd.read_csv("data/Online Retail.csv")

# ------------------------
# 1. Feature Engineering
# ------------------------

# Compute Total Spending per Customer
for df in [ecommerce_data_custom, ecommerce_data_large, online_retail]:
    if "Customer ID" in df.columns and "Total Purchase Amount" in df.columns:
        df["Total Spending"] = df.groupby("Customer ID")["Total Purchase Amount"].transform("sum")

# Compute Purchase Frequency (how many times a customer has purchased)
for df in [ecommerce_data_custom, ecommerce_data_large, online_retail]:
    if "Customer ID" in df.columns and "Purchase Date" in df.columns:
        df["Purchase Frequency"] = df.groupby("Customer ID")["Purchase Date"].transform("count")

# Compute Returns Percentage per Customer
for df in [ecommerce_data_custom, ecommerce_data_large]:
    if "Customer ID" in df.columns and "Returns" in df.columns:
        df["Return Rate"] = df.groupby("Customer ID")["Returns"].transform("mean")

# ------------------------
# 2. Merge Datasets
# ------------------------

# Merge E-commerce datasets on "Customer ID"
ecommerce_merged = pd.concat([ecommerce_data_custom, ecommerce_data_large, online_retail], ignore_index=True)

# Since Amazon Reviews does not have "Customer ID", we leave it separate for now
ecommerce_merged["Reviewer Name"] = ecommerce_merged.get("Reviewer Name", np.nan)
amazon_reviews["Customer ID"] = np.nan  # Placeholder for possible matching in the future

# Unified Customer Data (Final Merge)
unified_customer_data = pd.concat([ecommerce_merged, amazon_reviews], ignore_index=True)

# ------------------------
# 3. Save the Final Merged Dataset
# ------------------------
unified_customer_data.to_csv("data/unified_customer_data.csv", index=False)

# Display Final Dataset Info
print("\nUnified Customer Data Info:")
print(unified_customer_data.info())
