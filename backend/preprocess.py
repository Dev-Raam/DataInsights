import pandas as pd
import numpy as np

# Load CSV files 
amazon_reviews = pd.read_csv("data/Amazon_Reviews.csv")
ecommerce_data_custom = pd.read_csv("data/ecommerce_customer_data_custom_ratios.csv")
ecommerce_data_large = pd.read_csv("data/ecommerce_customer_data_large.csv")
online_retail = pd.read_csv("data/Online Retail.csv")
online_retail_sale = pd.read_csv("data/online_retail_sale.csv")


# Function to clean mixed-type columns
def clean_mixed_column(df, column_name):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce', downcast="integer")


# Handle mixed-type columns (Example: Assuming 'Customer ID' has mixed types)
for dataset in [ecommerce_data_custom, ecommerce_data_large, online_retail]:
    if "Customer ID" in dataset.columns:
        dataset = clean_mixed_column(dataset, "Customer ID")

# Handle missing values (Fill with mode for categorical, mean for numerical)
for df in [amazon_reviews, ecommerce_data_custom, ecommerce_data_large, online_retail, online_retail_sale]:
    for col in df.columns:
        if df[col].dtype == "object":  # Categorical
            df[col] = df[col].fillna(df[col].mode()[0])
        else:  # Numerical
            df[col] = df[col].fillna(df[col].mean())


# Drop duplicates
amazon_reviews.drop_duplicates(inplace=True)
ecommerce_data_custom.drop_duplicates(inplace=True)
ecommerce_data_large.drop_duplicates(inplace=True)
online_retail.drop_duplicates(inplace=True)
online_retail_sale.drop_duplicates(inplace=True)

# Display dataset info after cleaning
datasets = {
    "Amazon Reviews": amazon_reviews,
    "Ecommerce Data Custom": ecommerce_data_custom,
    "Ecommerce Data Large": ecommerce_data_large,
    "Online Retail": online_retail,
    "Online Retail Sale": online_retail_sale
}

for name, df in datasets.items():
    print(f"\n{name} Dataset Info:")
    print(df.info())
    print("-" * 50)
