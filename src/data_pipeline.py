import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Define the dataset path
csv_file_path = '/content/drive/MyDrive/scotch_review.csv'

# Step 2: Load and Clean the Dataset
try:
    df = pd.read_csv(csv_file_path)
    print(f"\nSuccessfully loaded {csv_file_path}. Found {len(df)} total entries.")

    # Data Cleaning: Drop rows where 'description' or 'price' is missing.
    df['price']=pd.to_numeric(df['price'],errors='coerce')
    df_cleaned = df.dropna(subset=['description', 'price'])
    if 'review.point' in df_cleaned.columns:
        df_cleaned = df_cleaned.rename(columns={'review.point': 'rating'})

    print(f"After cleaning, {len(df_cleaned)} entries remain.")

    # Step 3: Prepare the Text for Embedding
    df_cleaned['embedding_text'] = (
        "Whisky Name: " + df_cleaned['name'].astype(str) + ". " +
        "Category: " + df_cleaned['category'].astype(str) + ". " +
        "Review: " + df_cleaned['description'].astype(str)
    )

    # Store the cleaned DataFrame globally for subsequent cells
    globals()['df_cleaned'] = df_cleaned
    print("Cleaned DataFrame (df_cleaned) is ready for embedding generation.")

except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found. Please ensure it's in your Google Drive at the specified path.")
except Exception as e:
    print(f"An error occurred: {e}")