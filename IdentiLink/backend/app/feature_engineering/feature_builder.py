"""
Run this script ONCE to build final_features.csv from preprocessed_dataset.csv.
Usage (from backend/ folder):  python -m app.feature_engineering.feature_builder
"""
import pandas as pd
import numpy as np
import os
from app.feature_engineering.stylometry_features import extract_stylometry

input_path  = "dataset/processed/preprocessed_dataset.csv"
output_path = "dataset/processed/final_features.csv"

df = pd.read_csv(input_path)

# Apply stylometry row by row (returns list of 5 values per row)
stylo_cols = ["bio_length", "word_count", "avg_word_length", "uppercase_count", "digit_count"]
stylo_data = df["bio"].astype(str).apply(extract_stylometry).tolist()
df_stylo   = pd.DataFrame(stylo_data, columns=stylo_cols)

df_features = pd.concat([df, df_stylo], axis=1)

os.makedirs("dataset/processed", exist_ok=True)
df_features.to_csv(output_path, index=False)

print("✅ Features extracted successfully!")
print("📊 Total records:", len(df_features))
print("💾 Saved to:", output_path)
