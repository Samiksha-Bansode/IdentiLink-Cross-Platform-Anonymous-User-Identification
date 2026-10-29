"""
Run ONCE after combine_datasets.py.
Usage (from backend/ folder):  python -m app.dataset_pipeline.preprocess_dataset
"""
import pandas as pd
import os
import re

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_path = os.path.join(BASE_DIR, "dataset", "processed", "combined_dataset.csv")
out_path   = os.path.join(BASE_DIR, "dataset", "processed", "preprocessed_dataset.csv")

df = pd.read_csv(input_path)

df["username"] = df["username"].astype(str).str.lower()
df["bio"]      = df["bio"].astype(str).str.lower()
df["bio"]      = df["bio"].apply(lambda x: re.sub(r"[^a-z0-9 ]", "", x))
df = df.dropna(subset=["username", "bio"])

os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print(f"✅ Preprocessed {len(df)} records → {out_path}")
