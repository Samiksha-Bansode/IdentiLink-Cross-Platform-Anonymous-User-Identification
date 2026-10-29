"""
Run ONCE to merge platform CSVs.
Usage (from backend/ folder):  python -m app.dataset_pipeline.combine_datasets
Place raw CSVs in:  backend/dataset/raw/
"""
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR  = os.path.join(BASE_DIR, "dataset", "raw")
OUT_DIR  = os.path.join(BASE_DIR, "dataset", "processed")

platforms = [
    ("twitter_dataset.csv",   "Twitter"),
    ("reddit_dataset.csv",    "Reddit"),
    ("instagram_dataset.csv", "Instagram"),
]

frames = []
for fname, platform in platforms:
    path = os.path.join(RAW_DIR, fname)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "platform" not in df.columns:
            df["platform"] = platform
        frames.append(df)
    else:
        print(f"⚠️  Skipping (not found): {path}")

if not frames:
    raise FileNotFoundError("No CSVs found in dataset/raw/")

combined = pd.concat(frames, ignore_index=True)
os.makedirs(OUT_DIR, exist_ok=True)
output_path = os.path.join(OUT_DIR, "combined_dataset.csv")
combined.to_csv(output_path, index=False)
print(f"✅ Combined {len(combined)} records → {output_path}")
