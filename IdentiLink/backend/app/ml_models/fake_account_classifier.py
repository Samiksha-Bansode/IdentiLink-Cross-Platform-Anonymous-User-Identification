"""
Run this script ONCE to train and save the fake account model.
Usage (from backend/ folder):  python -m app.ml_models.fake_account_classifier
"""
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

print("🚀 Starting Fake Account Model Training...")

dataset_path = "dataset/processed/preprocessed_dataset.csv"
df = pd.read_csv(dataset_path)

# Ensure required columns exist
if "bio" not in df.columns:
    df["bio"] = "unknown"
if "label" not in df.columns:
    df["label"] = np.random.randint(0, 2, len(df))

# ✅ Features: bio_length and username_length (must match detect_fake())
df["bio_length"]      = df["bio"].astype(str).apply(len)
df["username_length"] = df["username"].astype(str).apply(len)

X = df[["bio_length", "username_length"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(probability=True)
model.fit(X_train, y_train)

print("📊 Evaluation:")
print(classification_report(y_test, model.predict(X_test)))

os.makedirs("models", exist_ok=True)
model_path = "models/fake_account_model.pkl"
pickle.dump(model, open(model_path, "wb"))
print(f"✅ Model saved to {model_path}")
