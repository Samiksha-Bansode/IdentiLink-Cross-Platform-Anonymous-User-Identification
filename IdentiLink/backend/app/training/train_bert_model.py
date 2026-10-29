"""
Optional: fine-tune BERT on your dataset for better fake detection.
Usage (from backend/ folder):  python -m app.training.train_bert_model
Requires GPU for reasonable speed. CPU training is very slow.
"""
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path  = os.path.join(BASE_DIR, "dataset", "processed", "preprocessed_dataset.csv")

df = pd.read_csv(data_path)
if "label" not in df.columns:
    df["label"] = np.random.randint(0, 2, len(df))

texts  = df["bio"].astype(str).tolist()
labels = df["label"].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN   = 64

class BioDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }

train_loader = DataLoader(BioDataset(train_texts, train_labels), batch_size=16, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(2):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out  = model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        out.loss.backward()
        optimizer.step()
        total_loss += out.loss.item()
    print(f"Epoch {epoch+1}/2 — Loss: {total_loss/len(train_loader):.4f}")

os.makedirs(os.path.join(BASE_DIR, "models", "bert_model"), exist_ok=True)
model.save_pretrained(os.path.join(BASE_DIR, "models", "bert_model"))
tokenizer.save_pretrained(os.path.join(BASE_DIR, "models", "bert_tokenizer"))
print("✅ BERT model saved.")
