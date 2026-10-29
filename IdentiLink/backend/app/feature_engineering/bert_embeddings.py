from transformers import BertTokenizer, BertModel
import torch

# Load once at startup
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Cache to avoid recomputing same bio multiple times
_cache = {}

def get_bert_embedding(text: str):
    """Convert text to BERT embedding vector."""
    text = str(text)
    if text in _cache:
        return _cache[text]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pool the last hidden state → shape (768,)
    embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
    _cache[text] = embedding
    return embedding

def precompute_dataset_embeddings(texts: list):
    """Pre-warm the cache for all dataset bios at startup."""
    print(f"⏳ Pre-computing BERT embeddings for {len(texts)} bios...")
    for text in texts:
        get_bert_embedding(str(text))
    print("✅ BERT cache ready.")
