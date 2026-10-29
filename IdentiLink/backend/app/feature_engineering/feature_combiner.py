import numpy as np
from app.feature_engineering.bert_embeddings import get_bert_embedding
from app.feature_engineering.stylometry_features import extract_stylometry

def combine_features(text: str) -> np.ndarray:
    """Concatenate BERT embedding + stylometry features into one vector."""
    bert  = get_bert_embedding(text)
    style = np.array(extract_stylometry(text), dtype=np.float32)
    return np.concatenate([bert, style])
