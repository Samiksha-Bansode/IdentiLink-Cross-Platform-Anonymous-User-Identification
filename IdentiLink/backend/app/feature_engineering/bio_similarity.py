from rapidfuzz import fuzz

def bio_similarity(a: str, b: str) -> float:
    """Token-set similarity between two bio strings, 0.0–1.0."""
    return fuzz.token_set_ratio(a, b) / 100
