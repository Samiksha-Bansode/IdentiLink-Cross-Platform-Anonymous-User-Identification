from rapidfuzz import fuzz

def username_similarity(a: str, b: str) -> float:
    """Character-level similarity between two usernames, 0.0–1.0."""
    return fuzz.ratio(a, b) / 100
