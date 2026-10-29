import numpy as np

def extract_stylometry(text: str) -> list:
    """
    Extract writing-style features from a plain text string.
    Returns a list of 5 numeric features.
    """
    text = str(text)
    words = text.split()

    bio_length      = len(text)
    word_count      = len(words)
    avg_word_length = float(np.mean([len(w) for w in words])) if words else 0.0
    uppercase_count = sum(1 for c in text if c.isupper())
    digit_count     = sum(1 for c in text if c.isdigit())

    return [bio_length, word_count, avg_word_length, uppercase_count, digit_count]
