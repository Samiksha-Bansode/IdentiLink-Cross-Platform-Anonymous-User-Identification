from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english")

def compute_tfidf(text_list: list):
    """Fit and return TF-IDF matrix as dense numpy array."""
    tfidf_matrix = vectorizer.fit_transform(text_list)
    return tfidf_matrix.toarray()
