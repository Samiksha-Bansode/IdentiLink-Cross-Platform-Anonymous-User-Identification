import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz

# ✅ Project utility imports
from app.feature_engineering.bert_embeddings import get_bert_embedding, precompute_dataset_embeddings
from app.feature_engineering.stylometry_features import extract_stylometry
from app.analysis.content_analysis import analyze_content

# ── Load dataset once at startup ──────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataset_path = os.path.join(BASE_DIR, "dataset", "processed", "combined_dataset.csv")
dataset      = pd.read_csv(dataset_path)

TEXT_COL = "bio"

# ── Pre-build TF-IDF matrix once ──────────────────────────────────────────────
_vectorizer  = TfidfVectorizer(stop_words="english")
tfidf_matrix = _vectorizer.fit_transform(dataset[TEXT_COL].astype(str)).toarray()

# ── Pre-warm BERT cache for all dataset bios ──────────────────────────────────
precompute_dataset_embeddings(dataset[TEXT_COL].tolist())

# ── Pre-build full BERT matrix ────────────────────────────────────────────────
print("⏳ Building BERT embedding matrix...")
bert_matrix = np.array([
    get_bert_embedding(str(bio)) for bio in dataset[TEXT_COL]
], dtype=np.float32)
print("✅ BERT matrix ready.")

# ── Pre-build stylometry matrix ───────────────────────────────────────────────
print("⏳ Pre-computing stylometry features...")
stylo_matrix = np.array([
    extract_stylometry(str(bio)) for bio in dataset[TEXT_COL]
], dtype=np.float32)
print("✅ Stylometry ready.")


def identify_user(username: str, min_match: float = 30.0):
    """
    Cross-platform identity matching for IdentiLink with functional UI Slider control.
    """
    results = []
    username_clean = str(username).lower().replace("@", "")

    # ── Find query user (exact match first, then fuzzy fallback) ──────────────
    row = dataset[dataset["username"].str.lower() == username_clean]

    is_fallback = False
    if row.empty:
        is_fallback = True
        scores = [(fuzz.ratio(username_clean, str(u).lower()), u) for u in dataset["username"]]
        best_score, best_match = max(scores, key=lambda x: x[0])
        if best_score < 60:
            return {"error": "username not found in dataset"}
        row = dataset[dataset["username"] == best_match]
        print(f"Using closest username fallback: {best_match} ({best_score}%)")

    # Use first occurrence of the query username
    query_row      = row.iloc[0]
    query_username = query_row["username"]
    query_platform = query_row["platform"]
    query_idx      = dataset.index.get_loc(row.index[0])
    query_username_lower = query_username.lower()
    query_bio      = str(query_row[TEXT_COL])

    # ── DYNAMIC EXTRACTOR: Get explicit root text components ──
    # If it's a fallback search (e.g. searching 'rohan' but finding 'rohit'),
    # use the originally inputted string 'rohan' as the root gate, not the found row!
    search_root = username_clean if is_fallback else query_username_lower
    query_tokens = re.sub(r'[^a-zA-Z0-9]', ' ', search_root).split()
    primary_query_token = query_tokens[0] if query_tokens else search_root

    # Pre-compute query vectors from global matrix arrays
    query_bert  = bert_matrix[query_idx].reshape(1, -1)
    query_tfidf = tfidf_matrix[query_idx].reshape(1, -1)
    query_stylo = stylo_matrix[query_idx]

    # Vectorized similarities across all rows
    bert_scores  = cosine_similarity(query_bert,  bert_matrix)[0]
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
    stylo_diff   = np.abs(stylo_matrix - query_stylo)
    stylo_scores = np.clip(1.0 - stylo_diff.mean(axis=1) / 100, 0, 1)

    usernames = dataset["username"].tolist()

    for i in range(len(dataset)):
        target_username = usernames[i]
        target_platform = dataset.iloc[i]["platform"]

        # ── Handle the Inputted/Fallback Target Row ──
        if i == query_idx:
            results.append({
                "username":         target_username,
                "bio":              str(dataset.iloc[i][TEXT_COL]),
                "platform":         target_platform,
                "match_percentage": 100.0 if not is_fallback else round(fuzz.ratio(username_clean, target_username.lower()), 2),
                "same_username":    not is_fallback,
                "sentiment":        analyze_content(str(dataset.iloc[i][TEXT_COL])),
            })
            continue

        target_lower = target_username.lower()

        # Prevent secondary identical handles from polluting the same platform
        if target_lower == query_username_lower and target_platform == query_platform:
            continue

        # ── Compute username string similarity ────────────────────────────────
        username_score = fuzz.token_sort_ratio(search_root, target_lower) / 100
        exact_match = (target_lower == search_root)

        # ── Strict Identity Core Gate ──────────────────────────────────────────
        contains_primary_name = (primary_query_token in target_lower)

        # Stricter fallback name gate to filter out 'kiran.ai' completely from 'rohan/rohit'
        if not exact_match and not contains_primary_name and username_score < 0.65:
            continue

        # ── Bio similarity scores ──
        bert_score  = float(bert_scores[i])
        tfidf_score = float(tfidf_scores[i])
        stylo_score = float(stylo_scores[i])
        bio_sim     = max(bert_score, tfidf_score)

        # ── Smart Bio Threshold ──
        if not exact_match:
            if contains_primary_name:
                if bio_sim < 0.05:
                    continue
            else:
                if bio_sim < 0.20:
                    continue

        # ── Final Score Compilation ──
        if exact_match:
            final_score = (0.30 * username_score + 0.35 * bert_score + 0.25 * tfidf_score + 0.10 * stylo_score)
        elif contains_primary_name:
            final_score = (0.55 * username_score + 0.20 * bert_score + 0.15 * tfidf_score + 0.10 * stylo_score)
        else:
            final_score = (0.40 * username_score + 0.30 * bert_score + 0.20 * tfidf_score + 0.10 * stylo_score)

        match_percentage = round(final_score * 100, 2)

        # ✅ FIX 2: Enforce the Slider Value Check Dynamically!
        if match_percentage < min_match:
            continue

        results.append({
            "username":         target_username,
            "bio":              str(dataset.iloc[i][TEXT_COL]),
            "platform":         target_platform,
            "match_percentage": match_percentage,
            "same_username":    exact_match,
            "sentiment":        analyze_content(str(dataset.iloc[i][TEXT_COL])),
        })

    # Sort results FIRST so that the highest matching elements survive deduplication
    results.sort(key=lambda x: x["match_percentage"], reverse=True)

    # ── Deduplicate by profile + platform ──
    seen_profiles = set()
    unique_results = []

    for match in results:
        profile_key = (match["username"].lower(), match["platform"].lower())
        
        if profile_key not in seen_profiles:
            seen_profiles.add(profile_key)
            unique_results.append(match)
            
        if len(unique_results) == 10:
            break

    return unique_results