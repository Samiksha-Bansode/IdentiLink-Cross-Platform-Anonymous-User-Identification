import os
import numpy as np
import pandas as pd

# ── Load dataset once at startup ──────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataset_path = os.path.join(BASE_DIR, "dataset", "processed", "combined_dataset.csv")
dataset      = pd.read_csv(dataset_path)

# NOTE: the old SVM was removed. The `label` column in the dataset has no
# relationship to any feature (cross-validated AUC ≈ 0.50), so the SVM learned
# to predict "real" for everyone. This scorer uses behavioural red flags instead.

FAKE_THRESHOLD = 0.58   # risk score at/above this => "fake"


def _pct(series: pd.Series, higher_is_riskier: bool = True) -> pd.Series:
    """Percentile rank (0-1) within a platform. NaN -> neutral 0.5."""
    r = series.rank(pct=True)
    if not higher_is_riskier:
        r = 1.0 - r + (1.0 / max(len(series), 1))
    return r.fillna(0.5).clip(0, 1)


def _build_risk_table() -> pd.DataFrame:
    df = dataset.copy()
    df["_bio"]  = df["bio"].astype(str)
    df["_user"] = df["username"].astype(str)

    # Platform-independent signals
    digits      = df["_user"].str.count(r"\d")
    df["s_digits"] = np.select([digits >= 3, digits == 2, digits == 1], [1.0, 0.7, 0.4], 0.0)
    dup         = df["_bio"].map(df["_bio"].value_counts())
    df["s_dupbio"] = _pct(dup)                         # templated / copy-pasted bio
    df["s_shortbio"] = _pct(df["_bio"].str.len(), higher_is_riskier=False)

    df["s_behaviour"] = 0.5
    for plat, idx in df.groupby("platform").groups.items():
        g = df.loc[idx]
        if g["followers"].notna().any():                # Twitter / Instagram
            follow_ratio = g["following"] / (g["followers"] + 1)    # follows many, few follow back
            post_rate    = g["posts"] / (g["followers"] + 1)        # lots of posts, no audience
            beh = 0.6 * _pct(follow_ratio) + 0.4 * _pct(post_rate)
        else:                                           # Reddit
            karma_per_comment = g["karma"] / (g["comments"] + 1)    # low karma for lots of comments
            beh = _pct(karma_per_comment, higher_is_riskier=False)
        df.loc[idx, "s_behaviour"] = beh

    df["risk"] = (
        0.45 * df["s_behaviour"]
        + 0.25 * df["s_digits"]
        + 0.20 * df["s_dupbio"]
        + 0.10 * df["s_shortbio"]
    ).clip(0, 1)
    return df


_risk = _build_risk_table()


def detect_fake(username: str) -> dict:
    name = str(username).lower().replace("@", "").strip()
    rows = _risk[_risk["_user"].str.lower() == name]
    if rows.empty:
        return {"error": "username not found"}

    r = rows.iloc[0]   # same "first occurrence" rule identify_user uses
    risk = float(r["risk"])
    return {
        "account_type":     "fake" if risk >= FAKE_THRESHOLD else "real",
        "fake_probability": round(risk, 3),
        "platform":         r["platform"],
        "signals": {
            "behaviour":     round(float(r["s_behaviour"]), 2),
            "username_digits": round(float(r["s_digits"]), 2),
            "generic_bio":   round(float(r["s_dupbio"]), 2),
            "short_bio":     round(float(r["s_shortbio"]), 2),
        },
    }