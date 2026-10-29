from app.services.identity_service import identify_user
from app.services.fake_detection_service import detect_fake


def trust_score(username: str) -> dict:
    matches = identify_user(username)
    fake    = detect_fake(username)

    if isinstance(matches, dict) and "error" in matches:
        return {"error": matches["error"]}
    if isinstance(fake, dict) and "error" in fake:
        return {"error": fake["error"]}

    # identify_user puts the query account itself first (100%). Exclude it so
    # the score reflects *other* accounts that corroborate this identity.
    others = [m for m in matches if not (
        m["username"].lower() == str(username).lower().replace("@", "")
        and m["platform"] == fake.get("platform")
    )]
    best       = others[0]["match_percentage"] if others else 0.0
    n_matches  = len(others)
    fake_prob  = fake["fake_probability"]

    # Corroboration: best cross-platform match (0-100) and how many matches exist
    corroboration = min(100.0, best * 0.8 + min(n_matches, 5) * 4)
    # Authenticity: inverse of fake risk
    authenticity  = (1.0 - fake_prob) * 100

    trust = 0.5 * authenticity + 0.5 * corroboration
    if fake["account_type"] == "fake":
        trust *= 0.7

    return {
        "username":         username,
        "trust_score":      round(max(0.0, min(100.0, trust)), 2),
        "account_type":     fake["account_type"],
        "fake_probability": fake_prob,
        "signals":          fake["signals"],
        "match_count":      n_matches,
        "top_match":        others[0] if others else None,
    }