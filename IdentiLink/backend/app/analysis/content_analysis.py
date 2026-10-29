from textblob import TextBlob

def analyze_content(text: str) -> str:
    """Return Positive / Negative / Neutral sentiment for a text string."""
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"
