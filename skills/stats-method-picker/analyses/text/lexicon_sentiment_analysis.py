"""Lexicon-based sentiment analysis (polarity dictionary, no model training -
classic approach for short texts)."""
import re

POSITIVE_LEXICON = {
    "good", "great", "excellent", "fast", "efficient", "recommend", "loved", "perfect",
    "satisfied", "pleasant", "reliable",
}
NEGATIVE_LEXICON = {
    "bad", "terrible", "slow", "delay", "problem", "complaint", "disappointed", "horrible",
    "dissatisfied", "defect", "canceled",
}
NEGATORS = {"not", "never", "no"}


def _tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def lexicon_sentiment(text):
    tokens = _tokenize(text)
    score = 0
    for i, token in enumerate(tokens):
        polarity = 0
        if token in POSITIVE_LEXICON:
            polarity = 1
        elif token in NEGATIVE_LEXICON:
            polarity = -1
        if polarity and i > 0 and tokens[i - 1] in NEGATORS:
            polarity *= -1
        score += polarity

    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"
    return {"score": score, "label": label}


if __name__ == "__main__":
    sentences = [
        "The product is great and the delivery was fast, I highly recommend it!",
        "Terrible service, I had a problem with a delivery delay.",
        "The service is not bad, but it is not excellent either.",
        "Delivery within the agreed deadline.",
    ]
    for sentence in sentences:
        print(lexicon_sentiment(sentence), "-", sentence)
