"""Basic text mining: tokenization, stopword removal,
term frequency and n-grams."""
import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for", "with", "by", "is",
    "was", "were", "are", "it", "its", "this", "that", "as", "but", "not", "from", "be", "had",
}


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def term_frequency(texts, remove_stopwords=True, top_n=10):
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        if remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        counter.update(tokens)
    return {"most_frequent_terms": counter.most_common(top_n), "vocabulary_size": len(counter)}


def ngrams(text, n=2):
    tokens = tokenize(text)
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


if __name__ == "__main__":
    corpus = [
        "The customer complained about the delivery delay of the product.",
        "The delivery was fast and the product arrived in perfect condition.",
        "Delivery delay caused a customer complaint again.",
    ]
    r = term_frequency(corpus)
    print("vocabulary:", r["vocabulary_size"], "top terms:", r["most_frequent_terms"][:5])
    print("bigrams of the first text:", ngrams(corpus[0])[:5])
