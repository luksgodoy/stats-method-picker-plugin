"""Word embeddings via co-occurrence matrix + SVD (lightweight analogue of
GloVe/LSA - avoids depending on downloading pre-trained models)."""
import re
import numpy as np
from collections import Counter


def _tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def train_embeddings(documents, dimensions=10, window=2):
    vocabulary = sorted({token for doc in documents for token in _tokenize(doc)})
    index = {word: i for i, word in enumerate(vocabulary)}
    n = len(vocabulary)
    cooccurrence = np.zeros((n, n))

    for doc in documents:
        tokens = _tokenize(doc)
        for i, target in enumerate(tokens):
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if i != j:
                    cooccurrence[index[target], index[tokens[j]]] += 1

    u, s, _ = np.linalg.svd(cooccurrence)
    effective_dimensions = min(dimensions, n)
    embeddings = u[:, :effective_dimensions] * s[:effective_dimensions]
    return {"vocabulary": vocabulary, "embeddings": embeddings, "index": index}


def similar_words(result, word, top_n=3):
    index = result["index"]
    embeddings = result["embeddings"]
    if word not in index:
        return []
    target_vector = embeddings[index[word]]
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target_vector) + 1e-9
    similarities = (embeddings @ target_vector) / norms
    ranking = sorted(
        ((w, float(similarities[i])) for w, i in index.items() if w != word),
        key=lambda pair: -pair[1],
    )
    return ranking[:top_n]


if __name__ == "__main__":
    documents = [
        "dog barks in the yard and plays with the ball",
        "cat meows in the yard and plays with the yarn",
        "dog and cat play together in the yard of the house",
        "car drives on the street and makes a lot of noise in the traffic",
        "bus drives on the street and carries passengers in the traffic",
    ]
    result = train_embeddings(documents, dimensions=5)
    print("vocabulary:", len(result["vocabulary"]), "words")
    print("similar to 'dog':", similar_words(result, "dog"))
