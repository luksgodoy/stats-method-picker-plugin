"""Text classification (e.g. spam/ham, ticket categories)
via TF-IDF + Multinomial Naive Bayes."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


def classify_text(documents, labels, max_features=500):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features)),
        ("clf", MultinomialNB()),
    ])
    cv_accuracy = cross_val_score(pipeline, documents, labels, cv=5).mean()
    pipeline.fit(documents, labels)
    return {"cv_accuracy": float(cv_accuracy), "classes": pipeline.named_steps["clf"].classes_.tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    positives = ["great product fast excellent recommend"] * 40
    negatives = ["terrible delay problem bad disappointed"] * 40
    documents = positives + negatives
    labels = ["positive"] * 40 + ["negative"] * 40
    indices = rng.permutation(len(documents))
    documents = [documents[i] for i in indices]
    labels = [labels[i] for i in indices]
    r = classify_text(documents, labels)
    print("CV accuracy:", r["cv_accuracy"], "classes:", r["classes"])
