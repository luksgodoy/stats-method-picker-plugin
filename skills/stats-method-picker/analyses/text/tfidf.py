"""TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_tfidf(documents, max_features=200, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    matrix = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()
    mean_weights = np.asarray(matrix.mean(axis=0)).ravel()
    ranking = sorted(zip(terms, mean_weights), key=lambda pair: -pair[1])[:10]
    return {"n_terms": len(terms), "most_relevant_terms": [(t, float(p)) for t, p in ranking],
            "matrix_shape": matrix.shape}


if __name__ == "__main__":
    corpus = [
        "statistical analysis of monthly sales data",
        "regression model for sales forecasting",
        "sentiment analysis of customer comments",
        "customer clustering by purchasing behavior",
    ]
    r = vectorize_tfidf(corpus)
    print("shape:", r["matrix_shape"], "terms:", r["n_terms"])
    print("most relevant:", r["most_relevant_terms"][:5])
