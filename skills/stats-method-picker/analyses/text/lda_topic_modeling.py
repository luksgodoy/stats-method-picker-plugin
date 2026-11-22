"""Topic modeling via LDA (Latent Dirichlet Allocation)."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def model_topics(documents, n_topics=3, n_words_per_topic=5):
    vectorizer = CountVectorizer(max_features=300)
    matrix = vectorizer.fit_transform(documents)
    terms = np.array(vectorizer.get_feature_names_out())

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0, max_iter=50)
    lda.fit(matrix)

    topics = []
    for component in lda.components_:
        top_indices = component.argsort()[::-1][:n_words_per_topic]
        topics.append(terms[top_indices].tolist())

    doc0_distribution = lda.transform(matrix[0]).ravel().tolist()
    return {"topics": topics, "first_document_distribution": doc0_distribution}


if __name__ == "__main__":
    documents = [
        "statistical analysis regression model data",
        "linear regression model statistical forecast",
        "soccer team game championship goal",
        "championship game team fans goal victory",
        "cooking recipe ingredients oven seasoning",
        "seasoning recipe dish ingredients kitchen",
    ]
    r = model_topics(documents, n_topics=3)
    for i, topic in enumerate(r["topics"]):
        print(f"topic {i}:", topic)
