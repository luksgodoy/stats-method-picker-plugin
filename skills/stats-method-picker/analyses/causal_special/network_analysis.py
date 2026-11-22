"""Network (graph) analysis: centrality metrics and community detection, via networkx."""
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


def network_metrics(graph):
    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "density": float(nx.density(graph)),
        "degree_centrality": {n: round(c, 3) for n, c in nx.degree_centrality(graph).items()},
        "betweenness_centrality": {n: round(c, 3) for n, c in nx.betweenness_centrality(graph).items()},
    }


def detect_communities(graph):
    communities = greedy_modularity_communities(graph)
    return {"n_communities": len(communities), "communities": [sorted(c) for c in communities]}


if __name__ == "__main__":
    graph = nx.Graph()
    graph.add_edges_from([
        ("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"),
        ("D", "E"), ("D", "F"), ("E", "F"), ("F", "G"),
    ])
    metrics = network_metrics(graph)
    print("nodes:", metrics["n_nodes"], "edges:", metrics["n_edges"], "density:", round(metrics["density"], 3))
    print("most central (degree):", max(metrics["degree_centrality"], key=metrics["degree_centrality"].get))

    communities = detect_communities(graph)
    print("communities detected:", communities["n_communities"], communities["communities"])
