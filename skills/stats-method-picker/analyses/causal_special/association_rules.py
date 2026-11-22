"""Association rules (Apriori) for market basket analysis:
finds items frequently bought together and derives rules with support/confidence/lift."""
from itertools import combinations
from collections import Counter


def _support(itemset, transactions):
    itemset = frozenset(itemset)
    return sum(1 for t in transactions if itemset.issubset(t)) / len(transactions)


def frequent_itemsets(transactions, min_support=0.2):
    transactions = [frozenset(t) for t in transactions]
    items = {item for t in transactions for item in t}
    current_candidates = [frozenset([item]) for item in items]
    frequent = {}

    size = 1
    while current_candidates:
        supports = {c: _support(c, transactions) for c in current_candidates}
        survivors = {c: s for c, s in supports.items() if s >= min_support}
        if not survivors:
            break
        frequent.update(survivors)

        size += 1
        surviving_items = {item for itemset in survivors for item in itemset}
        current_candidates = [frozenset(c) for c in combinations(sorted(surviving_items), size)]

    return frequent


def generate_rules(transactions, min_support=0.2, min_confidence=0.5):
    transactions_fs = [frozenset(t) for t in transactions]
    frequent = frequent_itemsets(transactions, min_support)
    rules = []
    for itemset, itemset_support in frequent.items():
        if len(itemset) < 2:
            continue
        for antecedent_size in range(1, len(itemset)):
            for antecedent in combinations(itemset, antecedent_size):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                antecedent_support = _support(antecedent, transactions_fs)
                if antecedent_support == 0:
                    continue
                confidence = itemset_support / antecedent_support
                if confidence >= min_confidence:
                    consequent_support = _support(consequent, transactions_fs)
                    lift = confidence / consequent_support if consequent_support > 0 else float("inf")
                    rules.append({"antecedent": set(antecedent), "consequent": set(consequent),
                                  "support": round(itemset_support, 3), "confidence": round(confidence, 3),
                                  "lift": round(lift, 3)})
    return sorted(rules, key=lambda r: -r["lift"])


if __name__ == "__main__":
    transactions = [
        {"bread", "milk", "butter"},
        {"bread", "milk"},
        {"bread", "butter"},
        {"milk", "butter", "coffee"},
        {"bread", "milk", "butter", "coffee"},
        {"coffee", "sugar"},
        {"bread", "milk", "butter"},
    ]
    rules = generate_rules(transactions, min_support=0.3, min_confidence=0.6)
    for rule in rules[:5]:
        print(f"{rule['antecedent']} -> {rule['consequent']} | "
              f"support={rule['support']} confidence={rule['confidence']} lift={rule['lift']}")
