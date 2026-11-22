#!/usr/bin/env python3
"""Filter the method cards by keywords and print only the best candidates (max 5).

Cards live one per file in data/methods/*.json; adding a card adds a method, no code change.

Usage: python3 query.py term1 term2 ...   (lowercase, 2-4 terms from the problem;
       a quoted phrase such as "count data" is split into words)
       python3 query.py --categories      (summary line + category | name of every card)
"""

import json
import sys
import unicodedata
from pathlib import Path

METHODS = Path(__file__).resolve().parent / "data" / "methods"

if sys.argv[1:] and sys.argv[1] in ("--help", "-h"):
    print(__doc__)
    sys.exit(0)

data = [json.load(open(p, encoding="utf-8")) for p in sorted(METHODS.glob("*.json"))]

if sys.argv[1:] == ["--categories"] or len(sys.argv) < 2:
    cats = sorted({e["category"] for e in data})
    print(f"{len(data)} methods in {len(cats)} categories: {', '.join(cats)}")
    for e in sorted(data, key=lambda e: (e["category"], e["name"])):
        print(e["category"], "|", e["name"])
    sys.exit(0)


def norm(x):
    """Lowercase and strip accents, so accented and plain spellings match."""
    return "".join(c for c in unicodedata.normalize("NFKD", x.lower())
                   if not unicodedata.combining(c))


# Split every argument into words, so "count data" and count data behave the same.
KW = [w for arg in sys.argv[1:] for w in norm(arg).split()]


def text(e):
    s = json.dumps([e["name"], e["when_to_use"], e["expected_input"],
                    e.get("plain_explanation", {}).get("purpose", "")],
                   ensure_ascii=False)
    return norm(s)


def score(e):
    t = text(e)
    return sum(k in t for k in KW)


hits = [e for e in data if score(e)]
hits.sort(key=lambda e: (-score(e), e["id"]))
for e in hits[:5]:
    print(e["id"], "|", e["name"], "|", e["category"], "|", e["when_to_use"], "|",
          e["expected_input"], "|", e["python_functions"])
