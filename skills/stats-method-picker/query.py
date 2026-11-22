#!/usr/bin/env python3
"""Filter the catalog by keywords and print only the best candidates (max 5).

Usage: python3 query.py term1 term2 ...   (lowercase, 2-4 terms from the problem;
       a quoted phrase such as "count data" is split into words)
       python3 query.py --categories      (list category + name of every entry)
"""

import json
import sys
import unicodedata
from pathlib import Path

CATALOG = Path(__file__).resolve().parent / "data" / "catalog.json"
data = json.load(open(CATALOG, encoding="utf-8"))

if sys.argv[1:] and sys.argv[1] in ("--help", "-h"):
    print(__doc__)
    sys.exit(0)

if sys.argv[1:] == ["--categories"] or len(sys.argv) < 2:
    for e in data:
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
hits.sort(key=lambda e: -score(e))
for e in hits[:5]:
    print(e["id"], "|", e["name"], "|", e["category"], "|", e["when_to_use"], "|",
          e["expected_input"], "|", e["python_functions"])
