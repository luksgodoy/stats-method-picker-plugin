---
name: stats-method-picker
description: Consult the Stats Method Picker (a catalog of statistical/ML techniques with Python wrappers) BEFORE choosing a method. Use when the user asks for statistical analysis, modeling, forecasting, hypothesis testing, correlation, regression, time series, clustering, classification, counting events (count data), causality, or "which technique/method should I use" on data. Do NOT use for tasks with no data analysis (code, docs, infrastructure, reports without statistics).
---

# Stats Method Picker — consult before choosing the method

Catalog: one JSON card per method in `${CLAUDE_PLUGIN_ROOT}/skills/stats-method-picker/data/methods/`
(fields `id, name, category, when_to_use, expected_input, python_functions, plain_explanation`,
plus optional `r_functions`, `code_status`, `notes` and `pdca_use`; see `docs/card-schema.md`).
The number of methods and the categories change as contributors add cards: never assume them,
`query.py --categories` prints the current count and category list.
Python wrappers: `${CLAUDE_PLUGIN_ROOT}/skills/stats-method-picker/analyses/<category>/<module>.py`.

## Steps

1. Understand the data (structure, variable types), the context and the goal. Ask if anything is missing.
2. **Never dump the catalog.** Filter by keyword and bring only the fields of the candidates
   (max ~5). Replace the terms with 2-4 lowercase words from the problem:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/skills/stats-method-picker/query.py" count poisson
```

   No results: try synonyms, or list only category + name with `query.py --categories`.
3. Judge by meaning (data structure before name) and discard the incompatible ones.
4. Answer **before choosing the method**: a short list (up to 3) with the technique, why it fits,
   and how to run it (`python_functions`: wrapper in `analyses/`; `r_functions` when present).
   If none fits, say "the catalog does not cover it" and proceed on your own.
5. The catalog only suggests candidates; the final choice belongs to whoever requested the analysis.

## Never
- Open CSV/Parquet/XLSX files or third-party sensitive data unless necessary: use aggregates, schema or a sample.
- Invent a function or source that is not in the catalog.
