---
name: stats-method-picker
description: Consult the Stats Method Picker (a catalog of 70 statistical/ML techniques with Python wrappers) BEFORE choosing a method. Use when the user asks for statistical analysis, modeling, forecasting, hypothesis testing, correlation, regression, time series, clustering, classification, counting events (count data), causality, or "which technique/method should I use" on data. Do NOT use for tasks with no data analysis (code, docs, infrastructure, reports without statistics).
---

# Stats Method Picker — consult before choosing the method

Catalog: `${CLAUDE_PLUGIN_ROOT}/skills/stats-method-picker/data/catalog.json` (70 entries; fields
`id, name, category, when_to_use, expected_input, python_functions, r_functions, plain_explanation`,
plus optional `notes` and `pdca_use`).
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
   and how to run it (`python_functions`: wrapper in `analyses/`; `r_functions` if there is none).
   If none fits, say "the catalog does not cover it" and proceed on your own.
5. The catalog only suggests candidates; the final choice belongs to whoever requested the analysis.

## Never
- Open CSV/Parquet/XLSX files or third-party sensitive data unless necessary: use aggregates, schema or a sample.
- Invent a function or source that is not in the catalog.
