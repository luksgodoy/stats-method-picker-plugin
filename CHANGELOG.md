# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org): a new method bumps the **minor** version, a fix to an
existing card or wrapper bumps the **patch** version, and a breaking change to the card schema or
the query interface bumps the **major** version.

## Unreleased

## 0.1.0

- First release: 70 methods across 8 categories (hypothesis tests, regression, multivariate,
  time series, supervised learning, text, optimization and simulation, causal and special).
- One JSON card per method in `skills/stats-method-picker/data/methods/`, Python wrappers in
  `skills/stats-method-picker/analyses/`.
- `query.py` keyword search and `--categories` listing, driven by the card folder.
- `scripts/validate_catalog.py`, CI workflow, contribution guide, card schema and method template.
