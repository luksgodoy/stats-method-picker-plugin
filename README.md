# Stats Method Picker (Claude Code plugin)

A skill that, when asked for a data analysis (hypothesis test, regression, time series,
clustering, classification, causality, etc.), consults a catalog of 70 statistical/ML
techniques and returns up to 3 candidates with the reasoning and how to run each. The Python
wrappers for the techniques live in `skills/stats-method-picker/analyses/`. The skill is
triggered by its description.

## Installation

```
/plugin marketplace add luksgodoy/stats-method-picker-plugin
/plugin install stats-method-picker@stats-method-picker-marketplace
```

## Dependencies

- Catalog lookup: Python 3 only (standard library).
- Running the wrappers: `pip install -r requirements.txt` (numpy, pandas, scipy, statsmodels,
  scikit-learn, arch, networkx, lifelines, umap-learn, prophet, xgboost; on macOS xgboost
  needs `brew install libomp`).

## Usage examples

```
python3 skills/stats-method-picker/query.py regression
python3 skills/stats-method-picker/query.py count poisson
python3 skills/stats-method-picker/query.py time series
python3 skills/stats-method-picker/query.py --categories
```

In Claude Code, just ask, e.g. "which technique should I use to model count data?" and the
skill queries the catalog before answering.

## License

[MIT](LICENSE).

## Contact

Questions, bugs or ideas: open an [issue](https://github.com/luksgodoy/stats-method-picker-plugin/issues) or reach out on GitHub ([@luksgodoy](https://github.com/luksgodoy)).
