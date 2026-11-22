# Stats Method Picker (Claude Code plugin)

A skill that, when asked for a data analysis (hypothesis test, regression, time series,
clustering, classification, causality, etc.), consults a community-extensible catalog of
statistical/ML techniques and returns up to 3 candidates with the reasoning and how to run each. The Python
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

## Updating

New methods arrive as new plugin versions. To receive them:

```
/plugin marketplace update stats-method-picker-marketplace
/plugin update stats-method-picker@stats-method-picker-marketplace
```

(or reinstall the plugin if your Claude Code version has no `update` subcommand). Versions follow
[Semantic Versioning](https://semver.org): a **new method bumps the minor version** in
`.claude-plugin/plugin.json`, a card or wrapper fix bumps the patch, a breaking schema change bumps
the major. See [CHANGELOG.md](CHANGELOG.md).

## Contributing

Methods are added by pull request: one JSON card in `skills/stats-method-picker/data/methods/` plus
one Python wrapper in `skills/stats-method-picker/analyses/<category>/`. No other file changes.
Start with [CONTRIBUTING.md](CONTRIBUTING.md), the [card schema](docs/card-schema.md) and the
[template](templates/new-method/). Before opening a pull request, run:

```
python3 scripts/validate_catalog.py
python3 -m compileall skills
```

## License

[MIT](LICENSE).

## Contact

Questions, bugs or ideas: open an [issue](https://github.com/luksgodoy/stats-method-picker-plugin/issues) or reach out on GitHub ([@luksgodoy](https://github.com/luksgodoy)).
