# Contributing a method

Anyone can add a statistical or ML method through a pull request. Adding a method means adding
**one card and one wrapper**; no other file needs to change (`query.py` and `SKILL.md` discover
cards on their own).

## Add a method in 5 steps

1. **Copy the template**: `templates/new-method/card.json` to
   `skills/stats-method-picker/data/methods/<id>.json` and `templates/new-method/wrapper.py` to
   `skills/stats-method-picker/analyses/<category>/<module>.py`. The `<id>` is a lowercase slug
   such as `quantile-regression`.
2. **Fill in the card** following [docs/card-schema.md](docs/card-schema.md). Set `category` to an
   existing folder under `analyses/`, or create a new folder (with an empty `__init__.py`).
3. **Write the wrapper**: module docstring, the functions named in `python_functions`, and a minimal
   runnable example under `if __name__ == "__main__":`.
4. **Validate**: `python3 scripts/validate_catalog.py` and `python3 -m compileall skills`. Then check
   that the method is found: `python3 skills/stats-method-picker/query.py <a keyword of yours>`.
5. **Open a pull request** and fill in the template. Add a line to `CHANGELOG.md` under
   `Unreleased`. The maintainer bumps the version when releasing.

## Quality rules

- **Source and plain explanation**: the card must explain the method in plain language
  (`plain_explanation`) and the wrapper must rely on a recognized implementation or reference
  (document it in `notes` when it is not obvious).
- **Wrapper**: docstring plus a minimal example that runs on simulated data.
- **Dependencies**: a new third-party package goes in `requirements.txt` (with a minimum version).
  Prefer packages already listed.
- **No personal or proprietary data**: examples use simulated or public data only. No local paths,
  e-mail addresses, credentials, client data or copyrighted material.
- **One method per pull request**, unless the methods are closely related.
- Everything in English.

## Reporting a missing method

Open an issue with the "New method" template if you want a method but cannot implement it.
