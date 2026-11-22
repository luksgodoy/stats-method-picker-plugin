# Method card schema

One method = one JSON file in `skills/stats-method-picker/data/methods/<id>.json`, plus one
Python wrapper in `skills/stats-method-picker/analyses/<category>/<module>.py`.
`python3 scripts/validate_catalog.py` enforces everything below.

## Required fields

| Field | Type | Rule |
|---|---|---|
| `id` | string | Lowercase slug (`[a-z0-9]` and `-`), unique, **equal to the file name** without `.json`. |
| `name` | string | Human-readable name. |
| `category` | string | `snake_case`; must equal the folder of the wrapper (`analyses/<category>/`). |
| `when_to_use` | string | Technical hint: data structure, variable types, question answered. |
| `sources` | list of strings | Paths **relative to `skills/stats-method-picker/`**, each an existing `analyses/<category>/<module>.py`. |
| `python_functions` | list of strings | Public functions of the wrapper, e.g. `"poisson_regression()"`; each must be defined in it. |
| `expected_input` | object | `data_structure` and `question` (strings) are required. |
| `plain_explanation` | object | `purpose`, `when_to_use_plain`, `data_needed`, `example` required; `diagram` optional. |

## Optional fields

| Field | Type | Meaning |
|---|---|---|
| `r_functions` | list of strings | R equivalents, if any. Skip it if you only use Python. |
| `code_status` | string | One of the values below. |
| `notes` | string | Caveats, extra dependencies, assumptions. |
| `pdca_use` | string | Where the method fits in a Plan-Do-Check-Act cycle. |

`expected_input` optional keys (all strings): `dependent_variable` (may be `null` for unsupervised
methods), `dependent_variable_type`, `problem_type`, `predictor_type`, `expected_scale`, `note`.
Unknown keys anywhere are rejected, to catch typos.

## `code_status` values

- `runnable`: the code runs as is.
- `own-implementation`: the wrapper is implemented in this repository on top of the listed dependencies.
- `needs-adaptation`: works as a starting point, but needs changes for real data or an extra setup.
- `conceptual`: explanation only, no executable code.

## Wrapper rules

- Module docstring (first statement) describing what it does.
- Parses with `ast.parse` (the validator never imports it, so heavy dependencies are fine).
- Defines every function named in `python_functions` at module top level.
- Includes a minimal `if __name__ == "__main__":` example on simulated data.

## Categories

A category is just a folder under `analyses/`. To create a new one, add the folder
(with an empty `__init__.py`, like the others) and use its name in the card. Prefer an existing
category when one fits. Categories show up automatically in `query.py --categories`.
