#!/usr/bin/env python3
"""Validate every method card in skills/stats-method-picker/data/methods/*.json.

Standard library only. Prints one error per problem, prefixed by the card file,
and exits 1 if any error was found. See docs/card-schema.md for the rules.

Usage: python3 scripts/validate_catalog.py
"""

import ast
import json
import re
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parent.parent / "skills" / "stats-method-picker"
METHODS = SKILL / "data" / "methods"
ANALYSES = SKILL / "analyses"

ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
CODE_STATUS = {"runnable", "own-implementation", "needs-adaptation", "conceptual"}
REQUIRED = {"id": str, "name": str, "category": str, "when_to_use": str,
            "sources": list, "python_functions": list,
            "expected_input": dict, "plain_explanation": dict}
OPTIONAL = {"r_functions": list, "code_status": str, "notes": str, "pdca_use": str}
EXPECTED_INPUT_REQUIRED = ("data_structure", "question")
EXPECTED_INPUT_OPTIONAL = ("dependent_variable", "dependent_variable_type", "problem_type",
                           "predictor_type", "expected_scale", "note")
PLAIN = ("purpose", "when_to_use_plain", "data_needed", "example")
PLAIN_OPTIONAL = ("diagram",)


def check_card(path, card, errors, used_sources):
    def err(msg):
        errors.append(f"{path.name}: {msg}")

    if not isinstance(card, dict):
        err("card must be a JSON object")
        return
    for key, typ in REQUIRED.items():
        if key not in card:
            err(f"missing required field '{key}'")
        elif not isinstance(card[key], typ):
            err(f"field '{key}' must be {typ.__name__}")
    for key, typ in OPTIONAL.items():
        if key in card and not isinstance(card[key], typ):
            err(f"field '{key}' must be {typ.__name__}")
    for key in card:
        if key not in REQUIRED and key not in OPTIONAL:
            err(f"unknown field '{key}'")
    for key in ("name", "when_to_use"):
        if isinstance(card.get(key), str) and not card[key].strip():
            err(f"'{key}' is empty")

    cid = card.get("id")
    if isinstance(cid, str):
        if not ID_RE.match(cid):
            err(f"id '{cid}' must be a lowercase slug like 'poisson-regression'")
        if cid != path.stem:
            err(f"id '{cid}' must match the file name '{path.stem}.json'")
    cat = card.get("category")
    if isinstance(cat, str):
        if not CATEGORY_RE.match(cat):
            err(f"category '{cat}' must be lowercase snake_case")
        elif not (ANALYSES / cat).is_dir():
            err(f"category folder analyses/{cat}/ does not exist")

    if "code_status" in card and card["code_status"] not in CODE_STATUS:
        err(f"code_status '{card['code_status']}' not in {sorted(CODE_STATUS)}")

    for key in ("python_functions", "r_functions"):
        v = card.get(key)
        if isinstance(v, list) and (not v and key == "python_functions"
                                    or not all(isinstance(x, str) and x for x in v)):
            err(f"'{key}' must be a list of non-empty strings"
                + (" (at least one)" if key == "python_functions" else ""))

    ei = card.get("expected_input")
    if isinstance(ei, dict):
        for k in EXPECTED_INPUT_REQUIRED:
            if not isinstance(ei.get(k), str) or not ei[k].strip():
                err(f"expected_input.{k} must be a non-empty string")
        for k, v in ei.items():
            if k not in EXPECTED_INPUT_REQUIRED and k not in EXPECTED_INPUT_OPTIONAL:
                err(f"expected_input has unknown key '{k}'")
            elif k in EXPECTED_INPUT_OPTIONAL and not (
                    isinstance(v, str) or (k == "dependent_variable" and v is None)):
                err(f"expected_input.{k} must be a string"
                    + (" or null" if k == "dependent_variable" else ""))
    pe = card.get("plain_explanation")
    if isinstance(pe, dict):
        for k in PLAIN:
            if not isinstance(pe.get(k), str) or not pe[k].strip():
                err(f"plain_explanation.{k} must be a non-empty string")
        for k in pe:
            if k not in PLAIN and k not in PLAIN_OPTIONAL:
                err(f"plain_explanation has unknown key '{k}'")

    sources = card.get("sources")
    if isinstance(sources, list):
        if not sources:
            err("'sources' must list at least one wrapper path")
        py_sources = []
        for s in sources:
            if not isinstance(s, str):
                err("'sources' entries must be strings")
                continue
            target = SKILL / s
            if not target.is_file():
                err(f"source '{s}' does not exist (paths are relative to {SKILL.name}/)")
                continue
            used_sources.add(target.resolve())
            parts = Path(s).parts
            if len(parts) != 3 or parts[0] != "analyses" or not s.endswith(".py"):
                err(f"source '{s}' must look like analyses/<category>/<module>.py")
                continue
            if isinstance(cat, str) and parts[1] != cat:
                err(f"category '{cat}' does not match the folder of '{s}'")
            py_sources.append(target)
        defined = set()
        for target in py_sources:
            defined |= check_wrapper(target, errors, path.name)
        pf = card.get("python_functions")
        if isinstance(pf, list) and py_sources:
            for fn in pf:
                if isinstance(fn, str) and fn.rstrip("()") not in defined:
                    err(f"python function '{fn}' is not defined in its wrapper")


def check_wrapper(target, errors, card_name):
    rel = target.relative_to(SKILL)
    try:
        tree = ast.parse(target.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        errors.append(f"{card_name}: wrapper {rel} has a syntax error: {exc}")
        return set()
    if not ast.get_docstring(tree):
        errors.append(f"{card_name}: wrapper {rel} has no module docstring")
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def main():
    errors, ids, used_sources = [], {}, set()
    files = sorted(METHODS.glob("*.json"))
    if not files:
        errors.append(f"no cards found in {METHODS}")
    for path in files:
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}: invalid JSON: {exc}")
            continue
        check_card(path, card, errors, used_sources)
        cid = card.get("id") if isinstance(card, dict) else None
        if cid in ids:
            errors.append(f"{path.name}: duplicate id '{cid}' (also in {ids[cid]})")
        elif cid is not None:
            ids[cid] = path.name
    for wrapper in sorted(ANALYSES.glob("*/*.py")):
        if wrapper.name != "__init__.py" and wrapper.resolve() not in used_sources:
            errors.append(f"{wrapper.relative_to(SKILL)}: wrapper has no card pointing to it")
    for e in errors:
        print("ERROR", e)
    n = len(files)
    if errors:
        print(f"FAILED: {len(errors)} error(s) across {n} card(s)")
        return 1
    print(f"OK: {n} card(s) valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
