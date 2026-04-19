#!/usr/bin/env python3
"""Validate DNDSR config files against their JSON Schemas.

Usage:
    python cases/validate_configs.py [cases_dir]

Reads every *.json config file under *cases_dir* (default: the directory
containing this script), resolves its ``$schema`` reference to a schema
file, and validates it with jsonschema (draft-07).

Exit codes:
    0  all configs valid (or only warnings)
    1  one or more configs failed validation
    2  usage / setup error

Requirements:
    pip install jsonschema
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from jsonschema import Draft7Validator
except ImportError:
    print("ERROR: jsonschema is not installed.  Run:  pip install jsonschema", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# JSON with // comments
# ---------------------------------------------------------------------------

def strip_comments(text: str) -> str:
    """Remove C-style ``//`` line comments outside of JSON strings."""
    result: list[str] = []
    i = 0
    in_string = False
    while i < len(text):
        c = text[i]
        if in_string:
            result.append(c)
            if c == "\\" and i + 1 < len(text):
                i += 1
                result.append(text[i])
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
                result.append(c)
            elif c == "/" and i + 1 < len(text) and text[i + 1] == "/":
                while i < len(text) and text[i] != "\n":
                    i += 1
                continue
            else:
                result.append(c)
        i += 1
    return "".join(result)


def load_json_with_comments(path: str | Path) -> dict:
    with open(path) as f:
        return json.loads(strip_comments(f.read()))


# ---------------------------------------------------------------------------
# Schema resolution
# ---------------------------------------------------------------------------

SCHEMA_PREFIX_ORDER = [
    "euler2EQ3D",
    "euler2EQ",
    "eulerSA3D",
    "eulerSA",
    "euler3D",
    "eulerEX",
    "euler",
]


def resolve_schema(config_path: Path, cases_dir: Path) -> Path | None:
    """Return the schema path for a config, or None if not resolvable.

    Prefers the ``$schema`` key inside the file.  Falls back to
    heuristic matching by directory / filename prefix.
    """
    try:
        data = load_json_with_comments(config_path)
    except (json.JSONDecodeError, OSError):
        return None

    # 1. Explicit $schema reference
    ref = data.get("$schema")
    if ref and not ref.startswith("http"):
        candidate = (config_path.parent / ref).resolve()
        if candidate.is_file():
            return candidate

    # 2. Heuristic fallback — match the longest prefix first.
    #    If a prefix matches but has no schema file, return None
    #    immediately (don't fall through to shorter prefixes).
    parent = config_path.parent.name
    stem = config_path.stem
    for prefix in SCHEMA_PREFIX_ORDER:
        matched = False
        if parent != cases_dir.name and parent.startswith(prefix):
            matched = True
        elif stem.startswith(prefix):
            matched = True
        if matched:
            schema_file = cases_dir / f"{prefix}_schema.json"
            return schema_file if schema_file.is_file() else None

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases_dir", nargs="?",
                        default=str(Path(__file__).resolve().parent),
                        help="Root directory to scan (default: directory of this script)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Only print failures and summary")
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir).resolve()
    if not cases_dir.is_dir():
        print(f"ERROR: {cases_dir} is not a directory", file=sys.stderr)
        return 2

    configs = sorted(p for p in cases_dir.rglob("*.json")
                     if "schema" not in p.name)

    schemas: dict[Path, dict] = {}
    passed = 0
    failed = 0
    skipped = 0
    parse_errors = 0
    failure_details: list[tuple[Path, Path, list]] = []

    for cp in configs:
        sp = resolve_schema(cp, cases_dir)
        if sp is None:
            if not args.quiet:
                print(f"  SKIP  {cp.relative_to(cases_dir)}  (no schema)")
            skipped += 1
            continue

        if sp not in schemas:
            with open(sp) as f:
                schemas[sp] = json.load(f)

        try:
            data = load_json_with_comments(cp)
        except json.JSONDecodeError as exc:
            print(f"  ERR   {cp.relative_to(cases_dir)}  JSON parse error: {exc}")
            parse_errors += 1
            continue

        validator = Draft7Validator(schemas[sp])
        errors = list(validator.iter_errors(data))
        if errors:
            rel = cp.relative_to(cases_dir)
            schema_rel = sp.relative_to(cases_dir)
            print(f"  FAIL  {rel}  ({schema_rel})")
            for err in errors:
                loc = ".".join(str(p) for p in err.absolute_path) if err.absolute_path else "(root)"
                print(f"        [{loc}] {err.message}")
            failure_details.append((cp, sp, errors))
            failed += 1
        else:
            if not args.quiet:
                print(f"  OK    {cp.relative_to(cases_dir)}")
            passed += 1

    # Summary
    total = passed + failed + skipped + parse_errors
    print(f"\n{'=' * 50}")
    print(f"  Total:        {total}")
    print(f"  Passed:       {passed}")
    print(f"  Failed:       {failed}")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Skipped:      {skipped}")
    print(f"{'=' * 50}")

    return 1 if (failed or parse_errors) else 0


if __name__ == "__main__":
    sys.exit(main())
