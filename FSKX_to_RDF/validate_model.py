#!/usr/bin/env python3
"""Validate unmapped JSON-LD models against generated fskxo.schema.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonschema


def _strip_jsonld_keywords(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k.startswith("@"):
                continue
            if k == "unit_label":
                continue
            out[k] = _strip_jsonld_keywords(v)
        return out
    if isinstance(obj, list):
        return [_strip_jsonld_keywords(x) for x in obj]
    return obj


def _validate_one(path: Path, validator: jsonschema.Draft7Validator) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    data = _strip_jsonld_keywords(data)
    errors = list(validator.iter_errors(data))
    if errors:
        first = errors[0]
        where = "/" + "/".join(str(p) for p in first.absolute_path) if first.absolute_path else "(root)"
        print(f"✗ {path.name}: {where} -> {first.message}")
        return False
    print(f"✓ {path.name}")
    return True


def validate_folder(folder: Path, schema_path: Path) -> int:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft7Validator(schema)
    failed = 0
    files = sorted(folder.glob("*.jsonld"))
    if not files:
        print(f"No JSON-LD files found in {folder}")
        return 1

    for path in files:
        if not _validate_one(path, validator):
            failed += 1

    if failed:
        print(f"\n{failed} file(s) failed schema validation")
        return 1
    print("\nAll files passed schema validation")
    return 0


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=str(here / "unmapped" / "jsonld"))
    parser.add_argument("--file", default="")
    parser.add_argument("--schema", default=str(here / "generated" / "fskxo.schema.json"))
    args = parser.parse_args()
    schema_path = Path(args.schema)
    if args.file:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = jsonschema.Draft7Validator(schema)
        return 0 if _validate_one(Path(args.file), validator) else 1
    return validate_folder(Path(args.folder), schema_path)


if __name__ == "__main__":
    raise SystemExit(main())
