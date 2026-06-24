#!/usr/bin/env python3
"""SHACL validation for generated Turtle output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdflib import Graph


def validate_turtle(data_file: Path, shapes_file: Path) -> tuple[bool, str]:
    try:
        from pyshacl import validate
    except ModuleNotFoundError as e:
        return False, f"pyshacl is required for SHACL validation but is not installed ({e})."

    data_graph = Graph().parse(str(data_file), format="turtle")
    shapes_graph = Graph().parse(str(shapes_file), format="turtle")
    conforms, _report_graph, report_text = validate(
        data_graph,
        shacl_graph=shapes_graph,
        advanced=True,
        debug=False,
    )
    return bool(conforms), str(report_text)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="")
    parser.add_argument("--dir", default=str(here / "mapped" / "turtle"))
    parser.add_argument("--shapes", default=str(here / "generated" / "fskxo-shapes.ttl"))
    args = parser.parse_args()

    shapes = Path(args.shapes)
    if not shapes.exists():
        print(f"Shapes file missing: {shapes}", file=sys.stderr)
        return 2

    if args.file:
        f = Path(args.file)
        ok, report = validate_turtle(f, shapes)
        if ok:
            print(f"✓ SHACL conformant: {f.name}")
            return 0
        print(f"✗ SHACL violations in {f.name}\n{report}")
        return 1

    ttl_dir = Path(args.dir)
    files = sorted(ttl_dir.glob("*.ttl"))
    if not files:
        print(f"No Turtle files found in {ttl_dir}", file=sys.stderr)
        return 1

    failed = 0
    for f in files:
        ok, report = validate_turtle(f, shapes)
        if ok:
            print(f"✓ {f.name}")
        else:
            failed += 1
            print(f"✗ {f.name}\n{report}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
