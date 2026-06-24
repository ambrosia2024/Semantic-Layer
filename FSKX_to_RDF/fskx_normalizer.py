"""Normalize extracted FSKX metadata before JSON-LD/RDF conversion."""

from __future__ import annotations

import copy
from typing import Any

from schema_loader import get_date_slots

_DEFAULT_DATE_SLOTS = {"creationDate", "modificationDate", "publicationDate"}


def _int_array_to_iso(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 3 and all(isinstance(v, int) for v in value):
        y, m, d = value
        return f"{y:04d}-{m:02d}-{d:02d}"
    return value


def _normalize_date(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) == 3 and all(isinstance(v, int) for v in value):
            return _int_array_to_iso(value)
        return [_int_array_to_iso(v) for v in value]
    return value


def _fix_contacts(entries: list[dict]) -> list[dict]:
    out = []
    for item in entries:
        obj = dict(item)
        email = obj.get("email")
        if isinstance(email, str) and not email.strip():
            obj.pop("email", None)
        out.append(obj)
    return out


def _fix_references(entries: list[dict]) -> list[dict]:
    out = []
    for item in entries:
        obj = dict(item)
        if "date" in obj and "publicationDate" not in obj:
            obj["publicationDate"] = obj.pop("date")
        elif "date" in obj:
            obj.pop("date", None)
        obj.setdefault("isReferenceDescription", False)
        obj.setdefault("doi", "")
        out.append(obj)
    return out


def normalize(model: dict[str, Any], schema_path: str | None = None) -> dict[str, Any]:
    date_slots = set(get_date_slots(schema_path)) if schema_path else set(_DEFAULT_DATE_SLOTS)
    if not date_slots:
        date_slots = set(_DEFAULT_DATE_SLOTS)

    m = copy.deepcopy(model)
    gi = m.get("generalInformation", {})

    for slot in date_slots & set(gi.keys()):
        gi[slot] = _normalize_date(gi[slot])

    for ref in gi.get("reference", []) or []:
        for slot in date_slots & set(ref.keys()):
            ref[slot] = _normalize_date(ref[slot])

    if isinstance(gi.get("author"), list):
        gi["author"] = _fix_contacts(gi["author"])
    if isinstance(gi.get("creator"), list):
        gi["creator"] = _fix_contacts(gi["creator"])
    if isinstance(gi.get("reference"), list):
        gi["reference"] = _fix_references(gi["reference"])

    m["generalInformation"] = gi
    return m
