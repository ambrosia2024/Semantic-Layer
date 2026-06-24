"""Schema utilities for FSKX→RDF migration.

This module makes ``fskxo.linkml.yaml`` the single source of truth for:
- JSON-LD context generation
- date slot discovery
- enum slot discovery
- slot IRI lookup
- model validation
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from linkml.generators.jsonldcontextgen import ContextGenerator
from linkml.generators.jsonschemagen import JsonSchemaGenerator
from linkml.generators.shaclgen import ShaclGenerator
from linkml_runtime.utils.schemaview import SchemaView
from rdflib import Graph, Namespace

_HERE = Path(__file__).resolve().parent
DEFAULT_SCHEMA = str(_HERE / "fskxo.linkml.yaml")
DEFAULT_ONTOLOGY = str(_HERE / "fskxo.owl")
DEFAULT_OUTPUT_DIR = str(_HERE / "mapped" / "turtle")
LEGACY_SCHEMA_HTTP = "http://schema.org/"
SCHEMA_HTTPS = "https://schema.org/"
LEGACY_CONTEXT_TERMS = {
    "label": {"@id": "schema:label"},
    "unit_label": {"@id": "schema:unit_label"},
}
SH = Namespace("http://www.w3.org/ns/shacl#")


def get_default_schema_path() -> str:
    return DEFAULT_SCHEMA


def get_default_vocab_path() -> str:
    return DEFAULT_ONTOLOGY


def get_default_output_dir() -> str:
    return DEFAULT_OUTPUT_DIR


@lru_cache(maxsize=8)
def load_schema(schema_path: str = DEFAULT_SCHEMA) -> SchemaView:
    return SchemaView(str(Path(schema_path).resolve()))


def _normalize_schema_iris(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(LEGACY_SCHEMA_HTTP, SCHEMA_HTTPS)
    if isinstance(value, dict):
        return {key: _normalize_schema_iris(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_normalize_schema_iris(inner) for inner in value]
    return value


def _relax_scalar_shapes_for_mapped_rdf(shapes_ttl: str) -> str:
    """Allow mapper-enriched scalar values to remain literals or become ontology IRIs."""
    graph = Graph()
    graph.parse(data=shapes_ttl, format="turtle")
    for prop_shape in list(graph.subjects(SH.path, None)):
        if (prop_shape, SH["class"], None) in graph:
            continue
        for predicate in (SH.datatype, SH.nodeKind, SH["in"]):
            for value in list(graph.objects(prop_shape, predicate)):
                graph.remove((prop_shape, predicate, value))
    return graph.serialize(format="turtle")


def build_jsonld_context(schema_path: str = DEFAULT_SCHEMA) -> dict:
    context = _normalize_schema_iris(json.loads(ContextGenerator(schema_path).serialize()))
    context_body = context.get("@context", context)
    if isinstance(context_body, dict):
        context_body.update(LEGACY_CONTEXT_TERMS)
    return context


def get_date_slots(schema_path: str = DEFAULT_SCHEMA) -> list[str]:
    sv = load_schema(schema_path)
    return [name for name, slot in sv.all_slots().items() if slot.range == "date"]


def get_enum_slots(schema_path: str = DEFAULT_SCHEMA) -> list[str]:
    sv = load_schema(schema_path)
    enums = set(sv.all_enums().keys())
    return [name for name, slot in sv.all_slots().items() if slot.range in enums]


def get_slot_uri(
    slot_name: str,
    class_name: str | None = None,
    schema_path: str = DEFAULT_SCHEMA,
) -> str | None:
    sv = load_schema(schema_path)
    try:
        slot = sv.induced_slot(slot_name, class_name) if class_name else sv.all_slots().get(slot_name)
    except Exception:
        return None
    if slot is None or not slot.slot_uri:
        return None
    try:
        return _normalize_schema_iris(sv.expand_curie(str(slot.slot_uri)))
    except Exception:
        return _normalize_schema_iris(str(slot.slot_uri))


def get_required_slots(schema_path: str = DEFAULT_SCHEMA) -> dict[str, list[str]]:
    sv = load_schema(schema_path)
    out: dict[str, list[str]] = {}
    for class_name in sv.all_classes():
        required = [
            slot_name
            for slot_name in sv.class_slots(class_name)
            if sv.induced_slot(slot_name, class_name).required
        ]
        if required:
            out[class_name] = required
    return out


def validate_model(
    model: dict[str, Any],
    schema_path: str = DEFAULT_SCHEMA,
    *,
    target_class: str = "GenericModel",
) -> list[str]:
    from linkml.validator import validate
    from linkml.validator.report import Severity

    report = validate(model, schema_path, target_class=target_class)
    return [issue.message for issue in report.results if issue.severity >= Severity.ERROR]


def generate_artifacts(
    schema_path: str = DEFAULT_SCHEMA,
    output_dir: str | None = None,
) -> dict[str, str]:
    out = Path(output_dir or (_HERE / "generated"))
    out.mkdir(parents=True, exist_ok=True)

    ctx_path = out / "fskxo.context.jsonld"
    schema_json_path = out / "fskxo.schema.json"
    shapes_path = out / "fskxo-shapes.ttl"

    ctx_path.write_text(
        json.dumps(build_jsonld_context(schema_path), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    schema_json_path.write_text(JsonSchemaGenerator(schema_path).serialize(), encoding="utf-8")
    shapes_ttl = _normalize_schema_iris(ShaclGenerator(schema_path, closed=False).serialize())
    shapes_path.write_text(_relax_scalar_shapes_for_mapped_rdf(shapes_ttl), encoding="utf-8")

    return {
        "context": str(ctx_path),
        "json_schema": str(schema_json_path),
        "shacl": str(shapes_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate context/schema/shapes from fskxo.linkml.yaml")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", default=str(_HERE / "generated"))
    args = parser.parse_args()

    paths = generate_artifacts(schema_path=args.schema, output_dir=args.output_dir)
    print(json.dumps(paths, indent=2))
