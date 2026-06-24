"""RDF-level vocabulary linking for FSKXO controlled vocabulary slots."""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, RDFS, SKOS

from schema_loader import DEFAULT_SCHEMA, get_enum_slots, get_slot_uri

SCHEMA_LABEL = URIRef("https://schema.org/label")
OBO_EXACT_SYNONYM = URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym")
OBO_HAS_SYNONYM = URIRef("http://www.geneontology.org/formats/oboInOwl#hasSynonym")
FSKXO = "http://semanticlookup.zbmed.de/km/fskxo/"
CONTROLLED_VOCAB_SLOTS = {
    "language",
    "software",
    "format",
    "publicationType",
    "modelClass",
    "modelSubClass",
    "basicProcess",
    "product",
    "hazard",
    "populationGroup",
    "classification",
    "unit",
    "unitCategory",
    "dataType",
    "modelType",
}


def _norm_label(value: str) -> str:
    return " ".join(value.split()).lower()


def build_label_index(vocab_path: str | Path) -> dict[str, str]:
    """Return a normalized label/synonym to IRI index for the ontology."""
    owl_graph = Graph()
    owl_graph.parse(str(vocab_path))

    label_index: dict[str, str] = {}
    label_predicates = (
        SKOS.prefLabel,
        RDFS.label,
        SKOS.altLabel,
        OBO_EXACT_SYNONYM,
        OBO_HAS_SYNONYM,
    )
    for predicate in label_predicates:
        for subject, _, label in owl_graph.triples((None, predicate, None)):
            if isinstance(label, Literal):
                key = _norm_label(str(label))
                if key:
                    label_index.setdefault(key, str(subject))
    return label_index


def default_replace_predicates(schema_path: str = DEFAULT_SCHEMA) -> list[URIRef]:
    """Return predicate IRIs for LinkML enum slots, plus legacy fallbacks."""
    slot_names = set(get_enum_slots(schema_path)) | CONTROLLED_VOCAB_SLOTS
    predicates = []
    seen = set()
    for slot in sorted(slot_names):
        iri = get_slot_uri(slot, schema_path=schema_path)
        if iri and iri not in seen:
            seen.add(iri)
            predicates.append(URIRef(iri))
    if predicates:
        return predicates

    return [
        DCTERMS.type,
        URIRef(FSKXO + "FSKXO_0000000039"),
        URIRef(FSKXO + "FSKXO_0000000040"),
        URIRef(FSKXO + "FSKXO_0000000041"),
    ]


def apply_vocab_linking(
    graph: Graph,
    vocab_path: str | Path,
    schema_path: str = DEFAULT_SCHEMA,
    *,
    add_labels: bool = True,
) -> int:
    """Resolve controlled-vocabulary literals to ontology IRIs in-place."""
    vocab_file = Path(vocab_path)
    if not vocab_file.exists():
        return 0

    label_index = build_label_index(vocab_file)
    changed = 0

    for predicate in default_replace_predicates(schema_path):
        for subject, _, literal in list(graph.triples((None, predicate, None))):
            if not isinstance(literal, Literal):
                continue
            original = str(literal).strip()
            iri = label_index.get(_norm_label(original))
            if not iri:
                continue

            target = URIRef(iri)
            graph.remove((subject, predicate, literal))
            graph.add((subject, predicate, target))
            if add_labels:
                graph.add((target, SCHEMA_LABEL, Literal(original)))
            changed += 1

    return changed
