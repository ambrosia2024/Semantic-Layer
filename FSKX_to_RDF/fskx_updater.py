import io
import os
import re
import zipfile
import json
import unicodedata
from pathlib import Path
from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import DCTERMS, RDF, SKOS, XSD, RDFS, OWL

# Define Namespaces
FSKXO = Namespace("http://semanticlookup.zbmed.de/km/fskxo/")
MODEL = Namespace("https://www.ambrosia-project.eu/model/")
VOCAB = Namespace("https://www.ambrosia-project.eu/vocab/")
AMBLINK = Namespace("https://www.ambrosia-project.eu/vocab/linking/")

# FSKX-O properties
FSKXO_HAS_PARAMETER = FSKXO.hasParameter

# AMBLINK properties
AMBLINK_HAS_INPUT_MAPPING = AMBLINK.hasInputMapping
AMBLINK_HAS_OUTPUT_MAPPING = AMBLINK.hasOutputMapping
AMBLINK_MAPS_PARAMETER = AMBLINK.mapsParameter

def _find_metadata_member(z: zipfile.ZipFile) -> str | None:
    candidates = [n for n in z.namelist() if n.lower().endswith("metadata.json")]
    return min(candidates, key=lambda n: n.count("/")) if candidates else None

def get_model_uuid_from_fskx(fskx_file_bytes: io.BytesIO, file_name: str) -> str:
    """Extracts the model UUID from an FSKX file."""
    try:
        with zipfile.ZipFile(fskx_file_bytes, 'r') as zip_ref:
            metadata_filename = _find_metadata_member(zip_ref)
            if not metadata_filename:
                return None
            with zip_ref.open(metadata_filename) as meta_file:
                fskx_content = json.load(meta_file)
            uuid_match = re.search(
                r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})',
                file_name,
                re.I
            )
            return uuid_match.group(1) if uuid_match else (
                fskx_content.get('id')
                or (fskx_content.get('generalInformation') or {}).get('identifier')
            )
    except Exception:
        return None

def find_existing_ttl_files(directory: str):
    """Finds fskx-models.ttl and wiring-instances.ttl in a directory."""
    skos_file = Path(directory) / "fskx-models.ttl"
    wiring_file = Path(directory) / "wiring-instances.ttl"
    return skos_file if skos_file.exists() else None, wiring_file if wiring_file.exists() else None

def check_if_model_exists(model_uri: URIRef, graph: Graph) -> bool:
    """Checks if a model with the given URI already exists in the graph."""
    return (model_uri, None, None) in graph

def purge_model_in_graphs(skos_graph: Graph, wiring_graph: Graph, model_uri: URIRef):
    # remove parameter subtrees + the model subject in SKOS graph
    for p in list(skos_graph.objects(model_uri, FSKXO_HAS_PARAMETER)):
        for t in list(skos_graph.triples((p, None, None))): skos_graph.remove(t)
        skos_graph.remove((model_uri, FSKXO_HAS_PARAMETER, p))
    for t in list(skos_graph.triples((model_uri, None, None))): skos_graph.remove(t)

    # remove wiring blank nodes for that model
    for bn in list(wiring_graph.objects(model_uri, AMBLINK_HAS_INPUT_MAPPING)):
        for t in list(wiring_graph.triples((bn, None, None))): wiring_graph.remove(t)
        wiring_graph.remove((model_uri, AMBLINK_HAS_INPUT_MAPPING, bn))

def append_to_ttl(skos_graph: Graph, wiring_graph: Graph, new_skos_g: Graph, new_wiring_g: Graph, model_uri: URIRef, overwrite: bool = False):
    """Appends new model data to existing graphs."""
    if overwrite:
        purge_model_in_graphs(skos_graph, wiring_graph, model_uri)
    
    skos_graph += new_skos_g
    wiring_graph += new_wiring_g
    
    return skos_graph, wiring_graph
