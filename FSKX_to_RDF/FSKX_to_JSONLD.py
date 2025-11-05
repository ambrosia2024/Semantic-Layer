#
# RAKIP FSKX Extractor
#
# Extracts the metadata of all published FSKX models from the RAKIP model repository,
# converts it to the JSON Linked Data format and saves it as files in a folder
#
# The JSON to JSONLD conversion was provided by Julian Schneider (julian.schneider@bfr.bund.de)

from pyld import jsonld
import zipfile
import json
import os
import sys
import argparse
import logging
import copy
import zipfile
import tempfile
import shutil
import hashlib
import re
import unicodedata
from xml.etree import ElementTree as ET
from pathlib import Path
from difflib import SequenceMatcher
from pyld import jsonld
from robust_metadata_matching import load_all_metadata_robust


# Folder for the FSKX models to be read
INPUT_FOLDER = "fskx_models"

# Folder for the resulting LD files
OUTPUT_FOLDER = "unmapped/jsonld"

# Base URI for generating IDs - aligned with fskx.owl ontology
BASE_URI = "http://semanticlookup.zbmed.de/km/fskxo/"

# Path to the fskx.owl ontology file
ONTOLOGY_FILE = "fskxo.owl"


#ZENODO_TOKEN = "QgH8nwT9xXzgn1ZeJafy4D88GMLRLNhu5ZL11IHpdpbBUR2d2nEsaYUOPw5M"

# Global dictionary to store ontology mappings
ONTOLOGY_CLASSES = {}
ONTOLOGY_PROPERTIES = {}
ONTOLOGY_INDIVIDUALS = {}  # Maps labels to IRIs for NamedIndividuals
ONTOLOGY_SYNONYMS = {}     # Maps synonyms to canonical labels

# ---- Matching strictness (tune here) ----
FUZZY_ENABLED = True              # set to False to disable fuzzy completely
FUZZY_MIN_SCORE = 0.96            # 0..1; higher = stricter
FUZZY_DISABLE_BELOW_LEN = 10      # don't fuzzy-match queries shorter than this many chars

PARTIAL_MIN_TOKEN_OVERLAP = 0.8   # Jaccard token overlap needed for partial match
PREFIX_MIN_CHARS = 6              # require at least this many leading chars to match for prefix pass

# Remove harmless tokens from comparison (expand if needed)
MATCH_STOPWORDS = {"sp", "spp", "cf", "strain", "subsp", "serovar", "serotype"}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Path Flattening Extractor ---

class FskxExtractorFlat:
    """Extracts an FSKX archive while flattening long directory paths."""
    def __init__(self):
        self.component_map = {}
    
    def shorten_component(self, name):
        if not name: return ""
        if name in self.component_map: return self.component_map[name]
        short = f"{name[:3].lower()}_{hashlib.md5(name.encode()).hexdigest()[:6]}"
        self.component_map[name] = short
        return short
    
    def extract(self, fskx_path: Path, extract_to: Path):
        with zipfile.ZipFile(fskx_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                parts = member.filename.replace('\\', '/').split('/')
                target_path = extract_to
                for part in parts[:-1]:
                    if part: target_path = target_path / self.shorten_component(part)
                final_component = parts[-1]
                if member.is_dir():
                    if final_component:
                        (target_path / self.shorten_component(final_component)).mkdir(parents=True, exist_ok=True)
                else:
                    target_path.mkdir(parents=True, exist_ok=True)
                    target_file = target_path / final_component
                    with zip_ref.open(member) as source, target_file.open('wb') as target:
                        shutil.copyfileobj(source, target)

# --- Core Data Classes ---

class Model:
    def __init__(self, model_id, source_file):
        self.id = model_id
        self.source_file = source_file
        self.metadata = {}

class ParameterLink:
    def __init__(self, target_model_alias, target_param_id, source_model_alias, source_param_id):
        self.target_model_alias, self.target_param_id = target_model_alias, target_param_id
        self.source_model_alias, self.source_param_id = source_model_alias, source_param_id

class CompositeModel:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.models, self.links, self.submodel_aliases = {}, [], {}
        self.main_sbml_file: Path = None

# --- Joined Model Processing Functions ---

def parse_manifest(manifest_path: Path):
    if not manifest_path.exists():
        logging.error(f"manifest.xml not found at {manifest_path}")
        return {}
    logging.info(f"Parsing manifest.xml at {manifest_path}...")
    files = {'sbml': [], 'metadata': []}
    try:
        tree = ET.parse(str(manifest_path))
        root = tree.getroot()
        ns_map = {'omex': 'http://identifiers.org/combine.specifications/omex-manifest'}
        content_elements = root.findall('omex:content', ns_map) or root.findall('content')
        for content in content_elements:
            location = (content.get('location') or "").replace('\\', '/').lstrip('./')
            if location.endswith('.sbml'): files['sbml'].append(location)
            elif location.endswith('metaData.json'): files['metadata'].append(location)
        logging.info(f"Found {len(files['sbml'])} SBML and {len(files['metadata'])} metadata files in manifest.")
        return files
    except ET.ParseError as e:
        logging.error(f"Error parsing manifest.xml: {e}")
        return {}

def discover_models_from_manifest(composite_model: CompositeModel, manifest_files: dict):
    """
    Discovers models by looking for metaData.json files in subdirectories,
    using the directory name as the model ID. This captures models not
    defined in the main SBML.
    """
    logging.info("Discovering models from manifest based on directory structure...")
    for meta_path in manifest_files.get('metadata', []):
        # Path components, ignoring './' prefix
        parts = Path(meta_path.replace('\\', '/')).parts
        if len(parts) > 1 and parts[-1] == 'metaData.json':
            model_id = parts[-2]
            if model_id not in composite_model.models:
                logging.info(f"  - Found new model by directory: '{model_id}'")
                # The source file is not critical here, as metadata is what matters.
                # We can use the metadata path itself as a reference.
                composite_model.models[model_id] = Model(model_id, source_file=meta_path)
            else:
                logging.info(f"  - Model '{model_id}' was already found via SBML, skipping.")

def parse_main_sbml(composite_model: CompositeModel):
    logging.info(f"Parsing main SBML file: {composite_model.main_sbml_file}...")
    try:
        tree = ET.parse(str(composite_model.main_sbml_file))
        root = tree.getroot()
        ns = {'comp': 'http://www.sbml.org/sbml/level3/version1/comp/version1'}
        comp_ns = ns['comp']

        for ext_model in root.findall(f'.//comp:externalModelDefinition', ns):
            model_id, source = ext_model.get(f'{{{comp_ns}}}id'), ext_model.get(f'{{{comp_ns}}}source')
            if model_id and source:
                composite_model.models[model_id] = Model(model_id, source.replace('\\', '/'))
        for submodel in root.findall(f'.//comp:submodel', ns):
            alias, model_ref = submodel.get(f'{{{comp_ns}}}id'), submodel.get(f'{{{comp_ns}}}modelRef')
            if alias and model_ref: composite_model.submodel_aliases[alias] = model_ref
        for param in root.findall('.//parameter'):
            replaced_by = param.find('comp:replacedBy', ns)
            if replaced_by:
                target_param_id = param.get('id')
                source_param_id, source_model_alias = replaced_by.get(f'{{{comp_ns}}}idRef'), replaced_by.get(f'{{{comp_ns}}}submodelRef')
                target_model_alias = next((a for a in composite_model.submodel_aliases if a != source_model_alias), None)
                if all([target_param_id, source_param_id, source_model_alias, target_model_alias]):
                    composite_model.links.append(ParameterLink(target_model_alias, target_param_id, source_model_alias, source_param_id))
    except (ET.ParseError, FileNotFoundError) as e:
        logging.error(f"Error parsing main SBML file: {e}", exc_info=True)

def generate_jsonld_for_composite(composite_model: CompositeModel, original_fskx_path: Path, context_doc: dict):
    """
    Generates the JSON-LD structure for a composite model.
    It processes each sub-model using the same rich transformation logic as single models.
    """
    logging.info("Generating JSON-LD output for composite model...")
    root_id = f"{BASE_URI}CompositeModel/{sanitize_iri_component(original_fskx_path.stem)}"
    
    # The final JSON-LD structure, without the context which will be added later.
    json_ld = {
        "@id": root_id,
        "@type": "fskx:CompositeModel",
        "name": original_fskx_path.stem,
        "hasPart": []
    }

    # Process each sub-model using the single-model transformation logic
    for model_id, model in composite_model.models.items():
        logging.info(f"  - Processing sub-model with rich context: {model.id}")
        
        # Use the single-model function to get a fully processed sub-model JSON-LD
        sub_model_ld = transformJsonToJsonLD(model.id, model.metadata, context_doc)
        
        if sub_model_ld:
            # The transform function adds a context, which we don't need on the sub-part.
            # The main context will be applied to the whole composite model later.
            sub_model_ld.pop("@context", None)
            
            # Ensure the @type is set correctly for a sub-model part
            sub_model_ld["@type"] = "fskx:PredictiveModel"
            json_ld["hasPart"].append(sub_model_ld)

    logging.info("Adding parameter links to composite model structure...")
    for link in composite_model.links:
        source_model_id = composite_model.submodel_aliases.get(link.source_model_alias)
        target_model_id = composite_model.submodel_aliases.get(link.target_model_alias)
        if not all([source_model_id, target_model_id]):
            continue

        composite_target_param_id = link.target_param_id
        inferred_target_param_id = re.sub(r'\d+$', '', composite_target_param_id)

        target_model_json = next((m for m in json_ld["hasPart"] if m.get("@id") == f"{BASE_URI}{sanitize_iri_component(target_model_id)}"), None)

        if target_model_json:
            # The parameter list might be under modelMath
            params_list = []
            if "modelMath" in target_model_json and isinstance(target_model_json["modelMath"], dict):
                params_list = target_model_json["modelMath"].get("parameter", [])
            
            target_param_json = next((p for p in params_list if p.get("id") == inferred_target_param_id), None)
            
            if target_param_json:
                source_param_id_enriched = f"{BASE_URI}{sanitize_iri_component(source_model_id)}/parameter/{sanitize_iri_component(link.source_param_id)}"
                logging.info(f"  - Linking '{link.source_param_id}' from '{source_model_id}' to inferred '{inferred_target_param_id}' in '{target_model_id}'")
                target_param_json["wasInformedBy"] = {"@id": source_param_id_enriched}
            else:
                logging.warning(f"  - Could not find inferred target parameter '{inferred_target_param_id}' (from '{composite_target_param_id}') in model '{target_model_id}' to create link.")
        else:
            logging.warning(f"  - Could not find target model '{target_model_id}' in JSON-LD to create link.")

    return json_ld

def sanitize_iri_component(text):
    """
    Remove or replace special characters from text to make it IRI-safe.
    Replaces spaces with underscores, removes special characters.
    """
    if not text:
        return ""

    # Convert to string if not already
    text = str(text)

    # Replace common special characters
    text = text.replace(' ', '_')
    text = text.replace('/', '_')
    text = text.replace('\\', '_')
    text = text.replace('@', '_at_')
    text = text.replace('.', '_')
    text = text.replace(',', '_')
    text = text.replace(':', '_')
    text = text.replace(';', '_')
    text = text.replace('!', '')
    text = text.replace('?', '')
    text = text.replace('&', '_and_')
    text = text.replace('%', '_percent_')
    text = text.replace('#', '_')
    text = text.replace('(', '_')
    text = text.replace(')', '_')
    text = text.replace('[', '_')
    text = text.replace(']', '_')
    text = text.replace('{', '_')
    text = text.replace('}', '_')
    text = text.replace('<', '_')
    text = text.replace('>', '_')
    text = text.replace('|', '_')
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace('*', '_')
    text = text.replace('+', '_plus_')
    text = text.replace('=', '_eq_')

    # Remove multiple consecutive underscores
    text = re.sub(r'_+', '_', text)

    # Remove leading/trailing underscores
    text = text.strip('_')

    # Limit length to reasonable size
    if len(text) > 100:
        text = text[:100]

    return text

def load_ontology_mappings(ontology_file):
    """
    Parse the fskx.owl ontology file and extract class, property, and NamedIndividual IRIs.
    Returns dictionaries mapping labels to IRIs.
    """
    if not os.path.exists(ontology_file):
        logging.warning(f"Ontology file {ontology_file} not found. Using default IRI generation.")
        return {}, {}, {}, {}

    try:
        tree = ET.parse(ontology_file)
        root = tree.getroot()

        # Define namespaces
        namespaces = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'owl': 'http://www.w3.org/2002/07/owl#',
            'oboInOwl': 'http://www.geneontology.org/formats/oboInOwl#'
        }

        classes = {}
        properties = {}
        individuals = {}
        synonyms = {}

        # Extract OWL Classes
        for owl_class in root.findall('.//owl:Class', namespaces):
            about = owl_class.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            label_elem = owl_class.find('rdfs:label', namespaces)
            if about and label_elem is not None:
                label = label_elem.text
                classes[label] = about

        # Extract Object Properties
        for obj_prop in root.findall('.//owl:ObjectProperty', namespaces):
            about = obj_prop.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            label_elem = obj_prop.find('rdfs:label', namespaces)
            if about and label_elem is not None:
                label = label_elem.text
                properties[label] = about

        # Extract NamedIndividuals with their labels and synonyms
        for named_ind in root.findall('.//owl:NamedIndividual', namespaces):
            about = named_ind.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            label_elem = named_ind.find('rdfs:label', namespaces)

            if about and label_elem is not None:
                label = label_elem.text
                individuals[label] = about

                # Also map normalized versions for fuzzy matching
                normalized_label = label.lower().strip()
                individuals[normalized_label] = about

                # Extract synonyms
                synonym_elems = named_ind.findall('oboInOwl:hasSynonym', namespaces)
                for syn_elem in synonym_elems:
                    synonym = syn_elem.text
                    if synonym:
                        synonyms[synonym.lower().strip()] = label
                        individuals[synonym.lower().strip()] = about

        logging.info(f"Loaded {len(classes)} classes, {len(properties)} properties, and {len(set(individuals.values()))} NamedIndividuals from ontology.")
        return classes, properties, individuals, synonyms

    except Exception as e:
        logging.error(f"Error parsing ontology file: {e}", exc_info=True)
        return {}, {}, {}, {}

def generate_hash_id(data):
    """Generate a short hash-based ID from object data."""
    if isinstance(data, dict):
        # Create a deterministic string representation
        sorted_items = sorted(data.items()) if data else []
        content = str(sorted_items)
    else:
        content = str(data)
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]

def find_ontology_iri(text, field_type=None, debug_label_for_logs=None):
    """
    High-precision matching:
    1) exact label
    2) synonym → label
    3) token-exact (same token set)
    4) prefix (leading chars) — cautious
    5) partial (high token overlap)
    6) fuzzy (optional, high threshold, disabled for short queries)
    """
    if not text:
        return None

    q = normalize(text)
    q_tokens = tokenize(text)
    q_token_set = set(q_tokens)

    # For vocabulary fields (unit, unitCategory, dataType, classification), search only in individuals
    if field_type in {"unit", "unitCategory", "dataType", "classification"}:
        spaces = [ONTOLOGY_INDIVIDUALS]
    # hazards/products/etc. search order
    elif field_type in {"hazard", "product", "populationGroup", "modelClass"}:
        spaces = [ONTOLOGY_CLASSES, ONTOLOGY_INDIVIDUALS]
    else:
        spaces = [ONTOLOGY_INDIVIDUALS, ONTOLOGY_CLASSES]

    # ---- 1) exact label
    for space in spaces:
        iri = space.get(q)
        if iri:
            return iri

    # ---- 2) synonym → label
    canonical = ONTOLOGY_SYNONYMS.get(q)
    if canonical:
        for space in spaces:
            iri = space.get(canonical)
            if iri:
                return iri

    # Build candidate list (label, iri, tokens) once
    candidates = []
    for space in spaces:
        for label, iri in space.items():
            toks = tokenize(label)
            candidates.append((label, iri, toks, set(toks)))

    # ---- 3) token-exact
    for label, iri, toks, tokset in candidates:
        if tokset == q_token_set and tokset:
            return iri

    # ---- 4) prefix (ensure sufficiently long to avoid overmatch)
    if len(q) >= PREFIX_MIN_CHARS:
        for label, iri, toks, tokset in candidates:
            if label.startswith(q) or q.startswith(label):
                # Require at least partial token alignment too
                if jaccard(q_token_set, tokset) >= PARTIAL_MIN_TOKEN_OVERLAP:
                    return iri

    # ---- 5) partial with high token overlap
    best_partial = (None, 0.0)
    for label, iri, toks, tokset in candidates:
        overlap = jaccard(q_token_set, tokset)
        if overlap > best_partial[1]:
            best_partial = (iri, overlap)
    if best_partial[1] >= PARTIAL_MIN_TOKEN_OVERLAP:
        return best_partial[0]

    # ---- 6) fuzzy (optional, strict threshold, avoid short queries)
    if FUZZY_ENABLED and len(q) >= FUZZY_DISABLE_BELOW_LEN:
        best_fuzzy = (None, 0.0)
        for label, iri, toks, tokset in candidates:
            s = SequenceMatcher(None, q, label).ratio()
            if s > best_fuzzy[1]:
                best_fuzzy = (iri, s)
        if best_fuzzy[1] >= FUZZY_MIN_SCORE:
            return best_fuzzy[0]

    if debug_label_for_logs:
        logging.debug(f"[match-miss] '{debug_label_for_logs}' -> no safe match "
              f"(tokens={q_tokens}, fuzzy_used={FUZZY_ENABLED and len(q) >= FUZZY_DISABLE_BELOW_LEN})")
    return None


def normalize(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def tokenize(s: str) -> list:
    """Alpha/num tokens, no punctuation; stopwords removed; normalized."""
    s = normalize(s)
    # keep letters/digits, split on non-alphanum
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t]
    toks = [t for t in toks if t not in MATCH_STOPWORDS]
    return toks

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union


def generate_creator_id(model_id, creator_data, index=0):
    """Generate opaque hashed ID for creator/author objects to protect personal information."""
    # Always use hash-based ID for privacy
    hash_id = generate_hash_id(creator_data)
    return f"{BASE_URI}{sanitize_iri_component(model_id)}/creator/{hash_id}"

def generate_reference_id(model_id, reference_data, index=0):
    """Generate ID for reference objects."""
    if isinstance(reference_data, dict):
        # Use DOI if available
        if reference_data.get('doi'):
            safe_doi = sanitize_iri_component(reference_data['doi'])
            return f"{BASE_URI}{sanitize_iri_component(model_id)}/reference/{safe_doi}"
        elif reference_data.get('title'):
            safe_title = sanitize_iri_component(reference_data['title'][:50])
            return f"{BASE_URI}{sanitize_iri_component(model_id)}/reference/{safe_title}"

    # Fallback to hash-based ID
    hash_id = generate_hash_id(reference_data)
    return f"{BASE_URI}{sanitize_iri_component(model_id)}/reference/{hash_id}"

def generate_parameter_id(model_id, parameter_data):
    """Generate full URI for parameter objects."""
    if isinstance(parameter_data, dict) and parameter_data.get('id'):
        param_id = sanitize_iri_component(parameter_data['id'])
        return f"{BASE_URI}{sanitize_iri_component(model_id)}/parameter/{param_id}"

    # Fallback to hash-based ID
    hash_id = generate_hash_id(parameter_data)
    return f"{BASE_URI}{sanitize_iri_component(model_id)}/parameter/{hash_id}"

def generate_scope_object_id(model_id, object_type, object_data, index=0):
    """
    Generate ID for scope objects (product, hazard, etc.).
    Prefers ontology IRIs from controlled vocabulary when available.
    """
    if isinstance(object_data, dict) and object_data.get('name'):
        # Try to find ontology IRI based on name
        ontology_iri = find_ontology_iri(object_data['name'], field_type=object_type)
        if ontology_iri:
            logging.info(f"  -> Matched '{object_data['name']}' to ontology IRI: {ontology_iri}")
            return ontology_iri

        # Fallback to sanitized name-based ID
        safe_name = sanitize_iri_component(object_data['name'])
        return f"{BASE_URI}{sanitize_iri_component(model_id)}/scope/{object_type}/{safe_name}"

    # Fallback to hash-based ID for all other cases
    hash_id = generate_hash_id(object_data)
    return f"{BASE_URI}{sanitize_iri_component(model_id)}/scope/{object_type}/{hash_id}"

def generate_model_category_id(model_id, category_data, index=0):
    """
    Generate ID for model category objects.
    Prefers ontology IRIs from controlled vocabulary when available.
    """
    if isinstance(category_data, dict):
        if category_data.get('modelClass'):
            # Try to find ontology IRI for the model class
            ontology_iri = find_ontology_iri(category_data['modelClass'], field_type='modelClass')
            if ontology_iri:
                logging.info(f"  -> Matched model class '{category_data['modelClass']}' to ontology IRI: {ontology_iri}")
                return ontology_iri

            safe_class = sanitize_iri_component(category_data['modelClass'])
            return f"{BASE_URI}{sanitize_iri_component(model_id)}/modelCategory/{safe_class}"

    # Fallback to hash-based ID
    hash_id = generate_hash_id(category_data)
    return f"{BASE_URI}{sanitize_iri_component(model_id)}/modelCategory/{hash_id}"

def generate_study_object_id(model_id, object_type, study_data, index=0):
    """Generate ID for study-related objects."""
    if isinstance(study_data, dict):
        # Use identifier, name, or title if available
        for field in ['identifier', 'name', 'title']:
            if study_data.get(field):
                safe_value = sanitize_iri_component(str(study_data[field])[:50])
                return f"{BASE_URI}{sanitize_iri_component(model_id)}/dataBackground/{object_type}/{safe_value}"

    # Fallback to hash-based ID
    hash_id = generate_hash_id(study_data)
    return f"{BASE_URI}{sanitize_iri_component(model_id)}/dataBackground/{object_type}/{hash_id}"

def generate_math_object_id(model_id, object_type, math_data, index=0):
    """Generate ID for modelMath-related objects."""
    if isinstance(math_data, dict):
        # Use name or identifier if available
        for field in ['name', 'identifier', 'id']:
            if math_data.get(field):
                safe_value = sanitize_iri_component(str(math_data[field])[:50])
                return f"{BASE_URI}{sanitize_iri_component(model_id)}/modelMath/{object_type}/{safe_value}"

    # Fallback to hash-based ID
    hash_id = generate_hash_id(math_data)
    return f"{BASE_URI}{sanitize_iri_component(model_id)}/modelMath/{object_type}/{hash_id}"

def generate_generic_id(model_id, path, data, index=0):
    """Generate a generic ID for any object based on its path and data."""
    # Create a safe path component
    safe_path = path.replace(".", "/")

    # Try to use meaningful identifiers from the data
    if isinstance(data, dict):
        # Common identifying fields
        identifiers = ['id', 'identifier', 'name', 'title', 'email', 'doi']
        for field in identifiers:
            if data.get(field):
                safe_value = sanitize_iri_component(str(data[field])[:50])
                return f"{BASE_URI}{sanitize_iri_component(model_id)}/{safe_path}/{safe_value}"

    # Fallback to hash-based ID
    hash_id = generate_hash_id(data)
    if index > 0:
        return f"{BASE_URI}{sanitize_iri_component(model_id)}/{safe_path}/{hash_id}_{index}"
    return f"{BASE_URI}{sanitize_iri_component(model_id)}/{safe_path}/{hash_id}"

def map_vocab_field_to_iri(field_name, field_value):
    """
    Map vocabulary fields (unit, unitCategory, dataType, classification) to ontology IRIs.
    Returns the original value if no mapping is found.
    """
    if not field_value or not isinstance(field_value, str):
        return field_value

    # Determine the field type for targeted search
    field_type_mapping = {
        'unit': 'unit',
        'unitCategory': 'unitCategory',
        'dataType': 'dataType',
        'classification': 'classification'
    }

    field_type = field_type_mapping.get(field_name)
    if not field_type:
        return field_value

    # Try to find ontology IRI
    iri = find_ontology_iri(field_value, field_type=field_type, debug_label_for_logs=f"{field_name}={field_value}")

    if iri:
        # Return as an object with @id and label
        return {
            "@id": iri,
            "label": field_value
        }

    # Return original value if no mapping found
    return field_value

def add_ids_to_structure(data, model_id, path=""):
    """Recursively add @id fields to nested structures."""
    if isinstance(data, dict):
        # Preserve original unit string
        if 'unit' in data and isinstance(data['unit'], str):
            data['unit_label'] = data['unit']

        # Map vocabulary fields to IRIs BEFORE processing structure
        for vocab_field in ['unit', 'unitCategory', 'dataType', 'classification']:
            if vocab_field in data and isinstance(data[vocab_field], str):
                mapped_value = map_vocab_field_to_iri(vocab_field, data[vocab_field])
                data[vocab_field] = mapped_value

        # Always add @id if not present (unless it's a simple value)
        if "@id" not in data and len(data) > 1:
            # Sanitize model_id for use in IRIs
            safe_model_id = sanitize_iri_component(model_id)

            # Handle main sections
            if path == "generalInformation":
                data["@id"] = f"{BASE_URI}{safe_model_id}/generalInformation"
                data["@type"] = "generalInformation"
            elif path == "scope":
                data["@id"] = f"{BASE_URI}{safe_model_id}/scope"
                data["@type"] = "scope"
            elif path == "dataBackground":
                data["@id"] = f"{BASE_URI}{safe_model_id}/dataBackground"
                data["@type"] = "dataBackground"
            elif path == "modelMath":
                data["@id"] = f"{BASE_URI}{safe_model_id}/modelMath"
                data["@type"] = "modelMath"

            # Handle specific object types with custom generators
            elif path == "generalInformation.creator":
                data["@id"] = generate_creator_id(model_id, data)
            elif path == "generalInformation.reference":
                data["@id"] = generate_reference_id(model_id, data)
            elif path == "generalInformation.modelCategory":
                data["@id"] = generate_model_category_id(model_id, data)
            elif path == "scope.product":
                data["@id"] = generate_scope_object_id(model_id, "product", data)
            elif path == "scope.hazard":
                data["@id"] = generate_scope_object_id(model_id, "hazard", data)
            elif path == "scope.populationGroup":
                data["@id"] = generate_scope_object_id(model_id, "populationGroup", data)
            elif path.startswith("modelMath.parameter"):
                data["@id"] = generate_parameter_id(model_id, data)
            elif path == "modelMath.qualityMeasures":
                data["@id"] = generate_math_object_id(model_id, "qualityMeasures", data)
            elif path == "modelMath.modelEquation":
                data["@id"] = generate_math_object_id(model_id, "modelEquation", data)
            elif path == "modelMath.exposure":
                data["@id"] = generate_math_object_id(model_id, "exposure", data)
            elif path == "dataBackground.study":
                data["@id"] = generate_study_object_id(model_id, "study", data)
            elif path == "dataBackground.studySample":
                data["@id"] = generate_study_object_id(model_id, "studySample", data)
            elif path == "dataBackground.dietaryAssessmentMethod":
                data["@id"] = generate_study_object_id(model_id, "dietaryAssessmentMethod", data)
            elif path == "dataBackground.laboratory":
                data["@id"] = generate_study_object_id(model_id, "laboratory", data)
            elif path == "dataBackground.assay":
                data["@id"] = generate_study_object_id(model_id, "assay", data)

            # Handle all other nested objects with generic ID generation
            elif path and "." in path:
                data["@id"] = generate_generic_id(model_id, path, data)

            # Handle direct children of main sections
            elif path in ["generalInformation", "scope", "dataBackground", "modelMath"]:
                pass  # Already handled above
            else:
                # Any other top-level or nested object
                if path:
                    data["@id"] = generate_generic_id(model_id, path, data)

        # Handle arrays within objects
        for key, value in data.items():
            if key != "@id" and key != "@type":  # Don't process @id and @type fields
                new_path = f"{path}.{key}" if path else key

                # Special handling for known array fields
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict) and "@id" not in item:
                            # Add index-specific IDs for array items
                            if key == "creator":
                                item["@id"] = generate_creator_id(model_id, item, i)
                            elif key == "reference":
                                item["@id"] = generate_reference_id(model_id, item, i)
                            elif key == "parameter":
                                item["@id"] = generate_parameter_id(model_id, item)
                            elif key == "author":
                                item["@id"] = generate_creator_id(model_id, item, i)
                            elif key == "modelCategory":
                                item["@id"] = generate_model_category_id(model_id, item, i)
                            elif key in ["product", "hazard", "populationGroup"]:
                                item["@id"] = generate_scope_object_id(model_id, key, item, i)
                            elif key in ["study", "studySample", "dietaryAssessmentMethod", "laboratory", "assay"]:
                                item["@id"] = generate_study_object_id(model_id, key, item, i)
                            elif key in ["qualityMeasures", "modelEquation", "exposure"]:
                                item["@id"] = generate_math_object_id(model_id, key, item, i)
                            else:
                                item["@id"] = generate_generic_id(model_id, new_path, item, i)

                        # Recursively process array items
                        add_ids_to_structure(item, model_id, new_path)
                else:
                    # Recursively process nested objects
                    add_ids_to_structure(value, model_id, new_path)

    elif isinstance(data, list):
        # Handle arrays at the top level
        for i, item in enumerate(data):
            if isinstance(item, dict) and "@id" not in item:
                item["@id"] = generate_generic_id(model_id, path, item, i)
            add_ids_to_structure(item, model_id, path)

def fix_ontology_iris(data):
    """
    Recursively traverse JSON-LD data and expand shortened ontology IRIs back to full URIs.
    This ensures that hazard, product, and other ontology-mapped entities use full URIs.
    """
    if isinstance(data, dict):
        # Check if this object has an @id that uses the fskxo prefix and should be expanded
        if "@id" in data:
            id_value = data["@id"]
            # If the ID starts with "fskxo:" and looks like an ontology IRI (contains FSKXO_)
            if isinstance(id_value, str) and id_value.startswith("fskxo:") and "FSKXO_" in id_value:
                # Expand it to the full URI
                expanded_id = id_value.replace("fskxo:", BASE_URI)
                data["@id"] = expanded_id

        # Recursively process all values
        for value in data.values():
            if isinstance(value, (dict, list)):
                fix_ontology_iris(value)

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                fix_ontology_iris(item)

def core_iri_from_model(model):
    """Generate core IRI from model identifier."""
    # Prefer a full IRI if present; else mint from base + identifier
    ident = (model.get("generalInformation") or {}).get("identifier")
    if not ident:
        return None
    if ident.startswith("http://") or ident.startswith("https://"):
        return ident
    # Sanitize the identifier to remove special characters
    safe_ident = sanitize_iri_component(ident.lstrip("/"))
    return BASE_URI + safe_ident

# reads the metadata for a single model from a local .fskx file
def getFSKXMetadata(model_filename):
    filepath = os.path.join(INPUT_FOLDER, model_filename)
    logging.info(f"Reading metadata from: {filepath}")
    try:
        with zipfile.ZipFile(filepath, 'r') as fskx_zip:
            # FSKX files contain the metadata in 'metaData.json'
            if 'metaData.json' in fskx_zip.namelist():
                with fskx_zip.open('metaData.json') as metadata_file:
                    result = json.load(metadata_file)
                logging.info(f"Metadata for id {model_filename} loaded successfully.")
                return result
            else:
                logging.error(f"'metaData.json' not found in {filepath}")
                return None
    except FileNotFoundError:
        logging.error(f"File not found at {filepath}")
        return None
    except zipfile.BadZipFile:
        logging.error(f"Not a valid zip file: {filepath}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from 'metaData.json' in {filepath}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading {filepath}: {e}", exc_info=True)
        return None

# extracts the models to be converted from the local fskx_models folder
def getModelList():
    if not os.path.isdir(INPUT_FOLDER):
        logging.error(f"Input directory '{INPUT_FOLDER}' not found.")
        return []
    
    modelIDs = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.fskx')]
    logging.info(f"{len(modelIDs)} models found in '{INPUT_FOLDER}'.")
    return modelIDs



# loads the context for the JSON to JSONLD conversion.
# the context contains the ontology links and has to be
# provided as a JSON file
def loadContext(contextFile):
    """
    Loads the JSON-LD context from a file.
    """
    try:
        with open(contextFile, 'r') as f:
            context = json.load(f)
        logging.info("Context loaded successfully.")
        return context
    except FileNotFoundError:
        logging.error(f"The context file {contextFile} was not found.")
        return None
    except json.JSONDecodeError:
        logging.error(f"The context file {contextFile} is not a valid JSON file.")
        return None

# transform the JSON metadata to JSONLD and
# writes the result to a file with the ID
# as the filename
def transformJsonToJsonLD(id, metadata, context_doc):
    if not metadata:
        logging.warning(f"Skipping transformation for id {id} due to missing metadata.")
        return None

    # Deep copy the metadata to avoid modifying the original
    data = copy.deepcopy(metadata)

    # Get the model identifier for ID generation
    model_identifier = (data.get("generalInformation") or {}).get("identifier") or id

    # Ensure the root has an @id (core IRI)
    root_id = core_iri_from_model(metadata)
    if root_id:
        data["@id"] = root_id
    else:
        logging.warning(f"No identifier for {id}; root will remain a blank node.")

    # Add @id fields to nested structures to avoid blank nodes
    logging.info(f"Adding IDs to nested structures for model {model_identifier}...")
    add_ids_to_structure(data, model_identifier)

    # Attach the context correctly
    merged = {"@context": context_doc["@context"], **data}
    
    logging.info(f"Metadata for id {id} transformed with enhanced IDs.")
    return merged

# retrieve, transform and write the metadata for one model
def getJsonLD(modelID, context):
    jsonData = getFSKXMetadata(modelID)
    jsonLDData = transformJsonToJsonLD(modelID, jsonData, context)
    return jsonLDData

def process_fskx_file(fskx_file_path: Path, context_doc: dict, override=False):
    """
    Main processing function for a single FSKX file.
    Detects if the model is single or joined and processes accordingly.
    """
    if not fskx_file_path.is_file():
        logging.error(f"Not a valid file: {fskx_file_path}")
        return

    output_filename = Path(OUTPUT_FOLDER) / f"{fskx_file_path.stem}.jsonld"
    if not override and output_filename.exists():
        logging.info(f"Skipping existing file: {output_filename}")
        return

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_path = Path(temp_dir_str).resolve()
        logging.info(f"Extracting FSKX archive to {temp_path}")
        extractor = FskxExtractorFlat()
        try:
            extractor.extract(fskx_file_path, temp_path)
            logging.info("FSKX file extracted successfully.")
        except Exception:
            return logging.error("An exception occurred during extraction.", exc_info=True)

        content_root = next((Path(root) for root, _, files in os.walk(temp_path) if "manifest.xml" in files), temp_path)
        manifest_files = parse_manifest(content_root / "manifest.xml")

        is_joined = False
        if len(manifest_files.get('metadata', [])) > 1:
            is_joined = True
        else:
            for sbml_path_rel in manifest_files.get('sbml', []):
                sbml_path_abs = content_root / sbml_path_rel
                try:
                    if "comp:listOfSubmodels" in sbml_path_abs.read_text(encoding='utf-8'):
                        is_joined = True
                        break
                except (FileNotFoundError, UnicodeDecodeError):
                    continue
        
        if is_joined:
            logging.info(f"Processing '{fskx_file_path.name}' as a JOINED model.")
            def translate_path(rel_path_str: str) -> Path:
                parts = rel_path_str.replace('\\', '/').split('/')
                physical_path = content_root
                for part in parts[:-1]:
                    if part: physical_path = physical_path / extractor.component_map.get(part, part)
                if parts[-1]: physical_path = physical_path / parts[-1]
                return physical_path

            composite_model = CompositeModel(content_root)
            for sbml_path_rel in manifest_files['sbml']:
                sbml_path_abs = translate_path(sbml_path_rel)
                try:
                    if "comp:listOfSubmodels" in sbml_path_abs.read_text(encoding='utf-8'):
                        composite_model.main_sbml_file = sbml_path_abs
                        break
                except (FileNotFoundError, UnicodeDecodeError):
                    continue
            
            if not composite_model.main_sbml_file:
                return logging.error("Failed to identify a main SBML file for joined model.")

            parse_main_sbml(composite_model)
            # --- New model discovery step ---
            discover_models_from_manifest(composite_model, manifest_files)
            # --- End new step ---
            logging.info(f"Composite model now contains {len(composite_model.models)} models:")
            for model_id in composite_model.models.keys():
                logging.info(f"  - {model_id}")
            load_all_metadata_robust(
                composite_model,
                manifest_files,
                base_path=temp_path,
                component_map=extractor.component_map,
                verbose=True
            )
            final_jsonld = generate_jsonld_for_composite(composite_model, fskx_file_path, context_doc)
            output_filename = Path(OUTPUT_FOLDER) / f"{fskx_file_path.stem}.jsonld"

        else:
            logging.info(f"Processing '{fskx_file_path.name}' as a SINGLE model.")
            metadata_path = content_root / manifest_files['metadata'][0] if manifest_files.get('metadata') else None
            if not metadata_path or not metadata_path.exists():
                return logging.error(f"metaData.json not found for single model '{fskx_file_path.name}'")
            
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            final_jsonld = transformJsonToJsonLD(fskx_file_path.stem, metadata, context_doc)

        compaction_context = context_doc
        if is_joined:
            # For joined models, we need a context that understands the linking properties
            compaction_context = copy.deepcopy(context_doc)
            compaction_context["@context"]["prov"] = "http://www.w3.org/ns/prov#"
            compaction_context["@context"]["hasPart"] = "https://schema.org/hasPart"
            compaction_context["@context"]["wasInformedBy"] = "http://www.w3.org/ns/prov#wasInformedBy"
            
            param_context = compaction_context["@context"]["modelMath"]["@context"]["parameter"]["@context"]
            param_context["wasInformedBy"] = {
                "@id": "http://www.w3.org/ns/prov#wasInformedBy",
                "@type": "@id"
            }

        if final_jsonld:
            try:
                # For joined models, the root object needs the context attached before processing.
                # The single model transform already includes it.
                if is_joined and "@context" not in final_jsonld:
                    final_jsonld = {"@context": context_doc["@context"], **final_jsonld}

                expanded = jsonld.expand(final_jsonld)
                compacted = jsonld.compact(expanded, compaction_context, options={'compactArrays': True, 'skipExpansion': False})
                fix_ontology_iris(compacted)
                
                # Ensure output_filename is set correctly for single models
                if not is_joined:
                    output_filename = Path(OUTPUT_FOLDER) / f"{fskx_file_path.stem}.jsonld"

                output_filename.write_text(json.dumps(compacted, indent=2, ensure_ascii=False), encoding='utf-8')
                logging.info(f"Successfully wrote linked JSON-LD to {output_filename}")
            except Exception as e:
                logging.error(f"Error during JSON-LD processing for {fskx_file_path.name}: {e}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert single or joined FSKX models to JSON-LD.")
    parser.add_argument("fskx_path", type=str, help="Path to a single .fskx file or a directory containing .fskx files.")
    parser.add_argument('--override', action='store_true', help='Override existing output files.')
    args = parser.parse_args()

    ont_classes, ont_properties, ont_individuals, ont_synonyms = load_ontology_mappings(ONTOLOGY_FILE)
    ONTOLOGY_CLASSES.update(ont_classes)
    ONTOLOGY_PROPERTIES.update(ont_properties)
    ONTOLOGY_INDIVIDUALS.update(ont_individuals)
    ONTOLOGY_SYNONYMS.update(ont_synonyms)

    context = loadContext("jsonld-context_fsk_enhanced.json")
    if not context:
        sys.exit("Failed to load context. Exiting.")

    fskx_path = Path(args.fskx_path)
    if fskx_path.is_dir():
        logging.info(f"Processing all .fskx files in directory: {fskx_path}")
        for fskx_file in fskx_path.glob('*.fskx'):
            process_fskx_file(fskx_file, context, args.override)
    elif fskx_path.is_file() and fskx_path.suffix == '.fskx':
        logging.info(f"Processing single .fskx file: {fskx_path}")
        process_fskx_file(fskx_path, context, args.override)
    else:
        logging.error(f"Invalid path provided: {fskx_path}. Must be a .fskx file or a directory.")
        sys.exit(1)
