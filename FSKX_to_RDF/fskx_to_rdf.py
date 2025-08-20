import streamlit as st
import pandas as pd
import zipfile
import json
import io
import os
from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import DCTERMS, RDF, SKOS, XSD, RDFS, OWL
import math
import re # Import regex module for UUID extraction
import unicodedata
import difflib
from pathlib import Path
from . import fskx_updater

def build_fskxo_label_graph(master_df: pd.DataFrame | None = None) -> Graph:
    g = Graph()
    g.bind("fskxo", FSKXO)
    g.bind("vocab", VOCAB)
    g.bind("rdfs", RDFS)
    g.bind("dct", DCTERMS)
    g.bind("amblink", AMBLINK)

    LABELS = {
        # Properties
        FSKXO.FSKXO_0000000039: "hasClassification",
        FSKXO.FSKXO_0000000041: "hasDataType",
        FSKXO.FSKXO_0000000017: "hasParameter",
        FSKXO.FSKXO_0000000040: "hasUnitCategory",
        FSKXO.FSKXO_0000000005: "hasModelCategory",
        FSKXO.FSKXO_0000000019: "hasModelEquation",
        FSKXO.FSKXO_0000000007: "hasProduct",
        FSKXO.FSKXO_0000000008: "hasHazard",
        FSKXO.FSKXO_0000000004: "hasReference",
        FSKXO.FSKXO_0000000088: "hasVariabilitySubject",

        # Classes
        FSKXO.FSKXO_0000018113: "Predictive model",
        FSKXO.FSKXO_0000018125: "Other Empirical Model",
        FSKXO.FSKXO_0000018205: "Dose-response Model",
        FSKXO.FSKXO_0000018131: "Process model",
        FSKXO.FSKXO_0000018214: "QRA model",
        FSKXO.FSKXO_0000018210: "Health metrics model",

        # Category individuals
        FSKXO.FSKXO_0000018108: "Predictive model",
        FSKXO.FSKXO_0000018107: "Other Empirical models",
        FSKXO.FSKXO_0000018103: "Consumption model",
        FSKXO.FSKXO_0000018104: "Dose-response model",
        FSKXO.FSKXO_0000018105: "Exposure model",
        FSKXO.FSKXO_0000018106: "Health metrics model",
        FSKXO.FSKXO_0000018109: "Process model",
        FSKXO.FSKXO_0000018110: "QRA model",
        FSKXO.FSKXO_0000018111: "Risk characterization model",
        FSKXO.FSKXO_0000018112: "Toxicological reference value model",

        # Parameter roles
        FSKXO.FSKXO_0000017481: "Input",
        FSKXO.FSKXO_0000017482: "Output",
        FSKXO.FSKXO_0000017480: "Constant",

        # Datatypes
        FSKXO.FSKXO_0000017803: "Double",
        FSKXO.FSKXO_0000017816: "Vector[number]",
        FSKXO.FSKXO_0000017807: "Integer",

        # Unit categories
        FSKXO.FSKXO_0000017795: "Temperature",
        FSKXO.FSKXO_0000017796: "Time",

        # Custom vocab
        VOCAB.minValue: "minValue",
        VOCAB.maxValue: "maxValue",
        VOCAB.defaultValue: "defaultValue",
        VOCAB.hasUnit: "hasUnit",
        VOCAB.ModelParameter: "ModelParameter",
        AMBLINK.hasInputMapping: "hasInputMapping",
        AMBLINK.mapsParameter: "mapsParameter",
        AMBLINK.isFulfilledBy: "isFulfilledBy",
        AMBLINK.sourceVariableName: "sourceVariableName",
        AMBLINK.expectsQuantityKind: "expectsQuantityKind",
        SKOS.exactMatch: "exactMatch",
        DCTERMS.created: "created",
        DCTERMS.identifier: "identifier",
        DCTERMS.title: "title",
        DCTERMS.description: "description",
    }
    for uri, label in LABELS.items():
        g.add((uri, RDFS.label, Literal(label)))

    # --- Add friendly labels from master_mapping.xlsx for FSKXO URIs (e.g., units) ---
    if master_df is not None and not master_df.empty:
        # Any rows that carry FSKXO URIs in URI1..3 (often ConceptGroup == 'Unit')
        for _, row in master_df.iterrows():
            term = str(row.get("Term", "")).strip()
            alt  = str(row.get("altLabels", "")).strip()
            for col in ("URI1", "URI2", "URI3"):
                uri = str(row.get(col, "")).strip()
                if uri.startswith(str(FSKXO)):  # only add labels for fskxo:*
                    uref = URIRef(uri)
                    if term:
                        g.add((uref, RDFS.label, Literal(term)))
                    if alt:
                        for a in (x.strip() for x in alt.split(",") if x.strip()):
                            g.add((uref, SKOS.altLabel, Literal(a)))
    return g

# Define Namespaces
FSKXO = Namespace("http://semanticlookup.zbmed.de/km/fskxo/")
MODEL = Namespace("https://www.ambrosia-project.eu/model/")
VOCAB = Namespace("https://www.ambrosia-project.eu/vocab/")
QUDT_UNIT = Namespace("http://qudt.org/vocab/unit/") # Keep QUDT for units
AMBLINK = Namespace("https://www.ambrosia-project.eu/vocab/linking/")

# --- Properties (keep hard-coded) ---
HAS_CLASSIFICATION   = FSKXO.FSKXO_0000000039  # hasClassification
HAS_DATATYPE         = FSKXO.FSKXO_0000000041  # hasDataType
HAS_PARAMETER        = FSKXO.FSKXO_0000000017  # hasParameter
HAS_UNIT_CATEGORY    = FSKXO.FSKXO_0000000040  # hasUnitCategory
HAS_MODEL_CATEGORY   = FSKXO.FSKXO_0000000005  # hasModelCategory
HAS_MODEL_EQUATION   = FSKXO.FSKXO_0000000019  # hasModelEquation
HAS_PRODUCT          = FSKXO.FSKXO_0000000007  # hasProduct
HAS_HAZARD           = FSKXO.FSKXO_0000000008  # hasHazard
HAS_REFERENCE        = FSKXO.FSKXO_0000000004  # hasReference
HAS_VARIABILITY_SUBJ = FSKXO.FSKXO_0000000088  # hasVariabilitySubject (datatype property)

# --- Model Classes (rdf:type) ---
CLASS_PREDICTIVE     = FSKXO.FSKXO_0000018113  # Predictive model (Class)
CLASS_OTHER_EMP      = FSKXO.FSKXO_0000018125  # Other Empirical Model (Class)
CLASS_DOSE_RESPONSE  = FSKXO.FSKXO_0000018205  # Dose-response Model (Class)

# --- Model Category Individuals (for hasModelCategory) ---
CAT_PREDICTIVE       = FSKXO.FSKXO_0000018108  # Predictive model (Individual)
CAT_OTHER_EMP        = FSKXO.FSKXO_0000018107  # Other Empirical models (Individual)

# --- Parameter classification (Individuals) ---
ROLE_INPUT           = FSKXO.FSKXO_0000017481  # Input
ROLE_OUTPUT          = FSKXO.FSKXO_0000017482  # Output
ROLE_CONSTANT        = FSKXO.FSKXO_0000017480  # Constant

# --- Datatypes (Individuals) ---
DT_DOUBLE            = FSKXO.FSKXO_0000017803  # Double
DT_VECTOR_NUMBER     = FSKXO.FSKXO_0000017816  # Vector[number]
DT_INTEGER           = FSKXO.FSKXO_0000017807  # Integer

# --- Unit Categories (Individuals) ---
UC_TEMPERATURE       = FSKXO.FSKXO_0000017795  # Temperature
UC_TIME              = FSKXO.FSKXO_0000017796  # Time

# Custom vocabulary properties (not in FSKXO)
MIN_VALUE = VOCAB.minValue
MAX_VALUE = VOCAB.maxValue
DEFAULT_VALUE = VOCAB.defaultValue
HAS_UNIT = VOCAB.hasUnit # For concrete units, linking to QUDT or FSKXO individuals

# Correct, verified class URIs
MODEL_CLASS_MAP = {
    "predictive model": FSKXO.FSKXO_0000018113,
    "dose-response model": FSKXO.FSKXO_0000018205,
    "process model": FSKXO.FSKXO_0000018131,
    "qra model": FSKXO.FSKXO_0000018214,
    "health metrics model": FSKXO.FSKXO_0000018210,
    "other empirical model": FSKXO.FSKXO_0000018125,   # NOTE: class (singular)
}
# Verified category individuals (instances of Model Class)
MODEL_CATEGORY_INDIVIDUAL = {
    "consumption model": FSKXO.FSKXO_0000018103,
    "dose-response model": FSKXO.FSKXO_0000018104,
    "exposure model": FSKXO.FSKXO_0000018105,
    "health metrics model": FSKXO.FSKXO_0000018106,
    "other empirical models": FSKXO.FSKXO_0000018107,  # NOTE: individual (plural)
    "predictive model": FSKXO.FSKXO_0000018108,
    "process model": FSKXO.FSKXO_0000018109,
    "qra model": FSKXO.FSKXO_0000018110,
    "risk characterization model": FSKXO.FSKXO_0000018111,
    "toxicological reference value model": FSKXO.FSKXO_0000018112,
}

# Hard-coded, stable FSKXO URIs for Classification Individuals
CLASS_MAP = {
    'input':   ROLE_INPUT,
    'output':  ROLE_OUTPUT,
    'constant': ROLE_CONSTANT,
}

# Hard-coded, stable FSKXO URIs for Datatype Individuals
DATATYPE_MAP = {
    "double": DT_DOUBLE,
    "vectorofnumbers": DT_VECTOR_NUMBER,
    "integer": DT_INTEGER,
}

# Hard-coded, stable FSKXO URIs for Unit Category Individuals
UNIT_CATEGORY_MAP = {
    "temperature": UC_TEMPERATURE,
    "time": UC_TIME,
}

MASTER_MAPPING_FILE = "Mapping_Tables/master_mapping.xlsx"

@st.cache_resource
def load_vocab_graph(path: str) -> Graph:
    """
    Loads a Turtle file into an rdflib Graph, correcting malformed NCBI URIs on the fly.
    The vocabulary file uses an invalid CURIE format like `ncbi:wwwtax.cgi?id=...`,
    which this function replaces with the full, valid URI.
    """
    g = Graph()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove the invalid prefix declaration
        content = re.sub(r'@prefix ncbi:.*\n', '', content)
        
        # Replace the invalid CURIEs with full URIs
        content = re.sub(r'ncbi:wwwtax.cgi\?id=', r'<https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=', content)
        content = re.sub(r'(\?id=[0-9]+)\s*\.', r'\1> .', content)


        g.parse(data=content, format="turtle")
    except Exception as e:
        st.error(f"Error parsing vocabulary graph at {path}: {e}")
        # Return an empty graph on error
        return Graph()
    return g

def _txt(v): 
    return str(v) if v is not None else ""

def _asciifold(s: str) -> str:
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")

def _norm(s: str) -> str:
    s = _asciifold(s).lower()
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _norm_bio(s: str) -> str:
    # Looser normalization for biological names: drop spp., sp., subsp., cf., aff.
    s = _norm(s)
    s = re.sub(r"\b(spp?|subsp|ssp|cf|aff)\b\.?", "", s)
    return re.sub(r"\s+", " ", s).strip()

def is_nan_or_empty(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False

def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    if not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)

def param_uri_for(model_uri, param_id):
    """Creates a consistent URI for a model parameter."""
    return URIRef(f"{model_uri}/parameter/{slugify(param_id)}")

def _get_string_from_field(value):
    """Safely extracts a string from a field that could be a string or a dict."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get('name') or value.get('label') or value.get('id') or ''
    return ''

def _get_text(v):
    """Return a best-effort string from string|dict|other."""
    if isinstance(v, str): return v.strip()
    if isinstance(v, dict):
        for k in ("name", "label", "value", "id"):
            if isinstance(v.get(k), str) and v[k].strip():
                return v[k].strip()
    return ""

def _get_display_label(term, alt_labels):
    """Creates a display label from a term and its alt labels, matching the UI."""
    alt_labels_str = str(alt_labels or '').strip()
    if alt_labels_str:
        return f"{term} ({alt_labels_str})"
    return str(term)

@st.cache_data
def build_master_mapping_indexes(df):
    """Builds lookup dictionaries from the master mapping DataFrame for fast matching."""
    term_map = {}       # norm(Term) -> prefLabel
    pref_label_map = {} # norm(prefLabel) -> prefLabel
    alt_label_map = {}  # norm(altLabel) -> prefLabel
    
    for _, row in df.iterrows():
        pref_label = row['prefLabel']
        
        # Index by Term
        term = _norm(row['Term'])
        if term:
            term_map[term] = pref_label
            
        # Index by prefLabel (the part before parentheses)
        base_pref_label = _norm(row['Term']) # 'Term' is the base part of prefLabel
        if base_pref_label:
            pref_label_map[base_pref_label] = pref_label
            
        # Index by altLabels
        alt_labels = str(row.get('altLabels', '')).strip()
        if alt_labels:
            for alt in alt_labels.split(','):
                norm_alt = _norm(alt)
                if norm_alt:
                    alt_label_map[norm_alt] = pref_label
                    
    return term_map, pref_label_map, alt_label_map

def find_best_master_mapping_match(search_term, term_map, pref_label_map, alt_label_map):
    """
    Finds the best match for a search term in the master mapping indexes.
    Returns the matching 'prefLabel' for the UI dropdown.
    """
    if not search_term:
        return None
        
    norm_search = _norm(search_term)
    
    # 1. Exact match on normalized term, prefLabel, or altLabel
    if norm_search in term_map:
        return term_map[norm_search]
    if norm_search in pref_label_map:
        return pref_label_map[norm_search]
    if norm_search in alt_label_map:
        return alt_label_map[norm_search]
        
    # 2. Substring matching (e.g., 'temp' in 'temperature')
    for key, val in pref_label_map.items():
        if norm_search in key:
            return val
    for key, val in alt_label_map.items():
        if norm_search in key:
            return val
            
    return None

def _split_compound(s: str):
    """Split on common separators and trim."""
    parts = re.split(r"[;,/|]+", s)
    return [p.strip() for p in parts if p.strip()]

def choose_class_and_category(raw_meta: dict):
    """
    Determines the model's rdf:type (Class) and its hasModelCategory (Individual)
    based on the 'modelType' and 'modelCategory' fields in the FSKX metadata.
    """
    gi = raw_meta.get('generalInformation') or {}
    
    model_type_val = gi.get('modelType', '')
    model_cat_val = gi.get('modelCategory', '')

    model_type_str = _norm(_get_string_from_field(model_type_val))
    model_cat_str = _norm(_get_string_from_field(model_cat_val))

    # Default to a generic class if nothing specific is found
    class_uri = MODEL_CLASS_MAP.get(model_type_str, FSKXO.FSKXO_0000018113) # Default to Predictive Model Class

    # Find the corresponding individual for the category
    category_ind = MODEL_CATEGORY_INDIVIDUAL.get(model_cat_str)

    # Log if a category was provided but not found
    if model_cat_val and not category_ind:
        st.info(f"Debug: Model category '{_get_string_from_field(model_cat_val)}' normalized to '{model_cat_str}', but no matching individual was found in the lookup table.")

    return class_uri, category_ind

def lit_double(val):
    """Safely creates a Literal(float) if value is convertible."""
    try:
        return Literal(float(val), datatype=XSD.double)
    except (ValueError, TypeError):
        return None

def _find_metadata_member(z: zipfile.ZipFile) -> str | None:
    # Find 'metadata.json' OR 'metaData.json', anywhere, case-insensitive
    candidates = [n for n in z.namelist() if n.lower().endswith("metadata.json")]
    return min(candidates, key=lambda n: n.count("/")) if candidates else None

def _collect_from_container(container, keys):
    """Yield strings from container[key] where value can be str|dict|list."""
    for k in keys:
        v = container.get(k)
        if not v:
            continue
        if isinstance(v, str):
            for p in _split_compound(v): 
                yield p
        elif isinstance(v, dict):
            t = _get_text(v)
            if t:
                for p in _split_compound(t):
                    yield p
        elif isinstance(v, list):
            for item in v:
                t = _get_text(item) if not isinstance(item, str) else item
                if t:
                    for p in _split_compound(t):
                        yield p

def extract_hazards_products_from_fskx(fskx_fileobj):
    """
    fskx_fileobj: a file-like object (Streamlit UploadedFile or open(...) handle)
    Returns:
      hazards_raw:   list[str]
      hazards_norm:  list[str]  # _norm_bio
      products_raw:  list[str]
      products_norm: list[str]
    """
    hazards, products = [], []
    try:
        with zipfile.ZipFile(fskx_fileobj, "r") as z:
            member = _find_metadata_member(z)
            if not member:
                return [], [], [], []
            with z.open(member) as f:
                meta = json.load(f)
    except Exception:
        return [], [], [], []

    scope = meta.get("scope") or {}
    gi    = meta.get("generalInformation") or {}

    # 1) scope.* (singular/plural/casing already handled by explicit key lists)
    hazards += list(_collect_from_container(scope, ["hazard", "hazards", "Hazard", "Hazards"]))
    products += list(_collect_from_container(scope, ["product", "products", "Product", "Products"]))

    # 2) fallbacks in generalInformation
    hazards += list(_collect_from_container(gi, ["hazardName", "targetOrganism", "biologicalAgent"]))
    products += list(_collect_from_container(gi, ["productName", "commodity", "matrix"]))

    # Deduplicate but keep order
    def _dedupe(seq):
        seen = set(); out = []
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    hazards  = _dedupe(hazards)
    products = _dedupe(products)

    hazards_norm  = [_norm_bio(h) for h in hazards]
    products_norm = [_norm_bio(p) for p in products]

    return hazards, hazards_norm, products, products_norm

def extract_fskx_data(uploaded_file_object, file_name: str):
    """
    Extracts model metadata and parameters from an uploaded .fskx (zip) file.
    :param uploaded_file_object: A file-like object (e.g., BytesIO).
    :param file_name: The original name of the uploaded file.
    """
    model_metadata = {}
    parameters = []
    mappable_term_entries = []
    extracted_units = []

    try:
        with zipfile.ZipFile(uploaded_file_object, 'r') as zip_ref:
            metadata_filename = _find_metadata_member(zip_ref)
            if not metadata_filename:
                st.error(f"Could not find metaData.json / metadata.json in {file_name}")
                return None, None, None, None

            with zip_ref.open(metadata_filename) as meta_file:
                fskx_content = json.load(meta_file)

            # --- Metadata Extraction ---
            model_metadata['__raw_meta__'] = fskx_content

            # UUID: prefer filename UUID, then content id/identifier
            uuid_match = re.search(
                r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})',
                file_name,
                re.I
            )
            model_metadata['uuid'] = uuid_match.group(1) if uuid_match else (
                fskx_content.get('id')
                or (fskx_content.get('generalInformation') or {}).get('identifier')
                or f"uuid-not-found-{file_name}"
            )

            gi = (fskx_content.get('generalInformation') or {})
            model_metadata['name']        = gi.get('name', 'Untitled Model')
            model_metadata['description'] = gi.get('description', '')
            model_metadata['keywords']    = gi.get('keywords', [])

            # Created date: accept several variants
            model_metadata['created_date'] = (
                gi.get('creationDate')                                      # often [YYYY,MM,DD] or string
                or (gi.get('created') or {}).get('date')                    # { "date": "YYYY-MM-DD" }
                or gi.get('created')                                        # sometimes just a string
            )

            # --- Parameters ---
            model_math = fskx_content.get('modelMath') or {}
            params = (model_math.get('parameter') or model_math.get('parameters') or [])
            if not isinstance(params, list):
                params = [params]

            parameters = []
            for p in params:
                # normalize unit (can be str/list/dict)
                unit_val = p.get('unit')
                if isinstance(unit_val, list):
                    unit_val = ", ".join(str(u) for u in unit_val)
                elif isinstance(unit_val, dict):
                    unit_val = unit_val.get('id') or unit_val.get('name') or unit_val.get('label')

                # normalize datatype field
                p_dt = p.get('dataType') or p.get('datatype')

                # pick value/defaultValue
                p_val = p.get('value')
                if p_val is None:
                    p_val = p.get('defaultValue')

                parameters.append({
                    'id':          p.get('id'),
                    'name':        p.get('name'),
                    'description': p.get('description'),
                    'classification': p.get('classification'),
                    'unit':        unit_val,
                    'unitCategory': p.get('unitCategory'),
                    'datatype':    p_dt,
                    'minValue':    p.get('minValue'),
                    'maxValue':    p.get('maxValue'),
                    'value':       p_val,
                })

                # mappable terms for parameters
                pid = p.get('id')
                if pid and _norm(p.get('classification')) != 'constant':
                    pref = p.get('name') or pid
                    mappable_term_entries.append({
                        'Term': pid,
                        'ConceptGroup': 'Model Parameters',
                        'prefLabel': pref,
                        'altLabels': '',
                        'ProviderDescription': p.get('description', '')
                    })

                # collect units for UI
                if unit_val and str(unit_val).strip().lower() not in {'no unit','dimensionless','unitless','none'}:
                    uid = str(unit_val).strip()
                    if not any(u['Unit ID'] == uid for u in extracted_units):
                        extracted_units.append({'Unit ID': uid, 'Unit Name': uid})


    except Exception as e:
        st.error(f"Error processing file {file_name}: {e}")
        return None, None, None, None

    return model_metadata, parameters, mappable_term_entries, extracted_units

@st.cache_data
def load_master_mapping(file_path):
    """
    Loads the master mapping Excel file into a DataFrame.
    Handles new columns for wiring.
    """
    # Define all expected columns, including new ones
    expected_cols = [
        'Term', 'ConceptGroup', 'prefLabel', 'altLabels', 'ProviderDescription',
        'URI1', 'Match Type1', 'SourceProvider1',
        'URI2', 'Match Type2', 'SourceProvider2',
        'URI3', 'Match Type3', 'SourceProvider3',
        'QuantityKind_URI', 'FulfillsBy_URI', 'SourceVariableName' # New columns
    ]
    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        df.columns = [col.strip() for col in df.columns]
        
        for col in expected_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('')
            else:
                # Add missing columns so the app doesn't break
                df[col] = ''
                
        return df
    except FileNotFoundError:
        st.warning(f"Master mapping file not found at '{file_path}'. Creating an empty mapping table.")
        return pd.DataFrame(columns=expected_cols)
    except Exception as e:
        st.error(f"Error loading master mapping file: {e}")
        return pd.DataFrame()

def create_new_terms_table(all_mappable_terms, master_df):
    """
    Compares extracted terms with the master mapping and returns an Excel file
    in memory for unmapped terms.
    """
    if master_df.empty:
        # If master is empty, all extracted terms are new
        new_terms_df = pd.DataFrame(all_mappable_terms)
    else:
        # Create a unique key for matching
        master_df['lookup_key'] = master_df['Term'].str.lower() + "::" + master_df['ConceptGroup'].str.lower()
        
        unmapped_terms = []
        for term in all_mappable_terms:
            lookup_key = str(term.get('Term', '')).lower() + "::" + str(term.get('ConceptGroup', '')).lower()
            if lookup_key not in master_df['lookup_key'].values:
                unmapped_terms.append(term)
        
        if not unmapped_terms:
            return None
        
        new_terms_df = pd.DataFrame(unmapped_terms)

    # Prepare DataFrame for Excel output
    output_df = new_terms_df[['Term', 'ConceptGroup', 'prefLabel', 'altLabels', 'ProviderDescription']].copy()
    output_df['URI1'] = ''
    output_df['Match Type1'] = 'exactMatch'
    output_df['SourceProvider1'] = ''
    output_df['ProviderDescription1'] = ''
    output_df['URI2'] = ''
    output_df['Match Type2'] = ''
    output_df['SourceProvider2'] = ''
    output_df['ProviderDescription2'] = ''
    output_df['URI3'] = ''
    output_df['Match Type3'] = ''
    output_df['SourceProvider3'] = ''
    output_df['ProviderDescription3'] = ''

    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        output_df.to_excel(writer, index=False, sheet_name='New Terms')
    return output.getvalue()

def _build_vocab_struct_impl(_g: Graph):
    """
    Returns:
      uris, display_by_uri, pref_norm_index, alt_norm_index
    Now supports entries with *only* rdfs:label.
    """
    uris = []
    display_by_uri = {}
    pref_norm_index = {}
    alt_norm_index  = {}

    # UNION of subjects that have prefLabel OR label
    subjects = set(_g.subjects(SKOS.prefLabel, None)) | set(_g.subjects(RDFS.label, None))

    for s in subjects:
        uri = str(s)
        # Prefer skos:prefLabel; fall back to rdfs:label
        prefs = [ _txt(l) for l in _g.objects(s, SKOS.prefLabel) ]
        if not prefs:
            prefs = [ _txt(l) for l in _g.objects(s, RDFS.label) ]
        pref = prefs[0] if prefs else None
        if not pref:
            continue

        # alt labels (if any)
        alts = [ _txt(a) for a in _g.objects(s, SKOS.altLabel) ]
        alts = [a for a in alts if a and a != pref]

        # Display text: "pref (alt1; alt2)" (up to 2 alts)
        display = f"{pref} ({'; '.join(alts[:2])})" if alts else pref

        # collect
        uris.append(uri)
        display_by_uri[uri] = display

        # **Bio-safe normalization** for matching
        pn = _norm_bio(pref)
        pref_norm_index.setdefault(pn, []).append(uri)
        for a in alts:
            an = _norm_bio(a)
            alt_norm_index.setdefault(an, []).append(uri)

    uris.sort(key=lambda u: display_by_uri[u].lower())
    return uris, display_by_uri, pref_norm_index, alt_norm_index

@st.cache_data(show_spinner=False)
def build_vocab_struct_for_path(path: str):
    g = load_vocab_graph(path)
    return _build_vocab_struct_impl(g)

def find_best_uri_matches(name: str, uris, display_by_uri, pref_idx, alt_idx, max_hits=3, fuzzy_threshold=0.72):
    target = _norm_bio(name)
    if not target:
        return []

    # exact on pref / alt
    exact = pref_idx.get(target, []) + alt_idx.get(target, [])
    if exact:
        return list(dict.fromkeys(exact))[:max_hits]

    # token-contains: target tokens subset of candidate pref/alt
    toks = set(target.split())
    contains = []
    for key, lst in pref_idx.items():
        if toks.issubset(set(key.split())):
            contains.extend(lst)
    for key, lst in alt_idx.items():
        if toks.issubset(set(key.split())):
            contains.extend(lst)
    if contains:
        return list(dict.fromkeys(contains))[:max_hits]

    # plain substring contains (after bio-normalization)
    contains2 = [u for k,lst in pref_idx.items() if target in k for u in lst]
    contains2 += [u for k,lst in alt_idx.items() if target in k for u in lst]
    if contains2:
        return list(dict.fromkeys(contains2))[:max_hits]

    # fuzzy on the human display strings
    labels = {u: display_by_uri[u] for u in uris}
    candidates = difflib.get_close_matches(name, list(labels.values()), n=max_hits, cutoff=fuzzy_threshold)
    rev = {v: k for k, v in labels.items()}
    return [rev[c] for c in candidates if c in rev]

def create_mapping_ui(map_type: str, model_key: str, names_from_fskx: list,
                      uris, display_by_uri, pref_idx, alt_idx):
    """
    map_type: 'Hazard' or 'Product'
    names_from_fskx: list of raw names extracted from the FSKX file.
    uris: list[str]         # URI strings
    display_by_uri: dict    # uri -> display text
    *_idx: indexes from build_vocab_struct
    """
    st.markdown(f"**Map {map_type}(s)**")
    state_key = f"{map_type.lower()}_uris_{model_key}"
    st.session_state.setdefault(state_key, [])

    # Use URIs as values; show nice labels via format_func
    selections = st.multiselect(
        f"{map_type}(s)",
        options=uris,
        default=st.session_state[state_key],
        format_func=lambda u: display_by_uri.get(u, u),
        key=f"{map_type.lower()}_ms_{model_key}",
    )
    st.session_state[state_key] = selections
    return selections  # list[str] of URIs

def index_master_units(master_mapping_df):
    units_df = master_mapping_df[master_mapping_df['ConceptGroup'].str.lower().isin(['unit','units','concept'])].copy()
    units_df['Term_norm'] = units_df['Term'].str.lower()
    alt_map = {}
    for _, row in units_df.iterrows():
        # split altLabels by comma
        for a in (row.get('altLabels') or '').split(','):
            a = a.strip().lower()
            if a:
                alt_map[a] = row['Term']
    primary_map = dict(zip(units_df['Term_norm'], units_df['Term']))
    return primary_map, alt_map  # norm(term)->Term, alt->Term

def prefill_units_for_model(data_entry, master_mapping_df):
    prim, alt = index_master_units(master_mapping_df)
    unit_mappings = st.session_state.setdefault('unit_mappings', {})
    for p in data_entry['parameters']:
        unit = (p.get('unit') or '').strip()
        if not unit or unit == 'no unit':
            continue
        key = unit   # you use the raw unit string as the unit_id in the UI
        cand = prim.get(unit.lower()) or alt.get(unit.lower())
        # minimal heuristics
        if not cand and any(t in unit.lower() for t in ['°c', 'deg c', 'celsius']):
            cand = prim.get('deg_c') or prim.get('°c') or prim.get('degree celsius')
        if not cand and unit.lower() in ['h', 'hour', 'hours']:
            cand = prim.get('hour') or alt.get('hour(s)')
        if cand:
            unit_mappings[key] = cand

def _to_iso_date(d):
    if isinstance(d, list) and len(d) == 3:
        y, m, day = d
        try:
            return f"{int(y):04d}-{int(m):02d}-{int(day):02d}"
        except Exception:
            return None
    if isinstance(d, str) and d.strip():
        return d.strip()
    return None

def _bind_common_namespaces(g: Graph):
    """Binds all common namespaces to a graph."""
    g.bind("fskxo", FSKXO)
    g.bind("model", MODEL)
    g.bind("vocab", VOCAB)
    g.bind("dct", DCTERMS)
    g.bind("skos", SKOS)
    g.bind("xsd", XSD)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("qudt-unit", QUDT_UNIT)
    g.bind("amblink", AMBLINK)
    g.bind("netcdf", Namespace("https://www.ambrosia-project.eu/vocab/concept/netcdf/"))
    g.bind("qk", Namespace("http://qudt.org/vocab/quantitykind/"))


def all_model_uris_from_graph(g: Graph):
    # Prefer subjects that have parameters (most robust)
    uris = set(g.subjects(HAS_PARAMETER, None))
    if not uris:
        # Fallback: any subject typed as a known model class
        for cls in MODEL_CLASS_MAP.values():
            uris |= set(g.subjects(RDF.type, cls))
    return sorted(uris, key=str)


def _split_prefixes_and_body(ttl_text: str):
    prefixes = []
    body = []
    for ln in ttl_text.splitlines():
        if ln.startswith("@prefix "):
            prefixes.append(ln.rstrip())
        else:
            body.append(ln)
    return prefixes, "\n".join(body).strip()

def get_label_for_uri(g: Graph, uri: URIRef) -> str:
    if isinstance(uri, Literal):
        return ""
    # Try explicit labels
    label = next(g.objects(uri, SKOS.prefLabel), None) or next(g.objects(uri, RDFS.label), None)
    if label:
        return str(label)

    u = str(uri)
    # Try CURIE; if rdflib made a synthetic prefix (ns\d+:), prefer the fragment
    try:
        q = g.namespace_manager.qname(uri)
        if re.match(r"^ns\d+:", q):
            frag = re.split(r"[#/]", u)[-1]
            return frag.replace("_", " ")
        return q
    except Exception:
        frag = re.split(r"[#/]", u)[-1]
        return frag.replace("_", " ")

_SUBJECT_LINE = re.compile(r'^\s*(?:<[^>]+>|[A-Za-z0-9_-]+:[A-Za-z0-9._-]+)\s+')
_TYPE_PATTERN = re.compile(r'\s+a\s+([A-Za-z0-9_-]+:[A-Za-z0-9._-]+)')

def _label_for_curie(curie: str, g: Graph) -> str:
    try:
        uri = g.namespace_manager.expand_curie(curie)
    except Exception:
        return curie
    lbl = next(g.objects(uri, SKOS.prefLabel), None) or next(g.objects(uri, RDFS.label), None)
    if lbl: return str(lbl)
    u = str(uri); return u.rsplit('/',1)[-1].rsplit('#',1)[-1] or curie

def serialize_skos_graph_with_comments(graph: Graph, model_uris: list = None) -> str:
    """
    Serialize one block per model, unioning used prefixes and adding inline comments.
    """
    if not model_uris:
        model_uris = all_model_uris_from_graph(graph)
    blocks = []
    all_used_prefixes = set()

    # 1. Serialize each model block to find out which prefixes it uses
    for model_uri in sorted(model_uris, key=lambda u: str(u)):
        if (model_uri, None, None) not in graph:
            continue

        # Create a temporary graph for just this model and its parameters
        block_graph = Graph()
        _bind_common_namespaces(block_graph)
        for p, o in graph.predicate_objects(model_uri):
            block_graph.add((model_uri, p, o))
        for param_uri in graph.objects(model_uri, HAS_PARAMETER):
            for p, o in graph.predicate_objects(param_uri):
                block_graph.add((param_uri, p, o))

        # Serialize to a string to get the raw turtle text for this block
        block_ttl = block_graph.serialize(format="turtle")
        if isinstance(block_ttl, bytes):
            block_ttl = block_ttl.decode("utf-8")

        # Extract prefixes used in this block
        block_prefixes, body = _split_prefixes_and_body(block_ttl)
        all_used_prefixes.update(p.split()[1] for p in block_prefixes) # "skos:"
        blocks.append((model_uri, body))

    # 2. Bind all namespaces to the main graph to build a complete prefix map
    _bind_common_namespaces(graph)
    ns_map = {prefix: str(url) for prefix, url in graph.namespace_manager.namespaces()}

    # 3. Build the final output string
    output = []
    # Add only the prefixes that were actually used across all blocks
    sorted_prefixes = sorted(list(all_used_prefixes))
    
    # Reconstruct the full @prefix lines from the complete namespace map
    final_prefix_lines = []
    for prefix_token, url in ns_map.items():
        if f"{prefix_token}:" in all_used_prefixes:
            final_prefix_lines.append(f"@prefix {prefix_token}: <{url}> .")
    
    output.extend(sorted(final_prefix_lines))
    output.append("")

    # 4. Add the global class definition for vocab:ModelParameter
    if (VOCAB.ModelParameter, RDF.type, RDFS.Class) in graph:
        output.append("vocab:ModelParameter a rdfs:Class ;")
        output.append('    rdfs:label "Model Parameter"@en .')
        output.append("")

    # 5. Process each block, adding headers and comments
    for model_uri, body in blocks:
        model_title = next(graph.objects(model_uri, DCTERMS.title), "")
        header_text = f" model {str(model_uri).split('/')[-1]} ({model_title}) "
        total_width = 80
        header = "\n".join([
            "#" * total_width,
            f"#{header_text:#^{total_width-2}}#",
            "#" * total_width,
        ])
        output.append(header)

        # Add comments to each line of the body, following specific formatting rules
        commented_body_lines = []
        current_pred_label = ""  # track last predicate label

        for raw in body.splitlines():
            line = raw.rstrip()
            sline = line.lstrip()

            if not sline or sline.startswith("@"):
                commented_body_lines.append(line)
                continue

            # SUBJECT line: (no leading indent)
            if len(line) == len(sline):
                current_pred_label = ""
                m = _TYPE_PATTERN.search(sline)
                if m:
                    obj_curie = m.group(1)
                    obj_label = _label_for_curie(obj_curie, graph)
                    if obj_label:
                        pad = " " * (50 - len(line)) if 50 > len(line) else "  "
                        commented_body_lines.append(f"{line}{pad}# {obj_label}")
                        continue
                commented_body_lines.append(line)
                continue

            # OBJECT CONTINUATION: line begins with a URI (typically "<...>")
            if sline.startswith("<"):
                obj_tok = sline.split()[0].rstrip(";,")
                obj_label = get_label_for_uri(graph, URIRef(obj_tok.strip("<>")))
                comment = f"{current_pred_label} -> {obj_label}" if (current_pred_label and obj_label) else (obj_label or "")
                if comment:
                    pad = " " * (50 - len(line)) if 50 > len(line) else "  "
                    commented_body_lines.append(f"{line}{pad}# {comment}")
                else:
                    commented_body_lines.append(line)
                continue

            # PREDICATE line (indented and starts with a CURIE, not a "<...>")
            first = sline.split()[0]
            if ":" in first and not first.startswith("<"):
                pred_label = _label_for_curie(first, graph)
                current_pred_label = pred_label or first

                # Try to label the first object on this line
                obj_label = ""
                parts = sline.split()
                if len(parts) > 1:
                    obj_tok = parts[1].rstrip(";,")
                    if obj_tok.startswith("<"):
                        obj_label = get_label_for_uri(graph, URIRef(obj_tok.strip("<>")))
                    elif ":" in obj_tok and not obj_tok.startswith(('"', "'")):
                        obj_label = _label_for_curie(obj_tok, graph)

                comment = f"{current_pred_label} -> {obj_label}" if obj_label else current_pred_label
                pad = " " * (50 - len(line)) if 50 > len(line) else "  "
                commented_body_lines.append(f"{line}{pad}# {comment}")
                continue

            # Fallback
            commented_body_lines.append(line)

        output.append("\n".join(commented_body_lines))
        output.append("")

    return "\n".join(output).rstrip() + "\n"

def build_label_graph_from_master(master_df: pd.DataFrame, only_ns: str | None = FSKXO):
    """
    Create a tiny graph of rdfs:label triples for any URIs listed in the master
    mapping. By default, only labels FSKXO URIs so we get nice names for
    fskxo:FSKXO_0000017611 etc. Pass only_ns=None to label *all* URIs.
    """
    g = Graph()
    _bind_common_namespaces(g)

    # Columns that may contain URIs
    uri_cols = ['URI1', 'URI2', 'URI3']

    # Use the authoritative human name from the 'Term' column
    # (not the UI 'prefLabel' which may include "(alt1; alt2)")
    def _should_include(u: str) -> bool:
        if not u or is_nan_or_empty(u):
            return False
        if only_ns is None:
            return True
        return str(u).startswith(str(only_ns))

    for _, row in master_df.iterrows():
        term = str(row.get('Term', '')).strip()
        if not term:
            continue
        for col in uri_cols:
            uri = str(row.get(col, '')).strip()
            if _should_include(uri):
                try:
                    g.add((URIRef(uri), RDFS.label, Literal(term, lang='en')))
                except Exception:
                    # Skip malformed URIs silently
                    pass
    return g

def serialize_wiring_graph_custom(graph: Graph, model_uris: list = None) -> str:
    if not model_uris:
        model_uris = all_model_uris_from_graph(graph)
    blocks = []
    used_prefix_lines = []
    seen_prefixes = set()

    for model_uri in sorted(model_uris, key=str):
        if (model_uri, None, None) not in graph:
            continue

        # header
        model_title = next(graph.objects(model_uri, DCTERMS.title), "")
        header_text = f" model {str(model_uri).split('/')[-1]} ({model_title}) "
        total_width = 80
        header = "\n".join([
            "#" * total_width,
            f"#{header_text:#^{total_width-2}}#",
            "#" * total_width,
            "",
        ])

        # collect one model’s wiring (hasInputMapping blank nodes)
        block = Graph()
        _bind_common_namespaces(block)

        for bn in graph.objects(model_uri, AMBLINK.hasInputMapping):
            block.add((model_uri, AMBLINK.hasInputMapping, bn))
            # copy the bnode subtree
            for p, o in graph.predicate_objects(bn):
                block.add((bn, p, o))

        ttl = block.serialize(format="turtle")
        if isinstance(ttl, bytes):
            ttl = ttl.decode("utf-8")
        prefixes, body = _split_prefixes_and_body(ttl)
        if body.strip():  # only add non-empty wiring blocks
            blocks.append((header, body))
            for pl in prefixes:
                token = pl.split()[1] if " " in pl else pl
                if token not in seen_prefixes:
                    seen_prefixes.add(token)
                    used_prefix_lines.append(pl)

    out = []
    out.extend(used_prefix_lines)
    out.append("")
    for header, body in blocks:
        out.append(header)
        out.append(body)
        out.append("")
    return "\n".join(out).rstrip() + "\n"

def generate_rdf_outputs(model_data, original_parameters, master_mapping_df, knowledge_graph,
                         hazards_sel=None, products_sel=None):
    """
    Generates two RDF graphs based on user mappings and automatic wiring discovery.
    1. g_skos: The main SKOS/FSKXO description of the model and its parameters.
    2. g_wiring: The wiring triples that link input parameters to data sources.
    """
    g_skos = Graph()
    g_wiring = Graph()
    wiring_log = []

    # --- Unit Mapping Preparation ---
    unit_mappings = st.session_state.get('unit_mappings', {})
    unit_uri_map = {}
    units_df = master_mapping_df[master_mapping_df['ConceptGroup'].isin(['Unit', 'Concept'])]
    for _, row in units_df.iterrows():
        term = row['Term']
        uri = next((row[col] for col in ['URI1', 'URI2', 'URI3'] if row[col] and not is_nan_or_empty(row[col])), None)
        if uri:
            unit_uri_map[term] = uri

    # Bind namespaces for both graphs
    for g in [g_skos, g_wiring]:
        _bind_common_namespaces(g)

    g_skos.add((VOCAB.ModelParameter, RDF.type, RDFS.Class))
    g_skos.add((VOCAB.ModelParameter, RDFS.label, Literal("Model Parameter", lang="en")))

    fskx_uuid = model_data.get('uuid')
    if not fskx_uuid:
        st.error("Model UUID not found. Cannot generate unique model URI.")
        return Graph(), Graph(), []

    model_uri = MODEL[fskx_uuid]
    g_skos.add((model_uri, DCTERMS.identifier, Literal(fskx_uuid)))

    raw_meta = model_data.get('__raw_meta__', {})
    class_uri, category_ind = choose_class_and_category(raw_meta)
    g_skos.add((model_uri, RDF.type, class_uri))
    if not category_ind:
        # fallback to class-aligned category where it exists
        fallback = {
          FSKXO.FSKXO_0000018113: FSKXO.FSKXO_0000018108,  # Predictive model
          FSKXO.FSKXO_0000018125: FSKXO.FSKXO_0000018107,  # Other Empirical models
          # add others as needed
        }.get(class_uri)
        category_ind = category_ind or fallback
    if category_ind:
        g_skos.add((model_uri, HAS_MODEL_CATEGORY, category_ind))

    if hazards_sel:
        for uri in hazards_sel:
            g_skos.add((model_uri, HAS_HAZARD, URIRef(uri)))
    if products_sel:
        for uri in products_sel:
            g_skos.add((model_uri, HAS_PRODUCT, URIRef(uri)))

    if model_data.get('name'):
        g_skos.add((model_uri, DCTERMS.title, Literal(model_data['name'], lang='en')))
    if model_data.get('description'):
        g_skos.add((model_uri, DCTERMS.description, Literal(model_data['description'], lang='en')))

    # ... (creator, keywords, equation, created_date logic remains the same)
    creators = []
    for person in ( (raw_meta.get('generalInformation') or {}).get('creator') or [] ):
        name = " ".join([person.get('givenName',''), person.get('familyName','')]).strip()
        if name and "AI Assistant" not in name:
            creators.append(name)
    
    # Deduplicate and add creators
    for name in sorted(list(set(creators))):
        g_skos.add((model_uri, DCTERMS.creator, Literal(name)))
    if model_data.get('keywords'):
        kw = model_data['keywords']
        if isinstance(kw, (list, tuple)):
            for k in kw:
                g_skos.add((model_uri, DCTERMS.subject, Literal(str(k))))
        else:
            g_skos.add((model_uri, DCTERMS.subject, Literal(str(kw))))
    mm = (raw_meta.get('modelMath') or {})
    eq = mm.get('modelEquation')
    eq_text = (eq[0].get('modelEquation') if isinstance(eq, list) and eq else eq) if isinstance(eq, (list, dict)) else eq
    if isinstance(eq, dict) and eq.get('modelEquation'):
        eq_text = eq['modelEquation']
    if isinstance(eq_text, str) and eq_text.strip():
        g_skos.add((model_uri, HAS_MODEL_EQUATION, Literal(eq_text)))
    iso = _to_iso_date(model_data.get('created_date'))
    if iso:
        g_skos.add((model_uri, DCTERMS.created, Literal(iso, datatype=XSD.date)))


    original_params_lookup = {p['id'].lower(): p for p in original_parameters if p.get('id')}
    parameter_uris_for_model = []

    # Get the parameter mappings for this specific model from session state
    param_mappings = st.session_state.get('parameter_mappings', {}).get(fskx_uuid, {})

    for original_param in original_parameters:
        param_id = original_param.get('id')
        if not param_id:
            continue

        param_uri = param_uri_for(model_uri, param_id)
        parameter_uris_for_model.append(param_uri)
        g_skos.add((param_uri, RDF.type, VOCAB.ModelParameter))

        # Use original name/id for the label
        original_label = original_param.get('name') or original_param.get('id')
        g_skos.add((param_uri, RDFS.label, Literal(original_label, lang='en')))

        # Look up the mapped concept label
        mapped_concept_label = param_mappings.get(param_id)
        
        # If mapped, add a mapping relation to the concept's URI
        if mapped_concept_label:
            mapping_row = master_mapping_df[
                (master_mapping_df['prefLabel'] == mapped_concept_label) &
                (master_mapping_df['ConceptGroup'].isin(['Model Parameters', 'Parameter']))
            ]
            if not mapping_row.empty:
                # Add skos:definition from ProviderDescription
                desc = mapping_row.iloc[0].get('ProviderDescription', '')
                if desc and not is_nan_or_empty(desc):
                    g_skos.add((param_uri, SKOS.definition, Literal(desc, lang='en')))

                uris = [mapping_row.iloc[0][c] for c in ('URI1','URI2','URI3') if mapping_row.iloc[0][c]]
                qk_uri   = next((u for u in uris if 'qudt.org/vocab/quantitykind' in u), None)
                unit_uri = next((u for u in uris if 'qudt.org/vocab/unit'         in u), None)
                concept  = next((u for u in uris if all(s not in u for s in ('qudt.org/vocab/quantitykind','qudt.org/vocab/unit'))), None)

                if concept:
                    g_skos.add((param_uri, SKOS.exactMatch, URIRef(concept)))
                if qk_uri:
                    g_skos.add((param_uri, AMBLINK.expectsQuantityKind, URIRef(qk_uri)))

        # Add FSKXO properties (classification, datatype, etc.)
        if original_param.get('classification'):
            role_uri = CLASS_MAP.get(_norm(original_param['classification']))
            if role_uri:
                g_skos.add((param_uri, HAS_CLASSIFICATION, role_uri))

        # --- AUTOMATIC WIRING LOGIC ---
        is_input = _norm(original_param.get('classification')) == 'input'
        if is_input and mapped_concept_label:
            # 1. Look up concept in master mapping to get QuantityKind_URI
            mapping_row = master_mapping_df[
                (master_mapping_df['prefLabel'] == mapped_concept_label) &
                (master_mapping_df['ConceptGroup'].isin(['Model Parameters', 'Parameter']))
            ]
            if not mapping_row.empty:
                # Find the first URI that is a QuantityKind URI
                quantity_kind_uri = None
                for uri_col in ['URI1', 'URI2', 'URI3']:
                    uri = mapping_row.iloc[0][uri_col]
                    if uri and 'qudt.org/vocab/quantitykind' in uri:
                        quantity_kind_uri = uri
                        break
                if quantity_kind_uri and not is_nan_or_empty(quantity_kind_uri):
                    wiring_log.append(f"INFO: Parameter '{param_id}' mapped to '{mapped_concept_label}', expects QuantityKind <{quantity_kind_uri}>.")

                    # 2. SPARQL query to find a data source with matching QuantityKind
                    query = """
                        SELECT ?dataSource ?sourceVarName WHERE {
                            ?dataSource amblink:expectsQuantityKind ?qk .
                            OPTIONAL { ?dataSource amblink:sourceVariableName ?sourceVarName . }
                        }
                    """
                    results = knowledge_graph.query(
                        query,
                        initBindings={'qk': URIRef(quantity_kind_uri)},
                        initNs = { "amblink": AMBLINK }
                    )

                    if len(results) > 0:
                        # 3. Generate wiring triples
                        # For simplicity, we take the first match. Real-world might need disambiguation.
                        first_result = list(results)[0]
                        data_source_uri = first_result.dataSource
                        source_var_name = first_result.sourceVarName or slugify(str(data_source_uri).split('/')[-1])

                        input_mapping_node = BNode()
                        g_wiring.add((model_uri, AMBLINK.hasInputMapping, input_mapping_node))
                        g_wiring.add((input_mapping_node, RDF.type, AMBLINK.InputMapping))
                        g_wiring.add((input_mapping_node, AMBLINK.mapsParameter, param_uri))
                        g_wiring.add((input_mapping_node, AMBLINK.isFulfilledBy, data_source_uri))
                        g_wiring.add((input_mapping_node, AMBLINK.sourceVariableName, Literal(source_var_name)))
                        
                        wiring_log.append(f"SUCCESS: Automatically wired '{param_id}' to data source <{data_source_uri}>.")
                    else:
                        wiring_log.append(f"WARNING: No data source found in the knowledge graph with QuantityKind <{quantity_kind_uri}> for parameter '{param_id}'.")
                else:
                    wiring_log.append(f"INFO: Parameter '{param_id}' mapped to '{mapped_concept_label}', but no QuantityKind_URI was found in the master mapping.")
            else:
                wiring_log.append(f"ERROR: Could not find the mapped concept '{mapped_concept_label}' in the master mapping table.")


        # Add other FSKXO properties and custom vocab properties
        if original_param:
            dt = original_param.get('datatype')
            if dt:
                dt_uri = DATATYPE_MAP.get(_norm(dt))
                if dt_uri:
                    g_skos.add((param_uri, HAS_DATATYPE, dt_uri))
            
            uc = original_param.get('unitCategory')
            if uc:
                uc_uri = UNIT_CATEGORY_MAP.get(_norm(uc))
                if uc_uri:
                    g_skos.add((param_uri, HAS_UNIT_CATEGORY, uc_uri))

            unit_val = original_param.get('unit')
            if unit_val:
                # Look up the unit in the user-provided mappings from the UI
                mapped_unit_term = unit_mappings.get(unit_val.strip())
                if mapped_unit_term:
                    # Get the final URI from the master mapping lookup
                    unit_uri = unit_uri_map.get(mapped_unit_term)
                    if unit_uri:
                        g_skos.add((param_uri, HAS_UNIT, URIRef(unit_uri)))

            min_val_lit = lit_double(original_param.get('minValue'))
            if min_val_lit:
                g_skos.add((param_uri, MIN_VALUE, min_val_lit))

            max_val_lit = lit_double(original_param.get('maxValue'))
            if max_val_lit:
                g_skos.add((param_uri, MAX_VALUE, max_val_lit))

            default_val_lit = lit_double(original_param.get('value'))
            if default_val_lit:
                g_skos.add((param_uri, DEFAULT_VALUE, default_val_lit))

    # Link all parameters to the model after the loop
    if parameter_uris_for_model:
        for param_uri in parameter_uris_for_model:
            g_skos.add((model_uri, HAS_PARAMETER, param_uri))

    return g_skos, g_wiring, wiring_log

def main():
    # This must be the first Streamlit command
    st.set_page_config(layout="wide", page_title="FSKX to RDF Generator")
    st.title("FSKX to RDF Generator")

    # Load master mapping once
    master_mapping_df = load_master_mapping(MASTER_MAPPING_FILE)

    # Create a display label for dropdowns, combining Term and altLabels.
    master_mapping_df['prefLabel'] = master_mapping_df.apply(
        lambda row: _get_display_label(row['Term'], row['altLabels']),
        axis=1
    )


    # --- Vocab paths (editable) ---
    st.sidebar.header("Vocab paths")
    default_pathogen = str(Path(__file__).parent / "../Vocabulary/ambrosia-pathogen-vocab.ttl")
    default_plant    = str(Path(__file__).parent / "../Vocabulary/ambrosia-plant-vocab.ttl")
    default_netcdf   = str(Path(__file__).parent / "../Vocabulary/ambrosia-netcdf-vocab.ttl")


    PATHOGEN_VOCAB = st.sidebar.text_input("Pathogen vocab (.ttl)", default_pathogen)
    PLANT_VOCAB    = st.sidebar.text_input("Plant / Product vocab (.ttl)", default_plant)
    NETCDF_VOCAB   = st.sidebar.text_input("NetCDF Data Source vocab (.ttl)", default_netcdf)


    colp1, colp2 = st.sidebar.columns(2)
    with colp1:
        if st.button("Reload vocabs"):
            load_vocab_graph.clear()
            build_vocab_struct_for_path.clear()
            st.rerun()
    with colp2:
        st.caption("Change paths, then click reload.")

    if st.sidebar.button("Reset all mappings"):
        for k in ("unit_mappings", "parameter_mappings"):
            st.session_state.pop(k, None)
        st.rerun()

    # --- Load and index vocabs ---
    (pathogen_uris, pathogen_display, pathogen_pref_idx, pathogen_alt_idx) = build_vocab_struct_for_path(PATHOGEN_VOCAB)
    (plant_uris,    plant_display,    plant_pref_idx,    plant_alt_idx)    = build_vocab_struct_for_path(PLANT_VOCAB)

    st.caption(f"Hazard vocab loaded: {len(pathogen_uris)} concepts")
    st.caption(f"Product vocab loaded: {len(plant_uris)} concepts")

    # Quick peek to ensure they look different
    def _peek_labels(display_by_uri, n=5):
        return [display_by_uri[u] for u in list(display_by_uri.keys())[:n]]


    # Warn if they are (almost) the same list (wrong path, copy/paste, etc.)
    if pathogen_uris and plant_uris:
        overlap = len(set(pathogen_uris) & set(plant_uris)) / float(min(len(pathogen_uris), len(plant_uris)))
        if overlap > 0.8:
            st.warning("⚠️ The plant/product vocabulary appears to overlap heavily with the pathogen vocabulary. Double-check PLANT_VOCAB points to the correct file.")

    # --- Step 1: FSKX File Upload and Data Extraction ---
    st.header("1. Upload .fskx Files")
    uploaded_fskx_files = st.file_uploader(
        "Upload one or more .fskx files", 
        type="fskx", 
        accept_multiple_files=True,
        help="Select one or more .fskx (ZIP) files to extract model metadata and parameters. Mappings will be looked up in 'master_mapping.xlsx'."
    )

    if uploaded_fskx_files:
        # Initialize session state keys if they don't exist
        if 'extracted_data' not in st.session_state:
            st.session_state.extracted_data = []
        if 'all_extracted_mappable_term_entries' not in st.session_state:
            st.session_state.all_extracted_mappable_term_entries = []
        if 'all_extracted_units_data' not in st.session_state:
            st.session_state.all_extracted_units_data = []
        if 'unit_mappings' not in st.session_state:
            st.session_state.unit_mappings = {}

        for fskx_file in uploaded_fskx_files:
            # Check if this file has already been processed in the current session
            if not any(d['file_name'] == fskx_file.name for d in st.session_state.extracted_data):
                # IMPORTANT: The file object is consumed by zipfile, so read it into memory once
                # to allow multiple functions to access it.
                fskx_file_bytes = io.BytesIO(fskx_file.getvalue())
                
                model_metadata, parameters, mappable_term_entries, extracted_units = extract_fskx_data(fskx_file_bytes, fskx_file.name)
                
                # Reset pointer and extract hazards/products
                fskx_file_bytes.seek(0)
                haz_raw, haz_norm, prod_raw, prod_norm = extract_hazards_products_from_fskx(fskx_file_bytes)

                if model_metadata and parameters:
                    st.session_state.extracted_data.append({
                        'file_name': fskx_file.name,
                        'model_metadata': model_metadata,
                        'parameters': parameters,
                        'mappable_term_entries': mappable_term_entries,
                        'hazards_raw': haz_raw,
                        'products_raw': prod_raw,
                        'hazards_norm': haz_norm,
                        'products_norm': prod_norm,
                    })
                    st.session_state.all_extracted_mappable_term_entries.extend(mappable_term_entries)
                    
                    # Add unique units to the session state list
                    existing_unit_ids = {u['Unit ID'] for u in st.session_state.all_extracted_units_data}
                    for unit in extracted_units:
                        if unit['Unit ID'] not in existing_unit_ids:
                            st.session_state.all_extracted_units_data.append(unit)
                            existing_unit_ids.add(unit['Unit ID'])

                    st.success(f"Successfully extracted data from {fskx_file.name}")
                else:
                    st.warning(f"No data extracted from {fskx_file.name}. Please check the file content.")
            else:
                st.info(f"Data from {fskx_file.name} already loaded.")
        
        if st.session_state.extracted_data:
            st.subheader("Extracted Models:")
            for i, data_entry in enumerate(st.session_state.extracted_data):
                st.write(f"**{i+1}. Model Name:** {data_entry['model_metadata'].get('name', 'N/A')}")
                with st.expander(f"Details for {data_entry['file_name']}"):
                    st.json(data_entry['model_metadata'])
                    st.write("Original Parameters:")
                    st.dataframe(pd.DataFrame(data_entry['parameters']))
                    st.write("Mappable Term Entries:")
                    st.dataframe(pd.DataFrame(data_entry['mappable_term_entries']))

    # --- Step 2: Map Hazards, Products, and Units ---
    # --- Step 2: Map Model Parameters (and Hazards/Products) ---
    st.header("2. Map Model Parameters and Scope")

    if 'extracted_data' in st.session_state and st.session_state.extracted_data:
        # --- Build Indexes for Prefill ---
        param_concepts_df = master_mapping_df[master_mapping_df['ConceptGroup'].isin(['Model Parameters', 'Parameter'])]
        unit_concepts_df = master_mapping_df[master_mapping_df['ConceptGroup'].isin(['Unit', 'Concept'])]
        
        param_term_map, param_pref_map, param_alt_map = build_master_mapping_indexes(param_concepts_df)
        unit_term_map, unit_pref_map, unit_alt_map = build_master_mapping_indexes(unit_concepts_df)

        # --- Prefill Button ---
        if st.button("Prefill ALL Mappings (Hazards, Products, Parameters, and Units)"):
            for data_entry in st.session_state.extracted_data:
                model_uuid = data_entry['model_metadata']['uuid']
                
                # Prefill Hazards
                hazard_uris = []
                for name in data_entry.get('hazards_raw', []):
                    hazard_uris.extend(find_best_uri_matches(name, pathogen_uris, pathogen_display, pathogen_pref_idx, pathogen_alt_idx))
                st.session_state[f"hazard_uris_{model_uuid}"] = list(dict.fromkeys(hazard_uris))[:5]

                # Prefill Products
                product_uris = []
                for name in data_entry.get('products_raw', []):
                    product_uris.extend(find_best_uri_matches(name, plant_uris, plant_display, plant_pref_idx, plant_alt_idx))
                st.session_state[f"product_uris_{model_uuid}"] = list(dict.fromkeys(product_uris))[:5]

                # Prefill Parameters
                param_mappings = st.session_state.setdefault('parameter_mappings', {}).setdefault(model_uuid, {})
                for param in data_entry['parameters']:
                    if _norm(param.get('classification')) != 'constant':
                        search_term = param.get('name') or param.get('id')
                        # Only suggest matches for terms with 4 or more characters
                        if search_term and len(search_term) >= 4:
                            match = find_best_master_mapping_match(search_term, param_term_map, param_pref_map, param_alt_map)
                            if match:
                                param_mappings[param['id']] = match
                
                # Prefill Units
                unit_mappings = st.session_state.setdefault('unit_mappings', {})
                for unit_data in st.session_state.get('all_extracted_units_data', []):
                    unit_id = unit_data['Unit ID']
                    match = find_best_master_mapping_match(unit_id, unit_term_map, unit_pref_map, unit_alt_map)
                    if match:
                        # We need to store the 'Term' from the master mapping, not the display label
                        selected_term_rows = unit_concepts_df[unit_concepts_df['prefLabel'] == match]
                        if not selected_term_rows.empty:
                            unit_mappings[unit_id] = selected_term_rows.iloc[0]['Term']

            st.rerun()

        # Initialize session state for parameter mappings
        st.session_state.setdefault('parameter_mappings', {})

        # Get the list of valid parameter concepts from the master mapping file
        parameter_options = [""] + sorted(param_concepts_df['prefLabel'].unique().tolist())

        # Get the list of valid unit concepts
        unit_options = [""] + sorted(unit_concepts_df['prefLabel'].unique().tolist())


        st.subheader("2a. Map Units")
        st.write("Map all unique units found across all uploaded files. This mapping is applied globally.")
        if 'all_extracted_units_data' in st.session_state and st.session_state.all_extracted_units_data:
            for unit in st.session_state.all_extracted_units_data:
                unit_id = unit['Unit ID']
                
                current_selection_term = st.session_state.unit_mappings.get(unit_id)
                current_selection_display = ""
                if current_selection_term:
                    selection_rows = unit_concepts_df[unit_concepts_df['Term'] == current_selection_term]
                    if not selection_rows.empty:
                        current_selection_display = selection_rows.iloc[0]['prefLabel']

                try:
                    current_index = unit_options.index(current_selection_display) if current_selection_display in unit_options else 0
                except ValueError:
                    current_index = 0

                selected_display_label = st.selectbox(
                    f"Map Unit: **{unit.get('Unit Name', unit_id)}** (ID: `{unit_id}`)",
                    unit_options,
                    index=current_index,
                    key=f"map_{unit_id}"
                )
                
                if selected_display_label:
                    selected_term_rows = unit_concepts_df[unit_concepts_df['prefLabel'] == selected_display_label]
                    if not selected_term_rows.empty:
                        selected_term = selected_term_rows.iloc[0]['Term']
                        st.session_state.unit_mappings[unit_id] = selected_term
                else:
                    if unit_id in st.session_state.unit_mappings:
                        del st.session_state.unit_mappings[unit_id]
        else:
            st.info("No unique units extracted from files to be mapped.")


        for i, data_entry in enumerate(st.session_state.extracted_data):
            model_name = data_entry['model_metadata'].get('name', 'N/A')
            model_uuid = data_entry['model_metadata'].get('uuid')
            
            st.session_state.parameter_mappings.setdefault(model_uuid, {})

            st.subheader(f"Mappings for model: **{model_name}**")
            st.write("For each FSKX parameter, map it to a standard conceptual term. Parameters classified as 'constant' are excluded from this mapping process.")

            for param in data_entry['parameters']:
                if _norm(param.get('classification')) == 'constant':
                    continue
                param_id = param['id']
                param_name = param.get('name', param_id)
                param_class = param.get('classification', 'N/A').title()
                
                # Get current selection from session state
                current_selection = st.session_state.parameter_mappings[model_uuid].get(param_id, "")
                try:
                    current_index = parameter_options.index(current_selection)
                except ValueError:
                    current_index = 0 # Default to blank if selection not in options

                selected_concept = st.selectbox(
                    f"Map FSKX Parameter: **{param_name}** (`{param_id}`) - *{param_class}*",
                    options=parameter_options,
                    index=current_index,
                    key=f"param_map_{model_uuid}_{param_id}",
                    help="Select the semantic concept that best describes this parameter."
                )
                
                # Store the user's choice
                st.session_state.parameter_mappings[model_uuid][param_id] = selected_concept

            st.subheader("2c. Map Hazards and Products")
            # The existing UI for hazards and products can be reused here
            hazards_sel = create_mapping_ui("Hazard", model_uuid, data_entry.get('hazards_raw', []), pathogen_uris, pathogen_display, pathogen_pref_idx, pathogen_alt_idx)
            products_sel = create_mapping_ui("Product", model_uuid, data_entry.get('products_raw', []), plant_uris, plant_display, plant_pref_idx, plant_alt_idx)
            
            # Store selections for RDF generation
            data_entry['hazards_sel'] = hazards_sel
            data_entry['products_sel'] = products_sel

    else:
        st.info("Upload FSKX files in Step 1 to see mapping options.")


    # --- Step 3: Output New Terms for Mapping ---
    st.header("3. Output New Terms for Mapping")
    if 'all_extracted_mappable_term_entries' in st.session_state and st.session_state.all_extracted_mappable_term_entries:
        # Combine parameters and units into one list for checking against the master mapping
        combined_mappable_terms = list(st.session_state.all_extracted_mappable_term_entries)
        if 'all_extracted_units_data' in st.session_state:
            for unit in st.session_state.all_extracted_units_data:
                combined_mappable_terms.append({
                    'Term': unit['Unit ID'],
                    'ConceptGroup': 'Unit',
                    'prefLabel': unit.get('Unit Name', unit['Unit ID']),
                    'altLabels': '',
                    'ProviderDescription': 'Unit extracted from FSKX file'
                })

        new_terms_excel_data = create_new_terms_table(combined_mappable_terms, master_mapping_df)
        if new_terms_excel_data:
            st.download_button(
                label="Download New Unmapped Terms (new_terms.xlsx)",
                data=new_terms_excel_data,
                file_name="new_terms.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="This file contains terms extracted from your FSKX files that are not yet in 'master_mapping.xlsx'. Fill in the ontology mappings and manually merge them into 'master_mapping.xlsx'."
            )
            st.info("Please download 'new_terms.xlsx', fill in the mappings, and manually merge them into your 'master_mapping.xlsx' file. Rerun the app to apply new mappings.")
        else:
            st.info("All extracted terms are already mapped in 'master_mapping.xlsx' or no new terms were extracted.")
    else:
        st.info("Upload FSKX files in Step 1 to identify new terms.")

    # --- Step 4: Finalize and Save ---
    st.header("4. Finalize and Save")

    if 'extracted_data' in st.session_state and st.session_state.extracted_data:
        ttl_directory = Path(__file__).parent / "../Vocabulary"
        os.makedirs(ttl_directory, exist_ok=True)
        skos_file_path, wiring_file_path = fskx_updater.find_existing_ttl_files(str(ttl_directory))

        # Determine default action
        action_options = ["Append to Existing File(s)", "Generate New File(s)"]
        default_index = 0 if (skos_file_path and wiring_file_path) else 1
        
        action = st.radio("Action:", action_options, index=default_index, key="action_radio")

        # Pre-check for existing models if appending
        models_to_process = []
        if action == "Append to Existing File(s)":
            skos_graph_for_check = Graph().parse(skos_file_path, format="turtle") if skos_file_path else Graph()
            wiring_graph_for_check = Graph().parse(wiring_file_path, format="turtle") if wiring_file_path else Graph()

            for data_entry in st.session_state.extracted_data:
                model_uuid = data_entry['model_metadata']['uuid']
                model_uri = MODEL[model_uuid]
                
                entry_info = {"data": data_entry, "process": True, "overwrite": False}

                model_exists = fskx_updater.check_if_model_exists(model_uri, skos_graph_for_check) or \
                               fskx_updater.check_if_model_exists(model_uri, wiring_graph_for_check)

                if model_exists:
                    st.markdown(f"---")
                    st.warning(f"Model **{data_entry['model_metadata'].get('name', model_uuid)}** already exists in the TTL file(s).")
                    choice = st.radio(
                        "Choose action for this model:",
                        ("Keep existing instance (skip)", "Overwrite existing instance"),
                        key=f"overwrite_{model_uuid}"
                    )
                    if choice == "Keep existing instance (skip)":
                        entry_info["process"] = False
                    else:
                        entry_info["overwrite"] = True
                
                models_to_process.append(entry_info)
        else:
            # If generating new files, all models are processed without overwrite checks
            models_to_process = [{"data": de, "process": True, "overwrite": False} for de in st.session_state.extracted_data]


        st.markdown(f"---")
        if st.button("Process Files"):
            if action == "Append to Existing File(s)":
                skos_graph = Graph().parse(skos_file_path, format="turtle") if skos_file_path else Graph()
                wiring_graph = Graph().parse(wiring_file_path, format="turtle") if wiring_file_path else Graph()
                
                # Ensure namespaces are bound to the loaded graphs
                _bind_common_namespaces(skos_graph)
                _bind_common_namespaces(wiring_graph)

                processed_count = 0
                processed_model_uris = []
                for model_info in models_to_process:
                    if not model_info["process"]:
                        continue

                    data_entry = model_info["data"]
                    model_uri = MODEL[data_entry['model_metadata']['uuid']]
                    processed_model_uris.append(model_uri)
                    
                    knowledge_graph = Graph() + load_vocab_graph(PATHOGEN_VOCAB) + load_vocab_graph(PLANT_VOCAB) + load_vocab_graph(NETCDF_VOCAB)
                    new_skos_g, new_wiring_g, _ = generate_rdf_outputs(
                        model_data=data_entry['model_metadata'],
                        original_parameters=data_entry['parameters'],
                        master_mapping_df=master_mapping_df,
                        knowledge_graph=knowledge_graph,
                        hazards_sel=data_entry.get('hazards_sel', []),
                        products_sel=data_entry.get('products_sel', [])
                    )
                    skos_graph, wiring_graph = fskx_updater.append_to_ttl(skos_graph, wiring_graph, new_skos_g, new_wiring_g, model_uri, model_info["overwrite"])
                    processed_count += 1

                if processed_count > 0:
                    # SKOS
                    all_model_uris = all_model_uris_from_graph(skos_graph)
                    label_context = (load_vocab_graph(PATHOGEN_VOCAB) +
                                     load_vocab_graph(PLANT_VOCAB) +
                                     load_vocab_graph(NETCDF_VOCAB) +
                                     build_fskxo_label_graph(master_mapping_df))
                    full_graph_for_labels = skos_graph + label_context
                    _bind_common_namespaces(full_graph_for_labels)
                    skos_out = serialize_skos_graph_with_comments(full_graph_for_labels, all_model_uris)
                    with open(str(ttl_directory / "fskx-models.ttl"), "w", encoding="utf-8") as f:
                        f.write(skos_out)

                    # WIRING — note we use the actual wiring_graph content (plus skos for label context),
                    # and we serialize for *all* models present after append
                    all_model_uris_wiring = all_model_uris_from_graph(skos_graph)  # order by models in SKOS
                    wiring_out = serialize_wiring_graph_custom(wiring_graph + skos_graph, all_model_uris_wiring)
                    with open(str(ttl_directory / "wiring-instances.ttl"), "w", encoding="utf-8") as f:
                        f.write(wiring_out)

                    st.session_state.generated_outputs = {'combined_skos': skos_graph, 'combined_wiring': wiring_graph}
                    st.success(f"Successfully processed and appended data for {processed_count} model(s)!")
                else:
                    st.info("No new models were appended based on your selections.")

            else:  # Generate New File(s)
                combined_skos_graph = Graph()
                combined_wiring_graph = Graph()
                _bind_common_namespaces(combined_skos_graph)
                _bind_common_namespaces(combined_wiring_graph)

                model_uris = []
                knowledge_graph = Graph() + load_vocab_graph(PATHOGEN_VOCAB) + load_vocab_graph(PLANT_VOCAB) + load_vocab_graph(NETCDF_VOCAB)

                for model_info in models_to_process:
                    data_entry = model_info["data"]
                    g_skos, g_wiring, _ = generate_rdf_outputs(
                        model_data=data_entry['model_metadata'],
                        original_parameters=data_entry['parameters'],
                        master_mapping_df=master_mapping_df,
                        knowledge_graph=knowledge_graph,
                        hazards_sel=data_entry.get('hazards_sel', []),
                        products_sel=data_entry.get('products_sel', [])
                    )
                    combined_skos_graph += g_skos
                    combined_wiring_graph += g_wiring
                    model_uris.append(MODEL[data_entry['model_metadata']['uuid']])

                # Build a label context so comments are meaningful
                label_context = (knowledge_graph + build_fskxo_label_graph(master_mapping_df))
                full_graph_for_labels = combined_skos_graph + label_context
                _bind_common_namespaces(full_graph_for_labels)

                st.session_state.generated_outputs = {'combined_skos': combined_skos_graph, 'combined_wiring': combined_wiring_graph}
                st.success("Files processed and are now ready for download in Step 5.")

    else:
        st.info("Upload FSKX files and map parameters to generate or append RDF.")


    # --- Step 5: Download RDF Files ---
    st.header("5. Download RDF Files")
    if 'generated_outputs' in st.session_state and st.session_state.generated_outputs:

        # --- Download for Combined SKOS file ---
        g_skos = st.session_state.generated_outputs.get('combined_skos')
        if g_skos and len(g_skos) > 0:
            output_filename_skos = "fskx-models.ttl"
            try:
                label_context = (load_vocab_graph(PATHOGEN_VOCAB) +
                                 load_vocab_graph(PLANT_VOCAB) +
                                 load_vocab_graph(NETCDF_VOCAB) +
                                 build_fskxo_label_graph(master_mapping_df))

                full_graph_for_labels = g_skos + label_context
                _bind_common_namespaces(full_graph_for_labels)

                commented_ttl = serialize_skos_graph_with_comments(full_graph_for_labels)
                rdf_output_bytes_skos = commented_ttl.encode('utf-8')

                st.download_button(
                    label=f"Download SKOS Model Description ({output_filename_skos})",
                    data=rdf_output_bytes_skos,
                    file_name=output_filename_skos,
                    mime="text/turtle",
                    key="download_skos_combined"
                )
            except Exception as e:
                st.error(f"Error serializing combined SKOS graph: {e}")

        # --- Download for Combined Wiring file ---
        g_wiring = st.session_state.generated_outputs.get('combined_wiring')
        if g_wiring and len(g_wiring) > 0:
            output_filename_wiring = "wiring-instances.ttl"
            try:
                combined = g_wiring + g_skos
                custom_wiring_output = serialize_wiring_graph_custom(combined)
                rdf_output_bytes_wiring = custom_wiring_output.encode('utf-8')

                st.download_button(
                    label=f"Download Wiring Instances ({output_filename_wiring})",
                    data=rdf_output_bytes_wiring,
                    file_name=output_filename_wiring,
                    mime="text/turtle",
                    key="download_wiring_combined"
                )
            except Exception as e:
                st.error(f"Error serializing combined wiring graph: {e}")
        elif g_wiring is not None:
             st.info("No input parameters were wired, so no wiring file was generated.")

    else:
        st.info("Generate RDF in Step 4 to enable downloads.")

if __name__ == "__main__":
    main()
