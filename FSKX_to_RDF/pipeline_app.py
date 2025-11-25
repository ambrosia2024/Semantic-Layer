import streamlit as st
import subprocess
import sys
import logging
from pathlib import Path
import os
import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import DCTERMS, RDF, SKOS, RDFS
from rdflib.plugins.sparql import prepareQuery
import re
import unicodedata
import difflib
import zipfile
import json
import io

# Get the directory of the current script to build robust paths
script_dir = Path(__file__).resolve().parent

# --- Configuration ---
FSKX_DIR = script_dir / "fskx_models"
MAPPED_TURTLE_DIR = script_dir / "mapped" / "turtle"
MASTER_MAPPING_FILE = script_dir / "master_mapping.xlsx"
VOCAB_DIR = script_dir.parent / "Vocabulary"


# --- Namespaces ---
FSKXO = Namespace("http://semanticlookup.zbmed.de/km/fskxo/")
MODEL = Namespace("https://www.ambrosia-project.eu/model/")
VOCAB = Namespace("https://www.ambrosia-project.eu/vocab/")
AMBLINK = Namespace("https://www.ambrosia-project.eu/vocab/linking/")
SCHEMA = Namespace("https://schema.org/")
QUDT_UNIT = Namespace("http://qudt.org/vocab/unit/")
QK = Namespace("http://qudt.org/vocab/quantitykind/")

# --- Mappings ---
CLASSIFICATION_MAP = {
    "FSKXO_0000017481": "Input",
    "FSKXO_0000017482": "Output",
    "FSKXO_0000017480": "Constant",
}

# --- Helper Functions ---

def get_model_statuses():
    """
    Scans FSKX and mapped turtle directories to determine model statuses.
    Returns new models and a dictionary of existing models with their titles from the FSKX file.
    """
    if not FSKX_DIR.exists():
        return [], {}
    fskx_files = {p.stem for p in FSKX_DIR.glob("*.fskx")}

    if not MAPPED_TURTLE_DIR.exists() or not any(MAPPED_TURTLE_DIR.iterdir()):
        return sorted(list(fskx_files)), {}

    mapped_files_stems = {p.stem for p in MAPPED_TURTLE_DIR.glob("*.ttl")}
    new_models = sorted(list(fskx_files - mapped_files_stems))
    
    existing_models_stems = sorted(list(fskx_files.intersection(mapped_files_stems)))
    existing_models_with_titles = {}
    for stem in existing_models_stems:
        title = "Title not found"
        try:
            fskx_file_path = FSKX_DIR / f"{stem}.fskx"
            if fskx_file_path.exists():
                with zipfile.ZipFile(fskx_file_path, 'r') as zip_ref:
                    metadata_filename = next((f for f in zip_ref.namelist() if f.lower().endswith("metadata.json")), None)
                    if metadata_filename:
                        with zip_ref.open(metadata_filename) as meta_file:
                            fskx_content = json.load(meta_file)
                            title = fskx_content.get('generalInformation', {}).get('name', 'Title not found in JSON')
        except Exception:
            title = "Error reading FSKX"
        existing_models_with_titles[stem] = title
            
    return new_models, existing_models_with_titles

def run_script(script_name, args=[]):
    st.write(f"--- Running {script_name} ---")
    script_path = Path(__file__).resolve().parent / script_name
    command = [sys.executable, str(script_path)] + args
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace',
        cwd=Path(__file__).resolve().parent # Run script from its own directory
    )
    output_box = st.empty()
    output_text = ""
    for line in iter(process.stdout.readline, ''):
        output_text += line
        output_box.code(output_text)
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        st.error(f"{script_name} failed with return code {return_code}.")
        return False
    st.success(f"{script_name} completed successfully.")
    return True

@st.cache_data
def load_master_mapping(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        df.columns = [col.strip() for col in df.columns]
        for col in df.columns:
            if 'URI' in col or 'Term' in col:
                # Fill NaNs first, then convert to string to avoid "nan" strings
                df[col] = df[col].fillna('').astype(str)
        return df
    except FileNotFoundError:
        st.warning(f"Master mapping file not found at '{file_path}'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading master mapping file: {e}")
        return pd.DataFrame()

def _get_display_label(term, alt_labels):
    alt_labels_str = str(alt_labels or '').strip()
    return f"{term} ({alt_labels_str})" if alt_labels_str else str(term)

def _txt(v): return str(v) if v is not None else ""

@st.cache_resource
def load_vocab_graph(path: str) -> Graph:
    g = Graph()
    try:
        g.parse(path, format="turtle")
    except Exception as e:
        st.error(f"Error parsing vocabulary graph at {path}: {e}")
    return g

def _build_vocab_struct_impl(_g: Graph):
    uris, display_by_uri = [], {}
    subjects = set(_g.subjects(SKOS.prefLabel, None)) | set(_g.subjects(RDFS.label, None))
    for s in subjects:
        uri = str(s)
        prefs = [_txt(l) for l in _g.objects(s, SKOS.prefLabel)] or [_txt(l) for l in _g.objects(s, RDFS.label)]
        if not (pref := prefs[0] if prefs else None): continue
        alts = [a for a in [_txt(a) for a in _g.objects(s, SKOS.altLabel)] if a and a != pref]
        display = f"{pref} ({'; '.join(alts[:2])})" if alts else pref
        uris.append(uri)
        display_by_uri[uri] = display
    uris.sort(key=lambda u: display_by_uri[u].lower())
    return uris, display_by_uri

@st.cache_data(show_spinner=False)
def build_vocab_struct_for_path(path: str):
    g = load_vocab_graph(path)
    return _build_vocab_struct_impl(g)

def extract_fskx_params(fskx_file_path):
    """
    Extracts raw parameter IDs from an FSKX file.
    Handles both single and composite (multi-model) FSKX files by reading all
    metaData.json files found in the archive.
    """
    param_ids = set()
    try:
        with zipfile.ZipFile(fskx_file_path, 'r') as zip_ref:
            # Find all metaData.json files in the archive
            metadata_files = [f for f in zip_ref.namelist() if f.lower().endswith("metadata.json")]
            if not metadata_files:
                st.warning(f"No metaData.json found in {fskx_file_path.name}")
                return set()

            for metadata_filename in metadata_files:
                with zip_ref.open(metadata_filename) as meta_file:
                    fskx_content = json.load(meta_file)
                
                model_math = fskx_content.get('modelMath', {})
                params = model_math.get('parameter') or model_math.get('parameters', [])
                if not isinstance(params, list):
                    params = [params]

                for p in params:
                    param_id = p.get('id')
                    if param_id:
                        param_ids.add(param_id)

    except Exception as e:
        st.error(f"Error reading FSKX file {fskx_file_path.name}: {e}")
    return param_ids

def determine_model_cascade(graph, models_data):
    """
    Determines the execution order of models in a composite model based on parameter mappings.
    """
    # Define namespaces for SPARQL query
    ns = {
        "fskxo": FSKXO,
        "amblink": AMBLINK,
        "schema": SCHEMA,
        "dcterms": DCTERMS
    }

    # Query to find connections: where an input of one model is fulfilled by a parameter from another
    q_connections = """
        SELECT ?source_model ?target_model ?source_param_id ?target_param_id
        WHERE {
            ?target_model amblink:hasInputMapping ?mapping .
            ?mapping amblink:mapsParameter ?target_param .
            ?target_param dcterms:identifier ?target_param_id .
            
            ?mapping amblink:isFulfilledBy ?source_param .
            ?source_model fskxo:FSKXO_0000000016 / fskxo:FSKXO_0000000017 ?source_param .
            ?source_param dcterms:identifier ?source_param_id .

            FILTER(?source_model != ?target_model)
        }
    """
    
    connections = list(graph.query(q_connections, initNs=ns))
    
    if not connections:
        return models_data # Return original if no connections found

    # Build a dependency graph
    model_uris = {m['uri'] for m in models_data}
    dependencies = {uri: set() for uri in model_uris}
    dependents = {uri: set() for uri in model_uris}

    for conn in connections:
        source_uri = str(conn.source_model)
        target_uri = str(conn.target_model)
        if source_uri in model_uris and target_uri in model_uris:
            dependencies[target_uri].add(source_uri)
            dependents[source_uri].add(target_uri)

    # Topological sort to find the cascade order
    sorted_order = []
    # Start with models that have no dependencies
    queue = [uri for uri in model_uris if not dependencies[uri]]
    
    while queue:
        current_uri = queue.pop(0)
        sorted_order.append(current_uri)
        
        # For each model that depends on the current one, remove the dependency
        for dependent_uri in list(dependents[current_uri]):
            dependencies[dependent_uri].remove(current_uri)
            # If the dependent model has no other dependencies, add it to the queue
            if not dependencies[dependent_uri]:
                queue.append(dependent_uri)

    # If the sorted order doesn't include all models, there might be a cycle or disconnected components
    if len(sorted_order) != len(model_uris):
        # Add remaining models that were not part of the sorted cascade
        remaining = [m for m in models_data if m['uri'] not in sorted_order]
        # Create a final list of model data in the determined order
        final_sorted_models = [m for uri in sorted_order for m in models_data if m['uri'] == uri] + remaining
        return final_sorted_models

    # Reorder the original models_data list based on the sorted URIs
    return [next(m for m in models_data if m['uri'] == uri) for uri in sorted_order]

def extract_data_from_turtle(graph):
    """
    Extracts data from a turtle graph.
    It finds all models with parameters and groups their parameters accordingly,
    ignoring any parameters with a numeric suffix in their ID.
    """
    data = {
        "models": [], "hazards": set(), "products": set(),
        "model_uri": None, "cascade": []
    }
    
    # Define namespaces for queries
    ns = {
        "fskxo": FSKXO,
        "schema": SCHEMA,
        "dcterms": DCTERMS,
        "rdf": RDF,
        "rdfs": RDFS,
        "skos": SKOS,
        "model": MODEL,
        "vocab": VOCAB,
        "amblink": AMBLINK,
        "qudt-unit": QUDT_UNIT,
        "qk": QK
    }

    # Find the main model URI (composite or single)
    q_main_model = "SELECT ?s WHERE { ?s a fskxo:CompositeModel . } LIMIT 1"
    res = list(graph.query(q_main_model, initNs=ns))
    if not res:
        q_main_model = "SELECT ?s WHERE { ?s fskxo:FSKXO_0000000016 ?math . } LIMIT 1"
        res = list(graph.query(q_main_model, initNs=ns))
    
    if not res:
        st.error("Could not determine the main model URI.")
        return data
    main_model_uri = res[0][0]
    data["model_uri"] = main_model_uri

    # Simplified logic to find all models that have parameters.
    q_models = """
        SELECT DISTINCT ?model ?name WHERE {
            ?model fskxo:FSKXO_0000000016 / fskxo:FSKXO_0000000017 ?param .
            OPTIONAL { ?model schema:name ?name_s . }
            OPTIONAL { ?model dcterms:title ?name_d . }
            OPTIONAL { 
                ?model fskxo:FSKXO_0000000003 ?info .
                ?info dcterms:title ?name_info .
            }
            BIND(COALESCE(?name_s, ?name_d, ?name_info, STRAFTER(STR(?model), "/")) as ?name)
        }
    """
    models_to_process = list(graph.query(q_models, initNs=ns))

    # If no models found with parameters, fall back to the main URI
    if not models_to_process and main_model_uri:
        name_res = graph.value(main_model_uri, DCTERMS.title) or graph.value(main_model_uri, SCHEMA.name)
        name = name_res if name_res else str(main_model_uri).split('/')[-1]
        models_to_process.append((main_model_uri, name))

    # Filter out composite models from the list of models to process for parameters.
    # A composite model's name will contain the names of the base models it's composed of.
    model_names = [str(m[1]) for m in models_to_process]
    base_models_to_process = []
    for model_uri, model_name_literal in models_to_process:
        model_name = str(model_name_literal)
        is_composite = False
        # Check if this model's name contains the name of any *other* model.
        for other_name in model_names:
            if model_name != other_name and other_name in model_name:
                is_composite = True
                break
        if not is_composite:
            base_models_to_process.append((model_uri, model_name_literal))

    # If filtering removed all models (e.g., single model case), fall back to the original list.
    if not base_models_to_process:
        base_models_to_process = models_to_process

    # Query for parameters for each base model
    for model_uri, model_name in base_models_to_process:
        model_data = {"uri": str(model_uri), "name": str(model_name), "parameters": []}
        
        q_params = """
            SELECT ?param ?id ?name ?classification ?unit_label WHERE {
                ?model fskxo:FSKXO_0000000016 / fskxo:FSKXO_0000000017 ?param .
                OPTIONAL { ?param dcterms:identifier ?id_dcterms . }
                OPTIONAL { ?param schema:id ?id_schema . }
                BIND(COALESCE(?id_dcterms, ?id_schema) AS ?id)
                FILTER(BOUND(?id))
                
                OPTIONAL { ?param schema:name ?name . }
                OPTIONAL { ?param fskxo:FSKXO_0000017519 ?classification . }
                OPTIONAL { ?param schema:unit_label ?unit_label . }
            }
        """
        params_res = graph.query(q_params, initNs=ns, initBindings={'model': model_uri})
        
        temp_params = {}
        for row in params_res:
            param_uri = str(row.param)
            if param_uri not in temp_params:
                param_id = str(row.id)
                if re.search(r'\d+$', param_id):
                    continue
                
                schema_name = str(row.name) if row.name else None
                
                temp_params[param_uri] = {
                    'uri': param_uri, 'id': param_id, 
                    'name': schema_name or param_id, # Display name with fallback
                    'schema_name': schema_name, # Raw schema:name
                    'classifications': set(),
                    'raw_unit_label': str(row.unit_label) if row.unit_label else None
                }
            
            if row.classification:
                cls_id = str(row.classification).split('/')[-1]
                if (mc := CLASSIFICATION_MAP.get(cls_id)):
                    temp_params[param_uri]['classifications'].add(mc)
        
        model_data["parameters"] = sorted(temp_params.values(), key=lambda p: p['name'])
        if model_data["parameters"]:
            data["models"].append(model_data)

    # Scope extraction
    q_scope = "SELECT ?h ?p WHERE { {?m fskxo:FSKXO_0000000008 ?h} UNION {?m fskxo:FSKXO_0000000007 ?p} }"
    for row in graph.query(q_scope, initNs=ns, initBindings={'m': main_model_uri}):
        if row.h: data["hazards"].add(str(row.h))
        if row.p: data["products"].add(str(row.p))

    # Determine model cascade if more than one model is present
    if len(data["models"]) > 1:
        data["cascade"] = determine_model_cascade(graph, data["models"])
    else:
        data["cascade"] = data["models"]
        
    return data

def _bind_common_namespaces(g: Graph):
    """Binds all common namespaces to a graph for clean serialization."""
    g.bind("fskxo", FSKXO)
    g.bind("model", MODEL)
    g.bind("vocab", VOCAB)
    g.bind("amblink", AMBLINK)
    g.bind("schema", SCHEMA)
    g.bind("dcterms", DCTERMS)
    g.bind("skos", SKOS)
    g.bind("rdfs", RDFS)
    g.bind("qudt-unit", QUDT_UNIT)
    g.bind("qk", QK)

def render_fskx_to_rdf_ui(embedded=False, key_ns="fskx_to_rdf"):
    if not embedded:
        st.set_page_config(layout="wide")
    
    st.title("FSKX to RDF")

    # --- Main Pipeline Runner ---
    st.sidebar.header("Controls")

    # Execution Mode Selection
    execution_mode = st.sidebar.radio(
        "Execution Mode",
        ["Process All Models", "Process Single Model"],
        key=f"{key_ns}_exec_mode"
    )

    target_model_file = None
    if execution_mode == "Process Single Model":
        # Gather available FSKX files
        if FSKX_DIR.exists():
            fskx_files = sorted([p.name for p in FSKX_DIR.glob("*.fskx")])
            target_model_file = st.sidebar.selectbox(
                "Select Model to Process",
                fskx_files,
                key=f"{key_ns}_target_model"
            )
        else:
            st.sidebar.error(f"FSKX Directory not found: {FSKX_DIR}")

    override_existing = st.sidebar.checkbox("Re-process/Overwrite", value=False, key=f"{key_ns}_override")
    run_pipeline = st.sidebar.button("▶️ Run Pipeline", key=f"{key_ns}_run")

    st.header("Model Status")
    new_models, existing_models_with_titles = get_model_statuses()
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe({"New Models": new_models or ["-"]}, use_container_width=True)
    with col2:
        processed_df = pd.DataFrame(
            {"Model ID": existing_models_with_titles.keys(), "Title": existing_models_with_titles.values()}
        )
        st.dataframe(processed_df, use_container_width=True)

    if run_pipeline:
        st.session_state.pipeline_run = True
        st.header("Pipeline Output")
        script_args = ["--override"] if override_existing else []
        
        with st.status("Running pipeline...", expanded=True) as status:
            if execution_mode == "Process Single Model" and target_model_file:
                st.write(f"Processing single model: **{target_model_file}**")
                model_stem = Path(target_model_file).stem
                
                # 1. FSKX_to_JSONLD.py
                fskx_path = FSKX_DIR / target_model_file
                fskx_args = [str(fskx_path)] + script_args
                if not run_script("FSKX_to_JSONLD.py", fskx_args): st.stop()
                
                # 2. ValidityChecker.py
                if not run_script("ValidityChecker.py"): st.stop()
                
                # 3. run_mapper.py
                # Need to point to the generated unmapped jsonld file
                unmapped_jsonld = Path("unmapped/jsonld") / f"{model_stem}.jsonld"
                mapper_args = ["-i", str(unmapped_jsonld)] + script_args
                if not run_script("run_mapper.py", mapper_args): st.stop()
                
                # 4. jsonld_serialization_converter.py
                # Need to point to the mapped jsonld file
                mapped_jsonld = Path("mapped/jsonld") / f"{model_stem}.jsonld"
                converter_args = ["-i", str(mapped_jsonld), "-f", "turtle", "-o", "mapped/turtle"] + script_args
                if not run_script("jsonld_serialization_converter.py", converter_args): st.stop()
            
            else:
                # Process All
                fskx_to_jsonld_args = [str(FSKX_DIR)] + script_args
                if not run_script("FSKX_to_JSONLD.py", fskx_to_jsonld_args): st.stop()
                if not run_script("ValidityChecker.py"): st.stop()
                if not run_script("run_mapper.py", script_args): st.stop()
                converter_args = ["-d", "mapped/jsonld", "-f", "turtle", "-o", "mapped/turtle"]
                if override_existing: converter_args.append("--override")
                if not run_script("jsonld_serialization_converter.py", converter_args): st.stop()

            status.update(label="Pipeline finished successfully!", state="complete", expanded=False)

    st.markdown("---")
    st.header("Interactive Mapping")
    st.info("ℹ️ **Note:** The fields below are pre-filled with the currently saved mappings for this model. Any changes you make will be applied when you click the 'Apply All Mappings' button at the bottom.")

    if not MAPPED_TURTLE_DIR.exists() or not any(MAPPED_TURTLE_DIR.iterdir()):
        st.info("No models to map. Run the pipeline first.")
        st.stop()

    master_mapping_df = load_master_mapping(MASTER_MAPPING_FILE)
    unit_df = pd.DataFrame()
    unit_options = [""]
    unit_label_to_uri = {}
    if not master_mapping_df.empty:
        master_mapping_df['prefLabel'] = master_mapping_df.apply(lambda r: _get_display_label(r['Term'], r['altLabels']), axis=1)
        unit_df = master_mapping_df[master_mapping_df['ConceptGroup'] == 'Unit'].copy()
        unit_options = [""] + sorted(unit_df['prefLabel'].unique().tolist())
        
        for _, row in unit_df.iterrows():
            # Prefer URI1, then URI2, then URI3
            uri = row['URI1'] if pd.notna(row['URI1']) and row['URI1'] else \
                  (row['URI2'] if pd.notna(row['URI2']) and row['URI2'] else row['URI3'])
            if uri:
                unit_label_to_uri[row['prefLabel']] = str(uri)

    (pathogen_uris, pathogen_display) = build_vocab_struct_for_path(str(VOCAB_DIR / "ambrosia-pathogen-vocab.ttl"))
    (plant_uris, plant_display) = build_vocab_struct_for_path(str(VOCAB_DIR / "ambrosia-plant-vocab.ttl"))

    _, existing_models_with_titles = get_model_statuses()
    existing_models_options = list(existing_models_with_titles.keys())

    def format_model_option(model_stem):
        """Custom function to format the display in the selectbox."""
        title = existing_models_with_titles.get(model_stem, "N/A")
        return f"{model_stem} ({title})"

    selected_model = st.selectbox(
        "Select a model to map:",
        existing_models_options,
        format_func=format_model_option,
        key=f"{key_ns}_model_select"
    )

    if selected_model:
        turtle_file = MAPPED_TURTLE_DIR / f"{selected_model}.ttl"
        fskx_file = FSKX_DIR / f"{selected_model}.fskx"
        if turtle_file.exists() and fskx_file.exists():
            g = Graph().parse(str(turtle_file), format="turtle")
            model_data = extract_data_from_turtle(g)

            # --- Automatic Prefill Option ---
            if st.button("Automatic Prefill Option", help="Attempt to automatically infer Hazards, Products, Parameters, and Units from the FSKX/Turtle files.", key=f"{key_ns}_auto_prefill"):
                st.info("Running automatic inference...")
                
                # Configure Logging
                log_file = script_dir / "inference_debug.log"
                logging.basicConfig(filename=str(log_file), level=logging.DEBUG, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
                logging.info(f"Starting automatic inference for model: {selected_model}")

                # 1. Infer Hazards and Products from Turtle (Matched against Vocabularies)
                inferred_hazards = set()
                inferred_products = set()
                
                # Load Vocabulary Graphs for matching
                pathogen_graph = load_vocab_graph(str(VOCAB_DIR / "ambrosia-pathogen-vocab.ttl"))
                plant_graph = load_vocab_graph(str(VOCAB_DIR / "ambrosia-plant-vocab.ttl"))
                logging.info("Vocabulary graphs loaded.")

                # Namespaces for query
                ns_query = {
                    "fskxo": FSKXO,
                    "schema": SCHEMA,
                    "dcterms": DCTERMS,
                    "skos": SKOS
                }
                
                # --- Hazard Inference ---
                # Look for ?h via FSKXO_0000000008, then get its label or name URI
                q_hazard = """
                    SELECT ?label ?name_uri ?h_node WHERE {
                        ?model fskxo:FSKXO_0000000008 ?h_node .
                        OPTIONAL { ?h_node schema:label ?label . }
                        OPTIONAL { ?h_node schema:name ?name_uri . }
                    }
                """
                for row in g.query(q_hazard, initNs=ns_query):
                    h_label = str(row.label).strip() if row.label else None
                    h_uris = []
                    if row.name_uri: h_uris.append(URIRef(row.name_uri))
                    if row.h_node and isinstance(row.h_node, URIRef): h_uris.append(row.h_node)
                    
                    logging.debug(f"Found Hazard Candidate in FSKX: Label='{h_label}', URIs={[str(u) for u in h_uris]}")

                    # Strategy 1: URI Match (Exact Match in Vocab)
                    # Check if extracted URI is the Subject in Vocab OR Object of skos:exactMatch
                    for uri in h_uris:
                        # Case A: URI is the concept URI in vocab
                        if (uri, RDF.type, SKOS.Concept) in pathogen_graph:
                            inferred_hazards.add(str(uri))
                            logging.info(f"Hazard Match (Direct URI): {uri}")
                        
                        # Case B: URI is mapped via skos:exactMatch/closeMatch/narrowMatch
                        q_match = prepareQuery("""
                            SELECT ?concept WHERE {
                                ?concept a skos:Concept .
                                { ?concept skos:exactMatch ?uri } UNION
                                { ?concept skos:closeMatch ?uri } UNION
                                { ?concept skos:narrowMatch ?uri }
                            }
                        """, initNs={"skos": SKOS})
                        for match in pathogen_graph.query(q_match, initBindings={'uri': uri}):
                            inferred_hazards.add(str(match.concept))
                            logging.info(f"Hazard Match (SKOS Match URI): {match.concept} matches {uri}")

                    # Strategy 2: Label Match
                    if h_label:
                         # Try exact match case-insensitive
                         q_label = prepareQuery("""
                            SELECT ?concept ?pref ?alt WHERE {
                                ?concept a skos:Concept .
                                OPTIONAL { ?concept skos:prefLabel ?pref }
                                OPTIONAL { ?concept skos:altLabel ?alt }
                                FILTER(LCASE(STR(?pref)) = LCASE(?target_label) || LCASE(STR(?alt)) = LCASE(?target_label))
                            }
                        """, initNs={"skos": SKOS})
                         for match in pathogen_graph.query(q_label, initBindings={'target_label': Literal(h_label)}):
                             inferred_hazards.add(str(match.concept))
                             logging.info(f"Hazard Match (Label): {match.concept} matches label '{h_label}'")


                # --- Product Inference ---
                # Look for ?p via FSKXO_0000000007
                q_product = """
                    SELECT ?label ?name_uri ?p_node WHERE {
                        ?model fskxo:FSKXO_0000000007 ?p_node .
                        OPTIONAL { 
                            ?p_node schema:name ?p_name_node . 
                            ?p_name_node schema:label ?label .
                        }
                        OPTIONAL { ?p_node schema:label ?label_direct . }
                        BIND(COALESCE(?label, ?label_direct) AS ?label)
                        
                        OPTIONAL { ?p_node schema:name ?name_uri . FILTER(isIRI(?name_uri)) }
                    }
                """
                for row in g.query(q_product, initNs=ns_query):
                    p_label = str(row.label).strip() if row.label else None
                    p_uris = []
                    if row.name_uri: p_uris.append(URIRef(row.name_uri))
                    # Check if p_node or p_name_node (if URI) is relevant
                    # Usually p_node is scope specific, but let's check just in case
                    if row.p_node and isinstance(row.p_node, URIRef): p_uris.append(row.p_node)

                    logging.debug(f"Found Product Candidate in FSKX: Label='{p_label}', URIs={[str(u) for u in p_uris]}")

                    # Strategy 1: URI Match
                    for uri in p_uris:
                        if (uri, RDF.type, SKOS.Concept) in plant_graph:
                            inferred_products.add(str(uri))
                            logging.info(f"Product Match (Direct URI): {uri}")
                        
                        q_match = prepareQuery("""
                            SELECT ?concept WHERE {
                                ?concept a skos:Concept .
                                { ?concept skos:exactMatch ?uri } UNION
                                { ?concept skos:closeMatch ?uri } UNION
                                { ?concept skos:narrowMatch ?uri }
                            }
                        """, initNs={"skos": SKOS})
                        for match in plant_graph.query(q_match, initBindings={'uri': uri}):
                            inferred_products.add(str(match.concept))
                            logging.info(f"Product Match (SKOS Match URI): {match.concept} matches {uri}")

                    # Strategy 2: Label Match
                    if p_label:
                         q_label = prepareQuery("""
                            SELECT ?concept ?pref ?alt WHERE {
                                ?concept a skos:Concept .
                                OPTIONAL { ?concept skos:prefLabel ?pref }
                                OPTIONAL { ?concept skos:altLabel ?alt }
                                FILTER(LCASE(STR(?pref)) = LCASE(?target_label) || LCASE(STR(?alt)) = LCASE(?target_label))
                            }
                        """, initNs={"skos": SKOS})
                         for match in plant_graph.query(q_label, initBindings={'target_label': Literal(p_label)}):
                             inferred_products.add(str(match.concept))
                             logging.info(f"Product Match (Label): {match.concept} matches label '{p_label}'")

                
                # Update Session State for Hazards/Products
                logging.info(f"Inferred Hazards: {inferred_hazards}")
                logging.info(f"Inferred Products: {inferred_products}")
                
                if inferred_hazards:
                    st.session_state.hazard_mappings[selected_model] = list(inferred_hazards)
                    st.session_state[f"{key_ns}_hazards"] = list(inferred_hazards)
                if inferred_products:
                    st.session_state.product_mappings[selected_model] = list(inferred_products)
                    st.session_state[f"{key_ns}_products"] = list(inferred_products)
                
                # Update Session State for Hazards/Products
                if inferred_hazards:
                    st.session_state.hazard_mappings[selected_model] = list(inferred_hazards)
                if inferred_products:
                    st.session_state.product_mappings[selected_model] = list(inferred_products)


                # 2. Infer Parameters and Units (Existing Logic)
                inferred_units_file = script_dir / "inferred_units.json"
                mapped_turtle_dir = script_dir / "mapped" / "turtle"
                dirs_to_scan = []
                if mapped_turtle_dir.exists(): dirs_to_scan.append(str(mapped_turtle_dir))
                
                if dirs_to_scan:
                    infer_args = [
                        "--input-dirs", *dirs_to_scan,
                        "--master-mapping", str(MASTER_MAPPING_FILE),
                        "--out", str(inferred_units_file)
                    ]
                    inference_success = run_script("infer_units_from_fskx.py", infer_args)
                    
                    if inference_success and inferred_units_file.exists():
                        try:
                            with open(inferred_units_file, 'r', encoding='utf-8') as f:
                                inferred_data = json.load(f)
                            
                            model_inferred = inferred_data.get(selected_model, {})
                            
                            # Helper to update if not empty
                            def update_if_val(target_dict, key, val):
                                if val: target_dict[key] = val

                            # Collect params
                            all_params_for_prefill = []
                            for model in model_data["cascade"]:
                                all_params_for_prefill.extend(model["parameters"])

                            for param in all_params_for_prefill:
                                p_id = param['id']
                                if p_id in model_inferred:
                                    inf = model_inferred[p_id]
                                    
                                    # Inputs
                                    if "Input" in param.get('classifications', set()):
                                        if 'mapped_unit_term' in inf:
                                            st.session_state.setdefault('input_units', {}).setdefault(selected_model, {})[p_id] = inf['mapped_unit_term']
                                        if 'mapped_parameter_term' in inf:
                                            st.session_state.setdefault('parameter_mappings', {}).setdefault(selected_model, {})[p_id] = inf['mapped_parameter_term']
                                    
                                    # Outputs
                                    if "Output" in param.get('classifications', set()):
                                        if 'mapped_unit_term' in inf:
                                            st.session_state.setdefault('output_units', {}).setdefault(selected_model, {})[p_id] = inf['mapped_unit_term']
                                        if 'mapped_parameter_term' in inf:
                                            st.session_state.setdefault('output_concepts', {}).setdefault(selected_model, {})[p_id] = inf['mapped_parameter_term']

                        except Exception as e:
                            st.error(f"Error reading inferred units JSON: {e}")
                
                st.success(f"Prefill complete. Found {len(inferred_hazards)} hazards, {len(inferred_products)} products.")
                st.rerun()

            # --- Pre-fill mappings from existing turtle file ---
            # Build reverse lookup maps from URI to list of rows for disambiguation
            param_uri_to_rows = {}
            param_concepts_df = master_mapping_df[master_mapping_df['ConceptGroup'].isin(['Model Parameters', 'Parameter', 'Climate Variables', 'InputParameter', 'OutputParameter'])]
            for _, row in param_concepts_df.iterrows():
                for uri_col in ['URI1', 'URI2', 'URI3']:
                    uri = row[uri_col]
                    if uri and pd.notna(uri):
                        if uri not in param_uri_to_rows:
                            param_uri_to_rows[uri] = []
                        param_uri_to_rows[uri].append(row)

            # Load NetCDF vocab for disambiguation context
            netcdf_vocab_path = str(VOCAB_DIR / "ambrosia-netcdf-vocab.ttl")
            netcdf_graph = Graph()
            if Path(netcdf_vocab_path).exists():
                try:
                    netcdf_graph.parse(netcdf_vocab_path, format="turtle")
                except Exception:
                    pass # Ignore errors here, just best effort

            # Prepare concept dataframes for matching
            input_concepts_df = master_mapping_df[master_mapping_df['ConceptGroup'] == 'InputParameter']
            output_concepts_df = master_mapping_df[master_mapping_df['ConceptGroup'] == 'OutputParameter']
            
            input_concept_options = [""] + sorted(input_concepts_df['prefLabel'].unique().tolist())
            output_concept_options = [""] + sorted(output_concepts_df['prefLabel'].unique().tolist())

            # Initialize session state for the selected model
            param_mappings = st.session_state.setdefault('parameter_mappings', {}).setdefault(selected_model, {})
            output_units = st.session_state.setdefault('output_units', {}).setdefault(selected_model, {})
            input_units = st.session_state.setdefault('input_units', {}).setdefault(selected_model, {})

            # Pre-fill parameter mappings by reading the graph
            for model in model_data["models"]:
                for param in model["parameters"]:
                    param_uri = URIRef(param['uri'])
                    
                    # --- Pre-fill Input Units from Graph ---
                    if "Input" in param.get('classifications', set()):
                        input_unit_uri = g.value(subject=param_uri, predicate=AMBLINK.preferredInputUnit)
                        if input_unit_uri:
                            u_match = unit_df[
                                (unit_df['URI1'] == str(input_unit_uri)) | 
                                (unit_df['URI2'] == str(input_unit_uri)) | 
                                (unit_df['URI3'] == str(input_unit_uri))
                            ]
                            if not u_match.empty:
                                input_units[param['id']] = u_match.iloc[0]['prefLabel']

                    # --- Pre-fill Output Units from Graph ---
                    if "Output" in param.get('classifications', set()):
                        # Find if there is an OutputMapping pointing to this parameter
                        om_node = next(g.subjects(AMBLINK.mapsParameter, param_uri), None)
                        if om_node and (om_node, RDF.type, AMBLINK.OutputMapping) in g:
                            unit_uri = g.value(om_node, AMBLINK.hasOutputUnit)
                            if unit_uri:
                                # Reverse lookup unit label from URI in unit_df
                                u_match = unit_df[
                                    (unit_df['URI1'] == str(unit_uri)) | 
                                    (unit_df['URI2'] == str(unit_uri)) | 
                                    (unit_df['URI3'] == str(unit_uri))
                                ]
                                if not u_match.empty:
                                    output_units[param['id']] = u_match.iloc[0]['prefLabel']

                    # Pre-fill parameter mapping
                    existing_param_match = g.value(subject=param_uri, predicate=SKOS.exactMatch)
                    
                    if existing_param_match:
                        uri_str = str(existing_param_match)
                        if uri_str in param_uri_to_rows:
                            rows = param_uri_to_rows[uri_str]
                            
                            # Default to the first one (or last one if we mimicked previous behavior, but let's try to be smarter)
                            # If multiple rows map to this URI, we need to choose.
                            selected_label = rows[0]['prefLabel']
                            
                            if len(rows) > 1:
                                # Ambiguity detected. Try to use existing wiring (source variable) to disambiguate.
                                source_var = None
                                for im in g.subjects(AMBLINK.mapsParameter, param_uri):
                                     # Check if this IM belongs to the selected model (optional but safer)
                                     if (URIRef(model_data['model_uri']), AMBLINK.hasInputMapping, im) in g:
                                         source_var = g.value(im, AMBLINK.isFulfilledBy)
                                         break
                                
                                if source_var:
                                    # Get labels for the wired NetCDF variable
                                    vocab_terms = []
                                    # Query vocab for labels of source_var
                                    for p in [SKOS.prefLabel, SKOS.altLabel, RDFS.label]:
                                        for o in netcdf_graph.objects(source_var, p):
                                            vocab_terms.append(str(o).lower())
                                    # Also add local name
                                    vocab_terms.append(str(source_var).split('/')[-1].lower())
                                    
                                    best_row = None
                                    best_score = -1
                                    
                                    for row in rows:
                                        search_terms = [str(val).lower() for val in [row['Term'], row['altLabels'], row.get('ProviderTerm1', '')] if pd.notna(val) and val]
                                        
                                        score = 0
                                        for s in search_terms:
                                            for t in vocab_terms:
                                                if t in s or s in t: score += 50
                                                if difflib.SequenceMatcher(None, s, t).ratio() > 0.6: score += 20
                                        
                                        if score > best_score:
                                            best_score = score
                                            best_row = row
                                    
                                    if best_row is not None:
                                        selected_label = best_row['prefLabel']
                            
                            param_mappings[param['id']] = selected_label
                        else:
                             st.warning(f"URI {uri_str} found in graph but not in Master Mapping.")
                    
                    # Note: In-memory fuzzy fallback removed to rely on the external inference script.


            # Pre-fill hazards and products, ensuring state is fresh for the selected model
            valid_hazards = [h for h in model_data["hazards"] if h in pathogen_uris]
            valid_products = [p for p in model_data["products"] if p in plant_uris]

            # Ensure the session state keys exist before assignment
            st.session_state.setdefault('hazard_mappings', {})
            st.session_state.setdefault('product_mappings', {})

            st.subheader("Map Hazards and Products")
            
            # --- Display Model Cascade ---
            if len(model_data.get("cascade", [])) > 1:
                st.subheader("Model Cascade")
                cascade_str = " → ".join([f"`{m['name']}`" for m in model_data["cascade"]])
                st.markdown(cascade_str)

            col_h, col_p = st.columns(2)
            with col_h:
                # By removing the 'key', the widget is recreated on each run,
                # correctly using the 'default' value from the newly selected model.
                selected_hazards = st.multiselect(
                    "Hazards",
                    pathogen_uris,
                    default=valid_hazards,
                    format_func=pathogen_display.get,
                    key=f"{key_ns}_hazards"
                )
                st.session_state.hazard_mappings[selected_model] = selected_hazards
            with col_p:
                selected_products = st.multiselect(
                    "Products",
                    plant_uris,
                    default=valid_products,
                    format_func=plant_display.get,
                    key=f"{key_ns}_products"
                )
                st.session_state.product_mappings[selected_model] = selected_products


            if model_data["models"]:
                # Unit options are already correctly filtered from 'Unit' ConceptGroup
                
                # --- Collect all params for prefill ---
                all_input_params_list = []
                all_output_params_list = []
                for model in model_data["cascade"]:
                    m_inputs = sorted([p for p in model["parameters"] if "Input" in p.get('classifications', set())], key=lambda x: x['name'])
                    for p in m_inputs: all_input_params_list.append((model['name'], p))
                    
                    m_outputs = sorted([p for p in model["parameters"] if "Output" in p.get('classifications', set())], key=lambda x: x['name'])
                    for p in m_outputs: all_output_params_list.append((model['name'], p))

                # Initialize session state
                input_units = st.session_state.setdefault('input_units', {}).setdefault(selected_model, {})
                output_units = st.session_state.setdefault('output_units', {}).setdefault(selected_model, {})

                st.subheader("Map Input Parameters")
                # Initialize session state for input units (already done above, but keep for safety if moved)
                input_units = st.session_state.setdefault('input_units', {}).setdefault(selected_model, {})

                any_inputs_found = False
                for model in model_data["cascade"]:
                    inputs = sorted([p for p in model["parameters"] if "Input" in p.get('classifications', set())], key=lambda x: x['name'])
                    if not inputs:
                        continue

                    any_inputs_found = True
                    if len(model_data["models"]) > 1:
                        st.markdown(f"#### Model: `{model['name']}`")
                    else:
                        st.markdown("#### Input Parameters")

                    for param in inputs:
                        col_concept, col_unit = st.columns(2)
                        with col_concept:
                            default_concept = st.session_state.parameter_mappings[selected_model].get(param['id'], "")
                            
                            # Construct label: Always show parenthesis if schema_name exists
                            label_str = f"Concept: **{param['id']}**"
                            if param.get('schema_name'):
                                label_str += f" ({param['schema_name']})"
                                
                            selected_concept = st.selectbox(
                                label_str,
                                input_concept_options,
                                index=input_concept_options.index(default_concept) if default_concept in input_concept_options else 0,
                                key=f"{key_ns}_input_concept_map_{selected_model}_{param['uri']}"
                            )
                            st.session_state.parameter_mappings[selected_model][param['id']] = selected_concept
                        
                        with col_unit:
                            # Pre-fill input units from graph if available
                            # This needs to be done during pre-fill logic, but for now, we'll assume fresh.
                            current_input_unit = st.session_state.input_units[selected_model].get(param['id'], "")
                            raw_unit_disp = f"({param.get('raw_unit_label', '')})" if param.get('raw_unit_label') else ""
                            new_input_unit = st.selectbox(
                                f"Unit for **{param['id']}** {raw_unit_disp}",
                                unit_options,
                                index=unit_options.index(current_input_unit) if current_input_unit in unit_options else 0,
                                key=f"{key_ns}_input_unit_map_{selected_model}_{param['uri']}"
                            )
                            st.session_state.input_units[selected_model][param['id']] = new_input_unit

                if not any_inputs_found:
                    st.info("No input parameters found to map in this model.")
            else:
                st.warning("No parameters found to map in this model.")

            st.subheader("Map Output Parameters")
            # Initialize session state for output concepts
            output_concepts = st.session_state.setdefault('output_concepts', {}).setdefault(selected_model, {})
            # Initialize session state for output units (already exists, but for clarity)
            output_units = st.session_state.setdefault('output_units', {}).setdefault(selected_model, {})

            # 1. Collect all output params first to see if we need to render anything
            all_output_params = []
            for model in model_data["cascade"]:
                 model_outputs = sorted([p for p in model["parameters"] if "Output" in p.get('classifications', set())], key=lambda x: x['name'])
                 for p in m_outputs: all_output_params.append((model['name'], p)) # Just using local var from above loop would be cleaner but let's stick to existing structure or reuse what we calculated above.
                 # Actually, we calculated all_output_params_list above. We can just check if it is empty.
                 # But to minimize change, I will let this loop run or reuse `all_output_params_list`.
                 # Let's reuse all_output_params_list for checking emptiness, but the loop structure below relies on `model['parameters']` iteration which is fine.
            
            # Just re-collect for local use in loop if needed, but we already did it above.
            # Let's stick to the original structure minus the button.
            
            if not all_output_params_list: # Use the list we calculated above
                st.info("No output parameters found.")
            else:
                # Pre-fill existing output concepts and units from graph
                for model in model_data["models"]:
                    for param in model["parameters"]:
                        param_uri = URIRef(param['uri'])
                        
                        # Pre-fill Output Concepts from Graph
                        existing_output_concept_match = g.value(subject=param_uri, predicate=SKOS.exactMatch)
                        if existing_output_concept_match:
                            uri_str = str(existing_output_concept_match)
                            if uri_str in param_uri_to_rows: # Use general param_uri_to_rows, as output concepts might also be there
                                rows = param_uri_to_rows[uri_str]
                                # Similar disambiguation logic as for input concepts if needed, for simplicity take first
                                output_concepts[param['id']] = rows[0]['prefLabel']
                            else:
                                st.warning(f"Output Concept URI {uri_str} found in graph but not in Master Mapping.")

                        # Pre-fill Output Units from Graph (Existing logic remains)
                        if "Output" in param.get('classifications', set()):
                            om_node = next(g.subjects(AMBLINK.mapsParameter, param_uri), None)
                            if om_node and (om_node, RDF.type, AMBLINK.OutputMapping) in g:
                                unit_uri = g.value(om_node, AMBLINK.hasOutputUnit)
                                if unit_uri:
                                    u_match = unit_df[
                                        (unit_df['URI1'] == str(unit_uri)) | 
                                        (unit_df['URI2'] == str(unit_uri)) | 
                                        (unit_df['URI3'] == str(unit_uri))
                                    ]
                                    if not u_match.empty:
                                        output_units[param['id']] = u_match.iloc[0]['prefLabel']
                
                # Render UI for Output Concepts and Units
                for model in model_data["cascade"]:
                    outputs = sorted([p for p in model["parameters"] if "Output" in p.get('classifications', set())], key=lambda x: x['name'])
                    if not outputs: continue
                    
                    if len(model_data["models"]) > 1:
                        st.markdown(f"#### Model: `{model['name']}`")
                    else:
                        st.markdown("#### Output Parameters")

                    for param in outputs:
                        p_id = param['id']
                        col_concept, col_unit = st.columns(2)
                        with col_concept:
                            current_concept = st.session_state.output_concepts[selected_model].get(p_id, "")
                            
                            # Construct label: Always show parenthesis if schema_name exists
                            label_str = f"Concept: **{p_id}**"
                            if param.get('schema_name'):
                                label_str += f" ({param['schema_name']})"

                            new_concept = st.selectbox(
                                label_str, 
                                output_concept_options, 
                                index=output_concept_options.index(current_concept) if current_concept in output_concept_options else 0,
                                key=f"output_concept_{selected_model}_{p_id}"
                            )
                            st.session_state.output_concepts[selected_model][p_id] = new_concept
                        
                        with col_unit:
                            current_unit = st.session_state.output_units[selected_model].get(p_id, "")
                            raw_unit_disp = f"({param.get('raw_unit_label', '')})" if param.get('raw_unit_label') else ""
                            new_unit = st.selectbox(
                                f"Unit for **{p_id}** {raw_unit_disp}", 
                                unit_options, 
                                index=unit_options.index(current_unit) if current_unit in unit_options else 0,
                                key=f"output_unit_{selected_model}_{p_id}"
                            )
                            st.session_state.output_units[selected_model][p_id] = new_unit

            st.markdown("---")
            if st.button(f"Apply All Mappings to {selected_model}", key=f"{key_ns}_apply_mappings"):
                # Validation removed as per user feedback: "it is okay that Units or Parameters are kept empty"
                
                model_uri = model_data["model_uri"]
                
                # 1. Hazards and Products
                if model_uri:
                    g.remove((model_uri, FSKXO.FSKXO_0000000008, None)); g.remove((model_uri, FSKXO.FSKXO_0000000007, None))
                    for uri in st.session_state.hazard_mappings[selected_model]: g.add((model_uri, FSKXO.FSKXO_0000000008, URIRef(uri)))
                    for uri in st.session_state.product_mappings[selected_model]: g.add((model_uri, FSKXO.FSKXO_0000000007, URIRef(uri)))

                    # Remove all existing OutputMappings and associated triples for the model to ensure idempotency
                    for s, p, o in list(g.triples((URIRef(model_uri), AMBLINK.hasOutputMapping, None))):
                        g.remove((s, p, o))     # Remove link from model
                        g.remove((o, None, None)) # Remove the OutputMapping node and its properties
                    
                    # Also remove existing SKOS.exactMatch for output parameters to allow recreation
                    for param in all_output_params:
                        param_uri = URIRef(param[1]['uri'])
                        g.remove((param_uri, SKOS.exactMatch, None))

                # Iterate through all parameters from all models
                all_params = [param for model in model_data["models"] for param in model["parameters"]]

                for param in all_params:
                    param_uri = URIRef(param['uri'])
                    
                    # --- Input Parameter Concept and Unit Mapping ---
                    if "Input" in param.get('classifications', set()):
                        # Remove existing mappings first
                        g.remove((param_uri, SKOS.exactMatch, None))
                        g.remove((param_uri, AMBLINK.expectsQuantityKind, None)) # This was removed, so ensure it's not re-added accidentally
                        g.remove((param_uri, AMBLINK.preferredInputUnit, None)) # Remove existing preferred input unit

                        mapped_concept = st.session_state.parameter_mappings[selected_model].get(param['id'])
                        mapped_unit_label = st.session_state.input_units[selected_model].get(param['id'])

                        # Add new Concept mapping
                        if mapped_concept and not input_concepts_df[input_concepts_df['prefLabel'] == mapped_concept].empty:
                            param_row = input_concepts_df[input_concepts_df['prefLabel'] == mapped_concept].iloc[0]
                            # Use all available URIs for exactMatch
                            raw_uris = param_row[['URI1', 'URI2', 'URI3']].values.flatten().tolist()
                            uris = [str(u).strip() for u in raw_uris if u and pd.notna(u) and str(u).lower() != 'nan' and str(u).strip() != '']
                            for u in uris:
                                g.add((param_uri, SKOS.exactMatch, URIRef(u)))
                        
                        # Add new Unit mapping
                        if mapped_unit_label:
                            unit_uri = unit_label_to_uri.get(mapped_unit_label)
                            if unit_uri:
                                g.add((param_uri, AMBLINK.preferredInputUnit, URIRef(unit_uri)))

                    # --- Output Mappings (Concept and Unit) ---
                    if "Output" in param.get('classifications', set()):
                        output_concept_label = st.session_state.output_concepts[selected_model].get(param['id'])
                        output_unit_label = st.session_state.output_units[selected_model].get(param['id'])
                        
                        # Create OutputMapping Node (includes concept and unit)
                        if output_concept_label or output_unit_label: # Create mapping node if either is present
                            om_node = BNode()
                            g.add((URIRef(model_uri), AMBLINK.hasOutputMapping, om_node))
                            g.add((om_node, RDF.type, AMBLINK.OutputMapping))
                            g.add((om_node, AMBLINK.mapsParameter, param_uri))

                            # Add Output Concept mapping
                            if output_concept_label and not output_concepts_df[output_concepts_df['prefLabel'] == output_concept_label].empty:
                                concept_row = output_concepts_df[output_concepts_df['prefLabel'] == output_concept_label].iloc[0]
                                # Use all available URIs for exactMatch
                                raw_uris = concept_row[['URI1', 'URI2', 'URI3']].values.flatten().tolist()
                                uris = [str(u).strip() for u in raw_uris if u and pd.notna(u) and str(u).lower() != 'nan' and str(u).strip() != '']
                                
                                # Link observed property on the mapping node (using the first valid URI)
                                if uris:
                                    g.add((om_node, AMBLINK.producesObservedProperty, URIRef(uris[0])))
                                
                                # Link exact matches on the parameter node
                                for u in uris:
                                    g.add((param_uri, SKOS.exactMatch, URIRef(u))) 
                            
                            # Add Output Unit mapping
                            if output_unit_label: 
                                unit_uri = unit_label_to_uri.get(output_unit_label)
                                if unit_uri:
                                    g.add((om_node, AMBLINK.hasOutputUnit, URIRef(unit_uri)))

                # --- New Wiring Logic based on fskx_to_rdf.py ---
                if model_uri:
                    try:
                        netcdf_vocab_path = str(VOCAB_DIR / "ambrosia-netcdf-vocab.ttl")
                        knowledge_graph = Graph()
                        if Path(netcdf_vocab_path).exists():
                            knowledge_graph.parse(netcdf_vocab_path, format="turtle")
                        else:
                            st.warning(f"NetCDF vocabulary not found at {netcdf_vocab_path}. Cannot perform automatic wiring.")

                        # Remove all existing input mappings to avoid duplicates
                        for s, p, o in list(g.triples((model_uri, AMBLINK.hasInputMapping, None))):
                            g.remove((s, p, o))
                            g.remove((o, None, None))

                        # Iterate through all parameters again to create wiring
                        for param in all_params:
                            if 'Input' not in param.get('classifications', set()):
                                continue

                            mapped_concept = st.session_state.parameter_mappings[selected_model].get(param['id'])
                            if not mapped_concept:
                                continue

                            param_rows = input_concepts_df[input_concepts_df['prefLabel'] == mapped_concept] # Use input_concepts_df here
                            if param_rows.empty:
                                continue
                            param_row = param_rows.iloc[0]

                            # 1. Gather Candidates via URI
                            raw_uris = param_row[['URI1', 'URI2', 'URI3']].values.flatten().tolist()
                            target_uris = [str(u).strip() for u in raw_uris if u and pd.notna(u) and str(u).lower() != 'nan' and str(u).strip() != '']
                            candidates = []
                            
                            for uri in target_uris:
                                # SPARQL: Find NetCDF vars matching this URI
                                q = prepareQuery("""SELECT ?ds ?varName ?pref ?alt WHERE { 
                                    ?ds skos:exactMatch ?uri . 
                                    OPTIONAL { ?ds amblink:sourceVariableName ?varName }
                                    OPTIONAL { ?ds skos:prefLabel ?pref } 
                                    OPTIONAL { ?ds skos:altLabel ?alt } 
                                }""", initNs={"skos": SKOS, "amblink": AMBLINK})
                                
                                results = list(knowledge_graph.query(q, initBindings={'uri': URIRef(uri)}))
                                for res in results:
                                    if any(c['ds'] == res.ds for c in candidates):
                                        continue
                                        
                                    cand = {
                                        'ds': res.ds,
                                        'varName': str(res.varName) if res.varName else str(res.ds).split('/')[-1],
                                        'pref': str(res.pref) if res.pref else "",
                                        'alt': str(res.alt) if res.alt else "",
                                        'id': str(res.ds).split('/')[-1]
                                    }
                                    candidates.append(cand)

                            # 2. Resolve Ambiguity
                            final_match = None
                            if len(candidates) == 0:
                                st.warning(f"No NetCDF variable found for parameter '{param['name']}' (mapped to '{mapped_concept}'). URIs checked: {target_uris}")
                                continue
                            elif len(candidates) == 1:
                                final_match = candidates[0]
                            else:
                                # Disambiguation (Fuzzy String Scoring)
                                # Gather search terms from Excel
                                provider_term = param_row['ProviderTerm1'] if 'ProviderTerm1' in param_row else ''
                                search_terms = [str(val).lower() for val in [param_row['Term'], param_row['altLabels'], provider_term] if pd.notna(val) and val]
                                
                                best_cand, best_score = None, -1
                                
                                for cand in candidates:
                                    # Gather target terms from RDF
                                    target_terms = [t.lower() for t in [cand['pref'], cand['alt'], cand['varName'], cand['id']] if t]
                                    
                                    # Simple Scoring
                                    score = 0
                                    for s in search_terms:
                                        for t in target_terms:
                                            if t in s or s in t: score += 50
                                            if difflib.SequenceMatcher(None, s, t).ratio() > 0.6: score += 20
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_cand = cand
                                
                                final_match = best_cand
                                st.info(f"Disambiguated '{param['name']}' -> {final_match['varName']} (Score: {best_score}) among {[c['varName'] for c in candidates]}")

                            # 3. Create Triples using final_match
                            if final_match:
                                input_mapping_node = BNode()
                                g.add((model_uri, AMBLINK.hasInputMapping, input_mapping_node))
                                g.add((input_mapping_node, RDF.type, AMBLINK.InputMapping))
                                g.add((input_mapping_node, AMBLINK.mapsParameter, URIRef(param['uri'])))
                                g.add((input_mapping_node, AMBLINK.isFulfilledBy, final_match['ds']))
                                g.add((input_mapping_node, AMBLINK.sourceVariableName, Literal(final_match['varName'])))
                                st.success(f"Successfully wired parameter '{param['name']}' to data source '{final_match['varName']}'.")

                    except Exception as e:
                        st.error(f"An error occurred during the wiring process: {e}")


                _bind_common_namespaces(g)
                g.serialize(destination=str(turtle_file), format="turtle")
                st.success(f"Mappings applied and saved to {turtle_file.name}")
                st.rerun()

if __name__ == "__main__":
    render_fskx_to_rdf_ui()
