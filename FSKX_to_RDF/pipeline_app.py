import streamlit as st
import subprocess
import sys
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
                df[col] = df[col].astype(str).fillna('')
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
            SELECT ?param ?id ?name ?classification WHERE {
                ?model fskxo:FSKXO_0000000016 / fskxo:FSKXO_0000000017 ?param .
                OPTIONAL { ?param dcterms:identifier ?id_dcterms . }
                OPTIONAL { ?param schema:id ?id_schema . }
                BIND(COALESCE(?id_dcterms, ?id_schema) AS ?id)
                FILTER(BOUND(?id))
                
                OPTIONAL { ?param schema:name ?name . }
                OPTIONAL { ?param fskxo:FSKXO_0000017519 ?classification . }
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
                temp_params[param_uri] = {
                    'uri': param_uri, 'id': param_id, 'name': str(row.name or param_id),
                    'classifications': set()
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
    override_existing = st.sidebar.checkbox("Re-process all existing models", value=False, key=f"{key_ns}_override")
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
        fskx_to_jsonld_args = [str(FSKX_DIR)] + script_args
        with st.status("Running pipeline...", expanded=True) as status:
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
    if not master_mapping_df.empty:
        master_mapping_df['prefLabel'] = master_mapping_df.apply(lambda r: _get_display_label(r['Term'], r['altLabels']), axis=1)

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

            # --- Pre-fill mappings from existing turtle file ---
            # Build reverse lookup maps from URI to prefLabel for pre-filling dropdowns
            param_uri_to_prefLabel = {}
            param_concepts_df = master_mapping_df[master_mapping_df['ConceptGroup'].isin(['Model Parameters', 'Parameter', 'Climate Variables'])]
            for _, row in param_concepts_df.iterrows():
                for uri_col in ['URI1', 'URI2', 'URI3']:
                    uri = row[uri_col]
                    if uri and pd.notna(uri):
                        param_uri_to_prefLabel[uri] = row['prefLabel']

            # Initialize session state for the selected model
            param_mappings = st.session_state.setdefault('parameter_mappings', {}).setdefault(selected_model, {})

            # Pre-fill parameter mappings by reading the graph
            for model in model_data["models"]:
                for param in model["parameters"]:
                    param_uri = URIRef(param['uri'])
                    
                    # Pre-fill parameter mapping
                    existing_param_match = g.value(subject=param_uri, predicate=SKOS.exactMatch)
                    if existing_param_match and str(existing_param_match) in param_uri_to_prefLabel:
                        param_mappings[param['id']] = param_uri_to_prefLabel[str(existing_param_match)]


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


            st.subheader("Map Climate Input Parameters")
            if model_data["models"]:
                param_concepts_df = master_mapping_df[master_mapping_df['ConceptGroup'].isin(['Model Parameters', 'Parameter', 'Climate Variables'])]
                param_options = [""] + sorted(param_concepts_df['prefLabel'].unique().tolist())

                # --- Render Input Parameters ---
                any_inputs_found = False
                # Use the cascade order for rendering
                for model in model_data["cascade"]:
                    inputs = sorted([p for p in model["parameters"] if "Input" in p.get('classifications', set())], key=lambda x: x['name'])
                    if not inputs:
                        continue

                    any_inputs_found = True
                    # Display a header for the model, especially if there's more than one.
                    if len(model_data["models"]) > 1:
                        st.markdown(f"#### Model: `{model['name']}`")
                    else:
                        st.markdown("#### Input Parameters")

                    for param in inputs:
                        default = st.session_state.parameter_mappings[selected_model].get(param['id'], "")
                        selected_concept = st.selectbox(
                            f"Map: **{param['name']}** (`{param['id']}`)",
                            param_options,
                            index=param_options.index(default) if default in param_options else 0,
                            key=f"{key_ns}_param_map_{selected_model}_{param['uri']}" # URI is unique across models
                        )
                        st.session_state.parameter_mappings[selected_model][param['id']] = selected_concept
                
                if not any_inputs_found:
                    st.info("No input parameters found to map in this model.")
            else:
                st.warning("No parameters found to map in this model.")

            st.markdown("---")
            if st.button(f"Apply All Mappings to {selected_model}", key=f"{key_ns}_apply_mappings"):
                model_uri = model_data["model_uri"]
                if model_uri:
                    g.remove((model_uri, FSKXO.FSKXO_0000000008, None)); g.remove((model_uri, FSKXO.FSKXO_0000000007, None))
                    for uri in st.session_state.hazard_mappings[selected_model]: g.add((model_uri, FSKXO.FSKXO_0000000008, URIRef(uri)))
                    for uri in st.session_state.product_mappings[selected_model]: g.add((model_uri, FSKXO.FSKXO_0000000007, URIRef(uri)))

                # Iterate through all parameters from all models
                all_params = [param for model in model_data["models"] for param in model["parameters"]]

                for param in all_params:
                    param_uri = URIRef(param['uri'])
                    
                    # --- Parameter Concept Mapping ---
                    # Always remove existing parameter mappings first.
                    g.remove((param_uri, SKOS.exactMatch, None))
                    g.remove((param_uri, AMBLINK.expectsQuantityKind, None))
                    mapped_concept = st.session_state.parameter_mappings[selected_model].get(param['id'])
                    # Add new mappings only if a valid one is selected.
                    if mapped_concept and not master_mapping_df[master_mapping_df['prefLabel'] == mapped_concept].empty:
                        param_row = master_mapping_df[master_mapping_df['prefLabel'] == mapped_concept].iloc[0]
                        uris = [u for u in param_row[['URI1', 'URI2', 'URI3']] if u]
                        concept_uri = next((u for u in uris if 'qudt.org' not in u), None)
                        qk_uri = next((u for u in uris if 'quantitykind' in u), None)
                        if concept_uri: g.add((param_uri, SKOS.exactMatch, URIRef(concept_uri)))
                        if qk_uri: g.add((param_uri, AMBLINK.expectsQuantityKind, URIRef(qk_uri)))

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

                            param_row = master_mapping_df[master_mapping_df['prefLabel'] == mapped_concept]
                            if param_row.empty:
                                continue

                            uris = [u for u in param_row.iloc[0][['URI1', 'URI2', 'URI3']] if u]
                            
                            # Iterate through all potential URIs from the master mapping to find a match.
                            wiring_successful = False
                            for potential_qk_uri in uris:
                                if not potential_qk_uri: continue

                                query = prepareQuery(
                                    "SELECT ?ds ?varName WHERE { ?ds amblink:expectsQuantityKind ?qk . OPTIONAL { ?ds amblink:sourceVariableName ?varName . } }",
                                    initNs={"amblink": AMBLINK}
                                )
                                results = list(knowledge_graph.query(query, initBindings={'qk': URIRef(potential_qk_uri)}))
                                
                                if results:
                                    first_result = results[0]
                                    data_source_uri = first_result.ds
                                    source_var_name = first_result.varName or Literal(str(data_source_uri).split('/')[-1])

                                    input_mapping_node = BNode()
                                    g.add((model_uri, AMBLINK.hasInputMapping, input_mapping_node))
                                    g.add((input_mapping_node, RDF.type, AMBLINK.InputMapping))
                                    g.add((input_mapping_node, AMBLINK.mapsParameter, URIRef(param['uri'])))
                                    g.add((input_mapping_node, AMBLINK.isFulfilledBy, data_source_uri))
                                    g.add((input_mapping_node, AMBLINK.sourceVariableName, source_var_name))
                                    st.info(f"Successfully wired parameter '{param['name']}' to data source '{source_var_name}'.")
                                    wiring_successful = True
                                    # Break after the first successful wiring for this parameter
                                    break
                            
                            if not wiring_successful:
                                st.warning(f"No data source found for any of the URIs associated with parameter '{param['name']}'. URIs checked: {uris}")
                    except Exception as e:
                        st.error(f"An error occurred during the wiring process: {e}")


                _bind_common_namespaces(g)
                g.serialize(destination=str(turtle_file), format="turtle")
                st.success(f"Mappings applied and saved to {turtle_file.name}")
                st.rerun()

if __name__ == "__main__":
    render_fskx_to_rdf_ui()
