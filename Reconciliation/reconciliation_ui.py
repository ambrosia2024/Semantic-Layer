# -*- coding: utf-8 -*-
import sys
import streamlit as st
import os
import pandas as pd
import time
import concurrent.futures
import logging

# --- Configure Logging (Console Only) ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Temporarily set to DEBUG for troubleshooting
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(stream_handler)
    logger.info("Console logging configured at DEBUG level for reconciliation_ui.")

# --- Custom Module Imports (using relative imports for package structure) ---
try:
    from . import processing_service # Explicit relative import
    from .processing_service import fetch_suggestions_for_term_from_provider # Import the new function
except ImportError as import_err:
    st.error(f"Critical Import Error (reconciliation_ui): Could not import '.processing_service'. Error: {import_err}")
    logger.critical(f"Critical Import Error (reconciliation_ui): Could not import '.processing_service'. Error: {import_err}", exc_info=True)
    # If this is the main entry point, stop. Otherwise, re-raise.
    if __name__ == "__main__":
        st.stop()
    else:
        raise

semantic_search = None
try:
    from . import semantic_search # Explicit relative import
    logger.info("Successfully imported '.semantic_search' module.")
except ImportError as import_err:
    logger.warning(f"Import Warning (reconciliation_ui): Could not import '.semantic_search'. Cosine similarity will be unavailable. Error: {import_err}")

# --- Custom Provider Ontology Fetchers ---
# Import get_available_ontologies from relevant providers
try:
    from .bioportal_provider import get_available_ontologies as get_bioportal_ontologies
    from .earthportal_provider import get_available_ontologies as get_earthportal_ontologies
    from .ols_provider import get_available_ontologies as get_ols_ontologies
    from .semlookp_provider import get_available_ontologies as get_semlookp_ontologies
    from .agroportal_provider import get_available_ontologies as get_agroportal_ontologies
    logger.info("Successfully imported ontology fetchers from provider modules.")
except ImportError as import_err:
    logger.warning(f"Import Warning (reconciliation_ui): Could not import one or more ontology fetchers. Ontology filtering may be limited. Error: {import_err}")

# --- Import from reconciliation_utils.py ---
try:
    from .reconciliation_utils import (
        NO_MATCH_URI, NO_MATCH_DISPLAY, CUSTOM_SPARQL_PROVIDER_NAME, DEFAULT_SPARQL_QUERY_TEMPLATE,
        load_config, CONFIG, USER_AGENT,
        calculate_levenshtein_score, format_suggestion_display, create_download_link,
        get_combined_and_sorted_suggestions, prefill_best_matches, render_pagination_controls_ui
    )
    logger.info("Successfully imported utilities from 'reconciliation_utils.py'.")
except ImportError as import_err:
    st.error(f"Critical Import Error (reconciliation_ui): Could not import from 'reconciliation_utils.py'. Error: {import_err}")
    logger.critical(f"Critical Import Error (reconciliation_ui): Could not import from 'reconciliation_utils.py'. Error: {import_err}", exc_info=True)
    if __name__ == "__main__":
        st.stop()
    else:
        raise

# --- Main UI Rendering Function ---
def render_reconciliation_ui():
    """Renders the entire Reconciliation Tool UI and handles its logic."""
    # st.set_page_config(layout="wide") # Removed as it should only be called once per app page

    # --- Session State Initialization (inside the main function) ---
    if 'df' not in st.session_state: st.session_state['df'] = None
    if 'suggestions' not in st.session_state: st.session_state['suggestions'] = {}
    if 'selected_uris' not in st.session_state: st.session_state['selected_uris'] = {} 
    if 'last_uploaded_filename' not in st.session_state: st.session_state['last_uploaded_filename'] = None
    if 'semantic_model' not in st.session_state: st.session_state['semantic_model'] = None
    if 'provider_queue' not in st.session_state: st.session_state['provider_queue'] = []
    if 'provider_status' not in st.session_state: st.session_state['provider_status'] = {} 
    if 'total_indices_to_process' not in st.session_state: st.session_state['total_indices_to_process'] = [] 
    if 'display_provider' not in st.session_state: st.session_state['display_provider'] = None 
    if 'display_mixed_results' not in st.session_state: st.session_state['display_mixed_results'] = False # New state for mixed results
    if 'provider_has_results' not in st.session_state: st.session_state['provider_has_results'] = set() 
    if 'processing_active' not in st.session_state: st.session_state['processing_active'] = False 
    if 'sidebar_provider_progress_text_ph_obj' not in st.session_state: st.session_state['sidebar_provider_progress_text_ph_obj'] = None
    if 'sidebar_provider_progress_bar_ph_obj' not in st.session_state: st.session_state['sidebar_provider_progress_bar_ph_obj'] = None
    if 'stop_processing_requested' not in st.session_state: st.session_state['stop_processing_requested'] = False # New state for stop button
    
    # New state variables for parallel processing
    if 'processed_terms_count' not in st.session_state: st.session_state['processed_terms_count'] = 0
    if 'current_term_index_processing' not in st.session_state: st.session_state['current_term_index_processing'] = 0 # Index within total_indices_to_process list

    if 'custom_sparql_enabled' not in st.session_state: st.session_state['custom_sparql_enabled'] = False
    if 'custom_sparql_endpoint' not in st.session_state: st.session_state['custom_sparql_endpoint'] = ""
    if 'custom_sparql_query_template' not in st.session_state: st.session_state['custom_sparql_query_template'] = DEFAULT_SPARQL_QUERY_TEMPLATE
    if 'custom_sparql_var_uri' not in st.session_state: st.session_state['custom_sparql_var_uri'] = "uri"
    if 'custom_sparql_var_label' not in st.session_state: st.session_state['custom_sparql_var_label'] = "label"
    if 'custom_sparql_var_description' not in st.session_state: st.session_state['custom_sparql_var_description'] = "description"
    if 'csv_load_error_message' not in st.session_state: st.session_state['csv_load_error_message'] = None
    if 'data_source_message' not in st.session_state: st.session_state['data_source_message'] = None # For messages about data source
    if 'linked_preprocessed_data_df' not in st.session_state: st.session_state['linked_preprocessed_data_df'] = None # For the preprocessed data from generator
    
    # NCBI specific state
    if 'ncbi_selected_databases' not in st.session_state:
        st.session_state['ncbi_selected_databases'] = ['taxonomy', 'bioproject', 'gene', 'protein', 'nuccore', 'biosample', 'sra', 'pubmed'] # Default selection
    
    # Dialog specific state
    if 'custom_search_terms' not in st.session_state: st.session_state['custom_search_terms'] = {} 
    if 'custom_search_results' not in st.session_state: st.session_state['custom_search_results'] = {} 
    if 'active_reconciliation_index' not in st.session_state: st.session_state['active_reconciliation_index'] = None
    if 'dialog_selected_suggestion_uri' not in st.session_state: st.session_state['dialog_selected_suggestion_uri'] = None 
    if 'dialog_selected_suggestion_source' not in st.session_state: st.session_state['dialog_selected_suggestion_source'] = None 
    
    if 'semantic_model_load_attempted' not in st.session_state: st.session_state['semantic_model_load_attempted'] = False
    if 'matching_strategy_radio' not in st.session_state: st.session_state['matching_strategy_radio'] = "API Ranking" # Default
    if 'suggestion_slider' not in st.session_state: st.session_state['suggestion_slider'] = 10 # Default
    if 'levenshtein_threshold_slider' not in st.session_state: st.session_state['levenshtein_threshold_slider'] = 0.7 # Default threshold
    if 'show_only_matched_terms' not in st.session_state: st.session_state['show_only_matched_terms'] = False
    if 'show_only_unreconciled_terms' not in st.session_state: st.session_state['show_only_unreconciled_terms'] = False # New state variable
    if 'items_per_page' not in st.session_state: st.session_state['items_per_page'] = 10 # Default items per page
    if 'current_page' not in st.session_state: st.session_state['current_page'] = 1
    if 'skos_matching_enabled' not in st.session_state: st.session_state['skos_matching_enabled'] = False

    # New state variables for ontology filtering
    if 'available_ontologies_by_provider' not in st.session_state: st.session_state['available_ontologies_by_provider'] = {}
    if 'selected_ontologies_by_provider' not in st.session_state: st.session_state['selected_ontologies_by_provider'] = {}
    if 'ontology_loading_status' not in st.session_state: st.session_state['ontology_loading_status'] = {} # e.g., {'BioPortal': 'loading', 'AgroPortal': 'loaded', 'OLS (EBI)': 'error'}

    st.title("Reconciliation Service")
    st.write("Upload CSV, select sources, manage queue via sidebar, reconcile terms.")

    if CONFIG is None:
        st.error("Critical Error: 'config.yaml' could not be loaded. App cannot continue.")
        return 

    # --- Helper function to reset state and load DataFrame ---
    def _reset_state_and_load_df(df_to_load, source_name_msg, is_from_shared_generator=False):
        logger.info(f"Resetting state and loading df from source: {source_name_msg}")
        st.session_state['suggestions'] = {}
        st.session_state['selected_uris'] = {}
        st.session_state['provider_queue'] = []
        st.session_state['provider_status'] = {}
        st.session_state['display_provider'] = None
        st.session_state['provider_has_results'] = set()
        st.session_state['processing_active'] = False
        st.session_state['semantic_model'] = None
        st.session_state['semantic_model_load_attempted'] = False
        st.session_state['csv_load_error_message'] = None
        st.session_state['stop_processing_requested'] = False
        st.session_state['processed_terms_count'] = 0
        st.session_state['current_term_index_processing'] = 0
        st.session_state['active_reconciliation_index'] = None

        st.session_state['df'] = df_to_load.copy().fillna('')
        st.session_state['last_uploaded_filename'] = source_name_msg # Use source_name as a placeholder or actual filename

        # Ensure required columns exist and are of correct type
        required_cols_check = ["Term", "URI", "RDF Role", "Match Type"]
        missing_cols = [col for col in required_cols_check if col not in st.session_state.df.columns]
        if missing_cols:
            err_msg = f"Loaded data is missing required columns: {', '.join(missing_cols)}"
            logger.error(err_msg)
            st.session_state['csv_load_error_message'] = err_msg
            st.session_state['df'] = None # Invalidate df
            return False # Indicate failure

        if 'URI' not in st.session_state.df.columns: st.session_state.df['URI'] = ''
        st.session_state.df['URI'] = st.session_state.df['URI'].astype(str)
        if 'Source Provider' not in st.session_state.df.columns: st.session_state.df['Source Provider'] = ''
        if 'Provider Term' not in st.session_state.df.columns: st.session_state.df['Provider Term'] = ''
        if 'Provider Description' not in st.session_state.df.columns: st.session_state.df['Provider Description'] = ''
        if 'Confirmed Display String' not in st.session_state.df.columns: st.session_state.df['Confirmed Display String'] = ''
        st.session_state.df = st.session_state.df.fillna({'URI': '', 'Source Provider': '', 'Provider Term': '', 'Provider Description': '', 'Confirmed Display String': ''})

        st.session_state['total_indices_to_process'] = list(
            st.session_state.df[
                (st.session_state.df['URI'] == '') |
                (st.session_state.df['URI'] == NO_MATCH_URI)
            ].index
        )
        logger.info(f"Loaded df from '{source_name_msg}'. Found {len(st.session_state.total_indices_to_process)} terms needing URI.")
        st.session_state['data_source_message'] = f"Data successfully loaded from: {source_name_msg}."

        # Store all unique terms for downstream apps (e.g., RDF Generator)
        if 'Term' in st.session_state.df.columns:
            all_terms_list = st.session_state.df['Term'].astype(str).unique().tolist()
            st.session_state['all_terms_for_reconciliation'] = all_terms_list
            logger.info(f"Stored {len(all_terms_list)} unique terms in 'all_terms_for_reconciliation'.")
        else:
            st.session_state['all_terms_for_reconciliation'] = []
            logger.warning("'Term' column not found in loaded DataFrame, 'all_terms_for_reconciliation' will be empty.")
        
        if not is_from_shared_generator: # If loading from file upload, ensure any old shared link is cleared
             st.session_state['linked_preprocessed_data_df'] = None

        return True # Indicate success
    # --- End Helper Function ---

    persistent_missing_configs = []
    ncbi_api_key = CONFIG.get('ncbi', {}).get('api_key')
    if not ncbi_api_key or ncbi_api_key == 'YourAPIKey':
        persistent_missing_configs.append("NCBI API Key (for NCBI provider)")
    bioportal_api_key = CONFIG.get('bioportal', {}).get('api_key')
    if not bioportal_api_key or bioportal_api_key == 'YourAPIKey': 
        persistent_missing_configs.append("BioPortal API Key (for BioPortal provider)")
    agroportal_api_key = CONFIG.get('agroportal', {}).get('api_key') # Add AgroPortal API Key check
    if not agroportal_api_key or agroportal_api_key == 'YourAPIKey':
        persistent_missing_configs.append("AgroPortal API Key (for AgroPortal provider)")
    geonames_username = CONFIG.get('geonames', {}).get('username')
    if persistent_missing_configs:
        st.warning(f"**Configuration Alert:** The following required items are missing from your `config.yaml` and are needed for full functionality: {'; '.join(persistent_missing_configs)}. Some providers may not work correctly.", icon="⚠️")

    # --- Data Input Selection ---
    st.subheader("1. Select Data Source for Reconciliation")

    data_loaded_this_run = False # Flag to prevent multiple loads/reruns in one script execution

    # Option 1: Use data from Matching Table Generator
    shared_matching_table_df_candidate = st.session_state.get('shared_matching_table')
    can_load_shared = False
    if shared_matching_table_df_candidate is not None and isinstance(shared_matching_table_df_candidate, pd.DataFrame) and not shared_matching_table_df_candidate.empty:
        required_cols_shared = ["Term", "URI", "RDF Role", "Match Type"] # Columns expected from generator
        if all(col in shared_matching_table_df_candidate.columns for col in required_cols_shared):
            can_load_shared = True
            if st.button("Load Data from Matching Table Generator", key="load_shared_data_button"):
                if _reset_state_and_load_df(shared_matching_table_df_candidate, "Matching Table Generator", is_from_shared_generator=True):
                    data_loaded_this_run = True
        else:
            st.warning("Data from Generator is available but missing required columns (Term, URI, RDF Role, Match Type). Cannot load.", icon="⚠️")
    
    if can_load_shared:
        st.markdown("--- OR ---")

    # Option 2: Upload a new CSV or Excel file
    uploaded_file = st.file_uploader("Upload New Matching Table (CSV, XLSX, XLS)", type=["csv", "xlsx", "xls"], key="file_uploader_csv_excel")

    should_process_upload = False
    if uploaded_file is not None and not data_loaded_this_run:
        if st.session_state.get('df') is None:
            should_process_upload = True
        elif uploaded_file.name != st.session_state.get('last_uploaded_filename'):
            should_process_upload = True

    if should_process_upload:
        logger.info(f"Processing uploaded file: {uploaded_file.name}")
        try:
            df_loaded = None
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            if file_extension == '.csv':
                common_seps = [',', ';', '\t']
                last_exception = None

                for sep_try in common_seps:
                    try:
                        uploaded_file.seek(0)
                        df_attempt = pd.read_csv(uploaded_file, sep=sep_try, encoding='utf-8', skipinitialspace=True)
                        if df_attempt.shape[1] > 1:
                            df_loaded = df_attempt
                            logger.info(f"Detected CSV separator: '{repr(sep_try)}'")
                            break 
                        else:
                            last_exception = ValueError(f"Only one column with sep '{repr(sep_try)}'.")
                    except Exception as parse_e:
                        last_exception = parse_e
                        logger.debug(f"CSV parsing failed for sep '{repr(sep_try)}': {parse_e}")
                        continue
                
                if df_loaded is None:
                    logger.info("Common separators failed. Trying pandas auto-detection (engine=python)...")
                    try:
                        uploaded_file.seek(0)
                        df_attempt = pd.read_csv(uploaded_file, encoding='utf-8', sep=None, engine='python', skipinitialspace=True)
                        if df_attempt.shape[1] > 1:
                            df_loaded = df_attempt
                            logger.info("Pandas auto-detection (UTF-8, python engine) successful.")
                        else:
                            last_exception = ValueError("Pandas Auto-Detect (UTF-8, python engine) resulted in <= 1 column.")
                    except Exception as auto_utf8_e:
                        logger.warning(f"UTF-8 auto-detect failed: {auto_utf8_e}. Trying without specified encoding (engine=python).")
                        last_exception = auto_utf8_e
                        try:
                            uploaded_file.seek(0)
                            df_attempt = pd.read_csv(uploaded_file, sep=None, engine='python', skipinitialspace=True)
                            if df_attempt.shape[1] > 1:
                                df_loaded = df_attempt
                                logger.info("Pandas auto-detection (no encoding, python engine) successful.")
                            else:
                                last_exception = ValueError("Pandas Auto-Detect (no encoding, python engine) resulted in <= 1 column.")
                        except Exception as final_auto_e:
                            last_exception = final_auto_e
                            logger.error(f"Final auto-detection attempt failed: {final_auto_e}")
                
                if df_loaded is None:
                    err_msg = f"Failed to parse CSV. Check separator/encoding. Last error: {last_exception or 'Unknown parsing error'}"
                    logger.error(err_msg)
                    st.session_state['csv_load_error_message'] = err_msg
                    st.session_state['df'] = None 
                else:
                    if _reset_state_and_load_df(df_loaded, uploaded_file.name, is_from_shared_generator=False):
                        data_loaded_this_run = True
            elif file_extension in ['.xlsx', '.xls']:
                try:
                    uploaded_file.seek(0)
                    df_loaded = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == '.xlsx' else 'xlrd')
                    if _reset_state_and_load_df(df_loaded, uploaded_file.name, is_from_shared_generator=False):
                        data_loaded_this_run = True
                except Exception as e:
                    err_msg = f"Error reading Excel file: {e}. Please ensure it's a valid Excel format."
                    logger.exception(err_msg)
                    st.session_state['csv_load_error_message'] = err_msg
                    st.session_state['df'] = None
                    data_loaded_this_run = True
            else:
                err_msg = f"Unsupported file type: {file_extension}. Please upload a CSV, XLSX, or XLS file."
                logger.error(err_msg)
                st.session_state['csv_load_error_message'] = err_msg
                st.session_state['df'] = None
                data_loaded_this_run = True
        except Exception as e: # Catch errors from parsing attempts
            err_msg = f"Unexpected error reading uploaded file: {e}"
            logger.exception(err_msg)
            st.session_state['csv_load_error_message'] = err_msg
            st.session_state['df'] = None # Ensure df is None on error
            data_loaded_this_run = True
        
        if data_loaded_this_run:
            st.rerun()

    if st.session_state.get('csv_load_error_message'):
        st.error(st.session_state.get('csv_load_error_message'))
        st.session_state.csv_load_error_message = None 
        return

    if st.session_state.get('data_source_message') and not st.session_state.get('df') is None :
        st.success(st.session_state.get('data_source_message'))
        st.session_state.data_source_message = None


    if st.session_state.get('df') is None:
        if not can_load_shared and uploaded_file is None :
             st.info("Please select a data source (from Generator or upload a CSV) to begin.")
        return

    with st.sidebar:
        st.header("Settings & Control")
        provider_tooltips = {
            "Wikidata": "A large, collaboratively edited multilingual knowledge graph hosted by the Wikimedia Foundation. Good for general concepts, people, places, organizations.",
            "NCBI": "National Center for Biotechnology Information. Provides access to biomedical and genomic information databases like PubMed, GenBank, MeSH, etc. Requires API Key.",
            "BioPortal": "A comprehensive repository of biomedical ontologies developed by Stanford University. Covers a wide range of life science domains. Requires API Key.",
            "OLS (EBI)": "Ontology Lookup Service from the European Bioinformatics Institute (EBI). Provides access to a vast collection of life science ontologies.",
            "SemLookP": "Semantic Lookup Platform for Life Sciences. Aggregates multiple biomedical resources for search and annotation.",
            "AgroPortal": "A portal for agricultural and food domain ontologies, providing access to various terminologies and classifications relevant to agriculture.",
            "EarthPortal": "A semantic artifact repository focused on Earth system and environmental science vocabularies, powered by OntoPortal.",
            "QUDT": "A comprehensive resource for Units of Measure, Quantity Kinds, and Dimensions, accessible via SPARQL endpoint.",
            CUSTOM_SPARQL_PROVIDER_NAME: "Connect to your own custom SPARQL endpoint. Requires Endpoint URL and a valid SPARQL query template."
        }
        st.subheader("Reconciliation Sources")
        standard_providers = ["Wikidata", "NCBI", "BioPortal", "OLS (EBI)", "AgroPortal", "EarthPortal", "SemLookP", "QUDT"]
        available_providers = standard_providers.copy()
        if st.session_state.get('custom_sparql_enabled'):
            available_providers.append(CUSTOM_SPARQL_PROVIDER_NAME)
        
        st.write("Select providers to add to the queue:")
        cols_prov = st.columns(2)
        for i, provider_name_iter in enumerate(available_providers):
            with cols_prov[i % 2]:
                st.checkbox(provider_name_iter, 
                            key=f"queue_cb_{provider_name_iter}", 
                            value=(provider_name_iter in st.session_state.get('provider_queue', [])), 
                            disabled=st.session_state.get('processing_active', False), 
                            help=provider_tooltips.get(provider_name_iter, ""))
        
        # NCBI Database Selection Expander
        with st.expander("NCBI Database Selection", expanded=False):
            ncbi_all_databases = ['taxonomy', 'bioproject', 'gene', 'protein', 'nuccore', 'biosample', 'sra', 'pubmed', 'assembly', 'blastdbinfo', 'books', 'cdd', 'clinvar', 'dbgap', 'domains', 'gap', 'gapplus', 'gds', 'geoprofiles', 'homologene', 'medgen', 'mesh', 'ncv', 'nlmcatalog', 'omim', 'pmc', 'popset', 'probe', 'proteinclusters', 'pubchem-compound', 'pubchem-substance', 'pubchem-assay', 'snp', 'sra', 'structure', 'taxonomy', 'unigene', 'unists']
            
            def _update_ncbi_dbs_callback():
                st.session_state['ncbi_selected_databases'] = st.session_state.ncbi_db_multiselect
                st.info("NCBI database selection updated. Confirm/Update Queue to apply changes.")
                st.rerun()

            current_ncbi_selection = st.session_state.get('ncbi_selected_databases', ncbi_all_databases)
            st.multiselect(
                "Select NCBI Databases to Query:",
                options=ncbi_all_databases,
                default=current_ncbi_selection,
                key="ncbi_db_multiselect",
                on_change=_update_ncbi_dbs_callback,
                help="Choose specific NCBI databases for the 'NCBI' provider to query. If none selected, all will be queried."
            )
            if not st.session_state.get('ncbi_selected_databases') and "NCBI" in st.session_state.get('provider_queue', []):
                st.warning("NCBI provider is selected but no databases are chosen. It will query all available databases.", icon="⚠️")

        if st.button("Confirm/Update Queue", key="confirm_queue_button", disabled=st.session_state.get('processing_active', False)):
            newly_confirmed_providers = []
            current_selection_from_checkboxes = [p_name for p_name in available_providers if st.session_state.get(f"queue_cb_{p_name}")]

            st.session_state['provider_queue'] = current_selection_from_checkboxes
            for provider_name_iter_update in st.session_state.get('provider_queue', []):
                st.session_state.provider_status[provider_name_iter_update] = {
                    'status': 'pending', 'results_count': 0, 'error_msg': '', 'progress': 0.0, 'processed_indices': set()
                }
                for idx_sugg_clear in list(st.session_state.get('suggestions', {}).keys()):
                    if provider_name_iter_update in st.session_state.get('suggestions', {}).get(idx_sugg_clear, {}):
                        del st.session_state.get('suggestions', {})[idx_sugg_clear][provider_name_iter_update]
                if provider_name_iter_update in st.session_state.get('provider_has_results', set()):
                    st.session_state.get('provider_has_results', set()).remove(provider_name_iter_update)
                if st.session_state.get('display_provider') == provider_name_iter_update:
                    st.session_state['display_provider'] = None
                newly_confirmed_providers.append(provider_name_iter_update)
            
            if newly_confirmed_providers: st.success(f"Queue updated with: {', '.join(newly_confirmed_providers)}.")
            else: st.info("No providers selected for the queue.")
            st.rerun() 

        if st.session_state.get('processing_active', False): st.caption("Processing is active. Cannot modify the queue now.")

        st.markdown("---")
        st.subheader("Ontology Filters")
        lookup_providers = ["BioPortal", "OLS (EBI)", "SemLookP", "AgroPortal", "EarthPortal"]
        
        for provider_name in st.session_state.get('provider_queue', []):
            if provider_name in lookup_providers:
                if st.session_state.get('ontology_loading_status', {}).get(provider_name) != 'loaded' and \
                   st.session_state.get('ontology_loading_status', {}).get(provider_name) != 'loading':
                    
                    st.session_state.ontology_loading_status[provider_name] = 'loading'
                    logger.info(f"Attempting to load ontologies for {provider_name}...")
                    
                    try:
                        ontologies_list = []
                        if provider_name == "BioPortal":
                            api_key = CONFIG.get('bioportal', {}).get('api_key')
                            if api_key and api_key != 'YourAPIKey':
                                ontologies_list = get_bioportal_ontologies(USER_AGENT, api_key)
                            else:
                                raise ValueError("BioPortal API Key is missing or default.")
                        elif provider_name == "EarthPortal":
                            api_key = CONFIG.get('earthportal', {}).get('api_key')
                            if api_key and api_key != 'YourAPIKey':
                                ontologies_list = get_earthportal_ontologies(USER_AGENT, api_key)
                            else:
                                raise ValueError("EarthPortal API Key is missing or default.")
                        elif provider_name == "OLS (EBI)":
                            ontologies_list = get_ols_ontologies(USER_AGENT)
                        elif provider_name == "SemLookP":
                            ontologies_list = get_semlookp_ontologies(USER_AGENT)
                        elif provider_name == "AgroPortal":
                            api_key = CONFIG.get('agroportal', {}).get('api_key')
                            if api_key and api_key != 'YourAPIKey':
                                ontologies_list = get_agroportal_ontologies(USER_AGENT, api_key)
                            else:
                                raise ValueError("AgroPortal API Key is missing or default.")
                        
                        st.session_state.available_ontologies_by_provider[provider_name] = sorted([o.upper() for o in ontologies_list])
                        st.session_state.ontology_loading_status[provider_name] = 'loaded'
                        logger.info(f"Successfully loaded {len(ontologies_list)} ontologies for {provider_name} and converted to uppercase.")
                        if provider_name not in st.session_state.selected_ontologies_by_provider:
                            st.session_state.selected_ontologies_by_provider[provider_name] = []
                        st.rerun()
                    except Exception as e:
                        st.session_state.ontology_loading_status[provider_name] = 'error'
                        st.session_state.available_ontologies_by_provider[provider_name] = []
                        logger.error(f"Error loading ontologies for {provider_name}: {e}", exc_info=True)
                        st.sidebar.warning(f"Could not load ontologies for {provider_name}. Check API key/config. Error: {e}", icon="⚠️")
                        st.rerun()

        for provider_name in lookup_providers:
            if provider_name in st.session_state.get('provider_queue', []):
                if st.session_state.get('ontology_loading_status', {}).get(provider_name) == 'loaded':
                    available_ontos = st.session_state.get('available_ontologies_by_provider', {}).get(provider_name, [])
                    current_selected_ontos = st.session_state.get('selected_ontologies_by_provider', {}).get(provider_name, [])

                    if available_ontos:
                        with st.expander(f"Filter {provider_name} Ontologies ({len(current_selected_ontos)} selected)", expanded=False):
                            config_key_for_provider = provider_name.lower().replace('(ebi)', '').strip().replace(' ', '_')
                            preferred_ontologies_str = CONFIG.get(config_key_for_provider, {}).get('preferred_ontologies', '')
                            preferred_ontologies_list_upper = [o.strip().upper() for o in preferred_ontologies_str.split(',') if o.strip()]

                            preferred_options_ordered = [o for o in preferred_ontologies_list_upper if o in available_ontos]
                            other_options_sorted = sorted([o for o in available_ontos if o not in preferred_options_ordered])

                            display_options = []
                            if preferred_options_ordered:
                                display_options.extend(preferred_options_ordered)
                                if other_options_sorted:
                                    display_options.append("--- All Other Ontologies (Alphabetical) ---")
                            display_options.extend(other_options_sorted)

                            default_selection_for_multiselect = [o for o in current_selected_ontos if o in available_ontos]

                            new_selection = st.multiselect(
                                f"Select ontologies for {provider_name}:",
                                options=display_options,
                                default=default_selection_for_multiselect,
                                key=f"filter_ontologies_{provider_name}",
                                help=f"Select specific ontologies to filter results from {provider_name}. If none selected, all ontologies will be included."
                            )
                            
                            final_new_selection = [o for o in new_selection if o != "--- All Other Ontologies (Alphabetical) ---"]

                            if set(final_new_selection) != set(current_selected_ontos):
                                st.session_state.selected_ontologies_by_provider[provider_name] = final_new_selection
                                logger.info(f"Updated selected ontologies for {provider_name}: {final_new_selection}")
                                st.rerun()
                    else:
                        st.info(f"No ontologies available for {provider_name} or failed to load.")
                elif st.session_state.get('ontology_loading_status', {}).get(provider_name) == 'loading':
                    st.info(f"Loading ontologies for {provider_name}...")
                elif st.session_state.get('ontology_loading_status', {}).get(provider_name) == 'error':
                    st.warning(f"Failed to load ontologies for {provider_name}. Check logs for details.", icon="⚠️")
                else:
                    st.info(f"Ontologies for {provider_name} will load when selected in the queue.")
        st.markdown("---")

        st.subheader("Custom SPARQL Provider")
        current_custom_sparql_enabled = st.session_state.get('custom_sparql_enabled', False)
        new_custom_sparql_enabled = st.checkbox("Enable Custom SPARQL Provider", value=current_custom_sparql_enabled, key="custom_sparql_enabled_widget", help="Enable and configure your SPARQL endpoint.")
        if new_custom_sparql_enabled != current_custom_sparql_enabled:
            st.session_state['custom_sparql_enabled'] = new_custom_sparql_enabled
            st.rerun()

        if st.session_state.get('custom_sparql_enabled'):
            st.session_state['custom_sparql_endpoint'] = st.text_input("SPARQL Endpoint URL", value=st.session_state.get('custom_sparql_endpoint', ""))
            st.session_state['custom_sparql_query_template'] = st.text_area("SPARQL Query Template", value=st.session_state.get('custom_sparql_query_template', DEFAULT_SPARQL_QUERY_TEMPLATE), height=150)
            cols_vars = st.columns(3)
            with cols_vars[0]: st.session_state['custom_sparql_var_uri'] = st.text_input("URI Var", value=st.session_state.get('custom_sparql_var_uri', "uri"))
            with cols_vars[1]: st.session_state['custom_sparql_var_label'] = st.text_input("Label Var", value=st.session_state.get('custom_sparql_var_label', "label"))
            with cols_vars[2]: st.session_state['custom_sparql_var_description'] = st.text_input("Desc Var", value=st.session_state.get('custom_sparql_var_description', "description"))
            if not st.session_state.get('custom_sparql_endpoint', "").strip() or not st.session_state.get('custom_sparql_query_template', "").strip():
                st.warning("Provide Endpoint URL and Query Template for Custom SPARQL.", icon="⚠️")
        st.markdown("---")

        st.subheader("Processing Queue Status")
        if not st.session_state.get('provider_queue', []): st.write("Queue is empty.")
        else:
            for i, pn_status in enumerate(st.session_state.get('provider_queue', [])):
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    s_info = st.session_state.get('provider_status', {}).get(pn_status, {'status': 'pending'})
                    s_text = s_info.get('status', 'pending').capitalize()
                    icon = "⏳" if s_info.get('status') == 'pending' else "⚙️" if s_info.get('status') == 'running' else "✅" if s_info.get('status') == 'completed' else "❌" if s_info.get('status') == 'error' else "🛑"
                    st.markdown(f"{icon} {pn_status} - {s_text} `(Pos: {i + 1})`")
                with col2:
                    if st.button("✖", key=f"remove_provider_{pn_status}", help=f"Remove {pn_status} from queue", disabled=st.session_state.get('processing_active', False)):
                        st.session_state.provider_queue.remove(pn_status)
                        if pn_status in st.session_state.provider_status:
                            del st.session_state.provider_status[pn_status]
                        if st.session_state.get('display_provider') == pn_status:
                            st.session_state['display_provider'] = None
                        logger.info(f"Removed provider '{pn_status}' from queue.")
                        st.rerun()
        st.markdown("---")
        
        st.subheader("Processing Progress")
        if st.session_state.get('sidebar_provider_progress_text_ph_obj') is None: 
            st.session_state['sidebar_provider_progress_text_ph_obj'] = st.empty()
        if st.session_state.get('sidebar_provider_progress_bar_ph_obj') is None: 
            st.session_state['sidebar_provider_progress_bar_ph_obj'] = st.empty()

        overall_progress_text_ph = st.session_state.get('sidebar_provider_progress_text_ph_obj')
        overall_progress_bar_ph = st.session_state.get('sidebar_provider_progress_bar_ph_obj')

        total_terms_for_progress = len(st.session_state.get('total_indices_to_process', []))
        processed_terms_for_progress = st.session_state.get('processed_terms_count', 0)
        current_progress_value = 0.0
        if total_terms_for_progress > 0:
            current_progress_value = min(1.0, processed_terms_for_progress / total_terms_for_progress)

        if st.session_state.get('processing_active', False):
            overall_progress_text_ph.markdown(f"Overall Progress: **{processed_terms_for_progress} / {total_terms_for_progress}** terms")
            overall_progress_bar_ph.progress(current_progress_value)
        else:
            current_status_text = "Idle"
            if total_terms_for_progress > 0 and processed_terms_for_progress >= total_terms_for_progress:
                current_status_text = "Finished"
            elif st.session_state.get('provider_queue') and any(st.session_state.provider_status.get(p, {}).get('status') in ['completed', 'error', 'stopped'] for p in st.session_state.provider_queue):
                if current_progress_value == 1.0:
                    current_status_text = "Finished"
                elif current_progress_value > 0:
                    current_status_text = "Stopped"
                else:
                    current_status_text = "Finished (No terms processed or all errored)"
            elif not st.session_state.get('provider_queue') and total_terms_for_progress > 0 and processed_terms_for_progress < total_terms_for_progress :
                 current_status_text = "Idle (Queue empty)"
            elif not total_terms_for_progress:
                 current_status_text = "Idle (No terms to process)"


            overall_progress_text_ph.markdown(f"Overall Progress: _({current_status_text})_ **{processed_terms_for_progress} / {total_terms_for_progress}** terms")
            overall_progress_bar_ph.progress(current_progress_value)
        st.markdown("---")

        st.subheader("Matching Strategy")
        strategy_options = ["API Ranking", "Levenshtein Similarity", "Cosine Similarity"]
        current_strategy = st.session_state.get('matching_strategy_radio', "API Ranking")
        try:
            current_strategy_idx = strategy_options.index(current_strategy)
        except ValueError:
            current_strategy_idx = 0
        
        chosen_strategy = st.radio(
            "Matching/Sorting Strategy", 
            options=strategy_options, 
            index=current_strategy_idx, 
            key="matching_strategy_radio_widget"
        )
        if chosen_strategy != current_strategy:
            st.session_state['matching_strategy_radio'] = chosen_strategy
            st.rerun()


        if st.session_state.get('matching_strategy_radio') == "Cosine Similarity":
            if not semantic_search:
                st.error("Semantic Search module not available. Cosine Similarity disabled.")
            elif st.session_state.get('semantic_model') is None and not st.session_state.get('semantic_model_load_attempted', False):
                st.session_state['semantic_model_load_attempted'] = True 
                with st.spinner("Loading semantic model..."):
                    try:
                        model_name = CONFIG.get('cosine_model', {}).get('name', 'all-MiniLM-L6-v2')
                        logger.info(f"Attempting to load semantic model: {model_name}")
                        st.session_state['semantic_model'] = semantic_search.load_model(model_name)
                        if st.session_state.get('semantic_model') and st.session_state.get('semantic_model') != "LOAD_FAILED":
                            st.sidebar.success("Semantic model loaded successfully.")
                            logger.info("Semantic model loaded successfully.")
                        else:
                            st.sidebar.error("Failed to load semantic model. It might be configured incorrectly or files are missing.")
                            logger.error("semantic_search.load_model returned None or LOAD_FAILED.")
                            st.session_state['semantic_model'] = "LOAD_FAILED" 
                    except RuntimeError as e_rt: 
                        st.sidebar.error(f"Runtime error loading semantic model: {e_rt}. Cosine similarity will not work.")
                        logger.error(f"Runtime error loading semantic model: {e_rt}", exc_info=True)
                        st.session_state['semantic_model'] = "LOAD_FAILED" 
                    except Exception as e_model:
                        st.sidebar.error(f"An unexpected error occurred loading semantic model: {e_model}. Cosine similarity will be unavailable.")
                        logger.error(f"Unexpected error loading semantic model: {e_model}", exc_info=True)
                        st.session_state['semantic_model'] = "LOAD_FAILED" 
                    st.rerun() 
            elif st.session_state.get('semantic_model') == "LOAD_FAILED":
                 st.sidebar.warning("Semantic model failed to load. Cosine similarity is unavailable. Check logs.", icon="⚠️")
            elif st.session_state.get('semantic_model'):
                 st.sidebar.caption("Semantic model loaded.") 
        
        st.subheader("Query Settings")
        current_slider_val = st.session_state.get('suggestion_slider', 10)
        new_slider_val = st.slider("Max Suggestions per Term", 1, 25, value=current_slider_val, key="suggestion_slider_widget")
        if new_slider_val != current_slider_val:
            st.session_state['suggestion_slider'] = new_slider_val
        

        st.subheader("Process Control")
        num_to_process_total = len(st.session_state.get('total_indices_to_process', []))
        if num_to_process_total > 0:
            queue_has_runnable = any(st.session_state.get('provider_status', {}).get(p, {}).get('status') == 'pending' for p in st.session_state.get('provider_queue', []))
            prereq_reason = ""
            if not st.session_state.get('provider_queue', []): prereq_reason = "Queue is empty."
            elif not queue_has_runnable and not st.session_state.get('processing_active', False): prereq_reason = "No pending providers."
            
            if prereq_reason and not st.session_state.get('processing_active', False): st.error(f"Cannot start: {prereq_reason}")
            
            start_disabled = st.session_state.get('processing_active', False) or not queue_has_runnable or bool(prereq_reason)
            if st.button("Start Processing Queue", key="process_queue_button", disabled=start_disabled):
                missing_configs = []
                if CONFIG: 
                    for provider_name in st.session_state.get('provider_queue', []):
                        if provider_name == "NCBI":
                            ncbi_key = CONFIG.get('ncbi', {}).get('api_key')
                            if not ncbi_key or ncbi_key == 'YourAPIKey': missing_configs.append("NCBI API Key")
                        elif provider_name == "BioPortal":
                            bioportal_key = CONFIG.get('bioportal', {}).get('api_key')
                            if not bioportal_key or bioportal_key == 'YourAPIKey': missing_configs.append("BioPortal API Key")
                        elif provider_name == "AgroPortal":
                            agroportal_key = CONFIG.get('agroportal', {}).get('api_key')
                            if not agroportal_key or agroportal_key == 'YourAPIKey': missing_configs.append("AgroPortal API Key")
                        elif provider_name == "EarthPortal":
                            earthportal_key = CONFIG.get('earthportal', {}).get('api_key')
                            if not earthportal_key or earthportal_key == 'YourAPIKey': missing_configs.append("EarthPortal API Key")
                        elif provider_name == "GeoNames":
                            geonames_user = CONFIG.get('geonames', {}).get('username')
                            if not geonames_user or geonames_user == 'YourUsername': missing_configs.append("GeoNames Username")
                
                if missing_configs:
                    st.sidebar.error(f"Cannot start: Missing configuration(s) in config.yaml: {', '.join(missing_configs)}")
                else: 
                    if st.session_state.get('df') is not None:
                        st.session_state['total_indices_to_process'] = list(
                            st.session_state.get('df')[
                                pd.isnull(st.session_state.get('df')['URI']) |
                                (st.session_state.get('df')['URI'] == '') |
                                (st.session_state.get('df')['URI'] == NO_MATCH_URI)
                            ].index
                        )
                        logger.info(f"Recalculated total_indices_to_process: {len(st.session_state.get('total_indices_to_process',[]))} terms need URI before starting new processing run.")
                        
                        if not st.session_state.get('total_indices_to_process', []):
                            st.info("All terms appear to have URIs. Nothing to process.")
                        else:
                            st.session_state['processing_active'] = True
                            st.session_state['display_provider'] = None
                            st.session_state['processed_terms_count'] = 0
                            st.session_state['current_term_index_processing'] = 0
                            st.session_state['stop_processing_requested'] = False

                            for p_name_btn in st.session_state.get('provider_queue', []):
                                st.session_state.provider_status[p_name_btn] = {
                                    'status': 'pending', 
                                    'results_count': 0, 
                                    'error_msg': '', 
                                    'progress': 0.0 
                                }
                                for idx_sugg in list(st.session_state.get('suggestions', {}).keys()):
                                    if p_name_btn in st.session_state.get('suggestions', {}).get(idx_sugg, {}):
                                        del st.session_state.get('suggestions', {})[idx_sugg][p_name_btn]
                                if p_name_btn in st.session_state.get('provider_has_results', set()):
                                    st.session_state.get('provider_has_results', set()).remove(p_name_btn)
                            logger.info("Starting parallel processing..."); st.info("🚀 Processing started (parallel)..."); st.rerun()
                    else:
                        st.session_state['total_indices_to_process'] = []
                        logger.warning("Attempted to start processing but DataFrame is None.")
                        st.error("Cannot start processing: DataFrame is not loaded.")
            
            if st.session_state.get('processing_active', False):
                if st.button("🛑 Stop Processing", key="stop_processing_button", type="primary"):
                    st.session_state['stop_processing_requested'] = True
                    st.session_state['processing_active'] = False
                    logger.info("Stop processing requested by user.")
                    st.warning("Processing stop requested. Finishing current term and then stopping.")
                    st.rerun()

            if st.session_state.get('processing_active', False):
                logger.info("Parallel processing active block entered.")

                if st.session_state.get('stop_processing_requested', False):
                    st.session_state['processing_active'] = False
                    logger.info("Processing stopped as per request before starting new term.")
                    for p_name_stop in st.session_state.get('provider_queue', []):
                        if st.session_state.provider_status.get(p_name_stop, {}).get('status') == 'running':
                            st.session_state.provider_status[p_name_stop]['status'] = 'stopped'
                    st.rerun()

                all_indices_to_process_list = st.session_state.get('total_indices_to_process', [])
                df_for_processing = st.session_state.get('df')

                if not all_indices_to_process_list or df_for_processing is None:
                    st.session_state['processing_active'] = False
                    logger.info("No terms to process or DataFrame is None. Deactivating processing.")
                    st.rerun()

                active_providers_for_run = [
                    p_name for p_name in st.session_state.get('provider_queue', [])
                    if st.session_state.get('provider_status', {}).get(p_name, {}).get('status') not in ['error', 'stopped']
                ]

                if not active_providers_for_run:
                    st.session_state['processing_active'] = False
                    st.warning("No active (non-errored) providers in the queue. Stopping processing.")
                    for p_name_q in st.session_state.get('provider_queue', []):
                        if st.session_state.get('provider_status', {}).get(p_name_q, {}).get('status') not in ['error', 'stopped']:
                             st.session_state.provider_status[p_name_q]['status'] = 'completed'
                    st.rerun()

                num_providers_to_query = len(active_providers_for_run)
                max_workers = min(num_providers_to_query, 8) if num_providers_to_query > 0 else 1

                current_processing_idx_val = st.session_state.get('current_term_index_processing', 0)

                if current_processing_idx_val < len(all_indices_to_process_list):
                    actual_df_index = all_indices_to_process_list[current_processing_idx_val]
                    term_to_process = str(df_for_processing.loc[actual_df_index, 'Term']).strip()

                    if not term_to_process:
                        logger.info(f"Skipping empty term at df index {actual_df_index} (processing index {current_processing_idx_val})")
                        st.session_state['current_term_index_processing'] += 1
                        st.session_state['processed_terms_count'] +=1
                    else:
                        logger.info(f"Processing term: '{term_to_process}' (DF Index: {actual_df_index}, Processing Index: {current_processing_idx_val}) across {len(active_providers_for_run)} providers.")

                        if actual_df_index not in st.session_state.suggestions:
                            st.session_state.suggestions[actual_df_index] = {}
                        
                        for prov_name_running_ui in active_providers_for_run:
                            if st.session_state.provider_status.get(prov_name_running_ui, {}).get('status') == 'pending':
                                st.session_state.provider_status[prov_name_running_ui]['status'] = 'running'

                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            future_to_provider = {}
                            for provider_name in active_providers_for_run:
                                current_config_for_provider = CONFIG.copy()
                                if provider_name == CUSTOM_SPARQL_PROVIDER_NAME and st.session_state.get('custom_sparql_enabled'):
                                    current_config_for_provider['custom_sparql'] = {
                                        'endpoint': st.session_state.get('custom_sparql_endpoint'),
                                        'query_template': st.session_state.get('custom_sparql_query_template'),
                                        'var_uri': st.session_state.get('custom_sparql_var_uri'),
                                        'var_label': st.session_state.get('custom_sparql_var_label'),
                                        'var_description': st.session_state.get('custom_sparql_var_description')
                                    }
                                elif provider_name == "NCBI":
                                    current_config_for_provider['ncbi_databases'] = st.session_state.get('ncbi_selected_databases', [])
                                
                                if provider_name in st.session_state.get('selected_ontologies_by_provider', {}):
                                    current_config_for_provider['selected_ontologies_by_provider'] = {
                                        provider_name: st.session_state.get('selected_ontologies_by_provider', {}).get(provider_name, [])
                                    }
                                
                                future = executor.submit(
                                    fetch_suggestions_for_term_from_provider,
                                    provider_name, term_to_process, current_config_for_provider,
                                    USER_AGENT, st.session_state.suggestion_slider
                                )
                                future_to_provider[future] = provider_name

                            for future in concurrent.futures.as_completed(future_to_provider):
                                provider_name_res = future_to_provider[future]
                                try:
                                    provider_suggestions = future.result()
                                    st.session_state.suggestions[actual_df_index][provider_name_res] = provider_suggestions
                                    if provider_suggestions:
                                        st.session_state.provider_has_results.add(provider_name_res)
                                        if 'results_count' not in st.session_state.provider_status[provider_name_res]:
                                             st.session_state.provider_status[provider_name_res]['results_count'] = 0
                                        st.session_state.provider_status[provider_name_res]['results_count'] += 1 
                                except Exception as exc:
                                    logger.error(f"Error fetching suggestions for term '{term_to_process}' from provider '{provider_name_res}': {exc}", exc_info=True)
                                    st.session_state.suggestions[actual_df_index][provider_name_res] = []
                                    st.session_state.provider_status[provider_name_res]['status'] = 'error'
                                    st.session_state.provider_status[provider_name_res]['error_msg'] = str(exc)
                        
                        st.session_state['processed_terms_count'] += 1
                        st.session_state['current_term_index_processing'] += 1
                    
                    total_terms_overall = len(all_indices_to_process_list)
                    processed_terms_val = st.session_state.get('processed_terms_count', 0)
                    overall_term_progress = (processed_terms_val / total_terms_overall) if total_terms_overall > 0 else 1.0
                    
                    if st.session_state.get('stop_processing_requested', False) or \
                       st.session_state.get('current_term_index_processing', 0) >= len(all_indices_to_process_list):
                        st.session_state['processing_active'] = False
                        if st.session_state.get('stop_processing_requested', False):
                            logger.info("Processing stopped by user request after completing current term.")
                            st.warning("Processing was stopped.")
                            for p_name_stop_final in st.session_state.get('provider_queue', []):
                                if st.session_state.provider_status.get(p_name_stop_final, {}).get('status') == 'running':
                                    st.session_state.provider_status[p_name_stop_final]['status'] = 'stopped'
                        else:
                            logger.info("All terms processed in parallel.")
                            st.success("Processing queue finished.")
                            for p_name_final in st.session_state.get('provider_queue', []): 
                                if st.session_state.provider_status.get(p_name_final, {}).get('status') not in ['error', 'stopped']:
                                    st.session_state.provider_status[p_name_final]['status'] = 'completed'
                        st.session_state['stop_processing_requested'] = False
                    
                    if st.session_state.get('processing_active', False):
                        time.sleep(0.1) 
                        st.rerun()
                    else:
                        st.rerun()

                elif not st.session_state.get('stop_processing_requested', False):
                    st.session_state['processing_active'] = False
                    logger.info("All terms appear to be processed. Deactivating processing.")
                    for p_name_final_else in st.session_state.get('provider_queue', []):
                         if st.session_state.provider_status.get(p_name_final_else, {}).get('status') not in ['error', 'stopped', 'completed']:
                             st.session_state.provider_status[p_name_final_else]['status'] = 'completed'
                    st.session_state['stop_processing_requested'] = False
                    st.rerun()
        else: 
            if not st.session_state.get('stop_processing_requested', False):
                st.info("Upload CSV with terms needing URIs to enable processing.")
        st.markdown("---")

    st.header("Current Data")
    if st.session_state.get('df') is not None:
        df_display_copy = st.session_state.get('df').copy()
        if 'URI' in df_display_copy.columns: 
            df_display_copy['URI'] = df_display_copy['URI'].replace(NO_MATCH_URI, "")
        
        columns_to_show_in_main_table = ["Term", "URI", "RDF Role", "Match Type", "Source Provider", "Provider Term", "Provider Description"]
        df_preview = df_display_copy[[col for col in columns_to_show_in_main_table if col in df_display_copy.columns]].copy()
        
        st.dataframe(df_preview, use_container_width=True, key="main_df_display_preview")
    else: st.info("No data loaded to display.")
    st.markdown(f"*Total terms requiring reconciliation: {len(st.session_state.get('total_indices_to_process',[]))}*")
    st.markdown("---")

    st.header("View Provider Results")
    st.write("Click a button below to view suggestions from that provider.")
    viewable_providers_exist_main = False
    provider_order_main = {name: i for i, name in enumerate(st.session_state.get('provider_queue', []))}
    sorted_provider_names_main = sorted(
        st.session_state.get('provider_status', {}).keys(),
        key=lambda name: (provider_order_main.get(name, float('inf')), name)
    )
    if not sorted_provider_names_main and not st.session_state.get('processing_active', False) :
        st.info("No provider results to display yet. Add providers and start processing.")
    else:
        num_displayable_buttons = 0
        for p_name_check in sorted_provider_names_main:
            status_check = st.session_state.get('provider_status', {}).get(p_name_check, {}).get('status')
            if status_check and status_check != 'pending': num_displayable_buttons +=1
        
        if num_displayable_buttons == 0 and not st.session_state.get('processing_active', False):
             st.info("No provider results to display yet (e.g., all pending or queue empty). Start processing.")
        elif num_displayable_buttons == 0 and st.session_state.get('processing_active', False):
             st.info("Processing active. Result buttons will appear here as providers complete.")
        else:
            show_mixed_results_button = len(st.session_state.get('provider_has_results', set())) >= 2

            total_cols_needed = num_displayable_buttons
            if show_mixed_results_button:
                total_cols_needed += 1

            cols_btns_main = st.columns(min(total_cols_needed, 4) if total_cols_needed > 0 else 1) 
            col_idx = 0

            if show_mixed_results_button:
                button_label = "📊 Mixed Results"
                button_key = "view_main_mixed_results"
                button_help = "Click to view combined and sorted results from all providers with results."
                button_type = "primary" if st.session_state.get('display_mixed_results') else "secondary"
                current_col = cols_btns_main[col_idx % len(cols_btns_main)]
                if current_col.button(button_label, key=button_key, help=button_help, type=button_type, use_container_width=True):
                    st.session_state['display_mixed_results'] = True
                    st.session_state['display_provider'] = None
                    st.rerun()
                col_idx += 1

            for p_btn_name in sorted_provider_names_main:
                s_info_btn = st.session_state.get('provider_status', {}).get(p_btn_name, {})
                status_label = s_info_btn.get('status', 'N/A')
                if not status_label or status_label == 'pending': continue
                viewable_providers_exist_main = True
                results_count_val = s_info_btn.get('results_count', 0)
                icon = "❓" 
                if status_label == 'completed': icon = "✅"
                elif status_label == 'error': icon = "❌"
                elif status_label == 'running': icon = "⚙️"
                elif status_label == 'stopped': icon = "🛑"
                btn_label = f"{icon} {p_btn_name} ({results_count_val} results)"
                button_key = f"view_main_{p_btn_name}"
                button_help = f"Click to view results/status from {p_btn_name}"
                button_type = "primary" if st.session_state.get('display_provider') == p_btn_name else "secondary"
                current_col = cols_btns_main[col_idx % len(cols_btns_main)]
                if current_col.button(btn_label, key=button_key, help=button_help, type=button_type, use_container_width=True):
                    st.session_state['display_provider'] = p_btn_name
                    st.session_state['display_mixed_results'] = False
                    st.rerun()
                col_idx += 1
            if not viewable_providers_exist_main and st.session_state.get('processing_active', False):
                 st.info("Processing active. Result buttons will appear here as providers complete.")
            elif not viewable_providers_exist_main and not show_mixed_results_button:
                 st.info("No provider results to display yet (e.g., all pending or queue empty). Start processing.")
    st.markdown("---")

    st.header("Reconcile Terms")

    MIXED_RESULTS_DISPLAY_MODE = "Mixed Results"

    current_display_mode = None
    if st.session_state.get('display_mixed_results'):
        current_display_mode = MIXED_RESULTS_DISPLAY_MODE
    elif st.session_state.get('display_provider'):
        current_display_mode = st.session_state.get('display_provider')

    if not current_display_mode:
        st.info("Select a provider or 'Mixed Results' from the 'View Provider Results' buttons above to enable reconciliation.")
    elif st.session_state.get('df') is None or st.session_state.get('df').empty:
        st.info("No data loaded. Please upload a CSV file.")
    else:
        current_skos_state = st.session_state.get('skos_matching_enabled', False)
        new_skos_state = st.toggle(
            "Enable SKOS Matching",
            value=current_skos_state,
            key="skos_matching_toggle_widget",
            help="Enable this to select a SKOS match type (e.g., exactMatch, closeMatch) for your mappings. If disabled, the 'Match Type' column will be empty."
        )

        if new_skos_state != current_skos_state:
            st.session_state['skos_matching_enabled'] = new_skos_state
            if not new_skos_state and st.session_state.get('df') is not None:
                st.session_state.df['Match Type'] = ''
                logger.info("SKOS matching disabled. Cleared all 'Match Type' values.")
            st.rerun()
            
        recon_controls_cols = st.columns([2,3])
        with recon_controls_cols[0]:
            new_items_per_page = st.number_input(
                "Terms per page:", 
                min_value=1, 
                max_value=100, 
                value=st.session_state.get('items_per_page', 10), 
                step=1, 
                key="items_per_page_input",
                help="Number of terms to display per page in the reconciliation list."
            )
            if new_items_per_page != st.session_state.get('items_per_page'):
                st.session_state['items_per_page'] = new_items_per_page
                st.session_state['current_page'] = 1
                st.rerun()

        with recon_controls_cols[1]:
            st.session_state['show_only_matched_terms'] = st.toggle(
                "Show only terms with suggestions",
                value=st.session_state.get('show_only_matched_terms', False),
                key="toggle_show_matched_terms",
                help=f"If enabled, only terms for which suggestions are available (from {current_display_mode}) will be displayed below."
            )
            if 'toggle_show_matched_terms_last_value' not in st.session_state:
                st.session_state.toggle_show_matched_terms_last_value = st.session_state.show_only_matched_terms
            if st.session_state.show_only_matched_terms != st.session_state.toggle_show_matched_terms_last_value:
                st.session_state.current_page = 1
                st.session_state.toggle_show_matched_terms_last_value = st.session_state.show_only_matched_terms
            
            st.session_state['show_only_unreconciled_terms'] = st.toggle(
                "Show only unreconciled terms",
                value=st.session_state.get('show_only_unreconciled_terms', False),
                key="toggle_show_unreconciled_terms",
                help="If enabled, only terms that do not yet have a URI assigned (or are marked 'No Match') will be displayed."
            )
            if 'toggle_show_unreconciled_terms_last_value' not in st.session_state:
                st.session_state.toggle_show_unreconciled_terms_last_value = st.session_state.show_only_unreconciled_terms
            if st.session_state.show_only_unreconciled_terms != st.session_state.toggle_show_unreconciled_terms_last_value:
                st.session_state.current_page = 1
                st.session_state.toggle_show_unreconciled_terms_last_value = st.session_state.show_only_unreconciled_terms


        reconcile_col, dialog_col = st.columns(2)

        with reconcile_col:
            st.subheader(f"Review and Reconcile with: {current_display_mode} (Strategy: {st.session_state.get('matching_strategy_radio')})")
            
            if st.button("Prefill with Best Match", key="prefill_best_match_button", help="Automatically prefill all unreconciled terms with the best available match based on the current sorting strategy. Will not overwrite existing selections. Also fills empty SKOS match types if SKOS matching is enabled."):
                prefill_best_matches()

            current_lev_threshold = st.session_state.get('levenshtein_threshold_slider', 0.7)
            new_lev_threshold = st.slider(
                "Levenshtein Match Threshold for Prefill", 
                min_value=0.0, 
                max_value=1.0, 
                value=current_lev_threshold, 
                step=0.01, 
                key="levenshtein_threshold_slider_widget",
                help="Minimum Levenshtein similarity score (0.0 to 1.0) required for a suggestion to be considered a valid match when prefilling. If the best match is below this, 'No Match' is applied."
            )
            if new_lev_threshold != current_lev_threshold:
                st.session_state['levenshtein_threshold_slider'] = new_lev_threshold
            
            st.markdown("---")

            cols_header = st.columns([3, 4, 3, 2])
            cols_header[0].markdown("**Term**")
            cols_header[1].markdown("**Select/Confirm URI**")
            cols_header[2].markdown("**Match Type**")
            cols_header[3].markdown("**Custom Search**")
            st.markdown("---")

            term_indices_to_iterate = []
            
            all_df_indices = list(st.session_state.get('df').index)
            
            if st.session_state.get('show_only_matched_terms', False):
                filtered_by_suggestions = []
                for idx_filter in all_df_indices:
                    if current_display_mode == MIXED_RESULTS_DISPLAY_MODE:
                        all_suggs_for_term = st.session_state.get('suggestions', {}).get(idx_filter, {})
                        valid_sugg_lists = [s_list for s_list in all_suggs_for_term.values() if s_list is not None]
                        if any(s for s_list in valid_sugg_lists for s in s_list if s is not None):
                            filtered_by_suggestions.append(idx_filter)
                    else:
                        suggestions_for_term_provider_filter = st.session_state.get('suggestions', {}).get(idx_filter, {}).get(current_display_mode, [])
                        if suggestions_for_term_provider_filter:
                            filtered_by_suggestions.append(idx_filter)
                term_indices_to_iterate = filtered_by_suggestions
            else:
                term_indices_to_iterate = all_df_indices

            if st.session_state.get('show_only_unreconciled_terms', False):
                filtered_by_unreconciled = []
                for idx_filter in term_indices_to_iterate:
                    current_uri = str(st.session_state.get('df').loc[idx_filter, 'URI']).strip()
                    current_match_type = str(st.session_state.get('df').loc[idx_filter, 'Match Type']).strip()

                    is_uri_unreconciled = (not current_uri or current_uri == NO_MATCH_URI)
                    is_skos_match_unreconciled = (st.session_state.get('skos_matching_enabled') and not current_match_type)

                    if is_uri_unreconciled or is_skos_match_unreconciled:
                        filtered_by_unreconciled.append(idx_filter)
                term_indices_to_iterate = filtered_by_unreconciled
            
            items_per_page_val = st.session_state.get('items_per_page', 10)
            total_terms_to_display = len(term_indices_to_iterate)
            total_pages = (total_terms_to_display + items_per_page_val - 1) // items_per_page_val if items_per_page_val > 0 else 1
            total_pages = max(1, total_pages)

            if st.session_state.current_page < 1: st.session_state.current_page = 1
            if st.session_state.current_page > total_pages: st.session_state.current_page = total_pages
            
            start_idx_slice = (st.session_state.current_page - 1) * items_per_page_val
            end_idx_slice = start_idx_slice + items_per_page_val
            paginated_term_indices = term_indices_to_iterate[start_idx_slice:end_idx_slice]

            if total_pages > 1:
                render_pagination_controls_ui(total_pages, 'current_page', 'pagination_top')

            displayed_term_count = 0
            for index_main in paginated_term_indices:
                term_main = str(st.session_state.get('df').loc[index_main, 'Term']).strip()

                current_uri_in_df = str(st.session_state.get('df').loc[index_main, 'URI']).strip()
                current_source_in_df = str(st.session_state.get('df').loc[index_main, 'Source Provider']).strip()
                confirmed_display_string_from_df = str(st.session_state.get('df').loc[index_main, 'Confirmed Display String']).strip()
                current_match_type_in_df = str(st.session_state.get('df').loc[index_main, 'Match Type']).strip()

                row_cols = st.columns([3, 4, 3, 2])
                
                with row_cols[0]:
                    st.markdown(f"`{term_main}` <br><small>(Row {index_main})</small>", unsafe_allow_html=True)
                
                with row_cols[1]:
                    inline_options_map = {NO_MATCH_DISPLAY: (NO_MATCH_URI, "", None)}

                    if current_display_mode == MIXED_RESULTS_DISPLAY_MODE:
                        all_suggs_for_term = st.session_state.get('suggestions', {}).get(index_main, {})
                        processed_suggestions = get_combined_and_sorted_suggestions(
                            term_main, all_suggs_for_term, st.session_state.suggestion_slider, st.session_state.get('matching_strategy_radio')
                        )
                    else:
                        original_suggestions = st.session_state.get('suggestions', {}).get(index_main, {}).get(current_display_mode, [])
                        processed_suggestions = []
                        if original_suggestions and semantic_search and st.session_state.get('matching_strategy_radio') == "Cosine Similarity":
                            if st.session_state.get('semantic_model') and st.session_state.get('semantic_model') != "LOAD_FAILED":
                                try:
                                    processed_suggestions = semantic_search.calculate_hybrid_scores(
                                        model=st.session_state.get('semantic_model'),
                                        input_term=term_main, suggestions=list(original_suggestions)
                                    )
                                except Exception as e: 
                                    processed_suggestions = original_suggestions
                                    logger.error(f"Error during hybrid scoring for original suggestions (term: '{term_main}'): {e}", exc_info=True)
                            else: 
                                processed_suggestions = original_suggestions
                        elif original_suggestions: 
                            processed_suggestions = original_suggestions
                    
                    for sugg in processed_suggestions:
                        s_uri = sugg.get('uri'); s_label = sugg.get('label')
                        if s_uri and s_label:
                            display_text = format_suggestion_display(sugg, st.session_state.get('matching_strategy_radio'))
                            if display_text not in inline_options_map:
                                source_for_map = sugg.get('source_provider') or sugg.get('db') or sugg.get('ontology') or sugg.get('source_db') or current_display_mode
                                inline_options_map[display_text] = (s_uri, source_for_map, sugg)
                    
                    current_selection_display = NO_MATCH_DISPLAY 

                    if confirmed_display_string_from_df and confirmed_display_string_from_df != NO_MATCH_DISPLAY:
                        if confirmed_display_string_from_df not in inline_options_map:
                            if current_uri_in_df and current_uri_in_df != NO_MATCH_URI:
                                inline_options_map[confirmed_display_string_from_df] = (current_uri_in_df, current_source_in_df, None)
                        current_selection_display = confirmed_display_string_from_df
                    elif current_uri_in_df and current_uri_in_df != NO_MATCH_URI:
                        found_match_in_current_provider = False
                        for display_key, (uri_val, source_val, _) in inline_options_map.items():
                            if uri_val == current_uri_in_df and source_val == current_source_in_df:
                                current_selection_display = display_key
                                found_match_in_current_provider = True
                                break
                        if not found_match_in_current_provider:
                            fallback_display = f"{current_uri_in_df} (from {current_source_in_df or 'previous selection'})"
                            if fallback_display not in inline_options_map:
                                inline_options_map[fallback_display] = (current_uri_in_df, current_source_in_df, None)
                            current_selection_display = fallback_display
                    
                    inline_select_key = f"inline_select_{index_main}_{current_display_mode}"
                    try:
                        if current_selection_display not in inline_options_map.keys():
                            current_selection_display = NO_MATCH_DISPLAY
                            if NO_MATCH_DISPLAY not in inline_options_map:
                                inline_options_map[NO_MATCH_DISPLAY] = (NO_MATCH_URI, "")
                        current_select_idx = list(inline_options_map.keys()).index(current_selection_display)
                    except ValueError: 
                        current_select_idx = 0 
                    
                    selected_display_option_inline = st.selectbox(
                        "Select URI:", options=list(inline_options_map.keys()),
                        index=current_select_idx, key=inline_select_key, label_visibility="collapsed"
                    )

                    chosen_uri_inline, chosen_source_inline, chosen_suggestion_obj = inline_options_map.get(selected_display_option_inline, (NO_MATCH_URI, "", None))

                    uri_changed = chosen_uri_inline != current_uri_in_df
                    source_changed = (chosen_uri_inline != NO_MATCH_URI and chosen_source_inline != current_source_in_df) or \
                                     (chosen_uri_inline == NO_MATCH_URI and current_source_in_df != "")
                    display_string_changed = selected_display_option_inline != confirmed_display_string_from_df

                    if uri_changed or source_changed or display_string_changed:
                        st.session_state.df.loc[index_main, 'URI'] = chosen_uri_inline
                        st.session_state.df.loc[index_main, 'Source Provider'] = chosen_source_inline if chosen_uri_inline != NO_MATCH_URI else ""
                        st.session_state.df.loc[index_main, 'Confirmed Display String'] = selected_display_option_inline if chosen_uri_inline != NO_MATCH_URI else NO_MATCH_DISPLAY
                        
                        if chosen_suggestion_obj:
                            st.session_state.df.loc[index_main, 'Provider Term'] = chosen_suggestion_obj.get('label', '')
                            st.session_state.df.loc[index_main, 'Provider Description'] = chosen_suggestion_obj.get('description', '')
                        else:
                            st.session_state.df.loc[index_main, 'Provider Term'] = ""
                            st.session_state.df.loc[index_main, 'Provider Description'] = ""
                        
                        if chosen_uri_inline != NO_MATCH_URI:
                            if st.session_state.get('skos_matching_enabled') and (not current_match_type_in_df or current_match_type_in_df == ""):
                                st.session_state.df.loc[index_main, 'Match Type'] = 'skos:exactMatch'
                        elif chosen_uri_inline == NO_MATCH_URI:
                            st.session_state.df.loc[index_main, 'Match Type'] = ''

                        logger.info(f"[InlineSelect] Term index {index_main} ('{term_main}') updated. URI: '{chosen_uri_inline}', Source: '{chosen_source_inline}', Display: '{st.session_state.df.loc[index_main, 'Confirmed Display String']}'.")
                        st.rerun()

                with row_cols[2]:
                    skos_match_types = ["", "skos:exactMatch", "skos:closeMatch", "skos:broadMatch", "skos:narrowMatch", "skos:relatedMatch"]
                    
                    try:
                        current_match_type_idx = skos_match_types.index(current_match_type_in_df)
                    except ValueError:
                        current_match_type_idx = 0 

                    is_disabled = not st.session_state.get('skos_matching_enabled')
                    
                    selected_match_type = st.selectbox(
                        "Select Match Type:", 
                        options=skos_match_types,
                        index=current_match_type_idx,
                        key=f"skos_match_type_{index_main}",
                        label_visibility="collapsed",
                        disabled=is_disabled
                    )

                    if selected_match_type != current_match_type_in_df:
                        final_match_type = "" if selected_match_type == "No Match" else selected_match_type
                        st.session_state.df.loc[index_main, 'Match Type'] = final_match_type
                        logger.info(f"[MatchTypeSelect] Term index {index_main} ('{term_main}') updated Match Type to: '{final_match_type}'.")
                        st.rerun()

                with row_cols[3]:
                    if st.button("New Search...", key=f"custom_search_modal_btn_{index_main}"):
                        logger.info(f"[NewSearchButton] Clicked for term index {index_main}.")
                        st.session_state['active_reconciliation_index'] = index_main
                        logger.debug(f"[NewSearchButton] Set active_reconciliation_index to: {st.session_state.get('active_reconciliation_index')}")
                        dialog_custom_search_term_key = f"dialog_custom_search_text_{index_main}_{current_display_mode}"
                        st.session_state.custom_search_terms[dialog_custom_search_term_key] = "" 
                        dialog_custom_search_results_key = (index_main, current_display_mode, "dialog_modal")
                        st.session_state.custom_search_results[dialog_custom_search_results_key] = [] 
                        st.rerun()
                st.markdown("---")
                displayed_term_count += 1
            
            if displayed_term_count == 0:
                if st.session_state.get('show_only_matched_terms', False):
                    st.info(f"No terms found with suggestions from '{current_display_mode}' on this page or in total. Disable the toggle or check other pages.")
                elif not term_indices_to_iterate:
                     st.info("No terms to display for reconciliation based on current filters.")
                else:
                     st.info(f"No terms to display on page {st.session_state.current_page}. Try other pages.")
            
            if total_pages > 1:
                st.markdown("---")
                render_pagination_controls_ui(total_pages, 'current_page', 'pagination_bottom')

        with dialog_col:
            active_idx_for_dialog = st.session_state.get('active_reconciliation_index')
            logger.debug(f"[DialogRender] Checking for active dialog in right column. active_reconciliation_index = {active_idx_for_dialog}")

            if active_idx_for_dialog is not None:
                if st.session_state.get('df') is not None and active_idx_for_dialog in st.session_state.get('df').index:
                    idx = active_idx_for_dialog
                    providers_for_dialog_search = []
                    is_mixed_search = False
                    if st.session_state.get('display_mixed_results'):
                        is_mixed_search = True
                        providers_for_dialog_search = [p for p in st.session_state.get('provider_queue', []) if st.session_state.get('provider_status', {}).get(p, {}).get('status') not in ['pending', 'error']]
                        if not providers_for_dialog_search:
                            st.warning("No active providers in the queue to perform a mixed search. Add and process providers first.")
                            st.session_state['active_reconciliation_index'] = None
                            st.rerun()
                            return
                    else:
                        provider_for_single_search = st.session_state.get('display_provider')
                        if provider_for_single_search:
                            providers_for_dialog_search.append(provider_for_single_search)
                        else:
                            st.warning("No provider selected for custom search.")
                            st.session_state['active_reconciliation_index'] = None
                            st.rerun()
                            return

                    if providers_for_dialog_search:
                        term_to_reconcile_in_dialog = str(st.session_state.df.loc[idx, 'Term']).strip()
                        search_target_display = "All Providers" if is_mixed_search else providers_for_dialog_search[0]
                        dialog_title = f"Custom Search for: '{term_to_reconcile_in_dialog}' (Row {idx}) with {search_target_display}"

                        with st.container(border=True):
                            st.subheader(dialog_title)
                            st.caption(f"Strategy: {st.session_state.get('matching_strategy_radio')}")

                            dialog_custom_search_term_key = f"dialog_custom_search_text_{idx}_{search_target_display}"
                            dialog_custom_search_results_key = (idx, search_target_display, "dialog_modal")

                            current_custom_search_text = st.session_state.custom_search_terms.get(dialog_custom_search_term_key, "")
                            new_custom_search_text = st.text_input(
                                "Enter new term for custom search:",
                                value=current_custom_search_text,
                                key=dialog_custom_search_term_key
                            )
                            if new_custom_search_text != current_custom_search_text:
                                st.session_state.custom_search_terms[dialog_custom_search_term_key] = new_custom_search_text

                            if st.button("Search", key=f"dialog_search_btn_{idx}"):
                                term_for_api = st.session_state.custom_search_terms.get(dialog_custom_search_term_key, "").strip()
                                if term_for_api:
                                    with st.spinner(f"Searching {search_target_display} for '{term_for_api}'..."):
                                        all_results = {}
                                        with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers_for_dialog_search)) as executor:
                                            future_to_provider = {}
                                            for p_name in providers_for_dialog_search:
                                                # Create a dynamic config for the dialog search, including current ontology selections
                                                dynamic_dialog_config = CONFIG.copy()
                                                if p_name in st.session_state.get('selected_ontologies_by_provider', {}):
                                                    dynamic_dialog_config['selected_ontologies_by_provider'] = {
                                                        p_name: st.session_state.get('selected_ontologies_by_provider', {}).get(p_name, [])
                                                    }
                                                if p_name == "NCBI":
                                                    dynamic_dialog_config['ncbi_databases'] = st.session_state.get('ncbi_selected_databases', [])
                                                if p_name == CUSTOM_SPARQL_PROVIDER_NAME and st.session_state.get('custom_sparql_enabled'):
                                                    dynamic_dialog_config['custom_sparql'] = {
                                                        'endpoint': st.session_state.get('custom_sparql_endpoint'),
                                                        'query_template': st.session_state.get('custom_sparql_query_template'),
                                                        'var_uri': st.session_state.get('custom_sparql_var_uri'),
                                                        'var_label': st.session_state.get('custom_sparql_var_label'),
                                                        'var_description': st.session_state.get('custom_sparql_var_description')
                                                    }

                                                future = executor.submit(
                                                    fetch_suggestions_for_term_from_provider,
                                                    p_name, term_for_api, dynamic_dialog_config, USER_AGENT, st.session_state.get('suggestion_slider', 10)
                                                )
                                                future_to_provider[future] = p_name
                                            for future in concurrent.futures.as_completed(future_to_provider):
                                                p_name_res = future_to_provider[future]
                                                try:
                                                    all_results[p_name_res] = future.result()
                                                except Exception as e:
                                                    logger.error(f"Error in custom search for '{p_name_res}': {e}", exc_info=True)
                                                    all_results[p_name_res] = []
                                        
                                        sorted_results = get_combined_and_sorted_suggestions(
                                            term_for_api, all_results, st.session_state.suggestion_slider, st.session_state.get('matching_strategy_radio')
                                        )
                                        st.session_state.custom_search_results[dialog_custom_search_results_key] = sorted_results
                                        st.success(f"Found {len(sorted_results)} combined results for '{term_for_api}'.")
                                    st.rerun()
                                else:
                                    st.warning("Please enter a term to search.")

                            dialog_results = st.session_state.custom_search_results.get(dialog_custom_search_results_key, [])
                            options_dialog_custom = {NO_MATCH_DISPLAY: (NO_MATCH_URI, "", None)}
                            if dialog_results:
                                for sugg in dialog_results:
                                    source_provider = sugg.get('source_provider') or sugg.get('db') or sugg.get('ontology') or sugg.get('source_db') or "Unknown"
                                    display_text = format_suggestion_display(sugg, st.session_state.get('matching_strategy_radio'))
                                    options_dialog_custom[display_text] = (sugg.get('uri'), source_provider, sugg)

                            selected_dialog_uri_display = st.selectbox(
                                "Select URI from custom search:", options=list(options_dialog_custom.keys()),
                                key=f"dialog_custom_select_{idx}", index=0
                            )

                            st.markdown("---")
                            d_cols = st.columns(2)
                            if d_cols[0].button("Confirm this URI & Close", key=f"dialog_confirm_btn_{idx}", type="primary"):
                                chosen_uri, chosen_source, chosen_suggestion_obj = options_dialog_custom.get(selected_dialog_uri_display, (NO_MATCH_URI, "", None))
                                if chosen_uri != NO_MATCH_URI:
                                    st.session_state.df.loc[idx, 'URI'] = chosen_uri
                                    st.session_state.df.loc[idx, 'Source Provider'] = chosen_source
                                    st.session_state.df.loc[idx, 'Confirmed Display String'] = selected_dialog_uri_display
                                    if chosen_suggestion_obj:
                                        st.session_state.df.loc[idx, 'Provider Term'] = chosen_suggestion_obj.get('label', '')
                                        st.session_state.df.loc[idx, 'Provider Description'] = chosen_suggestion_obj.get('description', '')
                                    else:
                                        st.session_state.df.loc[idx, 'Provider Term'] = ""
                                        st.session_state.df.loc[idx, 'Provider Description'] = ""
                                    logger.info(f"[DialogConfirm] Term index {idx} updated to URI '{chosen_uri}'.")
                                else:
                                    st.session_state.df.loc[idx, 'URI'] = NO_MATCH_URI
                                    st.session_state.df.loc[idx, 'Source Provider'] = ""
                                    st.session_state.df.loc[idx, 'Confirmed Display String'] = NO_MATCH_DISPLAY
                                    st.session_state.df.loc[idx, 'Provider Term'] = ""
                                    st.session_state.df.loc[idx, 'Provider Description'] = ""
                                    logger.info(f"[DialogConfirm] Term index {idx} set to NO_MATCH_URI.")
                                
                                st.session_state['active_reconciliation_index'] = None
                                st.rerun()

                            if d_cols[1].button("Cancel & Close", key=f"dialog_cancel_btn_{idx}"):
                                st.session_state['active_reconciliation_index'] = None
                                st.rerun()
                    else:
                        logger.warning("[DialogRender] No providers available for custom search. Clearing active_reconciliation_index.")
                        st.session_state['active_reconciliation_index'] = None
                        st.rerun()
                else:
                    logger.warning(f"[DialogRender] Conditions not met for dialog in right column. df is None: {st.session_state.get('df') is None}, active_idx_for_dialog: {active_idx_for_dialog}")
                    if active_idx_for_dialog is not None: 
                        st.session_state['active_reconciliation_index'] = None 
                        st.rerun()


    st.markdown("---"); st.header("Download Reconciled Data")
    if st.session_state.get('df') is not None:
        download_df_main = st.session_state.get('df').copy()
        
        cols_for_download = [
            "Term", "URI", "RDF Role", "Match Type",
            "Source Provider", "Provider Term", "Provider Description"
        ]
        final_cols_for_download = [col for col in cols_for_download if col in download_df_main.columns]
        download_df_main = download_df_main[final_cols_for_download]

        if 'URI' in download_df_main.columns: 
            download_df_main['URI'] = download_df_main['URI'].replace(NO_MATCH_URI, "")
        
        original_filename_main = st.session_state.get('last_uploaded_filename', "data.csv")
        download_filename_main = f"{os.path.splitext(original_filename_main)[0]}_reconciled.csv"
        
        st.session_state['shared_reconciled_matching_table'] = st.session_state.get('df').copy()
        logger.info("Saved reconciled matching table to 'shared_reconciled_matching_table' for RDF Generator.")
        
        create_download_link(download_df_main, download_filename_main, f"Download '{download_filename_main}'")
    else: st.info("Upload and process a CSV first.")
