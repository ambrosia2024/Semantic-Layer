# -*- coding: utf-8 -*-
import sys
import streamlit as st
import os
import math
import pandas as pd
import Levenshtein
from io import StringIO, BytesIO
import time
import traceback
import yaml
import logging
import concurrent.futures
import rdflib
from rdflib.util import guess_format

# --- Configure Logging (Console Only) ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Temporarily set to DEBUG for troubleshooting
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(stream_handler)
logger.info("Console logging configured at DEBUG level for reconciliation_ui.")

# --- Pure Python Levenshtein Fallback ---
def _pure_python_levenshtein_distance(s1, s2):
    """
    Calculates the Levenshtein distance between two strings.
    A pure Python fallback implementation.
    """
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    # Create two vectors (rows)
    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)

    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]
    return v1[len(s2)]

# --- Constants ---
NO_MATCH_URI = "No Match"
NO_MATCH_DISPLAY = f"--- {NO_MATCH_URI} ---"
CUSTOM_SPARQL_PROVIDER_NAME = "Custom SPARQL"
DEFAULT_SPARQL_QUERY_TEMPLATE = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT DISTINCT ?uri ?label ?description WHERE {{
  ?uri skos:prefLabel|rdfs:label ?label .
  FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{term}")))
  OPTIONAL {{ ?uri skos:definition|rdfs:comment ?description . }}
}}
LIMIT {limit}
"""

# --- Config Loading ---
@st.cache_data(ttl=3600)
def load_config(default_config_filename='config.yaml'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(script_dir, default_config_filename)
    logger.info(f"Attempting to load config from: {config_file_path}")
    if not os.path.exists(config_file_path):
        logger.error(f"Config file not found: {config_file_path}")
        return None
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded successfully from {config_file_path}")
        return config
    except Exception as e:
        logger.exception(f"Error loading/parsing config: {config_file_path}")
        return None

CONFIG = load_config() 

# --- User Agent Setup ---
USER_AGENT = "StreamlitReconApp/1.0 (Contact: your-email@example.com)" 
if CONFIG:
    try:
        contact_email = CONFIG.get('contact_email') or CONFIG.get('ncbi', {}).get('email', 'no-email-provided@example.com')
        app_version_str = CONFIG.get('app_version', 'N/A')
        ua_template = CONFIG.get('user_agent_template', "StreamlitReconApp (Contact: {email})")
        ua_template = ua_template.replace('{email}', contact_email)
        ua_template = ua_template.replace('{version}', app_version_str)
        USER_AGENT = ua_template
        logger.debug(f"User-Agent set to: {USER_AGENT}")
    except Exception as e:
        logger.error(f"Error creating User-Agent from config: {e}")
else:
    logger.warning("CONFIG is None, User-Agent will use default. Check config.yaml loading.")


# --- Utility Functions ---
def calculate_levenshtein_score(s1, s2):
    logger.debug(f"calculate_levenshtein_score inputs: s1='{s1}' (type {type(s1)}), s2='{s2}' (type {type(s2)})")
    if not isinstance(s1, str) or not isinstance(s2, str):
        logger.warning(f"Levenshtein calculation received non-string input: s1='{s1}' (type {type(s1)}), s2='{s2}' (type {type(s2)})")
        return 0.0

    s1_lower = s1.lower()
    s2_lower = s2.lower()
    logger.debug(f"calculate_levenshtein_score lowercased: s1_lower='{s1_lower}', s2_lower='{s2_lower}'")

    distance = None
    try:
        distance = Levenshtein.distance(s1_lower, s2_lower)
        logger.debug(f"Levenshtein.distance('{s1_lower}', '{s2_lower}') returned: {distance} (Type: {type(distance)})")
        if distance is None: # If the C-optimized version returns None, fall back
            logger.warning(f"Levenshtein.distance returned None for '{s1_lower}' and '{s2_lower}'. Falling back to pure Python implementation.")
            distance = _pure_python_levenshtein_distance(s1_lower, s2_lower)
            logger.debug(f"Pure Python Levenshtein distance for '{s1_lower}' and '{s2_lower}' returned: {distance} (Type: {type(distance)})")
    except Exception as e:
        logger.error(f"Error with Levenshtein.distance for '{s1_lower}' and '{s2_lower}': {e}. Falling back to pure Python implementation.", exc_info=True)
        distance = _pure_python_levenshtein_distance(s1_lower, s2_lower)
        logger.debug(f"Pure Python Levenshtein distance for '{s1_lower}' and '{s2_lower}' returned: {distance} (Type: {type(distance)})")

    if distance is None: # Should not happen if fallback works, but as a final safeguard
        logger.error(f"Both Levenshtein.distance and pure Python fallback returned None for '{s1_lower}' and '{s2_lower}'. Returning 0.0 as a last resort.")
        return 0.0

    max_len = max(len(s1_lower), len(s2_lower))

    if max_len == 0:
        score = 1.0 if distance == 0 else 0.0
        logger.debug(f"Max length is 0. Score: {score}")
        return score
    else:
        score = 1.0 - (distance / max_len)
        logger.debug(f"Calculated score: {score} (distance: {distance}, max_len: {max_len})")
        return score

def format_suggestion_display(suggestion, strategy='API Ranking', max_desc_length=100):
    label = suggestion.get('label', 'N/A'); desc = suggestion.get('description', ''); uri = suggestion.get('uri', '')
    
    # Prioritize the new 'source_provider' key for display
    source_provider_display = suggestion.get('source_provider')
    db_info = f" [{source_provider_display}]" if source_provider_display else ""
    
    score_info = ""; score = None; score_key = None; include_score = False; score_label_prefix = ""
    if strategy == "Levenshtein Similarity": score_key = 'levenshtein_score'; score_label_prefix = "Lev: "; include_score = True
    elif strategy == "Cosine Similarity": score_key = 'hybrid_score'; score_label_prefix = "Hybrid: "; include_score = True
    elif strategy == "API Ranking": score_key = 'score'; include_score = score_key in suggestion
    if include_score and score_key:
        score = suggestion.get(score_key)
        if score is not None: # Only one check needed
            try: 
                score_info = f" [{score_label_prefix}{float(score):.2f}]"
            except (ValueError, TypeError): 
                score_info = f" [{score_label_prefix}N/A]"
        else: # If score is None, explicitly set N/A
            score_info = f" [{score_label_prefix}N/A]"
    display_text = f"{label}{db_info}{score_info}"
    if desc:
        desc_str = str(desc)
        if len(desc_str) > max_desc_length: display_text += f" ({desc_str[:max_desc_length]}...)"
        else: display_text += f" ({desc_str})"
    if uri: display_text += f" <{uri}>"
    else: display_text += " <No URI>"
    return display_text

def create_download_link(df_to_download, filename, link_text="Download CSV"):
    csv = df_to_download.to_csv(index=False, encoding='utf-8-sig') 
    st.download_button(label=link_text, data=csv, file_name=filename, mime='text/csv')

def get_combined_and_sorted_suggestions(term_main, all_suggestions_for_term, max_suggestions_to_show, strategy):
    """
    Combines suggestions from all providers for a given term,
    calculates scores if needed, sorts by the chosen strategy, and returns top N.
    """
    combined_suggestions = []
    lookup_providers = ["BioPortal", "OLS (EBI)", "SemLookP", "AgroPortal", "EarthPortal"]

    for provider_name, suggestions_list in all_suggestions_for_term.items():
        if suggestions_list is None:
            continue
        
        # Apply ontology filter if this is a lookup provider and filters are selected
        if provider_name in lookup_providers:
            selected_ontologies = st.session_state.get('selected_ontologies_by_provider', {}).get(provider_name, [])
            if selected_ontologies: # If specific ontologies are selected, filter
                filtered_provider_suggestions = []
                for suggestion in suggestions_list:
                    if suggestion is None:
                        continue
                    # Check if the suggestion's ontology is in the selected list
                    # The 'ontology' key is expected from BioPortal, EarthPortal, OLS, SemLookP providers
                    sugg_ontology = suggestion.get('ontology') or suggestion.get('source_provider') # Fallback to source_provider if 'ontology' isn't explicit
                    if sugg_ontology and sugg_ontology.upper() in [o.upper() for o in selected_ontologies]: # Convert both to upper for comparison
                        filtered_provider_suggestions.append(suggestion)
                    else:
                        logger.debug(f"Filtering out suggestion from {provider_name} (ontology: {sugg_ontology}) not in selected list: {selected_ontologies}")
                suggestions_list = filtered_provider_suggestions
                logger.info(f"Applied ontology filter for {provider_name}. Original: {len(all_suggestions_for_term.get(provider_name, []))} -> Filtered: {len(suggestions_list)}")
            # If selected_ontologies is empty, no filter is applied, and all suggestions_list are kept.

        for suggestion in suggestions_list:
            if suggestion is None:
                continue
            if 'label' in suggestion and 'uri' in suggestion:
                sugg_copy = suggestion.copy()
                
                # Always calculate Levenshtein for potential display or fallback
                s_label = sugg_copy.get('label', '')
                if not isinstance(s_label, str): s_label = str(s_label)
                lev_score = calculate_levenshtein_score(term_main, s_label)
                sugg_copy['levenshtein_score'] = lev_score
                logger.debug(f"Calculated Levenshtein score for '{s_label}' (term '{term_main}'): {lev_score} (Type: {type(lev_score)})")
                logger.debug(f"Appending suggestion with Levenshtein score: {sugg_copy}")
                combined_suggestions.append(sugg_copy)
            else:
                logger.warning(f"Skipping malformed suggestion from {provider_name}: {suggestion}")
    
    logger.debug(f"Combined suggestions before filtering/sorting for '{term_main}': {combined_suggestions}")

    # Determine the sort key based on the strategy
    sort_key_str = 'score' # Default to API Ranking
    if strategy == "Levenshtein Similarity":
        sort_key_str = 'levenshtein_score'
    elif strategy == "Cosine Similarity":
        # This assumes hybrid scores have been pre-calculated for the Cosine strategy
        sort_key_str = 'hybrid_score'

    # Filter out any non-dict items just in case
    filtered_suggestions = [s for s in combined_suggestions if isinstance(s, dict)]
    logger.debug(f"Filtered suggestions (only dicts) for '{term_main}': {filtered_suggestions}")

    # All suggestions should already have a numeric levenshtein_score from the loop above.
    # The previous check and assignment to -1.0 is removed as it seems redundant and potentially problematic.
    # If calculate_levenshtein_score returns a non-numeric, it's already 0.0.

    # Sort the suggestions.
    # The key lambda now handles cases where the score value is None or not a number.
    def sort_key_func(x):
        val = x.get(sort_key_str)
        if isinstance(val, (int, float)):
            return val
        return -1.0 # Default value for sorting if score is missing or invalid

    sorted_suggestions = sorted(
        filtered_suggestions, 
        key=sort_key_func,
        reverse=True
    )
    
    logger.debug(f"Final sorted suggestions for '{term_main}' (using key '{sort_key_str}'):")
    for s in sorted_suggestions:
        logger.debug(f"  - Label: {s.get('label')}, URI: {s.get('uri')}, LevScore: {s.get('levenshtein_score')}, SortScore: {s.get(sort_key_str)}")
    
    return sorted_suggestions[:max_suggestions_to_show]

def prefill_best_matches():
    logger.info("Prefill with Best Match button clicked.")
    df_to_process = st.session_state.get('df')
    if df_to_process is None:
        st.warning("No data loaded to prefill.")
        logger.warning("Prefill aborted: No DataFrame found in session state.")
        return

    levenshtein_threshold = st.session_state.get('levenshtein_threshold_slider', 0.7)
    skos_enabled = st.session_state.get('skos_matching_enabled', False)
    logger.info(f"Prefill initiated with Levenshtein Threshold: {levenshtein_threshold:.2f}, SKOS Enabled: {skos_enabled}")

    indices_to_check = list(df_to_process.index)
    prefilled_count = 0

    for index_main in indices_to_check:
        term_main = str(df_to_process.loc[index_main, 'Term']).strip()
        current_uri_in_df = str(df_to_process.loc[index_main, 'URI']).strip()
        
        logger.debug(f"Prefill processing term '{term_main}' (index {index_main}). Current URI: '{current_uri_in_df}'")

        if current_uri_in_df and current_uri_in_df != NO_MATCH_URI:
            logger.debug(f"Skipping term '{term_main}' (index {index_main}) as it already has a URI.")
            continue

        if not term_main:
            logger.debug(f"Skipping empty term at index {index_main}.")
            continue

        all_suggs_for_term = st.session_state.get('suggestions', {}).get(index_main, {})
        
        # Always sort by Levenshtein for prefilling
        processed_suggestions = get_combined_and_sorted_suggestions(
            term_main, all_suggs_for_term, st.session_state.suggestion_slider, "Levenshtein Similarity"
        )
        
        best_match = None
        if processed_suggestions:
            potential_best_match = processed_suggestions[0]
            lev_score_for_check = potential_best_match.get('levenshtein_score')

            if isinstance(lev_score_for_check, (int, float)) and lev_score_for_check >= levenshtein_threshold:
                best_match = potential_best_match
        
        if best_match:
            chosen_uri = best_match.get('uri')
            chosen_source = best_match.get('source_provider') or best_match.get('db') or best_match.get('ontology') or "Unknown"
            chosen_label = best_match.get('label', '')
            chosen_description = best_match.get('description', '')
            
            # Use the matching strategy from the UI for display purposes only
            display_strategy = st.session_state.get('matching_strategy_radio', 'Levenshtein Similarity')
            best_match_display_text = format_suggestion_display(best_match, display_strategy)

            df_to_process.loc[index_main, 'URI'] = chosen_uri
            df_to_process.loc[index_main, 'Source Provider'] = chosen_source
            df_to_process.loc[index_main, 'Confirmed Display String'] = best_match_display_text
            df_to_process.loc[index_main, 'Provider Term'] = chosen_label
            df_to_process.loc[index_main, 'Provider Description'] = chosen_description
            prefilled_count += 1
        else:
            df_to_process.loc[index_main, 'URI'] = NO_MATCH_URI
            df_to_process.loc[index_main, 'Source Provider'] = ""
            df_to_process.loc[index_main, 'Confirmed Display String'] = NO_MATCH_DISPLAY
            df_to_process.loc[index_main, 'Provider Term'] = ""
            df_to_process.loc[index_main, 'Provider Description'] = ""

    if skos_enabled:
        for index_main in indices_to_check:
            current_uri = str(df_to_process.loc[index_main, 'URI']).strip()
            current_match_type = str(df_to_process.loc[index_main, 'Match Type']).strip()
            if current_uri and current_uri != NO_MATCH_URI and not current_match_type:
                df_to_process.loc[index_main, 'Match Type'] = 'skos:exactMatch'
    
    st.session_state['df'] = df_to_process 
    st.success(f"Prefilled {prefilled_count} terms with best matches and updated SKOS matches where applicable.")
    logger.info(f"Prefill process completed. {prefilled_count} terms prefilled.")
    st.rerun()

# --- Pagination Control UI Function ---
def parse_rdf_data(rdf_data, filename):
    """Parses RDF data from an uploaded file."""
    graph = rdflib.Graph()
    file_format = guess_format(filename)
    graph.parse(data=rdf_data, format=file_format)
    return graph

def render_pagination_controls_ui(total_pages, current_page_session_key, key_prefix):
    """Renders pagination controls horizontally using a single st.columns layout."""
    if total_pages <= 1:
        return

    current_page = st.session_state.get(current_page_session_key, 1)

    # --- Calculate Page Elements (numbers and ellipses) ---
    page_numbers_to_show = set()
    page_numbers_to_show.add(1)  # Always show first page
    page_numbers_to_show.add(total_pages)  # Always show last page
    # Show current page and +/- 1 page around it to keep it concise
    for i in range(max(1, current_page - 1), min(total_pages + 1, current_page + 2)):
        page_numbers_to_show.add(i)
    
    # Defensive check: Ensure all elements are integers before sorting
    cleaned_page_numbers = [p for p in page_numbers_to_show if isinstance(p, int)]
    sorted_page_numbers = sorted(list(cleaned_page_numbers))

    temp_display_elements = []
    last_page_shown = 0
    if sorted_page_numbers:
        for page_num in sorted_page_numbers:
            if page_num > last_page_shown + 1 and last_page_shown != 0:
                if not temp_display_elements or temp_display_elements[-1] != "...":
                    temp_display_elements.append("...")
            if not temp_display_elements or temp_display_elements[-1] != page_num: # Avoid duplicates if logic overlaps
                 temp_display_elements.append(page_num)
            last_page_shown = page_num
        
        # Refined ellipsis logic for the end
        if total_pages > last_page_shown:
            if total_pages > last_page_shown + 1: # Gap before last page
                 if not temp_display_elements or temp_display_elements[-1] != "...":
                    temp_display_elements.append("...")
            if not temp_display_elements or temp_display_elements[-1] != total_pages:
                 temp_display_elements.append(total_pages)

    # Final cleanup of ellipses
    final_page_elements = []
    if temp_display_elements:
        final_page_elements.append(temp_display_elements[0])
        for i in range(1, len(temp_display_elements)):
            # Skip duplicate ellipses
            if temp_display_elements[i] == "..." and final_page_elements[-1] == "...":
                continue
            # Skip ellipsis if it's between N and N+1
            if temp_display_elements[i] == "..." and i + 1 < len(temp_display_elements) and \
               isinstance(final_page_elements[-1], int) and isinstance(temp_display_elements[i+1], int) and \
               temp_display_elements[i+1] == final_page_elements[-1] + 1:
                continue
            final_page_elements.append(temp_display_elements[i])
    
    # --- Define Column Specification ---
    # Prev button: 1.5 units
    # Page info: 2 units
    # Page elements (numbers/ellipses): 1 unit each
    # Next button: 1.5 units
    col_spec = [1.5, 2]  # For Prev and Page Info
    for _ in final_page_elements:
        col_spec.append(1)  # Each page number or ellipsis
    col_spec.append(1.5)  # For Next

    cols = st.columns(col_spec)
    
    col_idx = 0

    # --- Previous Button ---
    if cols[col_idx].button("⬅️ Prev", key=f"{key_prefix}_prev_horiz_v2", disabled=(current_page <= 1), use_container_width=True):
        st.session_state[current_page_session_key] -= 1
        st.rerun()
    col_idx += 1
    
    # --- Page X of Y Display ---
    cols[col_idx].markdown(f"<div style='text-align: center; margin-top: 0.5em; white-space: nowrap;'>Page {current_page} of {total_pages}</div>", unsafe_allow_html=True)
    col_idx += 1
    
    # --- Page Number Buttons / Ellipses ---
    for elem in final_page_elements:
        if col_idx < len(cols) - 1:  # Ensure we don't write into the Next button's column slot
            current_column = cols[col_idx]
            if elem == "...":
                current_column.markdown("<div style='text-align: center; margin-top: 0.5em;'>...</div>", unsafe_allow_html=True)
            else:
                page_num = int(elem)
                is_current = (page_num == current_page)
                if current_column.button(
                    f"{page_num}",
                    key=f"{key_prefix}_page_horiz_v2_{page_num}",
                    type="primary" if is_current else "secondary",
                    disabled=is_current,
                    use_container_width=False # Changed to False for smaller page number buttons
                ):
                    st.session_state[current_page_session_key] = page_num
                    st.rerun()
            col_idx += 1
            
    # --- Next Button ---
    # This should always be the last column defined in col_spec
    if cols[-1].button("Next ➡️", key=f"{key_prefix}_next_horiz_v2", disabled=(current_page >= total_pages), use_container_width=True):
        st.session_state[current_page_session_key] += 1
        st.rerun()
