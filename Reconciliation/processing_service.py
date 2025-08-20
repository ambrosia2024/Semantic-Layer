# processing_service.py
import pandas as pd
import time
import logging
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions
from .reconciliation_utils import calculate_levenshtein_score, CUSTOM_SPARQL_PROVIDER_NAME # Import from utils

# --- Custom Provider Modules ---
# Import modules directly. If an import fails, Python will raise ImportError,
# which should be caught by the main app.py script.
try:
    from . import wikidata_provider
    from . import ncbi_provider
    from . import bioportal_provider
    from . import ols_provider
    from . import semlookp_provider
    from . import agroportal_provider
    from . import earthportal_provider # Import the new EarthPortal provider
    from . import qudt_provider # Import the new QUDT provider
except ImportError as e:
    logging.critical(f"Failed to import a required provider module: {e}", exc_info=True)
    raise ImportError(f"Missing provider module: {e}") from e


logger = logging.getLogger(__name__)

# --- Custom SPARQL Query Function ---
def query_custom_sparql(term: str, limit: int, config: dict, user_agent: str) -> list:
    """
    Queries a custom SPARQL endpoint defined in the config.
    """
    sparql_config = config.get('custom_sparql')
    if not sparql_config:
        logger.error("Custom SPARQL config missing in processing service config dict.")
        raise ValueError("Custom SPARQL configuration not found.")
    endpoint_url = sparql_config.get('endpoint')
    query_template = sparql_config.get('query_template')
    var_uri = sparql_config.get('var_uri', 'uri')
    var_label = sparql_config.get('var_label', 'label')
    var_desc = sparql_config.get('var_description', 'description')
    if not endpoint_url or not query_template:
        msg = f"Custom SPARQL endpoint URL or query template is empty for term '{term}'."
        logger.error(msg); raise ValueError(msg)
    sanitized_term = term.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
    try:
        query = query_template.format(term=sanitized_term, limit=limit)
        logger.debug(f"Formatted Custom SPARQL Query for '{term}':\n{query}")
    except KeyError as e:
        msg = f"Missing placeholder {{{e}}} in SPARQL query template. Template: {query_template}"
        logger.error(msg); raise ValueError(msg) from e
    except Exception as fmt_e:
        msg = f"Error formatting SPARQL query for term '{term}': {fmt_e}"
        logger.error(msg); raise ValueError(msg) from fmt_e
    results_list = []
    try:
        sparql = SPARQLWrapper(endpoint_url); sparql.setQuery(query); sparql.setReturnFormat(JSON)
        sparql.agent = user_agent; sparql.setTimeout(30)
        start_time = time.time(); logger.info(f"Querying custom SPARQL endpoint: {endpoint_url} for term '{term}'")
        results = sparql.query().convert(); duration = time.time() - start_time
        bindings = results.get("results", {}).get("bindings", [])
        logger.info(f"Custom SPARQL query for '{term}' took {duration:.2f}s, got {len(bindings)} bindings.")
        for result in bindings:
            try:
                uri = result.get(var_uri, {}).get("value"); label = result.get(var_label, {}).get("value")
                description = result.get(var_desc, {}).get("value") if var_desc else None
                if uri and label:
                    results_list.append({
                        "uri": uri,
                        "label": label,
                        "description": description or "",
                        "score": None,
                        "db": CUSTOM_SPARQL_PROVIDER_NAME,
                        "source_provider": CUSTOM_SPARQL_PROVIDER_NAME # Add source provider information
                    })
                else:
                    missing_vars = [v for v, val in [(var_uri, uri), (var_label, label)] if not val]; logger.warning(f"Missing expected variable(s) ({', '.join(missing_vars)}) in SPARQL result binding: {result}")
            except Exception as parse_e: logger.warning(f"Error parsing individual SPARQL result binding: {result}. Error: {parse_e}")
    except SPARQLExceptions.EndPointNotFound as e: logger.error(f"Custom SPARQL Endpoint not found or invalid: {endpoint_url}. Error: {e}"); raise ConnectionError(f"SPARQL Endpoint not found: {endpoint_url}") from e
    except SPARQLExceptions.QueryBadFormed as e: logger.error(f"Custom SPARQL query badly formed. Error: {e}\nQuery Attempted:\n{query}"); raise ValueError(f"Bad SPARQL Query. Check template/term syntax. Error: {e}") from e
    except ConnectionRefusedError as e: logger.error(f"Connection refused by SPARQL endpoint: {endpoint_url}. Error: {e}"); raise ConnectionError(f"Connection refused by SPARQL endpoint: {endpoint_url}") from e
    except Exception as e: logger.exception(f"An unexpected error occurred querying Custom SPARQL endpoint '{endpoint_url}' for term '{term}'."); raise ConnectionError(f"Failed to query Custom SPARQL {endpoint_url}: {e}") from e
    return results_list[:limit]


# --- Main Chunk Processing Function ---
def process_chunk_for_provider(
    current_provider_name: str,
    all_indices_to_process: list,
    processed_indices_for_this_provider: set,
    df: pd.DataFrame,
    config: dict, # The main config dictionary
    user_agent: str,
    num_suggestions: int,
    matching_strategy: str,
    semantic_model,
    chunk_size: int
) -> (dict, set, bool, str):
    """
    Processes the next chunk of terms for the specified provider by calling the
    appropriate function with the correct arguments.
    """
    indices_to_process_now = []
    processed_in_this_chunk = set()
    suggestions_for_chunk = {}
    error_message = None
    finished_provider = False

    count = 0
    for index in all_indices_to_process:
        if index not in processed_indices_for_this_provider:
            indices_to_process_now.append(index)
            count += 1
            if count >= chunk_size: break

    logger.info(f"Provider '{current_provider_name}': Processing chunk with {len(indices_to_process_now)} terms.")

    if not indices_to_process_now:
        all_done = all(idx in processed_indices_for_this_provider for idx in all_indices_to_process)
        logger.info(f"No more indices to process for provider '{current_provider_name}'. All done: {all_done}")
        return {}, set(), all_done, None

    provider_func = None
    try:
        if current_provider_name == "Wikidata":
            provider_func = wikidata_provider.query_wikidata
        elif current_provider_name == "NCBI":
            provider_func = ncbi_provider.query_ncbi
        elif current_provider_name == "BioPortal":
            provider_func = bioportal_provider.query_bioportal
        elif current_provider_name == "OLS (EBI)":
            provider_func = ols_provider.query_ols
        elif current_provider_name == "SemLookP":
            provider_func = semlookp_provider.query_semlookp
        elif current_provider_name == "AgroPortal":
            provider_func = agroportal_provider.query_agroportal
        elif current_provider_name == "EarthPortal":
            provider_func = earthportal_provider.query_earthportal
        elif current_provider_name == "QUDT":
            provider_func = qudt_provider.query_qudt
        elif current_provider_name == CUSTOM_SPARQL_PROVIDER_NAME:
            provider_func = query_custom_sparql
            if 'custom_sparql' not in config or not config['custom_sparql'].get('endpoint') or not config['custom_sparql'].get('query_template'):
                error_message = f"{CUSTOM_SPARQL_PROVIDER_NAME} selected but not configured correctly (Endpoint/Query missing)."
                logger.error(error_message); finished_provider = True; provider_func = None
        else:
            error_message = f"Unknown provider specified: {current_provider_name}"
            logger.error(error_message); finished_provider = True
    except AttributeError as e:
        error_message = f"Function definition for provider '{current_provider_name}' not found in its module. Check function name mapping. Error: {e}"
        logger.exception(error_message); finished_provider = True; provider_func = None

    if provider_func and not finished_provider:
        for index in indices_to_process_now:
            term = str(df.loc[index, 'Term']).strip() if index in df.index else None
            if not term:
                logger.debug(f"Skipping empty or invalid term at index {index} for {current_provider_name}")
                processed_in_this_chunk.add(index)
                continue

            kwargs_for_call = {'user_agent': user_agent}
            try:
                if current_provider_name == "Wikidata":
                    kwargs_for_call['limit'] = num_suggestions
                elif current_provider_name == "NCBI":
                    kwargs_for_call['limit_per_db'] = num_suggestions
                    kwargs_for_call['api_key'] = config.get('ncbi', {}).get('api_key')
                    if not kwargs_for_call['api_key']: raise ValueError("Required NCBI API Key not found in config.")
                    kwargs_for_call['databases_to_search'] = config.get('ncbi_databases', [])
                elif current_provider_name == "BioPortal":
                    kwargs_for_call['page_size'] = num_suggestions
                    kwargs_for_call['api_key'] = config.get('bioportal', {}).get('api_key')
                    if not kwargs_for_call['api_key']: raise ValueError("Required BioPortal API key not found in config.")
                elif current_provider_name == "OLS (EBI)":
                    pass
                elif current_provider_name == "SemLookP":
                    kwargs_for_call['limit'] = num_suggestions
                    kwargs_for_call['ontologies'] = config.get('semlookp', {}).get('ontologies', None)
                    custom_api_url = config.get('semlookp', {}).get('api_url', None)
                    if custom_api_url:
                         kwargs_for_call['api_url'] = custom_api_url
                         logger.debug(f"Using custom SemLookP API URL from config: {custom_api_url}")
                elif current_provider_name == "AgroPortal":
                    kwargs_for_call['page_size'] = num_suggestions
                    kwargs_for_call['api_key'] = config.get('agroportal', {}).get('api_key')
                    if not kwargs_for_call['api_key']: raise ValueError("Required AgroPortal API key not found in config.")
                elif current_provider_name == "EarthPortal":
                    kwargs_for_call['page_size'] = num_suggestions
                    kwargs_for_call['api_key'] = config.get('earthportal', {}).get('api_key')
                    if not kwargs_for_call['api_key']: raise ValueError("Required EarthPortal API key not found in config.")
                elif current_provider_name == "QUDT":
                    kwargs_for_call['limit'] = num_suggestions
                    kwargs_for_call['config'] = config # Pass the full config to access QUDT specific settings
                elif current_provider_name == CUSTOM_SPARQL_PROVIDER_NAME:
                    kwargs_for_call['limit'] = num_suggestions
                    kwargs_for_call['config'] = config

                logger.debug(f"Calling {current_provider_name} for term '{term}' (Index: {index}) with args: {list(kwargs_for_call.keys())}")
                term_suggestions = provider_func(term=term, **kwargs_for_call)

                suggestions_for_chunk[index] = {current_provider_name: term_suggestions}
                processed_in_this_chunk.add(index)

            except (TypeError, ConnectionError, ValueError, TimeoutError, Exception) as e:
                error_message = f"Error querying {current_provider_name} for term '{term}' (Index {index}): {type(e).__name__} - {e}"
                if isinstance(e, TypeError):
                     logger.error(f"TypeError calling '{provider_func.__name__}'. Check if function signature matches arguments provided: {list(kwargs_for_call.keys())}")
                logger.exception(error_message)
                finished_provider = True
                break

    all_processed_indices = processed_indices_for_this_provider.union(processed_in_this_chunk)
    remaining_indices_exist = any(idx not in all_processed_indices for idx in all_indices_to_process)
    if not remaining_indices_exist and not error_message:
        logger.info(f"All terms ({len(all_indices_to_process)}) processed for provider '{current_provider_name}'.")
        finished_provider = True
    elif error_message:
        logger.warning(f"Provider '{current_provider_name}' finishing chunk due to error: {error_message}")
        finished_provider = True

    logger.debug(f"Returning from process_chunk for '{current_provider_name}': "
                 f"Suggestions={len(suggestions_for_chunk)}, ProcessedNow={len(processed_in_this_chunk)}, "
                 f"Finished={finished_provider}, Error='{error_message}'")

    return suggestions_for_chunk, processed_in_this_chunk, finished_provider, error_message


# --- Function to fetch suggestions for a single term from a specific provider ---
def fetch_suggestions_for_term_from_provider(
    provider_name: str,
    term_to_search: str,
    config: dict,
    user_agent: str,
    num_suggestions: int
) -> list:
    """
    Fetches suggestions for a single term from a specified provider.
    """
    logger.info(f"[CustomSearch] Entering fetch_suggestions_for_term_from_provider for term='{term_to_search}', provider='{provider_name}', num_suggestions={num_suggestions}")
    suggestions = []
    error_message = None

    if not term_to_search or not term_to_search.strip():
        logger.warning("Term to search is empty. Returning no suggestions.")
        return []

    provider_func = None
    try:
        if provider_name == "Wikidata":
            provider_func = wikidata_provider.query_wikidata
        elif provider_name == "NCBI":
            provider_func = ncbi_provider.query_ncbi
        elif provider_name == "BioPortal":
            provider_func = bioportal_provider.query_bioportal
        elif provider_name == "OLS (EBI)":
            provider_func = ols_provider.query_ols
        elif provider_name == "SemLookP":
            provider_func = semlookp_provider.query_semlookp
        elif provider_name == "AgroPortal":
            provider_func = agroportal_provider.query_agroportal
        elif provider_name == "EarthPortal":
            provider_func = earthportal_provider.query_earthportal
        elif provider_name == "QUDT":
            provider_func = qudt_provider.query_qudt
        elif provider_name == CUSTOM_SPARQL_PROVIDER_NAME:
            provider_func = query_custom_sparql
            if 'custom_sparql' not in config or not config['custom_sparql'].get('endpoint') or not config['custom_sparql'].get('query_template'):
                error_message = f"{CUSTOM_SPARQL_PROVIDER_NAME} selected but not configured correctly (Endpoint/Query missing)."
                logger.error(error_message)
                raise ValueError(error_message)
        else:
            error_message = f"Unknown provider specified: {provider_name}"
            logger.error(f"[CustomSearch] {error_message}")
            raise ValueError(error_message)
    except AttributeError as e:
        error_message = f"Function definition for provider '{provider_name}' not found in its module. Error: {e}"
        logger.exception(f"[CustomSearch] {error_message}")
        raise ValueError(error_message) from e

    if provider_func:
        logger.debug(f"[CustomSearch] Selected provider function: {provider_func.__name__}")
        kwargs_for_call = {'user_agent': user_agent}
        try:
            if provider_name == "Wikidata":
                kwargs_for_call['limit'] = num_suggestions
            elif provider_name == "NCBI":
                kwargs_for_call['limit_per_db'] = num_suggestions
                kwargs_for_call['api_key'] = config.get('ncbi', {}).get('api_key')
                if not kwargs_for_call['api_key'] or kwargs_for_call['api_key'] == 'YourAPIKey':
                    raise ValueError("Required NCBI API Key not found or is default in config.")
                selected_ncbi_dbs = config.get('ncbi_databases')
                if selected_ncbi_dbs:
                    kwargs_for_call['databases_to_search'] = selected_ncbi_dbs
            elif provider_name == "BioPortal":
                kwargs_for_call['page_size'] = num_suggestions
                kwargs_for_call['api_key'] = config.get('bioportal', {}).get('api_key')
                if not kwargs_for_call['api_key'] or kwargs_for_call['api_key'] == 'YourAPIKey':
                    raise ValueError("Required BioPortal API key not found or is default in config.")
                selected_ontologies = config.get('selected_ontologies_by_provider', {}).get(provider_name, [])
                if selected_ontologies:
                    kwargs_for_call['ontologies'] = ",".join(selected_ontologies)
                    logger.debug(f"BioPortal: Filtering by ontologies: {kwargs_for_call['ontologies']}")
            elif provider_name == "OLS (EBI)":
                pass
            elif provider_name == "SemLookP":
                kwargs_for_call['limit'] = num_suggestions
                custom_api_url = config.get('semlookp', {}).get('api_url', None)
                if custom_api_url:
                    kwargs_for_call['api_url'] = custom_api_url
                pass
            elif provider_name == "AgroPortal":
                kwargs_for_call['page_size'] = num_suggestions
                kwargs_for_call['api_key'] = config.get('agroportal', {}).get('api_key')
                if not kwargs_for_call['api_key'] or kwargs_for_call['api_key'] == 'YourAPIKey':
                    raise ValueError("Required AgroPortal API key not found or is default in config.")
                selected_ontologies = config.get('selected_ontologies_by_provider', {}).get(provider_name, [])
                if selected_ontologies:
                    kwargs_for_call['ontologies'] = ",".join(selected_ontologies)
                    logger.debug(f"AgroPortal: Filtering by ontologies: {kwargs_for_call['ontologies']}")
            elif provider_name == "EarthPortal":
                kwargs_for_call['page_size'] = num_suggestions
                kwargs_for_call['api_key'] = config.get('earthportal', {}).get('api_key')
                if not kwargs_for_call['api_key'] or kwargs_for_call['api_key'] == 'YourAPIKey':
                    raise ValueError("Required EarthPortal API key not found or is default in config.")
                selected_ontologies = config.get('selected_ontologies_by_provider', {}).get(provider_name, [])
                if selected_ontologies:
                    kwargs_for_call['ontologies'] = ",".join(selected_ontologies)
                    logger.debug(f"EarthPortal: Filtering by ontologies: {kwargs_for_call['ontologies']}")
            elif provider_name == "QUDT":
                kwargs_for_call['limit'] = num_suggestions
                kwargs_for_call['config'] = config # Pass the full config to access QUDT specific settings
            elif provider_name == CUSTOM_SPARQL_PROVIDER_NAME:
                kwargs_for_call['limit'] = num_suggestions
                kwargs_for_call['config'] = config

            logger.debug(f"[CustomSearch] Preparing to call {provider_name} for term '{term_to_search}'. Args: {kwargs_for_call.keys()}")
            
            if provider_name in ["OLS (EBI)", "SemLookP"]:
                selected_ontologies = config.get('selected_ontologies_by_provider', {}).get(provider_name, [])
                if selected_ontologies:
                    all_provider_suggestions = []
                    for ontology_shortname in selected_ontologies:
                        logger.info(f"[CustomSearch] Calling {provider_name} for '{term_to_search}' with ontology '{ontology_shortname}'.")
                        try:
                            if provider_name == "OLS (EBI)":
                                current_suggestions = provider_func(term=term_to_search, user_agent=user_agent, ontologies=ontology_shortname)
                            elif provider_name == "SemLookP":
                                current_suggestions = provider_func(term=term_to_search, ontologies=[ontology_shortname], **kwargs_for_call)
                            all_provider_suggestions.extend(current_suggestions)
                        except Exception as e:
                            logger.warning(f"Error fetching from {provider_name} for ontology '{ontology_shortname}': {e}")
                    suggestions = all_provider_suggestions
                else:
                    logger.info(f"[CustomSearch] Calling {provider_name} for '{term_to_search}' without ontology filter.")
                    if provider_name == "OLS (EBI)":
                        suggestions = provider_func(term=term_to_search, user_agent=user_agent, ontologies=None)
                    elif provider_name == "SemLookP":
                        suggestions = provider_func(term=term_to_search, ontologies=None, **kwargs_for_call)
            else:
                logger.info(f"[CustomSearch] Calling provider function {provider_func.__name__} for '{term_to_search}'")
                suggestions = provider_func(term=term_to_search, **kwargs_for_call)
            
            logger.info(f"[CustomSearch] Received {len(suggestions)} suggestions for '{term_to_search}' from {provider_name}.")
            if suggestions:
                logger.debug(f"[CustomSearch] First suggestion example: {suggestions[0]}")

        except (TypeError, ConnectionError, ValueError, TimeoutError, Exception) as e:
            error_message = f"Error querying {provider_name} for term '{term_to_search}': {type(e).__name__} - {e}"
            if isinstance(e, TypeError):
                logger.error(f"[CustomSearch] TypeError calling '{provider_func.__name__}'. Check signature. Args: {list(kwargs_for_call.keys())}", exc_info=True)
            else:
                logger.error(f"[CustomSearch] {error_message}", exc_info=True)
            raise
    
    logger.info(f"[CustomSearch] Returning {len(suggestions)} suggestions from fetch_suggestions_for_term_from_provider for term '{term_to_search}', provider '{provider_name}'.")
    return suggestions
