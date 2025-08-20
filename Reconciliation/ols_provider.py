# -*- coding: utf-8 -*-
import requests
import time
# import streamlit as st # Only needed if mixing logging with UI feedback
import traceback
from urllib.parse import quote
import logging # Import logging module

# --- Constants ---
DEFAULT_OLS_API_URL = "https://www.ebi.ac.uk/ols4/api"
DEFAULT_QUERY_FIELDS = "label,synonym,description,short_form,obo_id,iri"
# Adjust fieldList to explicitly request 'synonym' if necessary
DEFAULT_INCLUDE_FIELDS = "iri,label,synonym,description,ontology_prefix"
# Recommended sleep time between requests (seconds)
SLEEP_TIME = 0.5

# Use cache_data if Streamlit context is available and desired
# from streamlit import cache_data
# @cache_data(ttl=3600) # Cache results for 1 hour

# If running outside Streamlit or caching isn't needed, define a dummy decorator
def cache_data(ttl=None):
    def decorator(func):
        return func
    return decorator

@cache_data(ttl=3600) # Use dummy if no Streamlit
def query_ols(
    term,
    ontologies=None,         # Comma-separated string or list of ontology acronyms
    base_url=DEFAULT_OLS_API_URL,
    rows=15,                 # Number of results
    exact=False,             # Whether to search for exact matches only
    query_fields=DEFAULT_QUERY_FIELDS,
    field_list=DEFAULT_INCLUDE_FIELDS,
    user_agent="DefaultStreamlitClient/OLS"
    # proxies parameter removed
    ):
    """
    Queries the EBI OLS v4 API (/search) to reconcile terms.

    Args:
        term (str): The term to search for.
        ontologies (str or list, optional): Acronyms of ontologies to search within.
                                             If None, searches all.
        base_url (str): The base URL of the OLS API (without /search).
        rows (int): Maximum number of hits.
        exact (bool): If True, search for exact matches only.
        query_fields (str): Comma-separated list of fields to search within.
        field_list (str): Comma-separated list of fields to return.
        user_agent (str): User-Agent for the HTTP header.

    Returns:
        list: A list of suggestion dictionaries
              [{'uri': ..., 'label': ..., 'description': ..., 'ontology': ...}].
              Returns an empty list on error.
    """
    if not term:
        logging.warning("OLS query attempted with empty term.")
        return []

    search_url = f"{base_url}/search"
    suggestions = []
    response = None

    # --- Prepare parameters ---
    params = {
        'q': term,
        'rows': rows,
        'exact': str(exact).lower(), # Must be 'true' or 'false' string
        'queryFields': query_fields,
        'fieldList': field_list,
    }
    if ontologies:
        if isinstance(ontologies, list): params['ontology'] = ",".join(ontologies)
        else: params['ontology'] = ontologies

    # --- Prepare headers ---
    headers = {
        'Accept': 'application/json',
        'User-Agent': user_agent
    }
    # No authentication needed

    # --- API Query ---
    try:
        logging.info(f"Querying OLS: term='{term}', ontologies='{params.get('ontology', 'ALL')}'")
        # proxies argument removed
        response = requests.get(search_url, params=params, headers=headers, timeout=30)

        # --- Throttling ---
        logging.debug(f"OLS request sent, sleeping for {SLEEP_TIME}s")
        time.sleep(SLEEP_TIME) # Wait after each request

        response.raise_for_status() # Check for HTTP 4xx/5xx errors

        data = response.json() # Can raise ValueError

        # --- Parse results ---
        # Structure is {'responseHeader': ..., 'response': {'numFound': ..., 'start': ..., 'docs': [...]}}
        if 'response' in data and 'docs' in data['response'] and isinstance(data['response']['docs'], list):
            logging.info(f"OLS returned {len(data['response']['docs'])} hits for '{term}'.")
            for item in data['response']['docs']:
                try:
                    suggestion = {}
                    suggestion['uri'] = item.get('iri')
                    suggestion['label'] = item.get('label', 'N/A')

                    # Descriptions (can be a list, take the first)
                    defs = item.get('description', [])
                    if isinstance(defs, list) and defs: suggestion['description'] = defs[0]
                    elif isinstance(defs, str): suggestion['description'] = defs
                    else: suggestion['description'] = ""

                    # Ontology acronym
                    suggestion['ontology'] = item.get('ontology_prefix', 'Unknown')

                    # Optional: Extract synonyms (field name 'synonym' or 'synonyms'?)
                    # syns = item.get('synonyms', item.get('synonym', []))
                    # if isinstance(syns, list) and syns: suggestion['synonyms'] = syns
                    # elif isinstance(syns, str): suggestion['synonyms'] = [syns]

                    # Only add if URI is present
                    if suggestion['uri']:
                        # Set source provider to just the ontology acronym
                        source_provider_name = suggestion.get('ontology', 'OLS (EBI)')
                        
                        # Use original description, no prefix
                        original_description = suggestion.get('description', '')

                        suggestions.append({
                            "uri": suggestion['uri'],
                            "label": suggestion['label'],
                            "description": original_description.strip(),
                            "source_provider": source_provider_name
                        })
                        logging.debug(f"OLS: Added suggestion for '{term}' ({suggestion['ontology']}): {suggestion['label']} <{suggestion['uri']}>")

                except Exception as parse_e:
                    logging.error(f"Error parsing an OLS hit for '{term}': {parse_e}. Item: {item}", exc_info=True)

        elif 'error' in data: # Check for explicit error message in JSON body
            logging.warning(f"OLS API reported error (Status {response.status_code}): {data.get('error', {}).get('msg', data)}")
        # else: # No 'docs' and no explicit error -> likely no hits
            # logging.info(f"OLS: No hits found for '{term}'.")
            pass # No hits is not an error

    # --- Error Handling ---
    except ValueError as e: # JSONDecodeError
        logging.warning(f"OLS: Error parsing JSON response for '{term}'. Status: {response.status_code if response else 'N/A'}", exc_info=True)
        if response is not None and hasattr(response, 'text'):
             try: error_data = response.json(); logging.error(f"OLS Server Error (Status {response.status_code}): {error_data}")
             except ValueError: logging.error(f"OLS Server response was not JSON (Status {response.status_code}). Response text (first 500 chars):\n{response.text[:500]}")
        else: logging.error(f"OLS: Could not analyze server response for '{term}'.")
        return []

    except requests.exceptions.HTTPError as e:
        logging.warning(f"OLS HTTP Error for '{term}': {e.response.status_code} {e.response.reason}", exc_info=True)
        error_detail = ""
        try:
            if hasattr(e.response, 'json'): error_data = e.response.json(); error_detail = str(error_data)
            elif hasattr(e.response, 'text'): error_data = e.response.text[:500]; error_detail = error_data
            logging.warning(f"OLS Server Response (excerpt): {error_detail}")
        except Exception as inner_e: logging.warning(f"OLS: Could not extract error details from HTTPError response: {inner_e}")
        # Consider retry logic for 5xx errors
        return []

    except requests.exceptions.RequestException as e:
        # Catches other network errors (ConnectionError, Timeout, etc.)
        logging.warning(f"OLS Network Error for '{term}': {e}", exc_info=True)
        # Proxy check removed
        return []

    except Exception as e:
        # Catch-all for any other unexpected errors
        logging.exception(f"OLS: Unexpected error processing '{term}'") # Log full traceback
        return []

    logging.info(f"OLS query for '{term}' finished, returning {len(suggestions)} suggestions.")
    return suggestions

@cache_data(ttl=3600 * 24) # Cache for 24 hours as ontology list doesn't change often
def get_available_ontologies(
    user_agent="DefaultStreamlitClient/OLS",
    base_url=DEFAULT_OLS_API_URL
):
    """
    Fetches a list of all available ontology prefixes (acronyms) from OLS.

    Args:
        user_agent (str): User-Agent for the HTTP header.
        base_url (str): The base URL of the OLS API.

    Returns:
        list: A list of ontology prefixes (strings). Returns empty list on error.
    """
    ontologies_url = f"{base_url}/ontologies"
    available_prefixes = []
    response = None

    headers = {
        'Accept': 'application/json',
        'User-Agent': user_agent
    }

    try:
        logging.info(f"Fetching available ontologies from OLS: {ontologies_url}")
        response = requests.get(ontologies_url, headers=headers, timeout=30)
        response.raise_for_status() # Check for HTTP 4xx/5xx errors

        data = response.json()

        # OLS /ontologies endpoint returns a _embedded.ontologies list
        if '_embedded' in data and 'ontologies' in data['_embedded'] and isinstance(data['_embedded']['ontologies'], list):
            for ontology_item in data['_embedded']['ontologies']:
                ontology_prefix = ontology_item.get('ontologyId') # OLS uses ontologyId as the prefix/acronym
                if ontology_prefix:
                    available_prefixes.append(ontology_prefix.strip()) # Ensure no leading/trailing whitespace
            logging.info(f"Successfully fetched {len(available_prefixes)} ontologies from OLS.")
        else:
            logging.warning(f"OLS /ontologies endpoint returned unexpected data format: {type(data)}")
            if 'error' in data or 'errors' in data:
                logging.error(f"OLS API reported error fetching ontologies: {data}")

    except requests.exceptions.HTTPError as e:
        logging.error(f"OLS HTTP Error fetching ontologies: {e.response.status_code} {e.response.reason}", exc_info=True)
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"OLS Network Error fetching ontologies: {e}", exc_info=True)
        return []
    except ValueError as e: # JSONDecodeError
        logging.error(f"OLS: Error parsing JSON response fetching ontologies: {e}", exc_info=True)
        return []
    except Exception as e:
        logging.exception(f"OLS: Unexpected error fetching ontologies.")
        return []

    return available_prefixes
