# -*- coding: utf-8 -*-
import requests
import time
import traceback
from urllib.parse import urlparse, quote
import logging

# --- Constants ---
# Default base URL for AgroPortal
DEFAULT_AGROPORTAL_API_URL = "https://data.agroportal.lirmm.fr" # Corrected to the official API base URL
# Default fields to include in the response
DEFAULT_INCLUDE_FIELDS = "prefLabel,synonym,definition"
# Recommended sleep time between requests (seconds)
SLEEP_TIME = 1.0
# Increased timeout for requests (seconds)
REQUEST_TIMEOUT = 60.0

def cache_data(ttl=None):
    def decorator(func):
        return func
    return decorator

@cache_data(ttl=3600)
def query_agroportal(
    term,
    api_key,                 # API Key is required for AgroPortal as well
    ontologies=None,         # Comma-separated string or list of ontology acronyms
    base_url=DEFAULT_AGROPORTAL_API_URL,
    include_fields=DEFAULT_INCLUDE_FIELDS,
    page_size=15,            # How many results max?
    user_agent="DefaultStreamlitClient/AgroPortal"
    ):
    """
    Queries the AgroPortal API (/search) to reconcile terms.

    Args:
        term (str): The term to search for.
        api_key (str): The AgroPortal API Key (required).
        ontologies (str or list, optional): Acronyms of ontologies to search within.
                                             If None, searches all (slower).
        base_url (str): The base URL of the AgroPortal API.
        include_fields (str): Comma-separated list of fields to include.
        page_size (int): Maximum number of hits.
        user_agent (str): User-Agent for the HTTP header.

    Returns:
        list: A list of suggestion dictionaries
              [{'uri': ..., 'label': ..., 'description': ..., 'ontology': ...}].
              Returns an empty list on error.
    """
    if not term:
        logging.warning("AgroPortal query attempted with empty term.")
        return []
    if not api_key:
        logging.error("AgroPortal API Key is required but was not provided.")
        return []

    search_url = f"{base_url}/search"
    suggestions = []
    response = None

    # --- Prepare parameters ---
    params = {
        'q': term,
        'include': include_fields,
        'page': 1, # Fetch only the first page
        'pagesize': page_size,
        'display_context': 'false', # Reduce response size
        'display_links': 'true'     # Need links for ontology info
    }
    if ontologies:
        if isinstance(ontologies, list):
            params['ontologies'] = ",".join(ontologies)
        else:
            params['ontologies'] = ontologies # Assume it's already a string

    # --- Prepare headers ---
    headers = {
        'Authorization': f"apikey token={api_key}", # Recommended authentication
        'Accept': 'application/json', # Request JSON
        'User-Agent': user_agent
    }

    # --- API Query ---
    try:
        logging.info(f"Querying AgroPortal: term='{term}', ontologies='{params.get('ontologies', 'ALL')}'")
        response = requests.get(search_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT) # Increased timeout

        # --- Throttling ---
        logging.debug(f"AgroPortal request sent, sleeping for {SLEEP_TIME}s")
        time.sleep(SLEEP_TIME) # Wait after each request

        response.raise_for_status() # Check for HTTP 4xx/5xx errors

        data = response.json() # Can raise ValueError

        # --- Parse results ---
        if 'collection' in data and isinstance(data['collection'], list):
            logging.info(f"AgroPortal returned {len(data['collection'])} hits for '{term}'.")
            for item in data['collection']:
                try:
                    suggestion = {}
                    suggestion['uri'] = item.get('@id')
                    suggestion['label'] = item.get('prefLabel', 'N/A')

                    # Definitions (often a list, take the first if available)
                    defs = item.get('definition', [])
                    if isinstance(defs, list) and defs:
                        suggestion['description'] = defs[0] # Take the first definition
                    elif isinstance(defs, str): # If it's just a string
                         suggestion['description'] = defs
                    else:
                        suggestion['description'] = ""

                    # Extract ontology acronym from the links
                    ontology_link = item.get('links', {}).get('ontology')
                    if ontology_link:
                        parsed_link = urlparse(ontology_link)
                        suggestion['ontology'] = parsed_link.path.split('/')[-1]
                    else:
                        suggestion['ontology'] = "Unknown"
                        logging.warning(f"Could not extract ontology link for hit: {item.get('@id')}")

                    # Only add if a URI was found
                    if suggestion['uri']:
                        # Set source provider to just the ontology acronym
                        source_provider_name = suggestion.get('ontology', 'AgroPortal')
                        
                        # Use original description, no prefix
                        original_description = suggestion.get('description', '')

                        suggestions.append({
                            "uri": suggestion['uri'],
                            "label": suggestion['label'],
                            "description": original_description.strip(),
                            "source_provider": source_provider_name
                        })

                except Exception as parse_e:
                    # Log error during parsing of a single item
                    logging.error(f"Error parsing an AgroPortal hit for '{term}': {parse_e}. Item: {item}", exc_info=True)

        # Handle cases where 'collection' is missing or not a list
        elif 'collection' not in data:
             # Check for error messages in the body if status was OK
             if 'error' in data or 'errors' in data:
                  logging.warning(f"AgroPortal API reported error (Status {response.status_code}): {data}")
             pass # No hits is not an error itself


    # --- Error Handling ---
    except ValueError as e: # JSONDecodeError
        logging.warning(f"AgroPortal: Error parsing JSON response for '{term}': {e}", exc_info=True)
        if response is not None and hasattr(response, 'text'):
             try:
                 error_data = response.json()
                 logging.error(f"AgroPortal Server Error (Status {response.status_code}): {error_data}")
             except ValueError: # If the error body isn't JSON either
                 logging.error(f"AgroPortal Server response was not JSON (Status {response.status_code}). Response text (first 500 chars):\n{response.text[:500]}")
        else:
            logging.error(f"AgroPortal: Could not analyze server response for '{term}'.")
        return []

    except requests.exceptions.HTTPError as e:
        logging.warning(f"AgroPortal HTTP Error for '{term}': {e.response.status_code} {e.response.reason}", exc_info=True)
        # Try to get more details from the response body
        error_detail = ""
        try:
            if hasattr(e.response, 'json'): error_data = e.response.json(); error_detail = str(error_data)
            elif hasattr(e.response, 'text'): error_data = e.response.text[:500]; error_detail = error_data
            logging.warning(f"AgroPortal Server Response (excerpt): {error_detail}")
        except Exception as inner_e:
            logging.warning(f"AgroPortal: Could not extract error details from HTTPError response: {inner_e}")

        # Specific messages for common errors
        if e.response.status_code == 401:
            logging.error("AgroPortal: Invalid or missing API Key!")
        elif e.response.status_code == 403:
            logging.error("AgroPortal: Access Denied. API Key valid, but forbidden (or rate limit exceeded?)")
        return []

    except requests.exceptions.RequestException as e:
        # Catches other network errors (ConnectionError, Timeout, etc.)
        logging.warning(f"AgroPortal Network Error for '{term}': {e}", exc_info=True)
        return []

    except Exception as e:
        # Catch-all for any other unexpected errors
        logging.exception(f"AgroPortal: Unexpected error processing '{term}'") # Log full traceback
        return []

    logging.info(f"AgroPortal query for '{term}' finished, returning {len(suggestions)} suggestions.")
    return suggestions

@cache_data(ttl=3600 * 24) # Cache for 24 hours as ontology list doesn't change often
def get_available_ontologies(
    user_agent="DefaultStreamlitClient/AgroPortal",
    api_key=None,
    base_url=DEFAULT_AGROPORTAL_API_URL
):
    """
    Fetches a list of all available ontology acronyms from AgroPortal.

    Args:
        user_agent (str): User-Agent for the HTTP header.
        api_key (str): The AgroPortal API Key (required).
        base_url (str): The base URL of the AgroPortal API.

    Returns:
        list: A list of ontology acronyms (strings). Returns empty list on error or if API key is missing.
    """
    if not api_key:
        logging.error("AgroPortal API Key is required to fetch available ontologies but was not provided.")
        return []

    ontologies_url = f"{base_url}/ontologies"
    available_acronyms = []
    response = None

    headers = {
        'Authorization': f"apikey token={api_key}",
        'Accept': 'application/json',
        'User-Agent': user_agent
    }

    try:
        logging.info(f"Fetching available ontologies from AgroPortal: {ontologies_url}")
        response = requests.get(ontologies_url, headers=headers, timeout=30)
        response.raise_for_status() # Check for HTTP 4xx/5xx errors

        data = response.json()

        if isinstance(data, list):
            for ontology_item in data:
                acronym = ontology_item.get('acronym')
                if acronym:
                    available_acronyms.append(acronym)
            logging.info(f"Successfully fetched {len(available_acronyms)} ontologies from AgroPortal.")
        else:
            logging.warning(f"AgroPortal /ontologies endpoint returned unexpected data format: {type(data)}")
            if 'error' in data or 'errors' in data:
                logging.error(f"AgroPortal API reported error fetching ontologies: {data}")

    except requests.exceptions.HTTPError as e:
        logging.error(f"AgroPortal HTTP Error fetching ontologies: {e.response.status_code} {e.response.reason}", exc_info=True)
        if e.response.status_code == 401:
            logging.error("AgroPortal: Invalid or missing API Key when fetching ontologies!")
        elif e.response.status_code == 403:
            logging.error("AgroPortal: Access Denied when fetching ontologies (rate limit?)")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"AgroPortal Network Error fetching ontologies: {e}", exc_info=True)
        return []
    except ValueError as e: # JSONDecodeError
        logging.error(f"AgroPortal: Error parsing JSON response fetching ontologies: {e}", exc_info=True)
        return []
    except Exception as e:
        logging.exception(f"AgroPortal: Unexpected error fetching ontologies.")
        return []

    return available_acronyms
