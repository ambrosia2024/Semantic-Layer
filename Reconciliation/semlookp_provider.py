# -*- coding: utf-8 -*-
import requests
import time
import traceback
import logging # Import logging module
from urllib.parse import quote # Needed for potential IRI encoding if retrieving single terms later

# Configure logging (basic example, adjust as needed for your app)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use cache_data if Streamlit context is available and desired
# from streamlit import cache_data
# @cache_data(ttl=3600) # Cache results for 1 hour

# If running outside Streamlit or caching isn't needed, define a dummy decorator
def cache_data(ttl=None):
    def decorator(func):
        return func
    return decorator

@cache_data(ttl=3600) # Use dummy if no Streamlit
def query_semlookp(
    term,
    limit=7,
    ontologies=None, # Optional: list of ontology IDs like ['mesh', 'snomedct']
    api_url="https://semanticlookup.zbmed.de/ols/api/select", # Use the /select endpoint
    user_agent="MyReconApp/1.0 (SemLookPClient; contact@example.com)", # CHANGE THIS!
    ):
    """
    Queries the SemLookP API (based on OLS) for entities using the /select endpoint.

    Args:
        term (str): The term to search for.
        limit (int): Maximum number of suggestions.
        ontologies (list, optional): A list of ontology IDs to restrict the search
                                      (e.g., ['mesh', 'snomedct']). Defaults to None (all).
        api_url (str): The URL of the SemLookP API /select endpoint.
        user_agent (str): The User-Agent string for the request. CHANGE THIS!

    Returns:
        list: A list of suggestion dictionaries [{'uri': ..., 'label': ..., 'description': ...}].
              Returns an empty list on error.
    """
    if not term:
        logging.warning("SemLookP query attempted with empty term.")
        return []

    params = {
        "q": term,
        "rows": limit,
        # Explicitly request fields needed, reduces response size slightly
        "fieldList": "iri,label,description,ontology_name,ontology_prefix",
        # Parameters below are relevant for /select
        "type": "class,property,individual", # Search across different entity types
        "local": "false", # Search across all ontologies where term might be referenced
        "obsoletes": "false" # Exclude obsolete terms
    }

    # Add ontology filter if provided
    if ontologies and isinstance(ontologies, list) and len(ontologies) > 0:
        params["ontology"] = ",".join(ontologies)
        logging.info(f"SemLookP: Filtering by ontologies: {params['ontology']}")

    # Standard headers
    headers = {
        'User-Agent': user_agent,
        'Accept': 'application/json'
        }

    suggestions = []
    response = None
    logging.info(f"Querying SemLookP: term='{term}', limit={limit}, params={params}")

    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=20) # Increased timeout slightly
        logging.debug(f"SemLookP request sent ({response.url}), sleeping briefly...")
        time.sleep(0.05) # Brief pause respectful of API rate limits (if any)
        response.raise_for_status() # Raises HTTPError for 4xx/5xx

        data = response.json()      # Can raise JSONDecodeError (subclass of ValueError)

        # --- Response Structure Check ---
        # OLS/Solr responses often have results under response.docs
        # Check the actual structure if this doesn't work. HAL might use _embedded.
        if "response" in data and "docs" in data["response"]:
            results = data["response"]["docs"]
            logging.info(f"SemLookP returned {len(results)} hits (found in response.docs) for '{term}'.")

            for result in results:
                uri = result.get("iri")
                label = result.get("label", "N/A")
                desc_list = result.get("description") # Description is an array
                ontology_prefix = result.get("ontology_prefix", "")
                ontology_name = result.get("ontology_name", "")

                # Extract description safely
                description = ""
                if isinstance(desc_list, list) and len(desc_list) > 0:
                    description = desc_list[0] # Take the first description

                # Add ontology context to description if helpful
                context = ontology_prefix or ontology_name
                if context:
                     full_description = f"[{context}] {description}" if description else f"[{context}]"
                else:
                     full_description = description

                if uri: # Only add if we have a valid URI (IRI)
                    # Set source provider to just the ontology prefix/name
                    source_provider_name = ontology_prefix or ontology_name or "SemLookP"

                    suggestions.append({
                        "uri": uri,
                        "label": label,
                        "description": description.strip(), # Use original description, no prefix
                        "source_provider": source_provider_name
                    })
                    logging.debug(f"SemLookP: Added suggestion for '{term}': {label} <{uri}>")
                else:
                    logging.warning(f"SemLookP hit for '{term}' skipped due to missing IRI: {result}")
        # Add check for HAL structure as fallback?
        # elif "_embedded" in data and "terms" in data["_embedded"]:
        #    results = data["_embedded"]["terms"]
        #    logging.info(f"SemLookP returned {len(results)} hits (found in _embedded.terms) for '{term}'.")
        #    # ... processing logic would need slight adjustment if fields differ here ...
        else:
            logging.info(f"SemLookP: No 'response.docs' (or expected structure) in response for '{term}'. Data keys: {data.keys()}")
            # Log part of the response if structure is unexpected
            logging.debug(f"SemLookP Response Snippet: {str(data)[:500]}")


    except ValueError as e: # JSONDecodeError
        status_code = response.status_code if response else 'N/A'
        logging.warning(f"SemLookP: Error parsing JSON response for '{term}'. Status: {status_code}", exc_info=True)
        if response is not None:
            logging.warning(f"SemLookP Response Text: {response.text[:300]}...")
        return []
    except requests.exceptions.HTTPError as e:
        logging.warning(f"SemLookP: HTTP Error for '{term}': {e}", exc_info=False) # Log less verbosely for HTTP errors initially
        if response is not None:
            logging.warning(f"SemLookP Response Status: {response.status_code}")
            try:
                error_data = response.json()
                error_message = error_data.get('message', response.text)
                logging.warning(f"SemLookP Error Details: {error_message}")
            except ValueError: # If error response isn't JSON
                logging.warning(f"SemLookP Response Text: {response.text[:300]}...")
        return []
    except requests.exceptions.RequestException as e: # Catches Timeout, ConnectionError etc.
         logging.warning(f"SemLookP: Network Error for '{term}': {e}", exc_info=True)
         return []
    except Exception as e:
        logging.exception(f"SemLookP: Unexpected error for '{term}'") # Log full traceback
        return []

    logging.info(f"SemLookP query for '{term}' finished, returning {len(suggestions)} suggestions.")
    return suggestions

@cache_data(ttl=3600 * 24) # Cache for 24 hours as ontology list doesn't change often
def get_available_ontologies(
    user_agent="MyReconApp/1.0 (SemLookPClient; contact@example.com)", # CHANGE THIS!
    api_url="https://semanticlookup.zbmed.de/ols/api/ontologies" # Use the /ontologies endpoint
):
    """
    Fetches a list of all available ontology prefixes (acronyms/IDs) from SemLookP.
    SemLookP uses an OLS-like API structure.

    Args:
        user_agent (str): The User-Agent string for the request.
        api_url (str): The URL of the SemLookP API /ontologies endpoint.

    Returns:
        list: A list of ontology prefixes (strings). Returns empty list on error.
    """
    available_prefixes = []
    response = None

    headers = {
        'User-Agent': user_agent,
        'Accept': 'application/json'
    }

    try:
        logging.info(f"Fetching available ontologies from SemLookP: {api_url}")
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status() # Check for HTTP 4xx/5xx errors

        data = response.json()

        # SemLookP /ontologies endpoint (OLS-like) returns a _embedded.ontologies list
        if '_embedded' in data and 'ontologies' in data['_embedded'] and isinstance(data['_embedded']['ontologies'], list):
            for ontology_item in data['_embedded']['ontologies']:
                ontology_prefix = ontology_item.get('ontologyId') # OLS uses ontologyId as the prefix/acronym
                if ontology_prefix:
                    available_prefixes.append(ontology_prefix)
            logging.info(f"Successfully fetched {len(available_prefixes)} ontologies from SemLookP.")
        else:
            logging.warning(f"SemLookP /ontologies endpoint returned unexpected data format: {type(data)}")
            if 'error' in data or 'errors' in data:
                logging.error(f"SemLookP API reported error fetching ontologies: {data}")

    except requests.exceptions.HTTPError as e:
        logging.error(f"SemLookP HTTP Error fetching ontologies: {e.response.status_code} {e.response.reason}", exc_info=True)
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"SemLookP Network Error fetching ontologies: {e}", exc_info=True)
        return []
    except ValueError as e: # JSONDecodeError
        logging.error(f"SemLookP: Error parsing JSON response fetching ontologies: {e}", exc_info=True)
        return []
    except Exception as e:
        logging.exception(f"SemLookP: Unexpected error fetching ontologies.")
        return []

    return available_prefixes

# # Example Usage (if run directly)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     search_term = "diabetes"
#     # search_term = "heart attack"
#     # ontos = ['mesh'] # Example: Search only in MeSH
#     ontos = None # Example: Search all
#     results = query_semlookp(search_term, limit=5, ontologies=ontos)
#
#     if results:
#         print(f"\nFound {len(results)} results for '{search_term}':")
#         for res in results:
#             print(f"- Label: {res['label']}")
#             print(f"  URI: {res['uri']}")
#             print(f"  Desc: {res['description']}")
#     else:
#         print(f"\nNo results found or error occurred for '{search_term}'.")
#
#     print("\n--- Testing get_available_ontologies ---")
#     available_ontos = get_available_ontologies()
#     if available_ontos:
#         print(f"Found {len(available_ontos)} available ontologies:")
#         print(available_ontos[:10]) # Print first 10
#     else:
#         print("Failed to fetch available ontologies.")
