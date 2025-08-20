# geonames_provider.py
import requests
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin

# Configure logger for this module
logger = logging.getLogger(__name__)

def query_geonames(term: str, limit: int, username: str, base_url: str, user_agent: str) -> List[Dict[str, Any]]:
    """
    Queries the GeoNames API (searchJSON endpoint) for place names based on a search term.

    Args:
        term: The search term (e.g., "Berlin", "Mount Everest").
        limit: The maximum number of results to return.
        username: Your registered GeoNames username (required for API access).
        base_url: The base URL for the GeoNames API (e.g., "https://secure.geonames.org").
        user_agent: The User-Agent string to identify your application.

    Returns:
        A list of dictionaries, where each dictionary represents a found place
        and contains the keys 'label', 'description', and 'uri'.
        Returns an empty list if no results are found or in case of API errors.
    """
    if not username:
        logger.error("GeoNames username is missing. Cannot query API.")
        return []
    if not term:
        logger.warning("Search term is empty. Skipping GeoNames query.")
        return []
    if not base_url:
        logger.error("GeoNames base_url is missing. Cannot query API.")
        return []

    # --- Prepare Request Parameters and Headers ---
    endpoint_path = "searchJSON"
    try:
        full_api_url = urljoin(base_url.strip('/') + '/', endpoint_path)
    except Exception as url_e:
        logger.error(f"Could not construct valid GeoNames API URL from base '{base_url}' and path '{endpoint_path}': {url_e}")
        return []


    params = {
        'q': term,
        'maxRows': limit,
        'username': username,
        'style': 'full'
    }

    headers = {
        'User-Agent': user_agent
    }

    # --- Make API Call and Process Response ---
    results: List[Dict[str, Any]] = []
    try:
        logger.debug(f"Querying GeoNames: URL='{full_api_url}', term='{term}', limit={limit}, username='{username}'")
        response = requests.get(full_api_url, params=params, headers=headers, timeout=15)

        response.raise_for_status()
        data = response.json()

        if 'status' in data and isinstance(data['status'], dict):
            api_message = data['status'].get('message', 'Unknown GeoNames API error')
            api_code = data['status'].get('value')
            logger.error(f"GeoNames API Error: {api_message} (Code: {api_code}) for term '{term}'")
            return []

        if 'geonames' not in data or not isinstance(data['geonames'], list):
            logger.warning(f"Unexpected GeoNames response for term '{term}': {data}")
            return []

        geonames_results = data['geonames']
        logger.info(f"GeoNames returned {len(geonames_results)} results for term '{term}'.")

        for item in geonames_results:
            try:
                geoname_id = item.get('geonameId')
                if not geoname_id: continue
                label = item.get('toponymName', item.get('name'))
                if not label: continue

                desc_parts = [item.get('fcodeName'), item.get('adminName1'), item.get('countryName')]
                description = ", ".join(filter(None, desc_parts))
                uri = f"http://sws.geonames.org/{geoname_id}/"

                result_entry: Dict[str, Any] = {
                    'label': label,
                    'description': description,
                    'uri': uri,
                    'source_provider': "GeoNames" # Add source provider information
                }
                results.append(result_entry)

            except Exception as item_e:
                geoname_id_str = item.get('geonameId', 'UNKNOWN_ID')
                logger.exception(f"Error processing GeoNames item {geoname_id_str}: {item_e}")

    except requests.exceptions.Timeout: logger.error(f"GeoNames request timed out for term '{term}'.")
    except requests.exceptions.HTTPError as http_err: logger.error(f"GeoNames HTTP Error for term '{term}': {http_err}")
    except requests.exceptions.RequestException as req_err: logger.error(f"GeoNames Request failed for term '{term}': {req_err}")
    except ValueError as json_err: logger.error(f"Failed JSON decode from GeoNames for term '{term}': {json_err}")
    except Exception as e: logger.exception(f"Unexpected error during GeoNames query for term '{term}': {e}")

    return results

# --- Example Usage (updated) ---
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
#     test_username = "your_geonames_username"
#     test_base_url = "https://secure.geonames.org" # Get from config usually
#     test_user_agent = "MyReconciliationTool/Test (mycontact@example.com)"
#
#     if test_username == "your_geonames_username":
#         print("\nWARNING: Replace 'your_geonames_username'.\n")
#     else:
#         search_terms = ["Berlin", "Mount Everest"]
#         for term in search_terms:
#             print(f"\n--- Querying for: {term} ---")
#             places = query_geonames(term, 5, test_username, test_base_url, test_user_agent) # Pass base_url
#             # ... (print results) ...
