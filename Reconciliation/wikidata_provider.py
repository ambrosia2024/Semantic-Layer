# -*- coding: utf-8 -*-
import requests
import time
# import streamlit as st # Only needed if mixing logging with UI feedback
import traceback
import logging # Import logging module

# Use cache_data if Streamlit context is available and desired
# from streamlit import cache_data
# @cache_data(ttl=3600) # Cache results for 1 hour

# If running outside Streamlit or caching isn't needed, define a dummy decorator
def cache_data(ttl=None):
    def decorator(func):
        return func
    return decorator

@cache_data(ttl=3600) # Use dummy if no Streamlit
def query_wikidata(
    term,
    limit=7,
    api_url="https://www.wikidata.org/w/api.php",
    user_agent="DefaultStreamlitClient/Wikidata"
    # proxies parameter removed
    ):
    """
    Queries the Wikidata API for entities (via GET).

    Args:
        term (str): The term to search for.
        limit (int): Maximum number of suggestions.
        api_url (str): The URL of the Wikidata API.
        user_agent (str): The User-Agent string for the request.

    Returns:
        list: A list of suggestion dictionaries [{'uri': ..., 'label': ..., 'description': ...}].
              Returns an empty list on error.
    """
    if not term:
        logging.warning("Wikidata query attempted with empty term.")
        return []

    params = {"action": "wbsearchentities", "search": term, "language": "en", "format": "json", "limit": limit}
    headers = {'User-Agent': user_agent}
    suggestions = []
    response = None
    logging.info(f"Querying Wikidata: term='{term}', limit={limit}")
    try:
        # proxies argument removed
        response = requests.get(api_url, params=params, headers=headers, timeout=15)
        logging.debug(f"Wikidata request sent, sleeping briefly...")
        time.sleep(0.05) # Brief pause
        response.raise_for_status() # Raises HTTPError for 4xx/5xx
        data = response.json()      # Can raise ValueError

        if "search" in data:
            logging.info(f"Wikidata returned {len(data['search'])} hits for '{term}'.")
            for result in data["search"]:
                # Construct URI if concepturi is missing
                entity_id = result.get('id')
                uri = result.get("concepturi")
                if not uri and entity_id:
                    uri = f"http://www.wikidata.org/entity/{entity_id}"

                if uri: # Only add if we have a valid URI
                    suggestions.append({
                        "uri": uri,
                        "label": result.get("label", "N/A"),
                        "description": result.get("description", ""),
                        "source_provider": "Wikidata" # Add source provider information
                    })
                    logging.debug(f"Wikidata: Added suggestion for '{term}': {result.get('label', 'N/A')} <{uri}>")
                else:
                    logging.warning(f"Wikidata hit for '{term}' skipped due to missing URI/ID: {result}")
        else:
            logging.info(f"Wikidata: No 'search' key in response for '{term}'.")


    except ValueError as e: # JSONDecodeError
        logging.warning(f"Wikidata: Error parsing JSON response for '{term}'. Status: {response.status_code if response else 'N/A'}", exc_info=True)
        if response is not None: logging.warning(f"Wikidata Response Text: {response.text[:200]}...")
        return []
    except requests.exceptions.RequestException as e: # Catches Timeout, HTTPError, ConnectionError etc.
         logging.warning(f"Wikidata: Network/HTTP Error for '{term}': {e}", exc_info=True)
         # Proxy check removed
         return []
    except Exception as e:
        logging.exception(f"Wikidata: Unexpected error for '{term}'") # Log full traceback
        return []

    logging.info(f"Wikidata query for '{term}' finished, returning {len(suggestions)} suggestions.")
    return suggestions
