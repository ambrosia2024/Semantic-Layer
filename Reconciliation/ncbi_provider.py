# -*- coding: utf-8 -*-
import requests
import time
# import streamlit as st # Only needed if mixing logging with UI feedback
import traceback
import urllib.parse
import logging # Import logging module

# --- Constants ---
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
ESEARCH_URL = EUTILS_BASE_URL + "esearch.fcgi"
ESUMMARY_URL = EUTILS_BASE_URL + "esummary.fcgi"

# --- Helper function to construct NCBI URLs ---
def _construct_ncbi_uri(db, item_id):
    """Constructs a standard URL for an NCBI object."""
    if not item_id:
        return None
    # Specific URLs for common databases
    if db == 'taxonomy':
        return f"https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id={item_id}"
    elif db == 'bioproject':
        return f"https://www.ncbi.nlm.nih.gov/bioproject/{item_id}"
    elif db == 'gene':
        return f"https://www.ncbi.nlm.nih.gov/gene/{item_id}"
    elif db == 'protein':
        return f"https://www.ncbi.nlm.nih.gov/protein/{item_id}"
    elif db == 'nuccore':
        return f"https://www.ncbi.nlm.nih.gov/nuccore/{item_id}"
    elif db == 'biosample':
        return f"https://www.ncbi.nlm.nih.gov/biosample/{item_id}"
    elif db == 'sra':
        return f"https://www.ncbi.nlm.nih.gov/sra?term={item_id}" # SRA uses term, not ID directly in URL
    elif db == 'pubmed':
        return f"https://pubmed.ncbi.nlm.nih.gov/{item_id}/"
    # Generic fallback for other databases
    else:
        return f"https://www.ncbi.nlm.nih.gov/{db}/{item_id}"


# If running outside Streamlit or caching isn't needed, define a dummy decorator
def cache_data(ttl=None):
    def decorator(func):
        return func
    return decorator

@cache_data(ttl=3600) # Use dummy if no Streamlit
def query_ncbi(
    term,
    databases_to_search=['taxonomy', 'bioproject', 'gene', 'protein', 'nuccore', 'biosample', 'sra', 'pubmed'], # Expanded default databases
    limit_per_db=10, # How many IDs to fetch from ESearch per DB?
    api_key=None,    # Your NCBI API Key (STRONGLY RECOMMENDED!)
    tool_name="StreamlitReconTool", # Your tool's name
    email="user@example.com",      # Your email address (IMPORTANT!)
    user_agent="DefaultStreamlitClient/NCBI" # User-Agent for header
    ):
    """
    Queries NCBI E-utilities (ESearch + ESummary) to reconcile terms.

    Args:
        term (str): The term to search for.
        databases_to_search (list): List of NCBI databases to search.
        limit_per_db (int): Maximum number of IDs ESearch should return per DB.
        api_key (str, optional): Your NCBI API Key.
        tool_name (str): Name of the tool for E-utilities parameters.
        email (str): User's email for E-utilities parameters.
        user_agent (str): User-Agent for the HTTP header.

    Returns:
        list: A list of suggestion dictionaries
              [{'uri': ..., 'label': ..., 'description': ..., 'db': ..., 'id': ...}].
              Returns an empty list on error.
    """
    if not term:
        logging.warning("NCBI query attempted with empty term.")
        return []
    if not email or "@example.com" in email or email == "no-email-provided@example.com":
         logging.warning(f"NCBI Provider: A valid email address must be provided (current: {email}). Continuing, but NCBI might block requests.")

    all_suggestions = []
    base_params = {
        'tool': tool_name,
        'email': email,
    }
    if api_key:
        base_params['api_key'] = api_key
        sleep_time = 0.11 # Rate limit for API Key: 10 requests/second
    else:
        logging.warning("NCBI Provider: No NCBI API Key provided. Rate limit restricted to 3 requests/second.")
        sleep_time = 0.34 # Rate limit without API Key: 3 requests/second

    headers = {'User-Agent': user_agent}

    # 1. Iterate through the databases to search
    for db in databases_to_search:
        search_ids = []
        esearch_response = None
        logging.info(f"NCBI: Starting ESearch for term '{term}' in database '{db}'")
        try:
            # 2. ESearch Request (GET)
            esearch_params = base_params.copy()
            esearch_params.update({
                'db': db,
                'term': term,
                'retmax': limit_per_db,
                'retmode': 'json',
            })

            logging.debug(f"NCBI ESearch ({db}) params: {esearch_params}")
            esearch_response = requests.get(ESEARCH_URL, params=esearch_params, headers=headers, timeout=20)
            logging.debug(f"NCBI ESearch ({db}) request sent, sleeping for {sleep_time}s")
            time.sleep(sleep_time) # Wait AFTER EACH request!
            esearch_response.raise_for_status()
            esearch_data = esearch_response.json()

            # Extract IDs
            if 'esearchresult' in esearch_data and 'idlist' in esearch_data['esearchresult']:
                search_ids = esearch_data['esearchresult']['idlist']
                logging.info(f"NCBI ESearch ({db}) for '{term}' found {len(search_ids)} IDs: {search_ids}")
            elif 'error' in esearch_data:
                 logging.warning(f"NCBI ESearch ({db}) API error for '{term}': {esearch_data.get('error')}")
                 continue
            elif 'warning' in esearch_data:
                 logging.warning(f"NCBI ESearch ({db}) API warning for '{term}': {esearch_data.get('warning')}")
                 if 'esearchresult' in esearch_data and 'idlist' in esearch_data['esearchresult']:
                    search_ids = esearch_data['esearchresult']['idlist']
                 else:
                     continue

        except ValueError as e:
            logging.warning(f"NCBI ESearch ({db}): Error parsing JSON response for '{term}'. Status: {esearch_response.status_code if esearch_response else 'N/A'}", exc_info=True)
            if esearch_response is not None: logging.warning(f"NCBI ESearch ({db}) Response Text: {esearch_response.text[:200]}...")
            continue
        except requests.exceptions.RequestException as e:
             logging.warning(f"NCBI ESearch ({db}): Network/HTTP Error for '{term}': {e}", exc_info=True)
             continue
        except Exception as e:
            logging.exception(f"NCBI ESearch ({db}): Unexpected error for '{term}'")
            continue


        # 3. ESummary Request (POST), if IDs were found
        if search_ids:
            esummary_response = None
            logging.info(f"NCBI: Starting ESummary for {len(search_ids)} IDs in database '{db}' for term '{term}'")
            try:
                esummary_payload = base_params.copy()
                esummary_payload.update({
                    'db': db,
                    'id': ",".join(search_ids),
                    'retmode': 'json',
                    'version': '2.0',
                })

                logging.debug(f"NCBI ESummary ({db}) payload keys: {list(esummary_payload.keys())}")
                esummary_response = requests.post(ESUMMARY_URL, data=esummary_payload, headers=headers, timeout=30)
                logging.debug(f"NCBI ESummary ({db}) request sent, sleeping for {sleep_time}s")
                time.sleep(sleep_time)
                esummary_response.raise_for_status()
                esummary_data = esummary_response.json()

                # 4. Parse ESummary JSON (database-specific!)
                if 'result' in esummary_data:
                    result_dict = esummary_data['result']
                    processed_uids = result_dict.get('uids', [])

                    logging.info(f"NCBI ESummary ({db}) received results for {len(processed_uids)} UIDs.")

                    for uid in processed_uids:
                        if uid in result_dict:
                            item = result_dict[uid]
                            suggestion = {'db': db}
                            label = f"Unknown {db} {uid}"
                            description = ""
                            item_id_for_uri = uid

                            try:
                                # Parsing logic per database
                                if db == 'taxonomy':
                                    label = item.get('scientificname', label)
                                    description = item.get('rank', '')
                                elif db == 'bioproject':
                                    accession = item.get('project_acc')
                                    if accession:
                                        item_id_for_uri = accession
                                    label = item.get('name', item.get('project_title', label))
                                    description = item.get('project_description', '')
                                elif db == 'gene':
                                    label = item.get('name', label)
                                    description = item.get('description', item.get('summary', ''))
                                elif db == 'protein':
                                    label = item.get('title', label)
                                    description = item.get('organism', '')
                                elif db == 'nuccore':
                                    label = item.get('title', label)
                                    description = item.get('organism', '')
                                elif db == 'biosample':
                                    label = item.get('accession', label) # BioSample uses accession
                                    item_id_for_uri = item.get('accession', uid)
                                    description = item.get('description', '')
                                elif db == 'sra':
                                    label = item.get('title', label)
                                    description = item.get('description', '')
                                elif db == 'pubmed':
                                    label = item.get('title', label)
                                    description = item.get('authors', [{}])[0].get('name', '') if item.get('authors') else ''
                                    description += f" ({item.get('pubdate', '')})" if item.get('pubdate') else ''

                                # Construct URI
                                uri = _construct_ncbi_uri(db, item_id_for_uri)
                                if uri:
                                    suggestion['uri'] = uri
                                    suggestion['label'] = label
                                    suggestion['description'] = description
                                    suggestion['id'] = item_id_for_uri
                                    suggestion['source_provider'] = f"NCBI {db}" # Add source provider with specific DB
                                    all_suggestions.append(suggestion)
                                    logging.debug(f"NCBI: Added suggestion for '{term}' ({db}): {label} <{uri}>")

                            except Exception as parse_e:
                                logging.warning(f"NCBI Provider: Error parsing ESummary item ({db}, UID: {uid}) for '{term}': {parse_e}. Item: {item}", exc_info=True)

                        else:
                             logging.warning(f"NCBI ESummary ({db}): UID '{uid}' from 'uids' list not found in 'result' dictionary for term '{term}'.")

                elif 'error' in esummary_data:
                     logging.warning(f"NCBI ESummary ({db}) API error for '{term}': {esummary_data.get('error')}")
                elif 'warning' in esummary_data:
                     logging.warning(f"NCBI ESummary ({db}) API warning for '{term}': {esummary_data.get('warning')}")

            except ValueError as e:
                logging.warning(f"NCBI ESummary ({db}): Error parsing JSON response for '{term}'. Status: {esummary_response.status_code if esummary_response else 'N/A'}", exc_info=True)
                if esummary_response is not None: logging.warning(f"NCBI ESummary ({db}) Response Text: {esummary_response.text[:200]}...")
            except requests.exceptions.RequestException as e:
                 logging.warning(f"NCBI ESummary ({db}): Network/HTTP Error for '{term}': {e}", exc_info=True)
            except Exception as e:
                logging.exception(f"NCBI ESummary ({db}): Unexpected error for '{term}'")

    logging.info(f"NCBI query for '{term}' finished across all specified databases.")
    return all_suggestions
