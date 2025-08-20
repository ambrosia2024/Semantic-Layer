# agrovoc_provider.py
import requests
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode # To properly encode SPARQL query for GET request

# Configure logger for this module
# It will inherit the configuration from the main app if logging is set up there before import
logger = logging.getLogger(__name__)

# AGROVOC SPARQL Endpoint URL (Default, can be overridden via config)
DEFAULT_AGROVOC_SPARQL_ENDPOINT = "https://agrovoc.fao.org/sparql"

# SPARQL Query Template with GROUP BY clause added
SPARQL_QUERY_TEMPLATE = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX skosxl: <http://www.w3.org/2008/05/skos-xl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT DISTINCT ?conceptURI ?prefLabel (SAMPLE(?definition) AS ?definitionSample) (SAMPLE(?scopeNote) AS ?scopeNoteSample)
WHERE {{
  GRAPH <http://aims.fao.org/aos/agrovoc/> {{
    # Search in preferred or alternative labels using skosxl literal form
    {{ ?conceptURI skosxl:prefLabel ?labelLit . }}
    UNION
    {{ ?conceptURI skosxl:altLabel ?labelLit . }}

    ?labelLit skosxl:literalForm ?labelValue .

    # Filter by term (case-insensitive contains) and language
    # Using STR() ensures we compare the string value
    # Using REGEX flags "i" for case-insensitivity
    FILTER(REGEX(STR(?labelValue), "{term}", "i"))
    FILTER(LANGMATCHES(LANG(?labelValue), "{lang}"))

    # Get the preferred label in the target language
    ?conceptURI skosxl:prefLabel ?prefLabelLit .
    ?prefLabelLit skosxl:literalForm ?prefLabel .
    FILTER(LANGMATCHES(LANG(?prefLabel), "{lang}"))

    # Optionally get definition in the target language
    OPTIONAL {{
      ?conceptURI skos:definition ?definition .
      FILTER(LANGMATCHES(LANG(?definition), "{lang}"))
    }}
    # Optionally get scope note in the target language
    OPTIONAL {{
      ?conceptURI skos:scopeNote ?scopeNote .
      FILTER(LANGMATCHES(LANG(?scopeNote), "{lang}"))
    }}

    # Ensure it's a concept
    ?conceptURI a skos:Concept .
  }}
}}
# --- ADDED GROUP BY Clause to fix "Non-group key variable" error ---
GROUP BY ?conceptURI ?prefLabel
# ------------------------------------------------------------------
LIMIT {limit}
"""

def query_agrovoc(
    term: str,
    limit: int,
    user_agent: str,
    sparql_endpoint: str = DEFAULT_AGROVOC_SPARQL_ENDPOINT, # Allow overriding endpoint
    lang: str = 'en'
    ) -> List[Dict[str, Any]]:
    """
    Queries the AGROVOC SPARQL endpoint to find concepts matching the term.

    Args:
        term: The search term (e.g., "Maize", "Soil erosion").
        limit: The maximum number of results to return.
        user_agent: The User-Agent string for the HTTP request header.
        sparql_endpoint: The URL of the AGROVOC SPARQL endpoint.
        lang: The preferred language code (e.g., 'en', 'es', 'fr').

    Returns:
        A list of dictionaries, where each dictionary represents a found concept
        and contains the keys 'label', 'description', and 'uri'.
        Returns an empty list if no results are found or in case of errors.
    """
    if not term:
        logger.warning("Search term is empty. Skipping AGROVOC SPARQL query.")
        return []
    if not sparql_endpoint:
        logger.error("AGROVOC SPARQL endpoint URL is missing.")
        return []

    # Basic escaping for the term within SPARQL REGEX
    safe_term = term.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")

    # Format the SPARQL query
    try:
        query = SPARQL_QUERY_TEMPLATE.format(term=safe_term, lang=lang, limit=limit)
    except KeyError as fmt_err:
        logger.error(f"Error formatting SPARQL query template: {fmt_err}")
        return []


    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": user_agent
    }
    # Parameters for GET request
    params = {"query": query}
    results: List[Dict[str, Any]] = []

    try:
        logger.debug(f"Querying AGROVOC SPARQL: lang='{lang}', limit={limit}, term='{term}'")
        logger.debug(f"SPARQL Query Snippet:\n{query[:500]}...")

        # Use GET request
        response = requests.get(sparql_endpoint, params=params, headers=headers, timeout=30)

        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

        data = response.json()

        # Check for SPARQL result structure
        if data and "results" in data and "bindings" in data["results"]:
            bindings = data["results"]["bindings"]
            logger.info(f"AGROVOC SPARQL returned {len(bindings)} potential results for term '{term}'.")

            for binding in bindings:
                try:
                    # Extract required fields from the binding
                    concept_uri = binding.get("conceptURI", {}).get("value")
                    pref_label = binding.get("prefLabel", {}).get("value")

                    # Extract description, prioritizing definition over scope note
                    definition = binding.get("definitionSample", {}).get("value")
                    scope_note = binding.get("scopeNoteSample", {}).get("value")
                    description = definition if definition else scope_note # Use definition if available, otherwise scope note

                    # Ensure essential fields (URI and Label) are present
                    if concept_uri and pref_label:
                        result_entry: Dict[str, Any] = {
                            'label': pref_label,
                            'description': description, # This will be None if neither found
                            'uri': concept_uri,
                            'source_provider': "AGROVOC" # Add source provider information
                        }
                        results.append(result_entry)
                    else:
                        logger.warning(f"Skipping SPARQL binding due to missing URI or Label: {binding}")

                except Exception as item_e:
                    # Catch errors processing a single binding but continue with others
                    uri_str = binding.get("conceptURI", {}).get("value", 'UNKNOWN_URI')
                    logger.exception(f"Error processing AGROVOC SPARQL binding {uri_str}: {item_e}")
        else:
            # Log unexpected response structure, might indicate an error not caught by status code
            logger.warning(f"Unexpected response structure or no results/bindings from AGROVOC SPARQL for term '{term}'. Response: {data}")

    except requests.exceptions.Timeout:
        logger.error(f"AGROVOC SPARQL request timed out for term '{term}'.")
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"AGROVOC SPARQL HTTP Error for term '{term}': {http_err}")
        # Log response body for debugging HTTP errors if possible
        try: logger.error(f"Response Body: {http_err.response.text[:500]}")
        except: pass
    except requests.exceptions.RequestException as req_err:
        logger.error(f"AGROVOC SPARQL Request failed for term '{term}': {req_err}")
    except ValueError as json_err: # Includes JSONDecodeError
        logger.error(f"Failed to decode JSON response from AGROVOC SPARQL for term '{term}': {json_err}")
        try: logger.error(f"Response text was: {response.text[:500]}...")
        except NameError: pass # response might not be defined if request failed earlier
    except Exception as e:
        logger.exception(f"An unexpected error occurred during AGROVOC SPARQL query for term '{term}': {e}")

    logger.debug(f"Returning {len(results)} processed results from AGROVOC for term '{term}'.")
    return results

# --- Example Usage ---
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
#     test_user_agent = "MyReconciliationTool/Test (mycontact@example.com)"
#
#     search_terms = ["Maize", "Soil erosion", "Wein", "Vitis Vinifera", "Chardonnay"]
#     for term in search_terms:
#         print(f"\n--- Querying AGROVOC for: {term} ---")
#         # concepts = query_agrovoc(term, 5, test_user_agent, lang='en') # Test English
#         concepts = query_agrovoc(term, 5, test_user_agent, lang='de') # Test German
#         if concepts:
#             for concept in concepts:
#                 print(f"  Label: {concept['label']}")
#                 print(f"  Desc:  {concept['description']}")
#                 print(f"  URI:   {concept['uri']}")
#         else:
#             print("  No results found or error occurred.")
