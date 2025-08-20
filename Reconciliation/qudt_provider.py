# -*- coding: utf-8 -*-
import logging
import time
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

logger = logging.getLogger(__name__)

def query_qudt(term: str, limit: int, user_agent: str, config: dict) -> list:
    """
    Queries the QUDT SPARQL endpoint for units or quantity kinds based on the term.
    """
    endpoint_url = "https://qudt.org/fuseki/qudt/query" # Hardcoded as it's a persistent public endpoint
    
    # No need to check if endpoint_url is empty as it's hardcoded

    # Basic sanitization for the term to prevent SPARQL injection issues
    sanitized_term = term.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")

    # QUDT has Units and QuantityKinds. We can search both.
    # This query searches for labels containing the term in English.
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX qudt: <http://qudt.org/schema/qudt/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX dcterms: <http://purl.org/dc/terms/>

    SELECT DISTINCT ?uri ?label ?description WHERE {{
      {{
        ?uri a qudt:Unit ;
             rdfs:label ?label .
        FILTER (CONTAINS(LCASE(STR(?label)), LCASE("{sanitized_term}")))
        OPTIONAL {{ ?uri dcterms:description ?description . }}
      }} UNION {{
        ?uri a qudt:QuantityKind ;
             rdfs:label ?label .
        FILTER (CONTAINS(LCASE(STR(?label)), LCASE("{sanitized_term}")))
        OPTIONAL {{ ?uri dcterms:description ?description . }}
      }}
      FILTER (lang(?label) = 'en')
    }}
    LIMIT {limit}
    """

    results_list = []
    try:
        sparql = SPARQLWrapper(endpoint_url)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        sparql.agent = user_agent
        sparql.setTimeout(30) # 30-second timeout

        logger.info(f"Querying QUDT endpoint: {endpoint_url} for term '{term}'")
        start_time = time.time()
        results = sparql.query().convert()
        duration = time.time() - start_time
        logger.info(f"QUDT query for '{term}' took {duration:.2f}s, got {len(results.get('results', {}).get('bindings', []))} bindings.")

        for result in results["results"]["bindings"]:
            uri = result.get("uri", {}).get("value")
            label = result.get("label", {}).get("value")
            description = result.get("description", {}).get("value", "")

            if uri and label:
                results_list.append({
                    "uri": uri,
                    "label": label,
                    "description": description,
                    "score": None, # QUDT SPARQL doesn't provide a score directly
                    "source_provider": "QUDT"
                })
            else:
                logger.warning(f"Missing URI or label in QUDT result binding: {result}")

    except SPARQLExceptions.EndPointNotFound as e:
        logger.error(f"QUDT Endpoint not found or invalid: {endpoint_url}. Error: {e}")
        raise ConnectionError(f"QUDT SPARQL Endpoint not found: {endpoint_url}") from e
    except SPARQLExceptions.QueryBadFormed as e:
        logger.error(f"QUDT SPARQL query badly formed. Error: {e}\nQuery Attempted:\n{query}")
        raise ValueError(f"Bad QUDT SPARQL Query. Check template/term syntax. Error: {e}") from e
    except ConnectionRefusedError as e:
        logger.error(f"Connection refused by QUDT endpoint: {endpoint_url}. Error: {e}")
        raise ConnectionError(f"Connection refused by QUDT endpoint: {endpoint_url}") from e
    except Exception as e:
        logger.exception(f"An unexpected error occurred querying QUDT endpoint '{endpoint_url}' for term '{term}'.")
        raise ConnectionError(f"Failed to query QUDT {endpoint_url}: {e}") from e

    return results_list[:limit]
