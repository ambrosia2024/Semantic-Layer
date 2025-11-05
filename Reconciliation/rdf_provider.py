import streamlit as st
from rdflib.namespace import RDFS, SKOS
import logging
from .reconciliation_utils import calculate_levenshtein_score
import re

logger = logging.getLogger(__name__)

def search_local_rdf(term, limit, graph, **kwargs):
    """Searches the local RDF graph for a given term using fuzzy matching."""
    if graph is None:
        logger.warning("Local RDF graph is None.")
        return []

    logger.info(f"Fuzzily searching local RDF graph ({len(graph)} triples) for term: '{term}'")
    
    # Escape the term to make it safe for use in a regex
    escaped_term = re.escape(term)
    
    # More robust SPARQL query using REGEX for case-insensitive matching
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>

    SELECT DISTINCT ?uri ?label ?description
    WHERE {{
        ?uri rdfs:label|skos:prefLabel|skos:altLabel|dc:title ?label .
        
        FILTER(REGEX(STR(?label), "{escaped_term}", "i"))
        
        OPTIONAL {{ 
            ?uri rdfs:comment|skos:definition ?description .
        }}
    }}
    LIMIT {limit}
    """
    
    all_possible_matches = []
    try:
        qres = graph.query(query)
        logger.info(f"SPARQL query returned {len(qres)} total labels to check for fuzzy matching.")
        
        for row in qres:
            label_text = str(row.label)
            score = calculate_levenshtein_score(term, label_text)
            
            # Set a threshold for what is considered a "similar" term
            if score > 0.6: # Threshold can be adjusted
                all_possible_matches.append({
                    "uri": str(row.uri),
                    "label": label_text,
                    "description": str(row.description) if row.description else "",
                    "source_provider": "Local RDF File",
                    "levenshtein_score": score # Add score for sorting
                })

    except Exception as e:
        logger.error(f"Error executing SPARQL query or processing results for term '{term}': {e}", exc_info=True)

    # Sort by score in descending order
    sorted_matches = sorted(all_possible_matches, key=lambda x: x['levenshtein_score'], reverse=True)
    
    logger.info(f"Found {len(sorted_matches)} fuzzy matches for '{term}' above threshold. Returning top {limit}.")
    
    return sorted_matches[:limit]
