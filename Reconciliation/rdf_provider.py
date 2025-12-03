import streamlit as st
from rdflib.namespace import RDFS, SKOS
import logging
from .reconciliation_utils import calculate_levenshtein_score
import re

logger = logging.getLogger(__name__)

def search_local_rdf(term, limit, graph, **kwargs):
    if graph is None:
        logger.warning("Local RDF graph is None.")
        return []

    logger.info(f"Fuzzily searching local RDF graph ({len(graph)} triples) for term: '{term}'")

    # Nur für die SPARQL-String-Literal-Sicherheit, nicht als Regex:
    pattern = term.replace("\\", "\\\\").replace('"', '\\"')

    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX dc:   <http://purl.org/dc/elements/1.1/>
    PREFIX owl:  <http://www.w3.org/2002/07/owl#>
    PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
    PREFIX IAO:  <http://purl.obolibrary.org/obo/IAO_>
    PREFIX terms: <http://purl.org/dc/terms/>
    PREFIX dct:   <http://purl.org/dc/terms/>

    SELECT DISTINCT ?uri ?displayLabel ?description ?ontologyTitle
    WHERE {{
        # Typ-Filter: Klassen, Konzepte UND Individuen zulassen
        {{
            ?uri a owl:Class .
        }} UNION {{
            ?uri a rdfs:Class .
        }} UNION {{
            ?uri a skos:Concept .
        }} UNION {{
            ?uri a owl:NamedIndividual .
        }}

        # Kandidaten-Labels: Label + Pref/Alt + OBO-Synonyme + Titel
        {{
            ?uri rdfs:label ?candidateLabel .
        }} UNION {{
            ?uri skos:prefLabel ?candidateLabel .
        }} UNION {{
            ?uri skos:altLabel ?candidateLabel .
        }} UNION {{
            ?uri dc:title ?candidateLabel .
        }} UNION {{
            ?uri oboInOwl:hasExactSynonym ?candidateLabel .
        }} UNION {{
            ?uri oboInOwl:hasRelatedSynonym ?candidateLabel .
        }}

        # Case-insensitive substring-Match, kein Regex
        FILTER(CONTAINS(LCASE(STR(?candidateLabel)), LCASE("{pattern}")))

        # Anzeigename: bevorzugt rdfs:label / skos:prefLabel
        OPTIONAL {{
            ?uri rdfs:label ?mainLabel .
        }}
        OPTIONAL {{
            ?uri skos:prefLabel ?prefLabel .
        }}
        BIND(COALESCE(?mainLabel, ?prefLabel, ?candidateLabel) AS ?displayLabel)

        # Optionale Beschreibung: rdfs:comment, skos:definition, IAO_0000115
        OPTIONAL {{ ?uri rdfs:comment ?desc1 . }}
        OPTIONAL {{ ?uri skos:definition ?desc2 . }}
        OPTIONAL {{ ?uri IAO:0000115 ?desc3 . }}
        BIND(COALESCE(?desc1, ?desc2, ?desc3) AS ?description)
    }}
    LIMIT {limit}
    """
    
    all_possible_matches = []
    try:
        qres = graph.query(query)
        logger.info(f"SPARQL query returned {len(qres)} total labels to check for fuzzy matching.")
        
        for row in qres:
            label_text = str(row.displayLabel)
            score = calculate_levenshtein_score(term, label_text)
            
            # Set a threshold for what is considered a "similar" term
            if score > 0.6: # Threshold can be adjusted
                all_possible_matches.append({
                    "uri": str(row.uri),
                    "label": label_text,
                    "description": str(row.description) if row.description else "",
                    "source_provider": "Local RDF File", # This will be overridden in reconciliation_ui.py
                    "ontology_title": str(row.ontologyTitle) if row.ontologyTitle else "", # New: for custom source provider
                    "levenshtein_score": score # Add score for sorting
                })

    except Exception as e:
        logger.error(f"Error executing SPARQL query or processing results for term '{term}': {e}", exc_info=True)

    # Sort by score in descending order
    sorted_matches = sorted(all_possible_matches, key=lambda x: x['levenshtein_score'], reverse=True)
    
    logger.info(f"Found {len(sorted_matches)} fuzzy matches for '{term}' above threshold. Returning top {limit}.")
    
    return sorted_matches[:limit]
