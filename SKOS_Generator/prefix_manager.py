from collections import defaultdict
import re
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_prefixes(file_path):
    """
    Extracts prefixes from a file, allowing for multiple prefixes to be associated with the same namespace URI.
    """
    prefixes = defaultdict(list)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prefix_regex = re.compile(r'^\s*(?:@prefix|PREFIX)\s+([\w\-]+):\s+<([^>]+)>', re.IGNORECASE)
            for line in f:
                match = prefix_regex.match(line)
                if match:
                    prefix, namespace = match.groups()
                    namespace = namespace.strip()
                    if prefix not in prefixes[namespace]:
                        prefixes[namespace].append(prefix)
    except FileNotFoundError:
        pass # Silently fail if the file doesn't exist
    except Exception:
        pass # Silently fail on other errors
    return dict(prefixes)

def find_relevant_prefixes(all_prefixes, full_uris):
    """
    Finds the best prefix for each URI by finding the longest matching namespace
    from the prefix file and then binding that prefix to the common base of the
    data URIs.
    """
    final_bindings = {}
    
    # Sort namespaces by length, descending, to ensure longest match is found first.
    sorted_ns = sorted(all_prefixes.items(), key=lambda item: len(item[0]), reverse=True)

    handled_uris = set()

    for file_ns, prefixes in sorted_ns:
        # Find all data URIs that match this namespace and haven't been handled yet
        matching_uris = [u for u in full_uris if u.startswith(file_ns) and u not in handled_uris]
        
        if not matching_uris:
            continue
        
        logging.info(f"Processing namespace: {file_ns} with potential prefixes: {prefixes}")
        logging.info(f"  > Found {len(matching_uris)} matching URIs.")
            
        # Mark these URIs as handled so they aren't processed by a shorter, less specific namespace
        for u in matching_uris:
            handled_uris.add(u)
            
        # For this group of URIs, find the best prefix
        chosen_prefix = None
        if len(prefixes) == 1:
            chosen_prefix = prefixes[0]
        else:
            # AMBIGUITY RESOLUTION (e.g., obo vs mondo for the same base)
            prefix_scores = defaultdict(int)
            for uri in matching_uris:
                local_id = uri[len(file_ns):]
                for p in prefixes:
                    if local_id.lower().startswith(p.lower()):
                        prefix_scores[p] += 1
            
            if prefix_scores:
                # Case 1: Direct match found (e.g., URI local part starts with 'foodon').
                # Choose the one with the highest score, using length as a tie-breaker.
                chosen_prefix = max(prefix_scores, key=lambda p: (prefix_scores[p], len(p)))
                logging.info(f"  > Direct match logic selected '{chosen_prefix}' based on scores: {dict(prefix_scores)}")
            else:
                # Case 2: No direct match. Check for aliases in a more constrained way.
                # An alias is only considered if it's part of the domain name, which is a safer heuristic.
                try:
                    domain = re.search(r'https?://([^/]+)', file_ns).group(1)
                    alias_candidates = [p for p in prefixes if p.lower() in domain.lower()]
                    if alias_candidates:
                        chosen_prefix = max(alias_candidates, key=len)
                        logging.info(f"  > No direct match. Domain-based alias logic selected '{chosen_prefix}' from candidates: {alias_candidates}")
                    else:
                        chosen_prefix = None
                        logging.info(f"  > No direct match and no domain-based alias found. Skipping prefix assignment.")
                except Exception:
                    chosen_prefix = None
                    logging.info(f"  > Could not parse domain for alias check. Skipping prefix assignment.")

        # If a prefix could not be determined, skip this namespace group.
        if not chosen_prefix:
            # Un-handle the URIs so they can be processed by another (less specific) namespace if one exists.
            for u in matching_uris:
                if u in handled_uris:
                    handled_uris.remove(u)
            continue
        
        # Determine the namespace to bind. If multiple URIs share a more specific
        # common path than the file_ns, use that. Otherwise, use file_ns.
        if len(matching_uris) > 1:
            common_base = os.path.commonprefix(matching_uris)
            # Only use the common_base if it's more specific than the file_ns
            if len(common_base) > len(file_ns):
                last_sep = max(common_base.rfind('/'), common_base.rfind('#'))
                # Ensure separator is found and is after the file_ns part
                if last_sep > len(file_ns) - 2:
                    final_namespace = common_base[:last_sep + 1]
                else:
                    final_namespace = file_ns # Fallback
            else:
                final_namespace = file_ns
        else:
            # With only one URI, the namespace from the prefix file is the correct one.
            final_namespace = file_ns

        if chosen_prefix and chosen_prefix not in final_bindings:
            final_bindings[chosen_prefix] = final_namespace
            logging.info(f"  > SUCCESS: Binding prefix '{chosen_prefix}' to namespace '{final_namespace}'")

    return final_bindings


def add_prefixes_to_rdf(rdf_content, relevant_prefixes):
    prefix_string = ""
    for prefix, namespace in relevant_prefixes.items():
        prefix_string += f"@prefix {prefix}: <{namespace}>.\\n"
    
    return prefix_string + rdf_content

if __name__ == '__main__':
    # Example Usage for testing
    file_path = "Prefixes/all.file.sparql.txt"
    prefixes = extract_prefixes(file_path)
    if prefixes:
        matching_table_uris = [
            "http://purl.obolibrary.org/obo/FOODON_123456",
            "http://purl.obolibrary.org/obo/NCIT_C175890",
            "http://snomed.info/id/361234006",
            "https://loinc.org/rdf/67413-5",
            "http://purl.obolibrary.org/obo/FBcv_0003024"
        ]
        relevant_prefixes = find_relevant_prefixes(prefixes, matching_table_uris)
        
        rdf_content = "<rdf:RDF>\\n  <rdf:Description rdf:about=\"http://example.com/resource\">\\n    <dc:title>Example Resource</dc:title>\\n  </rdf:Description>\\n</rdf:RDF>"
        
        updated_rdf_content = add_prefixes_to_rdf(rdf_content, relevant_prefixes)
        print("--- Generated RDF ---")
        print(updated_rdf_content)
