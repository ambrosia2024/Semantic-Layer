import argparse
import json
import pandas as pd
from pathlib import Path
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, SKOS, RDFS
import difflib
import sys
import os

# Global log file path
LOG_FILE = Path(__file__).parent / "inference_debug.log"

def log(msg):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")
    except Exception:
        pass # Fallback if logging fails

# Namespaces
FSKXO = Namespace("http://semanticlookup.zbmed.de/km/fskxo/")
SCHEMA = Namespace("https://schema.org/")
DCTERMS = Namespace("http://purl.org/dc/terms/")

def load_master_mapping(file_path):
    try:
        log(f"Loading master mapping from {file_path}...")
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        log(f"Loaded {len(df)} rows from master mapping.")
        return df
    except Exception as e:
        log(f"Error loading master mapping: {e}")
        return pd.DataFrame()

def get_candidates(df, concept_group):
    # Filter for ConceptGroup
    filtered_df = df[df['ConceptGroup'] == concept_group].copy()
    
    candidates = []
    for _, row in filtered_df.iterrows():
        term = str(row['Term']).strip()
        alt_labels = str(row['altLabels']).strip() if pd.notna(row['altLabels']) else ""
        
        # Display label construction similar to pipeline_app
        display_label = f"{term} ({alt_labels})" if alt_labels else term
        
        candidates.append({
            'term': term,
            'altLabels': alt_labels,
            'display_label': display_label,
            'row': row
        })
    return candidates

def fuzzy_match_term(term_str, candidates, threshold=0.7, is_unit=False):
    if not term_str:
        return None
    
    term_str = term_str.strip()
    term_str_lower = term_str.lower()

    # Special case: Dimensionless for units
    if is_unit and term_str == "[]":
        # 1. Try to find exact match for "[]" in candidates
        for cand in candidates:
            if cand['term'] == "[]":
                return cand['display_label']
        
        # 2. Try to find "Dimensionless" or "Unitless" or "1"
        dimensionless_keywords = ["dimensionless", "unitless", "1"]
        for cand in candidates:
            c_term = cand['term'].lower()
            c_alts = cand['altLabels'].lower() if cand['altLabels'] else ""
            
            for kw in dimensionless_keywords:
                if kw == c_term or kw in c_alts.split(','):
                    return cand['display_label']
                    
        # 3. Return "[]" as fallback if no mapping found, allowing user to see it
        return "[]"

    best_match = None
    best_score = -1

    for cand in candidates:
        # Gather target terms for matching
        target_terms = [cand['term'].lower()]
        if cand['altLabels']:
            target_terms.extend([al.strip().lower() for al in cand['altLabels'].split(',')])
        
        score = 0
        for t in target_terms:
            if not t: continue
            
            # Exact match
            if term_str_lower == t:
                score = 1.0
            # Containment
            elif t in term_str_lower or term_str_lower in t:
                score = max(score, 0.5)
            # Fuzzy
            else:
                ratio = difflib.SequenceMatcher(None, term_str_lower, t).ratio()
                score = max(score, ratio)
        
        if score > best_score:
            best_score = score
            best_match = cand['display_label']

    if best_score >= threshold:
        return best_match
    
    return None

def keyword_match_term(term_str, candidates, verbose=False):
    """
    Performs single word matchings for parameters.
    If the parameter has a word (e.g. "time") and a candidate also carries "time" 
    in "Term" or "Altlabels", then this is a match.
    """
    if not term_str:
        return None
    
    if verbose: log(f"  Matching term: '{term_str}'")

    # Normalize and tokenize parameter name
    # Remove common special chars and split
    term_str_clean = str(term_str).lower()
    for char in ['_', '-', '(', ')', '[', ']', '/', ',', '.']:
        term_str_clean = term_str_clean.replace(char, ' ')
    
    # Tokenize and filter short words/common stop words to avoid noisy matches
    param_tokens = set([t for t in term_str_clean.split() if len(t) > 2]) # filtering tokens <= 2 chars
    
    if not param_tokens:
        if verbose: log(f"  No valid tokens found in '{term_str}'")
        return None

    if verbose: log(f"  Tokens: {param_tokens}")

    best_match = None
    max_overlap = 0

    for cand in candidates:
        # Gather target terms for matching
        target_strings = [cand['term'].lower()]
        if cand['altLabels']:
            target_strings.extend([al.strip().lower() for al in cand['altLabels'].split(',')])
        
        cand_tokens = set()
        for s in target_strings:
            for char in ['_', '-', '(', ')', '[', ']', '/', ',', '.']:
                s = s.replace(char, ' ')
            cand_tokens.update([t for t in s.split() if len(t) > 2])
        
        # Check intersection or partial match
        score = 0
        match_details = []
        for pt in param_tokens:
            for ct in cand_tokens:
                # Exact token match
                if pt == ct:
                    score += 1.0
                    match_details.append(f"{pt}=={ct}")
                # Prefix match (e.g. 'temp' in 'temperature') - strict enough to avoid noise
                elif (pt in ct or ct in pt) and min(len(pt), len(ct)) >= 3:
                    score += 0.8
                    match_details.append(f"{pt}~{ct}")
        
        if score > 0 and verbose and score >= max_overlap:
             log(f"    Candidate '{cand['display_label']}' score: {score} ({', '.join(match_details)})")

        if score > max_overlap:
            max_overlap = score
            best_match = cand['display_label']
            # Tie-breaking logic could go here (e.g. prefer shorter terms, or exact matches)

    if best_match and verbose:
        log(f"  -> Best match: '{best_match}' (Score: {max_overlap})")
    
    return best_match

def process_turtle_files(input_dirs, unit_candidates, input_candidates, output_candidates):
    results = {}
    
    files_to_process = []
    for d in input_dirs:
        p = Path(d)
        if p.is_dir():
            files_to_process.extend(list(p.glob("*.ttl")))
        elif p.is_file() and p.suffix == '.ttl':
            files_to_process.append(p)
        else:
            log(f"Input path not found or invalid: {p}")

    for ttl_file in files_to_process:
        try:
            log(f"Processing {ttl_file.name}...")
            g = Graph()
            g.parse(str(ttl_file), format="turtle")
            
            model_stem = ttl_file.stem
            results[model_stem] = {}
            
            # Query for parameters, their units, and classification
            q = """
                SELECT ?param ?id ?unitText ?unitLabel ?name ?classification ?classRef WHERE {
                    ?model fskxo:FSKXO_0000000016 / fskxo:FSKXO_0000000017 ?param .
                    OPTIONAL { ?param dcterms:identifier ?id . }
                    OPTIONAL { ?param schema:unitText ?unitText . }
                    OPTIONAL { ?param schema:unit_label ?unitLabel . }
                    OPTIONAL { ?param schema:name ?name . }
                    OPTIONAL { ?param fskxo:FSKXO_0000000039 ?classification . }
                    OPTIONAL { ?param fskxo:FSKXO_0000017519 ?classRef . }
                }
            """
            
            # Aggregate results by parameter ID first
            params_data = {}
            for row in g.query(q, initNs={"fskxo": FSKXO, "dcterms": DCTERMS, "schema": SCHEMA}):
                param_id = str(row.id) if row.id else None
                if not param_id: continue
                
                if param_id not in params_data:
                    params_data[param_id] = {
                        'id': param_id,
                        'unit_str': str(row.unitText) if row.unitText else (str(row.unitLabel) if row.unitLabel else None),
                        'name': str(row.name) if row.name else None,
                        'classifications': set()
                    }
                
                # Add classifications from this row
                if row.classification:
                    params_data[param_id]['classifications'].add(str(row.classification).upper())
                if row.classRef:
                    params_data[param_id]['classifications'].add(str(row.classRef))

            # Process aggregated parameters
            for param_id, p_data in params_data.items():
                log(f"  [Param: {param_id}]")
                result_entry = {}

                # Unit matching
                unit_str = p_data['unit_str']
                if unit_str:
                    # User requested threshold 0.7 for units
                    mapped_unit = fuzzy_match_term(unit_str, unit_candidates, threshold=0.7, is_unit=True) 
                    if mapped_unit:
                        result_entry['original_unit'] = unit_str
                        result_entry['mapped_unit_term'] = mapped_unit
                        log(f"    Unit matched: '{unit_str}' -> '{mapped_unit}'")
                    else:
                        log(f"    Unit NOT matched: '{unit_str}'")

                # Parameter name matching
                param_name = p_data['name']
                
                # Determine classification
                is_input = False
                is_output = False
                
                for cls in p_data['classifications']:
                    if "INPUT" in cls or "FSKXO_0000017481" in cls: is_input = True
                    if "OUTPUT" in cls or "FSKXO_0000017482" in cls: is_output = True
                
                log(f"    Classification: {'Input' if is_input else ''} {'Output' if is_output else ''} (Raw: {p_data['classifications']})")

                candidates_list = []
                if is_input:
                    candidates_list = input_candidates
                elif is_output:
                    candidates_list = output_candidates
                
                if candidates_list:
                    mapped_param = None
                    # Try matching with param_name first, then param_id
                    terms_to_try = []
                    if param_name: terms_to_try.append(param_name)
                    if param_id: terms_to_try.append(param_id)
                    
                    for term in terms_to_try:
                        match = keyword_match_term(term, candidates_list, verbose=True)
                        if match:
                            mapped_param = match
                            result_entry['original_name'] = term
                            result_entry['mapped_parameter_term'] = mapped_param
                            break
                    
                    if mapped_param:
                        log(f"    Parameter matched: '{result_entry['original_name']}' -> '{mapped_param}'")
                    else:
                        log(f"    Parameter NOT matched for {terms_to_try}")
                else:
                    log(f"    Skipping parameter matching (not Input/Output or no candidates)")

                if result_entry:
                    results[model_stem][param_id] = result_entry
                        
        except Exception as e:
            log(f"Error processing {ttl_file.name}: {e}")
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Infer units from FSKX turtle files.")
    parser.add_argument("--input-dirs", required=True, nargs='+', help="Directories or files containing turtle files to process")
    parser.add_argument("--master-mapping", required=True, help="Path to master_mapping.xlsx")
    parser.add_argument("--out", required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    df = load_master_mapping(args.master_mapping)
    if df.empty:
        return
        
    unit_candidates = get_candidates(df, 'Unit')
    input_candidates = get_candidates(df, 'InputParameter')
    output_candidates = get_candidates(df, 'OutputParameter')
    
    inferred_data = process_turtle_files(args.input_dirs, unit_candidates, input_candidates, output_candidates)
    
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(inferred_data, f, indent=2)
        
    log(f"Inferred units saved to {args.out}")
    print(f"Inferred units saved to {args.out}")

if __name__ == "__main__":
    # Clear previous log
    if LOG_FILE.exists():
        try:
            os.remove(LOG_FILE)
        except: pass
    
    log("Starting inference...")
    main()
