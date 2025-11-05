import logging
import json
from pathlib import Path
from xml.etree import ElementTree as ET

def extract_model_id_from_sbml(sbml_path):
    """
    Extracts the model ID from an SBML file, checking multiple namespaces.
    """
    if not sbml_path.exists():
        logging.warning(f"SBML file not found at {sbml_path}")
        return None
    try:
        tree = ET.parse(sbml_path)
        root = tree.getroot()
        namespaces = [
            'http://www.sbml.org/sbml/level3/version2/core',
            'http://www.sbml.org/sbml/level3/version1/core',
            'http://www.sbml.org/sbml/level2/version4/core',
            'http://www.sbml.org/sbml/level2/version1/core',
            'http://www.sbml.org/sbml/level1/version2/core',
            ''  # No namespace
        ]
        for ns in namespaces:
            model_element = root.find(f".//{{{ns}}}model")
            if model_element is not None:
                return model_element.get('id')
    except ET.ParseError:
        return None
    return None

def find_nearest_metadata_fixed(sbml_file, metadata_files, base_path, component_map):
    """
    Finds the nearest metadata.json file to an SBML file by traversing up the directory tree.
    """
    sbml_path_resolved = resolve_path_with_component_map(sbml_file, base_path, component_map)
    if not sbml_path_resolved.exists():
        logging.warning(f"SBML not found: {sbml_file}")
        return None

    current_dir = sbml_path_resolved.parent
    
    while current_dir != current_dir.parent:
        for meta_file in metadata_files:
            meta_path_resolved = resolve_path_with_component_map(meta_file, base_path, component_map)
            if meta_path_resolved.exists() and meta_path_resolved.parent == current_dir:
                return meta_file
        current_dir = current_dir.parent
        
    return None

def resolve_path_with_component_map(original_path, base_path, component_map):
    """
    Resolves an original path from the manifest to its flattened path on disk.
    """
    parts = str(original_path).replace('\\', '/').split('/')
    resolved_path = Path(base_path)
    for part in parts:
        if part in component_map:
            resolved_path = resolved_path / component_map[part]
        else:
            resolved_path = resolved_path / part
    return resolved_path

def load_all_metadata_robust(composite_model, manifest_files, base_path, component_map, verbose=False):
    """
    Drop-in replacement for load_all_metadata that uses the fixed path resolution.
    """
    logging.info("Loading metadata for all sub-models using fixed path resolution...")

    sbml_files = manifest_files.get('sbml', [])
    metadata_files = manifest_files.get('metadata', [])

    for sbml_file in sbml_files:
        if verbose:
            logging.info(f"Processing SBML file: {sbml_file}")

        resolved_sbml_path = resolve_path_with_component_map(sbml_file, base_path, component_map)
        if not resolved_sbml_path.exists():
            if verbose:
                logging.warning(f"  - SBML file does not exist at resolved path: {resolved_sbml_path}")
            continue

        model_id = extract_model_id_from_sbml(resolved_sbml_path)
        if not model_id:
            if verbose:
                logging.warning(f"  - Could not extract model ID from {sbml_file}")
            continue
        
        if verbose:
            logging.info(f"  - Extracted model ID: {model_id}")

        if model_id not in composite_model.models:
            if verbose:
                logging.warning(f"  - Model ID '{model_id}' not found in composite_model.models")
            continue

        best_match = find_nearest_metadata_fixed(sbml_file, metadata_files, base_path, component_map)
        
        if best_match:
            if verbose:
                logging.info(f"  - Found nearest metadata file: {best_match}")
            try:
                meta_path_abs = resolve_path_with_component_map(best_match, base_path, component_map)
                metadata = json.loads(meta_path_abs.read_text(encoding='utf-8'))
                
                composite_model.models[model_id].metadata = metadata
                logging.info(f"  - Loaded metadata for '{model_id}' from {best_match}")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.warning(f"Could not load metadata from {best_match}: {e}")
        else:
            if verbose:
                logging.warning(f"  - No metadata file found for model with SBML file '{sbml_file}'")
