#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vocabulary Mapper for RAKIP LD Extractor
Handles mapping of JSON objects (units, classifications, etc.) to ontology IRIs
Provides caching functionality for FSK models to avoid repeated downloads
"""

import json
import os
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any, List
from xml.etree import ElementTree as ET
import re
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime, timedelta

# ========== CONFIGURATION ==========
CACHE_DIR = "./model_cache"
CACHE_EXPIRY_DAYS = 7  # Cache models for 7 days
MAPPING_CACHE_FILE = "./vocab_mappings.pkl"

# Vocabulary types to map
VOCAB_TYPES = {
    "units": ["unit", "units", "unitCategory"],
    "classifications": ["classification", "modelClass", "modelSubClass"],
    "parameters": ["parameterType", "parameterClassification"],
    "organisms": ["hazard", "product", "populationGroup"],
    "general": ["language", "software", "format", "publicationType"]
}


class VocabularyMapper:
    """Maps vocabulary terms to ontology IRIs"""

    def __init__(self, ontology_file: str):
        self.ontology_file = ontology_file
        self.classes = {}
        self.properties = {}
        self.individuals = {}
        self.synonyms = {}
        self.custom_mappings = {}
        self._load_ontology()
        self._load_custom_mappings()

    def _normalize_text(self, s: str) -> str:
        """Normalize text for comparison"""
        if not isinstance(s, str):
            s = str(s) if s is not None else ""
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    def _load_ontology(self):
        """Load ontology mappings from OWL file"""
        if not os.path.exists(self.ontology_file):
            print(f"Warning: Ontology file {self.ontology_file} not found.")
            return

        try:
            tree = ET.parse(self.ontology_file)
            root = tree.getroot()

            ns = {
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
                'owl': 'http://www.w3.org/2002/07/owl#',
                'oboInOwl': 'http://www.geneontology.org/formats/oboInOwl#',
                'skos': 'http://www.w3.org/2004/02/skos/core#',
            }

            def _labels_for(elem):
                labels = []
                for tag in ('rdfs:label', 'skos:prefLabel'):
                    for lab in elem.findall(tag, ns):
                        t = (lab.text or '').strip()
                        if t:
                            labels.append(t)
                return labels

            def _add(elem, store_dict):
                iri = elem.get('{%s}about' % ns['rdf'])
                if not iri:
                    return
                labels = _labels_for(elem)
                if not labels:
                    return

                for label in labels:
                    key = self._normalize_text(label)
                    if key:
                        store_dict[key] = iri

                syn_preds = [
                    'oboInOwl:hasExactSynonym',
                    'oboInOwl:hasRelatedSynonym',
                    'oboInOwl:hasBroadSynonym',
                    'oboInOwl:hasNarrowSynonym',
                    'oboInOwl:hasSynonym',
                ]
                canonical = self._normalize_text(labels[0])
                for sp in syn_preds:
                    for syn_elem in elem.findall(sp, ns):
                        syn_text = (syn_elem.text or '').strip()
                        if syn_text:
                            self.synonyms[self._normalize_text(syn_text)] = canonical

            for c in root.findall('.//owl:Class', ns):
                _add(c, self.classes)
            for p in root.findall('.//owl:ObjectProperty', ns):
                _add(p, self.properties)
            for ind in root.findall('.//owl:NamedIndividual', ns):
                _add(ind, self.individuals)

            print(f"[VocabMapper] Loaded {len(self.classes)} classes, "
                  f"{len(self.properties)} properties, {len(self.individuals)} individuals, "
                  f"{len(self.synonyms)} synonyms from ontology")

        except Exception as e:
            print(f"[VocabMapper] Error loading ontology: {e}")

    def _load_custom_mappings(self):
        """Load custom vocabulary mappings from cache file"""
        if os.path.exists(MAPPING_CACHE_FILE):
            try:
                with open(MAPPING_CACHE_FILE, 'rb') as f:
                    self.custom_mappings = pickle.load(f)
                print(f"[VocabMapper] Loaded {len(self.custom_mappings)} custom mappings from cache")
            except Exception as e:
                print(f"[VocabMapper] Error loading custom mappings: {e}")

    def save_custom_mappings(self):
        """Save custom mappings to cache file"""
        try:
            with open(MAPPING_CACHE_FILE, 'wb') as f:
                pickle.dump(self.custom_mappings, f)
            print(f"[VocabMapper] Saved {len(self.custom_mappings)} custom mappings to cache")
        except Exception as e:
            print(f"[VocabMapper] Error saving custom mappings: {e}")

    def find_iri(self, text: str, field_type: Optional[str] = None,
                 vocab_category: Optional[str] = None) -> Optional[str]:
        """
        Find ontology IRI for a given text

        Args:
            text: The text to map
            field_type: Type of field (e.g., 'hazard', 'unit', 'classification')
            vocab_category: Broader category (e.g., 'organisms', 'units')

        Returns:
            IRI if found, None otherwise
        """
        if not text:
            return None

        q = self._normalize_text(text)

        # Check custom mappings first
        cache_key = f"{vocab_category}:{field_type}:{q}" if vocab_category else f"{field_type}:{q}"
        if cache_key in self.custom_mappings:
            return self.custom_mappings[cache_key]

        # Determine search order based on field type
        if field_type in {"hazard", "product", "populationGroup", "modelClass"}:
            spaces = [self.classes, self.individuals]
        else:
            spaces = [self.individuals, self.classes]

        # Direct match
        for space in spaces:
            iri = space.get(q)
            if iri:
                self.custom_mappings[cache_key] = iri
                return iri

        # Synonym match
        canonical = self.synonyms.get(q)
        if canonical:
            for space in spaces:
                iri = space.get(canonical)
                if iri:
                    self.custom_mappings[cache_key] = iri
                    return iri

        # Substring match
        for space in spaces:
            for label, iri in space.items():
                if q in label or label in q:
                    self.custom_mappings[cache_key] = iri
                    return iri

        # Fuzzy match (threshold: 0.90)
        best_iri, best_score = None, 0.0
        for space in spaces:
            for label, iri in space.items():
                s = SequenceMatcher(None, q, label).ratio()
                if s > best_score:
                    best_iri, best_score = iri, s

        if best_score >= 0.90:
            self.custom_mappings[cache_key] = best_iri
            return best_iri

        return None

    def map_object(self, obj: Any, parent_field: str = "") -> Any:
        """
        Recursively map vocabulary terms in an object to IRIs

        Args:
            obj: Object to map (dict, list, or primitive)
            parent_field: Parent field name for context

        Returns:
            Mapped object with IRIs where applicable
        """
        if isinstance(obj, dict):
            mapped = {}
            for key, value in obj.items():
                # Skip already mapped fields
                if key in ["@id", "@type"]:
                    mapped[key] = value
                    continue

                # Determine field type and category
                field_type = key
                vocab_category = None
                for cat, fields in VOCAB_TYPES.items():
                    if key in fields or parent_field in fields:
                        vocab_category = cat
                        break

                # Map string values
                if isinstance(value, str) and vocab_category:
                    iri = self.find_iri(value, field_type=field_type,
                                       vocab_category=vocab_category)
                    if iri:
                        # Store both IRI and original value
                        mapped[key] = {
                            "@id": iri,
                            "label": value
                        }
                        continue

                # Recursively map nested objects
                mapped[key] = self.map_object(value, parent_field=key)

            return mapped

        elif isinstance(obj, list):
            return [self.map_object(item, parent_field=parent_field) for item in obj]

        else:
            return obj

    def add_custom_mapping(self, text: str, iri: str, field_type: str = "",
                          vocab_category: str = ""):
        """Add a custom vocabulary mapping"""
        q = self._normalize_text(text)
        cache_key = f"{vocab_category}:{field_type}:{q}" if vocab_category else f"{field_type}:{q}"
        self.custom_mappings[cache_key] = iri
        print(f"[VocabMapper] Added custom mapping: '{text}' -> {iri}")


class ModelCache:
    """Cache for FSK models to avoid repeated downloads"""

    def __init__(self, cache_dir: str = CACHE_DIR, expiry_days: int = CACHE_EXPIRY_DAYS):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiry_days = expiry_days
        self.index_file = self.cache_dir / "cache_index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        """Load cache index"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ModelCache] Error loading index: {e}")
        return {}

    def _save_index(self):
        """Save cache index"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            print(f"[ModelCache] Error saving index: {e}")

    def _get_cache_path(self, model_id: str) -> Path:
        """Get cache file path for a model"""
        safe_id = re.sub(r'[^\w\-]', '_', model_id)
        return self.cache_dir / f"{safe_id}.json"

    def _is_expired(self, cached_date: str) -> bool:
        """Check if cache entry is expired"""
        try:
            cached = datetime.fromisoformat(cached_date)
            expiry = timedelta(days=self.expiry_days)
            return datetime.now() - cached > expiry
        except Exception:
            return True

    def get(self, model_id: str) -> Optional[Dict]:
        """
        Get model from cache

        Args:
            model_id: Model identifier

        Returns:
            Model data if cached and not expired, None otherwise
        """
        if model_id not in self.index:
            return None

        entry = self.index[model_id]

        # Check expiry
        if self._is_expired(entry.get('cached_at', '')):
            print(f"[ModelCache] Cache expired for model {model_id}")
            self.delete(model_id)
            return None

        # Load from file
        cache_path = self._get_cache_path(model_id)
        if not cache_path.exists():
            print(f"[ModelCache] Cache file missing for model {model_id}")
            del self.index[model_id]
            self._save_index()
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                print(f"[ModelCache] Retrieved model {model_id} from cache")
                return json.load(f)
        except Exception as e:
            print(f"[ModelCache] Error reading cache for {model_id}: {e}")
            return None

    def set(self, model_id: str, data: Dict):
        """
        Store model in cache

        Args:
            model_id: Model identifier
            data: Model data
        """
        cache_path = self._get_cache_path(model_id)

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Update index
            self.index[model_id] = {
                'cached_at': datetime.now().isoformat(),
                'file': str(cache_path)
            }
            self._save_index()

            print(f"[ModelCache] Cached model {model_id}")

        except Exception as e:
            print(f"[ModelCache] Error caching model {model_id}: {e}")

    def delete(self, model_id: str):
        """Delete model from cache"""
        if model_id in self.index:
            cache_path = self._get_cache_path(model_id)
            if cache_path.exists():
                cache_path.unlink()
            del self.index[model_id]
            self._save_index()
            print(f"[ModelCache] Deleted cache for model {model_id}")

    def clear(self):
        """Clear entire cache"""
        for model_id in list(self.index.keys()):
            self.delete(model_id)
        print("[ModelCache] Cleared all cache")

    def cleanup_expired(self):
        """Remove expired cache entries"""
        expired = []
        for model_id, entry in self.index.items():
            if self._is_expired(entry.get('cached_at', '')):
                expired.append(model_id)

        for model_id in expired:
            self.delete(model_id)

        if expired:
            print(f"[ModelCache] Cleaned up {len(expired)} expired entries")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = len(self.index)
        expired = sum(1 for entry in self.index.values()
                     if self._is_expired(entry.get('cached_at', '')))

        return {
            'total_entries': total,
            'active_entries': total - expired,
            'expired_entries': expired,
            'cache_dir': str(self.cache_dir),
            'expiry_days': self.expiry_days
        }


# ========== UTILITY FUNCTIONS ==========

def extract_vocab_fields(data: Dict, vocab_category: str = None) -> List[str]:
    """
    Extract all vocabulary fields from a data structure

    Args:
        data: Data to extract from
        vocab_category: Optional category to filter by

    Returns:
        List of unique field values
    """
    fields = set()

    def _extract(obj, parent_key=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                # Check if this field is a vocabulary field
                is_vocab = False
                if vocab_category:
                    if key in VOCAB_TYPES.get(vocab_category, []):
                        is_vocab = True
                else:
                    for cat_fields in VOCAB_TYPES.values():
                        if key in cat_fields:
                            is_vocab = True
                            break

                if is_vocab and isinstance(value, str):
                    fields.add(value)

                _extract(value, key)

        elif isinstance(obj, list):
            for item in obj:
                _extract(item, parent_key)

    _extract(data)
    return sorted(fields)


if __name__ == '__main__':
    # Example usage
    ontology_file = "./fskxo.owl"

    # Initialize mapper
    mapper = VocabularyMapper(ontology_file)

    # Initialize cache
    cache = ModelCache()

    # Print statistics
    print("\nCache Statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Example: Map some vocabulary terms
    test_terms = [
        "Listeria monocytogenes",
        "Dose-response model",
        "celsius",
        "log",
        "CFU/g"
    ]

    print("\nTest Mappings:")
    for term in test_terms:
        iri = mapper.find_iri(term)
        if iri:
            print(f"  ✓ '{term}' -> {iri}")
        else:
            print(f"  ✗ '{term}' -> NOT FOUND")

    # Save any new mappings
    mapper.save_custom_mappings()
