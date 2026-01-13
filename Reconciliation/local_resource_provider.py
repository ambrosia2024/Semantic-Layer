# -*- coding: utf-8 -*-
"""
local_resource_provider.py

A provider that lets the reconciliation app search *user-uploaded* local resources:
- OWL / OBO / RDF / Turtle / JSON-LD / OBO-JSON (best-effort)
- SKOS thesauri (best-effort)
- CSV/TSV/XLSX tables (heuristic column detection)

Design goals:
- Robust: fail loudly with actionable errors when parsing/indexing is impossible.
- Consistent output: returns the same suggestion dict schema as the remote providers.
- Fast enough interactively: parse+index ONCE per uploaded file, then query via a lightweight token index.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import os
import re
import tempfile
import sqlite3
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Optional dependencies (kept optional so the whole app doesn't crash if missing)
try:
    import Levenshtein  # python-Levenshtein
except Exception:  # pragma: no cover
    Levenshtein = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import rdflib
    from rdflib.namespace import RDF, RDFS, SKOS, DCTERMS, DC
except Exception:  # pragma: no cover
    rdflib = None
    RDF = RDFS = SKOS = DCTERMS = DC = None

try:
    # Oaklib (Ontology Access Kit)
    from oaklib import get_adapter  # type: ignore
except Exception:  # pragma: no cover
    get_adapter = None


# -----------------------------
# Exceptions
# -----------------------------

class LocalResourceError(RuntimeError):
    """Error with context that should be shown to the user (message must be readable)."""

    def __init__(self, message: str, *, hint: str = "", details: str = ""):
        super().__init__(message)
        self.hint = hint
        self.details = details

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)


# -----------------------------
# Data structures
# -----------------------------

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str) -> List[str]:
    # Keep alphanumerics, split on non-word boundaries, ignore tiny tokens
    s = _norm(s)
    toks = re.split(r"[^\w]+", s, flags=re.UNICODE)
    return [t for t in toks if len(t) >= 3]


def _lev_sim(a: str, b: str) -> float:
    """
    Similarity in [0,1]. Uses python-Levenshtein if available, otherwise a cheap fallback.
    """
    a_n, b_n = _norm(a), _norm(b)
    if not a_n or not b_n:
        return 0.0
    if a_n == b_n:
        return 1.0

    if Levenshtein is not None:
        dist = Levenshtein.distance(a_n, b_n)
        denom = max(len(a_n), len(b_n))
        return 1.0 - (dist / denom if denom else 0.0)

    # Fallback: Jaccard over tokens
    a_t, b_t = set(_tokenize(a_n)), set(_tokenize(b_n))
    if not a_t or not b_t:
        return 0.0
    return len(a_t & b_t) / len(a_t | b_t)


@dataclass
class LocalEntity:
    uri: str
    label: str
    description: str = ""
    synonyms: List[str] = field(default_factory=list)


@dataclass
class LocalIndex:
    """
    In-memory search index for one uploaded local resource.
    """
    resource_name: str
    resource_path: str
    resource_hash: str
    parse_backend: str  # "oak" | "rdflib" | "tabular"
    entities: List[LocalEntity]
    ontology_title: str = ""
    token_to_entity_ids: Dict[str, List[int]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def build_token_index(self) -> None:
        tok_map: Dict[str, List[int]] = {}
        for i, ent in enumerate(self.entities):
            texts = [ent.label] + (ent.synonyms or [])
            for txt in texts:
                for tok in _tokenize(txt):
                    tok_map.setdefault(tok, []).append(i)

        # Deduplicate lists while preserving order (cheap)
        for tok, ids in tok_map.items():
            seen = set()
            dedup = []
            for x in ids:
                if x not in seen:
                    seen.add(x)
                    dedup.append(x)
            tok_map[tok] = dedup

        self.token_to_entity_ids = tok_map


# Module-level cache: path/hash -> LocalIndex
_INDEX_CACHE: Dict[str, LocalIndex] = {}


# -----------------------------
# Loading / parsing
# -----------------------------

AUTO_OAK_OWL_MAX_MB = 2  # only small OWLs are allowed to use OAK in auto mode (optional)

RDF_EXTENSIONS = {"owl", "rdf", "ttl", "nt", "nq", "trig", "jsonld", "xml"}
OBO_EXTENSIONS = {"obo"}
TABULAR_EXTENSIONS = {"csv", "tsv", "xlsx", "xls"}

def _file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0

def _looks_like_obograph_json(path: Path) -> bool:
    # only read the first 64KB to avoid slowing down for large files
    try:
        with open(path, "rb") as f:
            head = f.read(65536)
        s = head.decode("utf-8", errors="ignore")
        return ('"graphs"' in s and '"nodes"' in s) or ('"obographs"' in s)
    except Exception:
        return False

def _auto_select_backend(path: Path, *, oak_available: bool) -> tuple[str, str]:
    ext = path.suffix.lower().lstrip(".")
    size_mb = _file_size_mb(path)

    if ext in TABULAR_EXTENSIONS:
        return "tabular", f"File extension .{ext} is tabular."

    if ext in OBO_EXTENSIONS:
        if oak_available:
            return "oak", "OBO format -> OAK adapter."
        return "rdflib", "OBO format but OAK not available; will likely fail without oaklib."

    if ext == "json" and _looks_like_obograph_json(path):
        if oak_available:
            return "oak", "Detected OBO Graph JSON -> OAK adapter."
        return "rdflib", "Detected OBO Graph JSON but OAK not available."

    if ext in RDF_EXTENSIONS:
        # OWL: nur wenn winzig, darf Auto zu OAK (optional)
        if ext == "owl" and oak_available and size_mb <= AUTO_OAK_OWL_MAX_MB:
            return "oak", f"Small OWL ({size_mb:.1f} MB) -> OAK for convenience APIs."
        return "rdflib", f"RDF/OWL serialization (.{ext}) -> rdflib extraction (size {size_mb:.1f} MB)."

    # unknown: fallback
    if oak_available:
        return "oak", f"Unknown extension .{ext}; trying OAK auto-detection."
    return "rdflib", f"Unknown extension .{ext}; trying rdflib."


RDF_LABEL_PREDICATES = [
    # common
    ("rdfs:label", lambda: RDFS.label if RDFS else None),
    ("skos:prefLabel", lambda: SKOS.prefLabel if SKOS else None),
    ("skos:altLabel", lambda: SKOS.altLabel if SKOS else None),
    # Dublin Core-ish
    ("dcterms:title", lambda: DCTERMS.title if DCTERMS else None),
    ("dc:title", lambda: DC.title if DC else None),
]

RDF_DESC_PREDICATES = [
    ("dcterms:description", lambda: DCTERMS.description if DCTERMS else None),
    ("dc:description", lambda: DC.description if DC else None),
    ("rdfs:comment", lambda: RDFS.comment if RDFS else None),
    ("skos:definition", lambda: SKOS.definition if SKOS else None),
    ("skos:note", lambda: SKOS.note if SKOS else None),
]


def _pick_literal(literals: Sequence[Any], prefer_langs: Sequence[str] = ("en", "")) -> str:
    """
    Pick the best literal:
    - prefer English if available
    - otherwise first literal
    """
    if not literals:
        return ""
    # rdflib Literals have .language and .value
    try:
        for lang in prefer_langs:
            for lit in literals:
                if getattr(lit, "language", None) == (lang or None):
                    return str(getattr(lit, "value", str(lit)))
    except Exception:
        pass
    return str(getattr(literals[0], "value", str(literals[0])))


def load_local_resource_index(
    resource_path: str,
    *,
    resource_name: Optional[str] = None,
    prefer_langs: Sequence[str] = ("en", ""),
    max_entities: Optional[int] = None,
    force_backend: str = "auto",  # "auto" | "oak" | "rdflib" | "tabular"
    progress_callback: Optional[callable] = None,
) -> LocalIndex:
    """
    Parse + index a local ontology/vocabulary file. Intended to be called once per upload.
    """
    p = Path(resource_path)
    if not p.exists():
        raise LocalResourceError(
            f"Local resource file not found: {resource_path}",
            hint="Ensure the upload was saved to disk and the path is correct."
        )

    resource_name = resource_name or p.name
    file_hash = _sha256_file(str(p))
    cache_key = f"{resource_name}:{file_hash}"

    if cache_key in _INDEX_CACHE:
        return _INDEX_CACHE[cache_key]

    backend = force_backend
    if backend == "auto":
        backend, reason = _auto_select_backend(p, oak_available=(get_adapter is not None))
        if progress_callback:
            try:
                progress_callback({"stage": "auto_backend", "backend": backend, "reason": reason})
            except Exception:
                pass

    if backend == "oak":
        if get_adapter is None:
            raise LocalResourceError(
                "OAK (oaklib) is not installed, but force_backend='oak' was requested.",
                hint="Install oaklib (and optional extras) or use force_backend='rdflib'/'tabular'."
            )
        idx = _load_via_oak(str(p), resource_name=resource_name, prefer_langs=prefer_langs, max_entities=max_entities, progress_callback=progress_callback)
    elif backend == "rdflib":
        idx = _load_via_rdflib(str(p), resource_name=resource_name, prefer_langs=prefer_langs, max_entities=max_entities, progress_callback=progress_callback)
    elif backend == "tabular":
        idx = _load_via_tabular(str(p), resource_name=resource_name, max_entities=max_entities, progress_callback=progress_callback)
    else:
        raise LocalResourceError(
            f"Unknown backend '{backend}'.",
            hint="Use one of: auto, oak, rdflib, tabular."
        )

    if not idx.entities:
        raise LocalResourceError(
            f"Parsed '{resource_name or p.name}' but extracted 0 entities.",
            hint="Try a different backend (oak vs rdflib) or provide CSV with label/uri columns."
        )

    idx.build_token_index()
    _INDEX_CACHE[cache_key] = idx
    return idx


def _load_via_oak(path: str, *, resource_name: str, prefer_langs: Sequence[str], max_entities: Optional[int], progress_callback: Optional[callable] = None) -> LocalIndex:
    if get_adapter is None:
        raise LocalResourceError("oaklib not installed", hint="Install oaklib or use rdflib backend.")

    try:
        adapter = get_adapter(path)
    except Exception as e:
        raise LocalResourceError(
            f"Failed to create OAK adapter for '{resource_name}'.",
            hint="Try force_backend='rdflib' for RDF/OWL files.",
            details=str(e)
        )

    entities = []
    warnings = []

    ent_iter = getattr(adapter, "entities", None)
    if not callable(ent_iter):
        raise LocalResourceError(
            "OAK adapter does not expose entities().",
            hint="Try force_backend='rdflib'."
        )

    for i, ent_id in enumerate(adapter.entities()):
        if progress_callback and i % 50 == 0:
            progress_callback(i)

        if max_entities is not None and i >= max_entities:
            warnings.append(f"Entity extraction truncated at max_entities={max_entities}.")
            break

        label = ""
        try:
            label = adapter.label(ent_id) or ""
        except Exception:
            pass
        if not label:
            continue

        # synonyms
        syns = []
        for attr in ("entity_aliases", "aliases", "synonyms"):
            fn = getattr(adapter, attr, None)
            if callable(fn):
                try:
                    syns = [s for s in (fn(ent_id) or []) if s]
                    break
                except Exception:
                    continue

        # description
        desc = ""
        for attr in ("definition", "comment", "description"):
            fn = getattr(adapter, attr, None)
            if callable(fn):
                try:
                    desc = fn(ent_id) or ""
                    if desc:
                        break
                except Exception:
                    continue

        # URI expand (optional)
        uri = str(ent_id)
        try:
            curie_to_uri = getattr(adapter, "curie_to_uri", None)
            if callable(curie_to_uri):
                uri = curie_to_uri(ent_id) or uri
        except Exception:
            pass

        entities.append(LocalEntity(uri=uri, label=str(label), description=str(desc), synonyms=[str(s) for s in syns]))

    return LocalIndex(
        resource_name=resource_name,
        resource_path=path,
        resource_hash=_sha256_file(path),
        parse_backend="oak",
        entities=entities,
        warnings=warnings,
    )


def _load_via_rdflib(
    path: str,
    *,
    resource_name: str,
    prefer_langs: Sequence[str],
    max_entities: Optional[int],
    progress_callback: Optional[callable] = None,
) -> LocalIndex:
    """
    Load with rdflib and extract:
    - subject URI
    - label from rdfs:label or skos:prefLabel (and maybe title)
    - synonyms from skos:altLabel
    - description from dc/dcterms/rdfs/skos definitions/comments
    """
    if rdflib is None:
        raise LocalResourceError(
            "rdflib is not installed, but an RDF-based ontology/thesaurus was provided.",
            hint="Install rdflib or install oaklib and use backend='oak'."
        )

    warnings: List[str] = []
    g = rdflib.Graph()

    # Format guess is fine most of the time; users can convert if it fails
    try:
        g.parse(path)
    except Exception as e:
        raise LocalResourceError(
            f"Failed to parse RDF file '{resource_name}' with rdflib.",
            hint="Convert to Turtle (.ttl) or RDF/XML (.rdf/.owl) if parsing fails. "
                 "If the file is not RDF, use a tabular loader (csv/xlsx).",
            details=str(e)
        )

    # Collect label/desc predicates that exist in this environment
    label_preds = [pred_fn() for _, pred_fn in RDF_LABEL_PREDICATES if pred_fn() is not None]
    desc_preds = [pred_fn() for _, pred_fn in RDF_DESC_PREDICATES if pred_fn() is not None]

    # Candidate subjects: anything with a label predicate
    subjects = set()
    for pred in label_preds:
        for s in g.subjects(predicate=pred):
            subjects.add(s)

    entities: List[LocalEntity] = []
    for i, s in enumerate(subjects):
        if progress_callback and i % 50 == 0:
            progress_callback(i)

        if max_entities is not None and i >= max_entities:
            warnings.append(f"Entity extraction truncated at max_entities={max_entities}.")
            break

        # label: prefer rdfs:label, then skos:prefLabel, then any title
        label_lits = []
        for pred in label_preds:
            label_lits.extend(list(g.objects(subject=s, predicate=pred)))
        label = _pick_literal(label_lits, prefer_langs=prefer_langs)
        if not label:
            continue

        # synonyms: skos:altLabel
        syns = []
        if SKOS is not None:
            syn_lits = list(g.objects(subject=s, predicate=SKOS.altLabel))
            syns = [str(getattr(l, "value", str(l))) for l in syn_lits]

        # description
        desc_lits = []
        for pred in desc_preds:
            desc_lits.extend(list(g.objects(subject=s, predicate=pred)))
        desc = _pick_literal(desc_lits, prefer_langs=prefer_langs)

        entities.append(LocalEntity(uri=str(s), label=label, description=desc, synonyms=syns))

    return LocalIndex(
        resource_name=resource_name,
        resource_path=path,
        resource_hash=_sha256_file(path),
        parse_backend="rdflib",
        entities=entities,
        warnings=warnings,
    )


TABULAR_URI_COL_CANDIDATES = {"uri", "iri", "id", "identifier", "term_id", "concept_id"}
TABULAR_LABEL_COL_CANDIDATES = {"label", "term", "name", "prefLabel", "preferred_label", "preferredlabel", "title"}
TABULAR_DESC_COL_CANDIDATES = {"description", "definition", "comment", "notes", "note"}
TABULAR_SYNONYM_COL_CANDIDATES = {"synonym", "synonyms", "altLabel", "altlabel", "aliases", "alias"}


def _choose_col(columns: Sequence[str], candidates: set) -> Optional[str]:
    lower_to_orig = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_to_orig:
            return lower_to_orig[cand.lower()]
    return None


def _split_synonyms(val: Any) -> List[str]:
    if val is None or (isinstance(val, float) and math.isnan(val)):  # type: ignore
        return []
    s = str(val).strip()
    if not s:
        return []
    # common separators
    parts = re.split(r"\s*[\|;,\t]\s*", s)
    return [p.strip() for p in parts if p.strip()]


def _load_via_tabular(
    path: str,
    *,
    resource_name: str,
    max_entities: Optional[int],
    progress_callback: Optional[callable] = None,
) -> LocalIndex:
    """
    Load CSV/TSV/XLSX. Heuristically detects columns.
    """
    warnings: List[str] = []
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")

    if pd is None and ext in {"xlsx", "xls"}:
        raise LocalResourceError(
            "pandas is required to read Excel files (.xlsx/.xls).",
            hint="Install pandas+openpyxl, or convert your file to CSV."
        )

    # Read into a DataFrame for uniform handling
    try:
        if ext == "csv":
            df = pd.read_csv(path) if pd is not None else None
        elif ext == "tsv":
            df = pd.read_csv(path, sep="\t") if pd is not None else None
        elif ext in {"xlsx", "xls"}:
            df = pd.read_excel(path)  # type: ignore
        else:
            # last resort: try csv with delimiter sniffing
            if pd is None:
                raise LocalResourceError(
                    "pandas is not installed; cannot parse tabular file robustly.",
                    hint="Install pandas or provide RDF/OWL/OBO and use OAK/rdflib."
                )
            df = pd.read_csv(path)
    except Exception as e:
        raise LocalResourceError(
            f"Failed to read tabular file '{resource_name}'.",
            hint="Ensure the file is a valid CSV/TSV/XLSX and uses UTF-8 (for CSV).",
            details=str(e)
        )

    if df is None or df.empty:
        raise LocalResourceError(
            f"Tabular file '{resource_name}' is empty.",
            hint="Provide at least columns for label and id/uri."
        )

    cols = list(df.columns)
    uri_col = _choose_col(cols, TABULAR_URI_COL_CANDIDATES)
    label_col = _choose_col(cols, TABULAR_LABEL_COL_CANDIDATES)
    desc_col = _choose_col(cols, TABULAR_DESC_COL_CANDIDATES)
    syn_col = _choose_col(cols, TABULAR_SYNONYM_COL_CANDIDATES)

    if label_col is None:
        raise LocalResourceError(
            f"Could not find a label column in '{resource_name}'.",
            hint=f"Add a column named one of: {sorted(TABULAR_LABEL_COL_CANDIDATES)}"
        )

    if uri_col is None:
        warnings.append(
            "No explicit URI/ID column detected. Will generate synthetic URIs based on row number."
        )

    entities: List[LocalEntity] = []
    for i, row in df.iterrows():
        if progress_callback and i % 50 == 0:
            progress_callback(i)

        if max_entities is not None and len(entities) >= max_entities:
            warnings.append(f"Entity extraction truncated at max_entities={max_entities}.")
            break

        label = str(row.get(label_col, "")).strip()
        if not label:
            continue

        if uri_col is not None:
            uri = str(row.get(uri_col, "")).strip()
        else:
            uri = f"urn:local:{resource_name}#{i}"

        desc = str(row.get(desc_col, "")).strip() if desc_col is not None else ""
        syns = _split_synonyms(row.get(syn_col)) if syn_col is not None else []

        entities.append(LocalEntity(uri=uri, label=label, description=desc, synonyms=syns))

    return LocalIndex(
        resource_name=resource_name,
        resource_path=path,
        resource_hash=_sha256_file(path),
        parse_backend="tabular",
        entities=entities,
        warnings=warnings,
    )


# -----------------------------
# Search / provider entrypoint
# -----------------------------

def _search_index(idx: LocalIndex, query: str, limit: int) -> List[Dict[str, Any]]:
    """
    Search a LocalIndex and return suggestion dicts consistent with other providers.
    """
    q = query.strip()
    if not q:
        return []

    # Candidate selection via token index
    toks = _tokenize(q)
    candidate_ids: List[int] = []
    if toks:
        seen = set()
        for t in toks:
            for ent_id in idx.token_to_entity_ids.get(t, []):
                if ent_id not in seen:
                    seen.add(ent_id)
                    candidate_ids.append(ent_id)

    # If token lookup fails (short query etc.), fall back to scanning a limited prefix subset
    if not candidate_ids:
        # scan first N entities as a last resort (still deterministic)
        candidate_ids = list(range(min(len(idx.entities), 5000)))

    scored: List[Tuple[float, int, str]] = []
    for ent_id in candidate_ids:
        ent = idx.entities[ent_id]
        best = _lev_sim(q, ent.label)
        for syn in (ent.synonyms or []):
            best = max(best, _lev_sim(q, syn))

        # simple boosts: substring match
        qn = _norm(q)
        ln = _norm(ent.label)
        if qn and ln and qn in ln:
            best = min(1.0, best + 0.08)

        scored.append((best, ent_id, ent.label))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(limit, 1)]

    suggestions: List[Dict[str, Any]] = []
    for score, ent_id, _ in top:
        ent = idx.entities[ent_id]
        # Use ontology title if extracted, else fallback to resource name
        display_name = idx.ontology_title or idx.resource_name
        suggestions.append(
            {
                "uri": ent.uri,
                "label": ent.label,
                "description": ent.description or "",
                "score": float(round(score, 6)),
                # make the file visible in the UI list
                "source_provider": display_name,
            }
        )
    return suggestions


def query_local_resources(
    term: str,
    *,
    limit: int = 10,
    config: Optional[Dict[str, Any]] = None,
    user_agent: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Provider function compatible with processing_service.fetch_suggestions_for_term_from_provider().

    Expected config:
      config["local_resources"] = [
        { "name": "my.owl", "path": "/tmp/...", "index": LocalIndex (optional), "backend": "auto|oak|rdflib|tabular" (optional) }
      ]
    """
    _ = user_agent  # unused, kept for signature compatibility

    if config is None:
        raise LocalResourceError(
            "Local Ontology provider called without config.",
            hint="Pass config['local_resources'] from the UI."
        )

    resources = config.get("local_resources") or []
    if not resources:
        raise LocalResourceError(
            "Local Ontology provider is selected but no local resources were uploaded.",
            hint="Upload an ontology/thesaurus file in the sidebar and confirm the queue."
        )

    prefer_langs = config.get("local_prefer_langs", ("en", ""))
    max_entities = config.get("local_max_entities", None)

    all_suggestions: List[Dict[str, Any]] = []

    # Search each loaded resource and merge results
    for res in resources:
        try:
            name = res.get("name") or res.get("resource_name") or "local_resource"
            path = res.get("path") or res.get("resource_path")
            backend = res.get("backend", "auto")

            if res.get("index") is not None:
                idx = res["index"]
            else:
                if not path:
                    raise LocalResourceError(
                        f"Local resource '{name}' is missing 'path'.",
                        hint="Upload must be saved to disk and passed through config."
                    )
                idx = load_local_resource_index(
                    path,
                    resource_name=name,
                    prefer_langs=prefer_langs,
                    max_entities=max_entities,
                    force_backend=backend,
                )

            # basic sanity: token index should exist
            if not idx.token_to_entity_ids:
                idx.build_token_index()

            all_suggestions.extend(_search_index(idx, term, limit))
        except LocalResourceError:
            raise
        except Exception as e:
            raise LocalResourceError(
                f"Local resource '{res}' failed during search.",
                hint="Check logs for the stack trace and verify the file format.",
                details=str(e)
            )

    # When multiple resources are loaded, keep only global top-N by score
    all_suggestions.sort(key=lambda d: d.get("score", -1), reverse=True)
    return all_suggestions[: max(limit, 1)]
