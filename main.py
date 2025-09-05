from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
from dotenv import load_dotenv

from contextlib import asynccontextmanager
import requests
from fastapi import FastAPI, HTTPException, Query, Depends, Request

# load environment variables from .env if present
load_dotenv()

# =============== ENV Config ===============
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "").rstrip("/")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "umls_labs")
OPENSEARCH_VERIFY_SSL = (os.getenv("OPENSEARCH_VERIFY_SSL", "true").lower() != "false")

# =============== Logging ===============
log = logging.getLogger("umls_api")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# =============== Enum for code systems ==============
class CodeSystem(str, Enum):
    loinc = "loinc"
    snomed = "snomed"
    icd10 = "icd10"
    other = "other"

# ======== Enums for UTS-like search ============
class InputType(str, Enum):
    atom = "atom"
    code = "code"
    sourceConcept = "sourceConcept"
    sourceDescriptor = "sourceDescriptor"
    sourceUi = "sourceUi"
    tty = "tty"  # not parsing by TTY; treat like 'atom'

class SearchType(str, Enum):
    words = "words"                 # default
    exact = "exact"
    rightTruncation = "rightTruncation"
    leftTruncation = "leftTruncation"      # fallback -> words
    normalizedString = "normalizedString"  # fallback -> words
    normalizedWords = "normalizedWords"    # fallback -> words


# =============== Minimal OpenSearch client ===============
class OpenSearchLabMapper:
    """
    Query an ingested OpenSearch index.
    Preserve schema and output format as in the full codebase.
    """
    def __init__(self, base_url: str, index_name: str, user: str, password: str, verify_ssl: bool = True):
        self.base_url = base_url.rstrip("/")
        self.index = index_name
        self.session = requests.Session()
        self.session.auth = (user, password)
        self.session.verify = verify_ssl
        self.timeout = 30

    # ---- helpers ----
    def _search(self, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{self.index}/_search"
        r = self.session.post(url, json=body, timeout=self.timeout)
        try:
            data = r.json()
        except Exception:
            raise HTTPException(status_code=502, detail=f"OpenSearch bad response: {r.status_code}")
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=data)
        return data

    def _get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/{self.index}/_doc/{doc_id}"
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code == 404:
            return None
        try:
            data = r.json()
        except Exception:
            raise HTTPException(status_code=502, detail=f"OpenSearch bad response: {r.status_code}")
        if r.status_code != 200 or not data.get("found"):
            return None
        return data.get("_source")

    # ---- query builders: concept text (legacy) ----
    def _build_dismax_query(self, q: str, size: int) -> Dict[str, Any]:
        fields_boosted = [
            "preferred_term^5",
            "synonyms^3",
        ]
        q_lower = q.lower()
        query: Dict[str, Any] = {
            "size": size,
            "_source": True,
            "query": {
                "dis_max": {
                    "tie_breaker": 0.0,
                    "queries": [
                        {"multi_match": {"query": q, "type": "phrase", "fields": fields_boosted, "slop": 0, "boost": 10.0}},
                        {"multi_match": {"query": q, "type": "best_fields", "fields": fields_boosted, "operator": "and", "boost": 8.0}},
                        {"multi_match": {"query": q, "type": "phrase_prefix", "fields": fields_boosted, "slop": 2, "boost": 7.0}},
                        {"bool": {"should": [
                            {"prefix": {"preferred_term": {"value": q_lower, "boost": 7.0}}},
                            {"prefix": {"synonyms": {"value": q_lower, "boost": 7.0}}},
                        ], "minimum_should_match": 1}},
                        {"multi_match": {"query": q, "fields": fields_boosted, "fuzziness": "AUTO", "operator": "and", "boost": 5.0}},
                    ],
                }
            },
            "track_total_hits": True,
        }
        return query

    # ---- query builders: UTS-like text (new, with paging) ----
    def _build_text_query_uts(self, q: str, size: int, from_: int, search_type: SearchType) -> Dict[str, Any]:
        fields = ["preferred_term^5", "synonyms^3"]
        q_lower = (q or "").lower()

        def dismax(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {"dis_max": {"tie_breaker": 0.0, "queries": queries}}

        if search_type == SearchType.exact:
            queries = [
                {"multi_match": {"query": q, "type": "phrase", "fields": fields, "slop": 0, "boost": 10.0}},
                {"multi_match": {"query": q, "type": "best_fields", "fields": fields, "operator": "and", "boost": 8.0}},
            ]
        elif search_type == SearchType.rightTruncation:
            queries = [
                {"multi_match": {"query": q, "type": "phrase_prefix", "fields": fields, "slop": 2, "boost": 8.0}},
                {"bool": {"should": [
                    {"prefix": {"preferred_term": {"value": q_lower, "boost": 7.0}}},
                    {"prefix": {"synonyms": {"value": q_lower, "boost": 6.0}}},
                ], "minimum_should_match": 1}},
            ]
        else:  # words + fallbacks
            queries = [
                {"multi_match": {"query": q, "type": "best_fields", "fields": fields, "operator": "and", "boost": 8.0}},
                {"multi_match": {"query": q, "type": "bool_prefix", "fields": fields, "boost": 6.0}},
            ]

        return {
            "from": from_,
            "size": size,
            "_source": True,
            "track_total_hits": True,
            "query": dismax(queries),
        }

    # ---- query builders: code search (legacy) ----
    def _build_code_query(self, code: str, size: int, system: Optional[Union[str, CodeSystem]]) -> Dict[str, Any]:
        sys_str = (system.value if isinstance(system, CodeSystem) else (system or "")).strip().lower()
        allowed_paths_all = ["loinc_codes", "snomed_codes", "icd10_codes", "other_codes"]
        if sys_str == "loinc":
            paths = ["loinc_codes"]
        elif sys_str == "snomed":
            paths = ["snomed_codes"]
        elif sys_str == "icd10":
            paths = ["icd10_codes"]
        elif sys_str == "other":
            paths = ["other_codes"]
        else:
            paths = allowed_paths_all

        code_lower = code.lower()
        nested_queries: List[Dict[str, Any]] = []
        for p in paths:
            per_path_dismax = {
                "dis_max": {
                    "tie_breaker": 0.0,
                    "queries": [
                        {"term": {f"{p}.code": {"value": code, "boost": 10.0}}},
                        {"prefix": {f"{p}.code": {"value": code_lower, "boost": 7.0}}},
                        {"fuzzy":  {f"{p}.code": {"value": code, "fuzziness": "AUTO", "boost": 5.0}}},
                    ]
                }
            }
            nested_queries.append({"nested": {"path": p, "query": per_path_dismax}})

        query: Dict[str, Any] = {
            "size": size,
            "_source": True,
            "query": {"dis_max": {"tie_breaker": 0.0, "queries": nested_queries}},
            "track_total_hits": True,
        }
        return query

    # ---- query builders: code search for UTS (new, with paging) ----
    def _build_code_query_uts(self, code: str, size: int, from_: int, paths: List[str]) -> Dict[str, Any]:
        code_lower = (code or "").lower()
        nested_queries: List[Dict[str, Any]] = []
        for p in paths:
            per_path = {
                "dis_max": {
                    "tie_breaker": 0.0,
                    "queries": [
                        {"term": {f"{p}.code": {"value": code, "boost": 10.0}}},
                        {"prefix": {f"{p}.code": {"value": code_lower, "boost": 7.0}}},
                        {"fuzzy":  {f"{p}.code": {"value": code, "fuzziness": "AUTO", "boost": 4.0}}},
                    ]
                }
            }
            nested_queries.append({"nested": {"path": p, "query": per_path}})

        return {
            "from": from_,
            "size": size,
            "_source": True,
            "track_total_hits": True,
            "query": {"dis_max": {"tie_breaker": 0.0, "queries": nested_queries}},
        }

    # ---- UTS-like search (reduced): always return CUI-level ----
    def uts_search(
        self,
        string: str,
        *,
        search_type: SearchType,
        input_type: InputType,
        sabs: Optional[List[str]],
        page_number: int,
        page_size: int,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Return (total_approx, rows) where rows follow UTS searchResults (CUI-level).
        """
        # SAB filter
        sab_filter = None
        if sabs:
            sab_filter = {s.strip().upper() for s in sabs if s and s.strip()}

        from_ = max(0, (page_number - 1) * page_size)

        def concept_hits_to_rows(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            for h in hits:
                src = h.get("_source", {})
                # if SAB filter is provided: keep concept that has at least one code in those SABs
                if sab_filter:
                    found = False
                    for arr in ("loinc_codes", "snomed_codes", "icd10_codes", "other_codes"):
                        for sc in src.get(arr, []) or []:
                            if sc.get("sab") in sab_filter:
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        continue
                cui = src.get("cui")
                name = src.get("preferred_term") or (src.get("synonyms") or [None])[0] or ""
                root = (src.get("source_vocabularies") or [None])[0] or "UMLS"
                rows.append({
                    "ui": cui,
                    "rootSource": root,
                    "uri": f"/content/current/CUI/{cui}",
                    "name": name,
                })
            return rows

        # code-like input -> prioritize code search first
        is_code_like = input_type in {InputType.code, InputType.sourceUi, InputType.sourceConcept, InputType.sourceDescriptor}
        if is_code_like:
            paths = ["loinc_codes", "snomed_codes", "icd10_codes", "other_codes"]
            if sab_filter:
                # simplify: if the SAB filter is entirely within LOINC/SNOMED/ICD10, narrow paths accordingly
                if sab_filter.issubset({"LNC", "LOINC"}):
                    paths = ["loinc_codes"]
                elif sab_filter.issubset({"SNOMEDCT_US", "SNOMEDCT"}):
                    paths = ["snomed_codes"]
                elif sab_filter.issubset({"CCSR_ICD10CM", "CCSR_ICD10PCS", "ICD10PCS", "ICD10", "ICD10CM"}):
                    paths = ["icd10_codes"]
            body = self._build_code_query_uts(string, page_size, from_, paths)
            data = self._search(body)
            total = data.get("hits", {}).get("total", {}).get("value", 0)
            hits = data.get("hits", {}).get("hits", [])
            rows = concept_hits_to_rows(hits)
            return total, rows

        # text search
        body = self._build_text_query_uts(string, page_size, from_, search_type)
        data = self._search(body)
        total = data.get("hits", {}).get("total", {}).get("value", 0)
        hits = data.get("hits", {}).get("hits", [])
        rows = concept_hits_to_rows(hits)
        return total, rows

    # ---- API-shape methods (unchanged) ----
    def map_lab_name_to_codes(self, lab_name: str, size: int = 25) -> Dict[str, Any]:
        body = self._build_dismax_query(lab_name, size=size)
        res = self._search(body)
        hits = res.get("hits", {}).get("hits", [])
        total = res.get("hits", {}).get("total", {})
        total_val = total.get("value", len(hits))

        matched_concepts: List[Dict[str, Any]] = []
        for h in hits:
            src = h.get("_source", {})
            total_codes = (
                len(src.get("loinc_codes", []))
                + len(src.get("snomed_codes", []))
                + len(src.get("icd10_codes", []))
                + len(src.get("other_codes", []))
            )
            matched_concepts.append({
                "cui": src.get("cui"),
                "preferred_term": src.get("preferred_term"),
                "synonyms": src.get("synonyms", [])[:10],
                "semantic_types": src.get("semantic_types", []),
                "loinc_codes": src.get("loinc_codes", []),
                "snomed_codes": src.get("snomed_codes", []),
                "icd10_codes": src.get("icd10_codes", []),
                "other_codes": src.get("other_codes", []),
                "total_codes": total_codes,
            })

        best_match = matched_concepts[0] if matched_concepts else None
        return {
            "query_term": lab_name,
            "matched_concepts": matched_concepts,
            "best_match": best_match,
            "total": total_val,
        }

    def search_by_term(self, term: str, size: int = 25) -> List[Dict[str, Any]]:
        out = self.map_lab_name_to_codes(term, size=size)
        return out["matched_concepts"]

    def search_by_code(self, code: str, size: int = 25, system: Optional[Union[str, CodeSystem]] = None) -> Dict[str, Any]:
        body = self._build_code_query(code, size=size, system=system)
        res = self._search(body)
        hits = res.get("hits", {}).get("hits", [])
        total = res.get("hits", {}).get("total", {})
        total_val = total.get("value", len(hits))

        matched_concepts: List[Dict[str, Any]] = []
        for h in hits:
            src = h.get("_source", {})
            total_codes = (
                len(src.get("loinc_codes", []))
                + len(src.get("snomed_codes", []))
                + len(src.get("icd10_codes", []))
                + len(src.get("other_codes", []))
            )
            matched_concepts.append({
                "cui": src.get("cui"),
                "preferred_term": src.get("preferred_term"),
                "synonyms": src.get("synonyms", [])[:10],
                "semantic_types": src.get("semantic_types", []),
                "loinc_codes": src.get("loinc_codes", []),
                "snomed_codes": src.get("snomed_codes", []),
                "icd10_codes": src.get("icd10_codes", []),
                "other_codes": src.get("other_codes", []),
                "total_codes": total_codes,
            })

        best_match = matched_concepts[0] if matched_concepts else None
        return {
            "query_term": code,
            "matched_concepts": matched_concepts,
            "best_match": best_match,
            "total": total_val,
        }

    def get_concept(self, cui: str) -> Optional[Dict[str, Any]]:
        src = self._get_by_id(cui)
        if not src:
            return None
        return src


# =============== FastAPI (lifespan) ===============
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (OPENSEARCH_URL and OPENSEARCH_USER and OPENSEARCH_PASSWORD):
        log.error("Missing OpenSearch credentials. Set OPENSEARCH_URL, OPENSEARCH_USER, OPENSEARCH_PASSWORD.")
        app.state.mapper = None
        yield
        return

    log.info(f"Connecting OpenSearch: {OPENSEARCH_URL} index={OPENSEARCH_INDEX}")
    mapper = OpenSearchLabMapper(
        base_url=OPENSEARCH_URL,
        index_name=OPENSEARCH_INDEX,
        user=OPENSEARCH_USER,
        password=OPENSEARCH_PASSWORD,
        verify_ssl=OPENSEARCH_VERIFY_SSL,
    )
    app.state.mapper = mapper
    try:
        yield
    finally:
        app.state.mapper = None
        log.info("OpenSearch client closed.")


app = FastAPI(title="UMLS Lab Code Mapper", version="1.0", lifespan=lifespan)


def get_mapper(request: Request) -> OpenSearchLabMapper:
    mapper = getattr(request.app.state, "mapper", None)
    if mapper is None:
        raise HTTPException(status_code=503, detail="Search backend not available.")
    return mapper


# ==================== NEW: UTS-like search (reduced) ====================
@app.get("/search-uts")
async def uts_like_search(
    string: str = Query(..., description="Term or code"),
    inputType: InputType = Query(InputType.atom, description="atom | code | sourceConcept | sourceDescriptor | sourceUi | tty"),
    sabs: Optional[str] = Query(None, description="Comma-separated RSABs (e.g., LNC,SNOMEDCT_US)"),
    searchType: SearchType = Query(SearchType.words, description="words | exact | rightTruncation | leftTruncation | normalizedString | normalizedWords"),
    pageNumber: int = Query(1, ge=1),
    pageSize: int = Query(25, ge=1, le=200),
    mapper: OpenSearchLabMapper = Depends(get_mapper),
):
    """
    Return a schema similar to UTS /search (reduced, CUI-level only):
    {
      "pageSize": n,
      "pageNumber": m,
      "result": {"classType":"searchResults","results":[{"ui": CUI, "rootSource": ..., "uri": ..., "name": ...}, ...]}
    }
    No results / end of pages: results = [{"ui":"NONE","name":"NO RESULTS"}]
    """
    sab_list = [s.strip().upper() for s in (sabs.split(",") if sabs else []) if s.strip()]

    total, rows = mapper.uts_search(
        string=string,
        search_type=searchType,
        input_type=inputType,
        sabs=sab_list or None,
        page_number=pageNumber,
        page_size=pageSize,
    )

    if not rows:
        return {
            "pageSize": pageSize,
            "pageNumber": pageNumber,
            "result": {
                "classType": "searchResults",
                "results": [{"ui": "NONE", "name": "NO RESULTS"}],
            },
        }

    return {
        "pageSize": pageSize,
        "pageNumber": pageNumber,
        "result": {
            "classType": "searchResults",
            "results": rows,
        },
    }


# =============== Endpoints (unchanged schema) ===============
@app.get("/map-lab")
async def map_lab_name(
    lab_name: str = Query(..., description="Lab test name to map to standard codes"),
    limit: int = Query(25, ge=1, le=200, description="Max results"),
    mapper: OpenSearchLabMapper = Depends(get_mapper),
):
    out = mapper.map_lab_name_to_codes(lab_name, size=limit)
    if not out["matched_concepts"]:
        raise HTTPException(status_code=404, detail=f"No UMLS concepts found for '{lab_name}'")
    return {
        "query": out["query_term"],
        "total_matches": out["total"],
        "best_match": out["best_match"],
        "all_matches": out["matched_concepts"],
    }


@app.get("/concept/{cui}")
async def get_concept_by_cui(
    cui: str,
    mapper: OpenSearchLabMapper = Depends(get_mapper),
):
    doc = mapper.get_concept(cui)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Concept {cui} not found")
    return doc


@app.get("/search")
async def search_concepts(
    term: str = Query(..., description="Search term"),
    limit: int = Query(10, ge=1, le=200, description="Max results"),
    mapper: OpenSearchLabMapper = Depends(get_mapper),
):
    results = mapper.search_by_term(term, size=limit)
    return {"query": term, "count": len(results), "results": results[:limit]}


@app.get("/search-code")
async def search_code(
    code: str = Query(..., description="Code to lookup (e.g., LOINC/SNOMED/ICD-10-PCS/CCSR)"),
    system: Optional[CodeSystem] = Query(
        None,
        description="Restrict to a code system"
    ),
    limit: int = Query(25, ge=1, le=200, description="Max results"),
    mapper: OpenSearchLabMapper = Depends(get_mapper),
):
    """
    Find concepts by a code. Ranking priority: exact > prefix > fuzzy.
    - system: dropdown {loinc | snomed | icd10 | other}
    """
    out = mapper.search_by_code(code, size=limit, system=system)
    if not out["matched_concepts"]:
        raise HTTPException(status_code=404, detail=f"No concepts found for code '{code}'")
    return {
        "query": out["query_term"],
        "total_matches": out["total"],
        "best_match": out["best_match"],
        "all_matches": out["matched_concepts"],
    }
