from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from contextlib import asynccontextmanager
import requests
from fastapi import FastAPI, HTTPException, Query, Depends, Request

# nạp biến môi trường từ .env nếu có
load_dotenv()

# =============== Config từ ENV =================
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "").rstrip("/")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "umls_labs")
OPENSEARCH_VERIFY_SSL = (os.getenv("OPENSEARCH_VERIFY_SSL", "true").lower() != "false")

# =============== Logging =======================
log = logging.getLogger("umls_api")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)


# =============== OpenSearch client mảnh gọn ===============
class OpenSearchLabMapper:
    """
    Truy vấn trực tiếp index OpenSearch đã ingest.
    Giữ nguyên schema và output format như code full.
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

    # ---- query builders ----
    def _build_dismax_query(self, q: str, size: int) -> Dict[str, Any]:
        """
        Ưu tiên:
          1) MATCH tuyệt đối (cao nhất):
             - match_phrase (exact phrase)
             - multi_match (operator=and) — tất cả terms phải khớp
          2) PREFIX:
             - match_phrase_prefix
             - multi_match bool_prefix
             - prefix explicit (preferred_term / synonyms)
          3) FUZZY (thấp nhất):
             - multi_match fuzziness=AUTO
        Chỉ đánh vào preferred_term và synonyms.
        """
        fields_boosted = [
            "preferred_term^5",
            "synonyms^3",
        ]

        q_lower = q.lower()  # cho prefix (không qua analyzer) nếu field đã lowercased khi index

        query: Dict[str, Any] = {
            "size": size,
            "_source": True,
            "query": {
                "dis_max": {
                    "tie_breaker": 0.0,
                    "queries": [
                        # (1) MATCH tuyệt đối — cao nhất
                        {
                            "multi_match": {
                                "query": q,
                                "type": "phrase",         # match_phrase
                                "fields": fields_boosted,
                                "slop": 0,
                                "boost": 10.0
                            }
                        },
                        {
                            "multi_match": {
                                "query": q,
                                "type": "best_fields",
                                "fields": fields_boosted,
                                "operator": "and",        # tất cả terms phải khớp
                                "boost": 8.0
                            }
                        },

                        # (2) PREFIX — mức ưu tiên kế tiếp
                        {
                            "multi_match": {
                                "query": q,
                                "type": "phrase_prefix",
                                "fields": fields_boosted,
                                "slop": 2,
                                "boost": 7.0
                            }
                        },
                        {
                            "bool": {
                                "should": [
                                    {"prefix": {"preferred_term": {"value": q_lower, "boost": 7.0}}},
                                    {"prefix": {"synonyms": {"value": q_lower, "boost": 7.0}}},
                                ],
                                "minimum_should_match": 1
                            }
                        },

                        # (3) FUZZY — thấp nhất
                        {
                            "multi_match": {
                                "query": q,
                                "fields": fields_boosted,
                                "fuzziness": "AUTO",
                                "operator": "and",
                                "boost": 5.0
                            }
                        },
                    ],
                }
            },
            "track_total_hits": True,
        }
        return query

    # ---- API-shape methods (giữ nguyên output format) ----
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

    def get_concept(self, cui: str) -> Optional[Dict[str, Any]]:
        src = self._get_by_id(cui)
        if not src:
            return None
        return src


# =============== FastAPI (lifespan) =================
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


# =============== Endpoints (giữ nguyên schema output) ===============
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
