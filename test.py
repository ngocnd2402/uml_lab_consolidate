from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field


# ================== Helpers ==================

def _read_rrf_rows(path: Path):
    """
    Read rows from an RRF file.  
    Trailing empty fields caused by ending '|' are trimmed.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        for row in reader:
            if not row:
                continue
            if row[-1] == "":
                row = row[:-1]
            yield row


# ================== Models ==================

class SourceCode(BaseModel):
    """Represents an individual code from a source vocabulary."""

    code: str
    source_vocab: str  # SAB value (e.g., "LNC", "SNOMEDCT_US")
    term_type: str     # TTY value (e.g., "PT", "SY")
    string: str        # Human-readable text
    is_preferred: bool = False


class Concept(BaseModel):
    """Represents a UMLS Concept with enhanced metadata."""

    cui: str
    preferred_term: Optional[str] = None
    definitions: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)
    semantic_types: List[str] = Field(default_factory=list)
    source_vocabularies: List[str] = Field(default_factory=list)
    relationships: Dict[str, List[str]] = Field(default_factory=dict)
    source_codes: List[SourceCode] = Field(default_factory=list)

    # Backward compatibility arrays
    loinc_codes: List[str] = Field(default_factory=list)
    snomed_codes: List[str] = Field(default_factory=list)
    icd10_codes: List[str] = Field(default_factory=list)


class LabMappingResult(BaseModel):
    """Result of mapping a lab name to standardized codes."""

    query_term: str
    matched_concepts: List[Dict[str, Any]]
    best_match: Optional[Dict[str, Any]] = None


# ================== Parser ==================

class UMLSLabMapper:
    """Parser for UMLS META files with lab-mapping utilities."""

    def __init__(self, meta_dir: str):
        self.meta_dir = Path(meta_dir)
        self.concepts: Dict[str, Concept] = {}
        self.string_index: Dict[str, Set[str]] = defaultdict(set)
        self.debug_sab = False

    # ---------- Internal Helpers ----------

    def _normalize_string(self, text: str) -> str:
        return text.lower().strip()

    def _concept(self, cui: str) -> Concept:
        if cui not in self.concepts:
            self.concepts[cui] = Concept(cui=cui)
        return self.concepts[cui]

    # ---------- Parsers ----------

    def parse_mrconso(self, limit: Optional[int] = None) -> None:
        """
        Parse MRCONSO.RRF:
        1. Index all strings
        2. Store codes with metadata
        3. Build preferred term hierarchy
        """
        file_path = self.meta_dir / "MRCONSO.RRF"
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        sab_counter = defaultdict(int)
        processed_count = 0

        for i, row in enumerate(_read_rrf_rows(file_path)):
            if limit and i >= limit:
                break
            if len(row) < 18:
                continue

            (
                cui, lat, ts, lui, stt, sui, ispref, aui,
                saui, scui, sdui, sab, tty, code, str_text,
                srl, suppress, cvf,
            ) = row[:18]

            if lat != "ENG":
                continue
            if suppress in {"Y", "E", "O"}:
                continue

            sab_counter[sab] += 1
            processed_count += 1

            concept = self._concept(cui)

            if sab and sab not in concept.source_vocabularies:
                concept.source_vocabularies.append(sab)

            if ispref == "Y" and ts == "P" and str_text:
                if not concept.preferred_term or stt == "PF":
                    concept.preferred_term = str_text
            elif not concept.preferred_term and str_text:
                concept.preferred_term = str_text

            if str_text and str_text not in concept.synonyms:
                concept.synonyms.append(str_text)

            normalized = self._normalize_string(str_text)
            self.string_index[normalized].add(cui)

            if code and code.strip():
                source_code = SourceCode(
                    code=code.strip(),
                    source_vocab=sab,
                    term_type=tty,
                    string=str_text,
                    is_preferred=(ispref == "Y"),
                )
                if not any(
                    sc.code == source_code.code and sc.source_vocab == source_code.source_vocab
                    for sc in concept.source_codes
                ):
                    concept.source_codes.append(source_code)

            if sab in {"LNC", "LOINC"}:
                if code not in concept.loinc_codes:
                    concept.loinc_codes.append(code)
            elif sab in {"SNOMEDCT_US", "SNOMEDCT"}:
                if code not in concept.snomed_codes:
                    concept.snomed_codes.append(code)
            elif sab in {"ICD10CM", "ICD10", "ICD10PCS"}:
                if code not in concept.icd10_codes:
                    concept.icd10_codes.append(code)

        if self.debug_sab:
            print(f"\nProcessed {processed_count:,} entries")
            print("Top 20 SAB values:")
            for sab, count in sorted(sab_counter.items(), key=lambda x: x[1], reverse=True)[:20]:
                print(f" {sab:20s}: {count:,}")

    def parse_mrsty(self, limit: Optional[int] = None) -> None:
        """Parse MRSTY.RRF for semantic types."""
        file_path = self.meta_dir / "MRSTY.RRF"
        if not file_path.exists():
            return

        for i, row in enumerate(_read_rrf_rows(file_path)):
            if limit and i >= limit:
                break
            if len(row) < 4:
                continue

            cui, tui, stn, sty = row[:4]
            concept = self.concepts.get(cui)
            if concept and sty not in concept.semantic_types:
                concept.semantic_types.append(sty)

    def parse_mrdef(self, limit: Optional[int] = None) -> None:
        """Parse MRDEF.RRF for definitions."""
        file_path = self.meta_dir / "MRDEF.RRF"
        if not file_path.exists():
            return

        for i, row in enumerate(_read_rrf_rows(file_path)):
            if limit and i >= limit:
                break
            if len(row) < 6:
                continue

            cui, aui, atui, satui, sab, definition = row[:6]
            concept = self.concepts.get(cui)
            if concept and definition and definition not in concept.definitions:
                concept.definitions.append(definition)

    # ---------- Query Methods ----------

    def is_lab_related(self, concept: Concept) -> bool:
        lab_types = {
            "Laboratory Procedure",
            "Diagnostic Procedure",
            "Laboratory or Test Result",
            "Clinical Finding",
            "Finding",
            "Quantitative Concept",
            "Substance",
        }
        return any(st in lab_types for st in concept.semantic_types)

    def map_lab_name_to_codes(self, lab_name: str) -> LabMappingResult:
        """
        Map a lab name to CUIs and standardized codes:
        1. Lookup in MRCONSO
        2. Get CUIs
        3. Extract codes from vocabularies
        """
        normalized_query = self._normalize_string(lab_name)
        matched_concepts: List[Dict[str, Any]] = []
        matching_cuis: Set[str] = set()

        if normalized_query in self.string_index:
            matching_cuis.update(self.string_index[normalized_query])

        for indexed_string, cuis in self.string_index.items():
            if normalized_query in indexed_string or indexed_string in normalized_query:
                matching_cuis.update(cuis)

        for cui in matching_cuis:
            concept = self.concepts.get(cui)
            if not concept:
                continue

            if concept.semantic_types and not self.is_lab_related(concept):
                continue

            loinc_codes, snomed_codes, icd10_codes, other_codes = [], [], [], []

            for source_code in concept.source_codes:
                code_info = {
                    "code": source_code.code,
                    "source_vocab": source_code.source_vocab,
                    "term_type": source_code.term_type,
                    "string": source_code.string,
                    "is_preferred": source_code.is_preferred,
                }
                if source_code.source_vocab in {"LNC", "LOINC"}:
                    loinc_codes.append(code_info)
                elif source_code.source_vocab in {"SNOMEDCT_US", "SNOMEDCT"}:
                    snomed_codes.append(code_info)
                elif source_code.source_vocab in {
                    "ICD10CM", "ICD10", "ICD10PCS", "CCSR_ICD10CM", "CCSR_ICD10PCS"
                }:
                    icd10_codes.append(code_info)
                else:
                    other_codes.append(code_info)

            concept_result = {
                "cui": cui,
                "preferred_term": concept.preferred_term,
                "synonyms": concept.synonyms[:10],
                "semantic_types": concept.semantic_types,
                "loinc_codes": loinc_codes,
                "snomed_codes": snomed_codes,
                "icd10_codes": icd10_codes,
                "other_codes": other_codes,
                "total_codes": len(concept.source_codes),
            }
            matched_concepts.append(concept_result)

        matched_concepts.sort(
            key=lambda x: (
                len(x["loinc_codes"])
                + len(x["snomed_codes"])
                + len(x["icd10_codes"])
            ),
            reverse=True,
        )
        best_match = matched_concepts[0] if matched_concepts else None

        return LabMappingResult(
            query_term=lab_name,
            matched_concepts=matched_concepts,
            best_match=best_match,
        )

    def search_by_term(self, term: str) -> List[Dict[str, Any]]:
        return self.map_lab_name_to_codes(term).matched_concepts

    def get_concept(self, cui: str) -> Optional[Dict[str, Any]]:
        concept = self.concepts.get(cui)
        if not concept:
            return None
        return {
            "cui": concept.cui,
            "preferred_term": concept.preferred_term,
            "synonyms": concept.synonyms,
            "semantic_types": concept.semantic_types,
            "source_vocabularies": concept.source_vocabularies,
            "definitions": concept.definitions,
            "source_codes": [sc.model_dump() for sc in concept.source_codes],
            "loinc_codes": concept.loinc_codes,
            "snomed_codes": concept.snomed_codes,
            "icd10_codes": concept.icd10_codes,
        }


# ================== FastAPI ==================

META_DIR = os.getenv("UMLS_META_DIR", r"D:\Work\Out\wrs\lab_data_proj\data\2025AA\META")
MRCONSO_LIMIT = int(os.getenv("UMLS_MRCONSO_LIMIT", "500000"))
MRSTY_LIMIT = int(os.getenv("UMLS_MRSTY_LIMIT", "200000"))
MRDEF_LIMIT = int(os.getenv("UMLS_MRDEF_LIMIT", "100000"))
DEBUG_SAB = os.getenv("UMLS_DEBUG_SAB", "true").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading UMLS data...")
    mapper = UMLSLabMapper(META_DIR)
    mapper.debug_sab = DEBUG_SAB
    mapper.parse_mrconso(limit=MRCONSO_LIMIT)
    mapper.parse_mrsty(limit=MRSTY_LIMIT)
    mapper.parse_mrdef(limit=MRDEF_LIMIT)
    print(f"Loaded {len(mapper.concepts):,} concepts")
    app.state.mapper = mapper
    try:
        yield
    finally:
        app.state.mapper = None
        print("UMLS mapper unloaded.")


app = FastAPI(title="UMLS Lab Code Mapper", version="1.0", lifespan=lifespan)


def get_mapper(request: Request) -> UMLSLabMapper:
    mapper = getattr(request.app.state, "mapper", None)
    if mapper is None:
        raise HTTPException(status_code=503, detail="Mapper not loaded")
    return mapper


@app.get("/map-lab")
async def map_lab_name(
    lab_name: str = Query(..., description="Lab test name to map"),
    mapper: UMLSLabMapper = Depends(get_mapper),
):
    result = mapper.map_lab_name_to_codes(lab_name)
    if not result.matched_concepts:
        raise HTTPException(status_code=404, detail=f"No UMLS concepts for '{lab_name}'")
    return {
        "query": result.query_term,
        "total_matches": len(result.matched_concepts),
        "best_match": result.best_match,
        "all_matches": result.matched_concepts,
    }


@app.get("/concept/{cui}")
async def get_concept_by_cui(
    cui: str,
    mapper: UMLSLabMapper = Depends(get_mapper),
):
    concept = mapper.get_concept(cui)
    if not concept:
        raise HTTPException(status_code=404, detail=f"Concept {cui} not found")
    return concept


@app.get("/search")
async def search_concepts(
    term: str = Query(..., description="Search term"),
    limit: int = Query(10, description="Max results"),
    mapper: UMLSLabMapper = Depends(get_mapper),
):
    results = mapper.search_by_term(term)
    return {"query": term, "count": len(results), "results": results[:limit]}


# ================== CLI Demo ==================

if __name__ == "__main__":
    print("Initializing UMLS Lab Mapper (CLI)...")
    mapper = UMLSLabMapper(META_DIR)
    mapper.debug_sab = DEBUG_SAB

    print("Parsing MRCONSO...")
    mapper.parse_mrconso(limit=MRCONSO_LIMIT)
    print("Parsing semantic types and definitions...")
    mapper.parse_mrsty(limit=MRSTY_LIMIT)
    mapper.parse_mrdef(limit=MRDEF_LIMIT)

    print(f"\nLoaded {len(mapper.concepts):,} concepts")
    print("\n" + "=" * 80)
    print("TESTING LAB NAME MAPPING WORKFLOW")
    print("=" * 80)

    test_terms = ["Hemoglobin A1c", "HbA1c", "glucose", "cholesterol", "creatinine"]
    for term in test_terms:
        print(f"\n{'=' * 50}\nMapping: {term}\n{'=' * 50}")
        result = mapper.map_lab_name_to_codes(term)
        if result.best_match:
            best = result.best_match
            print("Best Match:")
            print(f" CUI: {best['cui']}")
            print(f" Preferred Term: {best['preferred_term']}")
            print(f" Semantic Types: {', '.join(best['semantic_types'])}")
            print(f"\n LOINC Codes: {len(best['loinc_codes'])}")
            print(f"\n SNOMED CT Codes: {len(best['snomed_codes'])}")
            print(f"\n ICD-10 Codes: {len(best['icd10_codes'])}")
            print(f"\n Total Matches: {len(result.matched_concepts)}")
        else:
            print(" No matches found")

    print(f"\n{'=' * 80}\nSAMPLE API RESPONSES\n{'=' * 80}")
    sample_result = mapper.map_lab_name_to_codes(test_terms[0])
    print(
        json.dumps(
            {
                "query": sample_result.query_term,
                "total_matches": len(sample_result.matched_concepts),
                "best_match": sample_result.best_match,
            },
            indent=2,
        )
    )
