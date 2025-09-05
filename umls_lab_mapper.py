import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from pydantic import BaseModel, Field

# ========= Small helpers =========
def _read_rrf_rows(path: Path):
    """
    Đọc .RRF với delimiter '|', UMLS kết thúc mỗi dòng bằng '|' → trường cuối rỗng.
    Ta cắt trường rỗng cuối để số cột khớp schema.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        for row in reader:
            if not row:
                continue
            if row and row[-1] == "":
                row = row[:-1]
            yield row


# ========= Models =========
class SourceCode(BaseModel):
    """Atom/code từ 1 nguồn (SAB) cụ thể"""
    code: str
    source_vocab: str   # SAB (vd: "LNC", "SNOMEDCT_US")
    term_type: str      # TTY (vd: "PT", "SY")
    string: str         # Text (STR)
    is_preferred: bool = False  # ISPREF == 'Y'


class Concept(BaseModel):
    cui: str
    preferred_term: Optional[str] = None
    definitions: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)
    semantic_types: List[str] = Field(default_factory=list)
    source_vocabularies: List[str] = Field(default_factory=list)
    relationships: Dict[str, List[str]] = Field(default_factory=dict)

    # Mọi codes (đầy đủ metadata)
    source_codes: List[SourceCode] = Field(default_factory=list)

    # Mảng truy cập nhanh (tuỳ chọn)
    loinc_codes: List[str] = Field(default_factory=list)
    snomed_codes: List[str] = Field(default_factory=list)
    icd10_codes: List[str] = Field(default_factory=list)


class LabMappingResult(BaseModel):
    """Kết quả mapping lab name → bộ code chuẩn"""
    query_term: str
    matched_concepts: List[Dict[str, Any]]
    best_match: Optional[Dict[str, Any]] = None


# ========= Parser / Mapper =========
class UMLSLabMapper:
    """
    - parse_mrconso(): build preferred term, synonyms, index STR → CUI, thu thập codes (SAB, TTY, CODE, STR)
    - parse_mrsty(): nạp semantic types cho CUI
    - parse_mrdef(): nạp definitions
    - map_lab_name_to_codes(): workflow như mô tả
    """
    def __init__(self, meta_dir: str):
        self.meta_dir = Path(meta_dir)
        self.concepts: Dict[str, Concept] = {}
        self.string_index: Dict[str, Set[str]] = defaultdict(set)  # normalized STR → set CUI
        self.debug_sab = False

    def _normalize_string(self, text: str) -> str:
        return text.lower().strip()

    def _concept(self, cui: str) -> Concept:
        if cui not in self.concepts:
            self.concepts[cui] = Concept(cui=cui)
        return self.concepts[cui]

    def parse_mrconso(self, limit: Optional[int] = None) -> None:
        path = self.meta_dir / "MRCONSO.RRF"
        if not path.exists():
            raise FileNotFoundError(path)

        sab_counter = defaultdict(int)
        processed = 0

        for i, row in enumerate(_read_rrf_rows(path)):
            if limit and i >= limit:
                break
            if len(row) < 18:
                continue

            (
                cui, lat, ts, lui, stt, sui, ispref, aui, saui, scui, sdui,
                sab, tty, code, str_text, srl, suppress, cvf
            ) = row[:18]

            # English only
            if lat != "ENG":
                continue
            # Loại suppressed
            if suppress in {"Y", "E", "O"}:
                continue

            processed += 1
            sab_counter[sab] += 1

            c = self._concept(cui)

            # nguồn
            if sab and sab not in c.source_vocabularies:
                c.source_vocabularies.append(sab)

            # preferred
            if ispref == "Y" and ts == "P" and str_text:
                if not c.preferred_term or stt == "PF":
                    c.preferred_term = str_text
            elif not c.preferred_term and str_text:
                c.preferred_term = str_text

            # synonyms
            if str_text and str_text not in c.synonyms:
                c.synonyms.append(str_text)

            # index string
            if str_text:
                self.string_index[self._normalize_string(str_text)].add(cui)

            # codes
            if code and code.strip():
                sc = SourceCode(
                    code=code.strip(),
                    source_vocab=sab,
                    term_type=tty,
                    string=str_text or "",
                    is_preferred=(ispref == "Y"),
                )
                if not any(x.code == sc.code and x.source_vocab == sc.source_vocab for x in c.source_codes):
                    c.source_codes.append(sc)

                # quick arrays
                if sab in ("LNC", "LOINC"):
                    if sc.code not in c.loinc_codes:
                        c.loinc_codes.append(sc.code)
                elif sab in ("SNOMEDCT_US", "SNOMEDCT"):
                    if sc.code not in c.snomed_codes:
                        c.snomed_codes.append(sc.code)
                elif sab in ("ICD10CM", "ICD10"):
                    if sc.code not in c.icd10_codes:
                        c.icd10_codes.append(sc.code)

        if self.debug_sab:
            print(f"\n[MRCONSO] processed: {processed:,}")
            for sab, cnt in sorted(sab_counter.items(), key=lambda x: x[1], reverse=True)[:20]:
                print(f"  {sab:20s}: {cnt:,}")

    def parse_mrsty(self, limit: Optional[int] = None) -> None:
        path = self.meta_dir / "MRSTY.RRF"
        if not path.exists():
            return
        for i, row in enumerate(_read_rrf_rows(path)):
            if limit and i >= limit:
                break
            if len(row) < 4:
                continue
            cui, tui, stn, sty = row[:4]
            c = self.concepts.get(cui)
            if c and sty not in c.semantic_types:
                c.semantic_types.append(sty)

    def parse_mrdef(self, limit: Optional[int] = None) -> None:
        path = self.meta_dir / "MRDEF.RRF"
        if not path.exists():
            return
        for i, row in enumerate(_read_rrf_rows(path)):
            if limit and i >= limit:
                break
            if len(row) < 6:
                continue
            cui, aui, atui, satui, sab, definition = row[:6]
            c = self.concepts.get(cui)
            if c and definition and definition not in c.definitions:
                c.definitions.append(definition)

    # ---------- Helpers ----------
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

    # ---------- Queries ----------
    def map_lab_name_to_codes(self, lab_name: str) -> LabMappingResult:
        q = self._normalize_string(lab_name)
        matched: List[Dict[str, Any]] = []

        # exact
        cuis: Set[str] = set(self.string_index.get(q, set()))
        # partial (đơn giản)
        if not cuis:
            for s, ids in self.string_index.items():
                if q in s or s in q:
                    cuis.update(ids)

        for cui in cuis:
            c = self.concepts.get(cui)
            if not c:
                continue
            if c.semantic_types and not self.is_lab_related(c):
                continue

            loinc_codes, snomed_codes, icd10_codes, other_codes = [], [], [], []
            for sc in c.source_codes:
                item = {
                    "code": sc.code,
                    "source_vocab": sc.source_vocab,
                    "term_type": sc.term_type,
                    "string": sc.string,
                    "is_preferred": sc.is_preferred,
                }
                if sc.source_vocab in ("LNC", "LOINC"):
                    loinc_codes.append(item)
                elif sc.source_vocab in ("SNOMEDCT_US", "SNOMEDCT"):
                    snomed_codes.append(item)
                elif sc.source_vocab in ("ICD10CM", "ICD10"):
                    icd10_codes.append(item)
                else:
                    other_codes.append(item)

            matched.append({
                "cui": cui,
                "preferred_term": c.preferred_term,
                "synonyms": c.synonyms[:10],
                "semantic_types": c.semantic_types,
                "loinc_codes": loinc_codes,
                "snomed_codes": snomed_codes,
                "icd10_codes": icd10_codes,
                "other_codes": other_codes,
                "total_codes": len(c.source_codes),
            })

        matched.sort(key=lambda x: len(x["loinc_codes"]) + len(x["snomed_codes"]) + len(x["icd10_codes"]), reverse=True)
        best = matched[0] if matched else None
        return LabMappingResult(query_term=lab_name, matched_concepts=matched, best_match=best)

    def search_by_term(self, term: str) -> List[Dict[str, Any]]:
        return self.map_lab_name_to_codes(term).matched_concepts

    def get_concept(self, cui: str) -> Optional[Dict[str, Any]]:
        c = self.concepts.get(cui)
        if not c:
            return None
        return {
            "cui": c.cui,
            "preferred_term": c.preferred_term,
            "synonyms": c.synonyms,
            "semantic_types": c.semantic_types,
            "source_vocabularies": c.source_vocabularies,
            "definitions": c.definitions,
            "source_codes": [sc.model_dump() for sc in c.source_codes],
            "loinc_codes": c.loinc_codes,
            "snomed_codes": c.snomed_codes,
            "icd10_codes": c.icd10_codes,
        }
