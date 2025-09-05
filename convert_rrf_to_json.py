import json
from pathlib import Path
from typing import Dict, List, Any
from umls_lab_mapper import UMLSLabMapper, Concept


# ====================== SAB names ======================
def _load_sab_names(meta_dir: Path) -> Dict[str, str]:
    """
    Trả về map RSAB -> SAB_NAME (SON). Heuristic phổ biến MRSAB: row[3]=RSAB, row[4]=SON.
    """
    defaults = {
        "LNC": "LOINC",
        "LOINC": "LOINC",
        "SNOMEDCT_US": "SNOMED CT US Edition",
        "SNOMEDCT": "SNOMED CT",
        "ICD10CM": "ICD-10-CM",
        "ICD10": "ICD-10",
        "ICD10PCS": "ICD-10-PCS",
        "CCSR_ICD10CM": "CCSR (ICD-10-CM)",
        "CCSR_ICD10PCS": "CCSR (ICD-10-PCS)",
        "RXNORM": "RxNorm",
        "MSH": "MeSH",
    }
    m = dict(defaults)
    p = meta_dir / "MRSAB.RRF"
    if not p.exists():
        return m

    import csv
    def _read_rrf_rows(path: Path):
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
            for row in reader:
                if not row:
                    continue
                if row and row[-1] == "":
                    row = row[:-1]
                yield row

    for row in _read_rrf_rows(p):
        if len(row) >= 5:
            rsab = row[3].strip()
            son = row[4].strip()
            if rsab and son:
                m[rsab] = son
    return m


# ====================== Bucket helpers (no pattern) ======================
def _is_loinc(sab: str) -> bool:
    return sab in ("LNC", "LOINC")

def _is_snomed(sab: str) -> bool:
    return sab in ("SNOMEDCT_US", "SNOMEDCT")

def _is_icd10_group(sab: str) -> bool:
    return sab in ("CCSR_ICD10CM", "CCSR_ICD10PCS", "ICD10PCS")


# ====================== Build doc per CUI ======================
def build_opensearch_doc(
    sab_name_map: Dict[str, str],
    concept: Concept,
    *,
    max_synonyms: int = 60
) -> Dict[str, Any]:
    loinc_codes: List[Dict[str, Any]] = []
    snomed_codes: List[Dict[str, Any]] = []
    icd10_codes: List[Dict[str, Any]] = []
    other_codes: List[Dict[str, Any]] = []

    for sc in concept.source_codes:
        sab = sc.source_vocab
        sab_name = sab_name_map.get(sab, sab)

        item = {
            "sab": sab,
            "sab_name": sab_name,
            "code": sc.code or "",
            "term_type": sc.term_type,
            "string": sc.string,
            "is_preferred": sc.is_preferred,
        }

        if _is_loinc(sab):
            loinc_codes.append(item)
        elif _is_snomed(sab):
            snomed_codes.append(item)
        elif _is_icd10_group(sab):
            icd10_codes.append(item)
        else:
            # Lưu ý: ICD10 / ICD10CM (diagnosis) theo yêu cầu mới vẫn vào other_codes
            other_codes.append(item)

    syns = concept.synonyms[:max_synonyms]

    return {
        "cui": concept.cui,
        "preferred_term": concept.preferred_term,
        "synonyms": syns,
        "semantic_types": concept.semantic_types,
        "definitions": concept.definitions,
        "source_vocabularies": concept.source_vocabularies,
        "loinc_codes": loinc_codes,
        "snomed_codes": snomed_codes,
        "icd10_codes": icd10_codes,
        "other_codes": other_codes,
        "search_text": " ".join([concept.preferred_term or ""] + syns).strip(),
    }


def export_concepts_to_ndjson(
    meta_dir: Path,
    concepts: Dict[str, Concept],
    out_path: Path,
    index_name: str,
    *,
    labs_only: bool = False,
    max_synonyms: int = 60,
    min_nonempty_fields: int = 0
) -> None:
    """
    Xuất NDJSON theo định dạng OpenSearch _bulk:
      {"index":{"_index":"<index>","_id":"<CUI>"}}\n
      {document}\n
    """
    sab_name_map = _load_sab_names(meta_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _is_lab(con: Concept) -> bool:
        lab_types = {
            "Laboratory Procedure", "Diagnostic Procedure", "Laboratory or Test Result",
            "Clinical Finding", "Finding", "Quantitative Concept", "Substance"
        }
        return any(st in lab_types for st in con.semantic_types)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for cui, con in concepts.items():
            if labs_only and not _is_lab(con):
                continue

            doc = build_opensearch_doc(
                sab_name_map=sab_name_map,
                concept=con,
                max_synonyms=max_synonyms,
            )

            if min_nonempty_fields > 0:
                score = sum(bool(doc[k]) for k in (
                    "preferred_term", "synonyms", "loinc_codes", "snomed_codes", "icd10_codes"
                ))
                if score < min_nonempty_fields:
                    continue

            meta = {"index": {"_index": index_name, "_id": cui}}
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1

    print(f"[OpenSearch] wrote {written:,} documents → {out_path}")


# ====================== (Optional) mapping gợi ý ======================
OPENSEARCH_INDEX_MAPPING: Dict[str, Any] = {
    "settings": {
        "analysis": {
            "analyzer": {
                "lab_text_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"]
                }
            }
        }
    },
    "mappings": {
        "dynamic": "true",
        "properties": {
            "cui": {"type": "keyword"},
            "preferred_term": {"type": "text", "analyzer": "lab_text_analyzer"},
            "synonyms": {"type": "text", "analyzer": "lab_text_analyzer"},
            "semantic_types": {"type": "keyword"},
            "definitions": {"type": "text", "analyzer": "lab_text_analyzer"},
            "source_vocabularies": {"type": "keyword"},
            "loinc_codes": {
                "type": "nested",
                "properties": {
                    "sab": {"type": "keyword"},
                    "sab_name": {"type": "keyword"},
                    "code": {"type": "keyword"},
                    "term_type": {"type": "keyword"},
                    "string": {"type": "text", "analyzer": "lab_text_analyzer"},
                    "is_preferred": {"type": "boolean"}
                }
            },
            "snomed_codes": {
                "type": "nested",
                "properties": {
                    "sab": {"type": "keyword"},
                    "sab_name": {"type": "keyword"},
                    "code": {"type": "keyword"},
                    "term_type": {"type": "keyword"},
                    "string": {"type": "text", "analyzer": "lab_text_analyzer"},
                    "is_preferred": {"type": "boolean"}
                }
            },
            "icd10_codes": {
                "type": "nested",
                "properties": {
                    "sab": {"type": "keyword"},
                    "sab_name": {"type": "keyword"},
                    "code": {"type": "keyword"},
                    "term_type": {"type": "keyword"},
                    "string": {"type": "text", "analyzer": "lab_text_analyzer"},
                    "is_preferred": {"type": "boolean"}
                }
            },
            "other_codes": {
                "type": "nested",
                "properties": {
                    "sab": {"type": "keyword"},
                    "sab_name": {"type": "keyword"},
                    "code": {"type": "keyword"},
                    "term_type": {"type": "keyword"},
                    "string": {"type": "text", "analyzer": "lab_text_analyzer"},
                    "is_preferred": {"type": "boolean"}
                }
            },
            "search_text": {"type": "text", "analyzer": "lab_text_analyzer"}
        }
    }
}


# ====================== CLI ======================
if __name__ == "__main__":
    # Paths (chỉnh cho phù hợp)
    META_DIR = Path(r"D:\Work\Out\wrs\lab_data_proj\dev\2025AA\META")
    OUT_PATH = Path(r"D:\Work\Out\wrs\lab_data_proj\dev\exports\umls_labs.ndjson")
    INDEX_NAME = "umls_labs"

    mapper = UMLSLabMapper(str(META_DIR))
    mapper.parse_mrconso(limit=None)   # gom đủ code
    mapper.parse_mrsty(limit=None)
    mapper.parse_mrdef(limit=100_000)

    export_concepts_to_ndjson(
        meta_dir=META_DIR,
        concepts=mapper.concepts,
        out_path=OUT_PATH,
        index_name=INDEX_NAME,
        labs_only=True,
        max_synonyms=60,
        min_nonempty_fields=0,
    )

    mapping_path = OUT_PATH.parent / f"{INDEX_NAME}_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(OPENSEARCH_INDEX_MAPPING, f, ensure_ascii=False, indent=2)
    print(f"[OpenSearch] mapping saved → {mapping_path}")
