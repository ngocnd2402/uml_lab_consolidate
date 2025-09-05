from __future__ import annotations

import os
import sys
import json
import time
import gzip
import itertools
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import requests

from dotenv import load_dotenv
load_dotenv()


# ============================ Config & Logging ============================

logger = logging.getLogger("ingest_ndjson")
handler = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ============================ Helpers ============================

def iter_ndjson_lines(path: Path) -> Iterator[str]:
    """
    Stream NDJSON lines without stripping content (except trailing newline).
    Skip completely empty lines to avoid sending blank actions to _bulk.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                continue
            yield line.rstrip("\n")


def chunk_lines_by_count(lines: Iterator[str], lines_per_chunk: int) -> Iterator[List[str]]:
    """
    Group lines by a fixed count. The _bulk API expects action+source pairs,
    but since NDJSON may contain different actions (index, create, delete, etc.),
    we batch purely by number of lines and do not inspect content.
    """
    while True:
        batch = list(itertools.islice(lines, lines_per_chunk))
        if not batch:
            break
        yield batch


def _extract_first_error(data: dict) -> Optional[dict]:
    """
    Extract the first item-level error (if any) from a bulk response.
    """
    items = data.get("items", [])
    for item in items:
        # item is like {"index": {...}} or {"create": {...}} etc.
        for _, content in item.items():
            if isinstance(content, dict) and content.get("error"):
                return content["error"]
    return None


def post_bulk(
    session: requests.Session,
    base_url: str,
    ndjson_payload: bytes,
    *,
    gzip_enabled: bool = False,
    timeout: int = 60,
) -> Tuple[bool, dict]:
    """
    Call OpenSearch _bulk and return (ok, response_json).
    If gzip_enabled=True, set header Content-Encoding: gzip.
    """
    url = base_url.rstrip("/") + "/_bulk"
    headers = {"Content-Type": "application/x-ndjson"}
    if gzip_enabled:
        headers["Content-Encoding"] = "gzip"

    try:
        resp = session.post(url, data=ndjson_payload, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        return False, {"_status_code": -1, "_error": f"request_exception: {e.__class__.__name__}: {e}"}

    try:
        data = resp.json()
    except Exception:
        data = {"_status_code": resp.status_code, "_text": resp.text[:1000]}

    # Success for bulk is HTTP 200 and "errors": false
    ok = (resp.status_code == 200) and (isinstance(data, dict)) and (data.get("errors") is False)
    if not ok and isinstance(data, dict) and "errors" in data:
        # Attach a concise first error to help debugging
        first_err = _extract_first_error(data)
        if first_err and "_first_error" not in data:
            data["_first_error"] = first_err

    return ok, data


def backoff_sleep(attempt: int, base: float = 1.5, cap: float = 30.0) -> None:
    """
    Simple exponential backoff with a cap.
    attempt: 0,1,2,...  -> delays: base^(attempt+1) up to cap
    """
    delay = min(cap, base ** (attempt + 1))
    time.sleep(delay)


def ensure_index(
    session: requests.Session,
    base_url: str,
    index_name: str,
    mapping_path: Optional[Path] = None,
    timeout: int = 30,
) -> None:
    """
    Create an index if it does not exist. If mapping_path is provided and exists,
    it will be used as the creation body. Safe to call if index already exists.
    """
    idx_url = base_url.rstrip("/") + f"/{index_name}"
    r = session.get(idx_url, timeout=timeout)
    if r.status_code == 200:
        logger.info(f"[index] '{index_name}' already exists — skipping creation.")
        return

    body = None
    if mapping_path and mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8") as f:
            body = json.load(f)

    logger.info(f"[index] creating '{index_name}'" + (f" with mapping '{mapping_path.name}'" if body else ""))

    r = session.put(idx_url, json=body, timeout=timeout)
    if r.status_code not in (200, 201):
        try:
            j = r.json()
        except Exception:
            j = r.text
        raise RuntimeError(f"Create index failed ({r.status_code}): {j}")
    logger.info(f"[index] OK: {index_name}")


def sniff_first_index_from_ndjson(path: Path, max_lines: int = 1000) -> Optional[str]:
    """
    Try to infer the index name from action lines ({"index": {...}} / {"create": {...}}).
    Returns the first index name found, or None if not present.
    """
    for i, line in enumerate(iter_ndjson_lines(path)):
        if i >= max_lines:
            break
        if not line.startswith('{"index"') and not line.startswith('{"create"'):
            continue
        try:
            obj = json.loads(line)
            if "index" in obj and isinstance(obj["index"], dict):
                idx = obj["index"].get("_index")
                if idx:
                    return str(idx)
            if "create" in obj and isinstance(obj["create"], dict):
                idx = obj["create"].get("_index")
                if idx:
                    return str(idx)
        except Exception:
            continue
    return None


# ============================ Main ingest ============================

def ingest_ndjson(
    ndjson_path: Path,
    *,
    base_url: str,
    username: str,
    password: str,
    verify_ssl: bool = True,
    lines_per_chunk: int = 10_000,
    max_retries: int = 5,
    gzip_enabled: bool = False,
    mapping_path: Optional[Path] = None,
    create_index_if_missing: bool = False,
    explicit_index: Optional[str] = None,
) -> None:
    """
    Stream an NDJSON file into OpenSearch via _bulk.

    Behavior:
    - If create_index_if_missing=True: create the target index first.
      - Priority for target index: explicit_index, otherwise sniff from NDJSON.
    - If mapping_path is provided, it is used when creating the index.
    - Batches are formed by a fixed number of lines (no content inspection).
    """
    session = requests.Session()
    session.auth = (username, password)
    session.verify = verify_ssl

    # Create index if requested
    if create_index_if_missing:
        index_to_create = explicit_index
        if not index_to_create:
            index_to_create = sniff_first_index_from_ndjson(ndjson_path)
            if not index_to_create:
                logger.warning("[index] No _index found in NDJSON action lines; skipping index creation.")
        if index_to_create:
            ensure_index(session, base_url, index_to_create, mapping_path=mapping_path)

    total_lines = 0
    total_items = 0  # approximate number of action items returned by the server
    start_time = time.time()

    for batch_idx, lines in enumerate(chunk_lines_by_count(iter_ndjson_lines(ndjson_path), lines_per_chunk), start=1):
        # Assemble payload
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        if gzip_enabled:
            payload = gzip.compress(payload)

        # Retry with backoff for transient failures
        attempt = 0
        while True:
            ok, data = post_bulk(session, base_url, payload, gzip_enabled=gzip_enabled)
            if ok:
                batch_lines = len(lines)
                total_lines += batch_lines

                items = data.get("items", []) if isinstance(data, dict) else []
                total_items += len(items)

                if batch_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[bulk] batch={batch_idx} OK — lines={batch_lines:,} "
                        f"total_lines={total_lines:,} items={len(items):,} elapsed={elapsed:.1f}s"
                    )
                break

            # Not ok → prepare diagnostics and retry if allowed
            status_code = data.get("_status_code") if isinstance(data, dict) else None
            first_err = data.get("_first_error") if isinstance(data, dict) else None
            logger.warning(
                f"[bulk][attempt={attempt}] NOT OK status={status_code} "
                f"first_error={first_err or data}"
            )
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Bulk failed after {max_retries} retries. "
                    f"Last status={status_code}, error={first_err or data}"
                )

            backoff_sleep(attempt)
            attempt += 1

    elapsed = time.time() - start_time
    logger.info(f"[done] total_lines={total_lines:,} approx_actions={total_items:,} in {elapsed:.1f}s")


# ============================ CLI Entry ============================

if __name__ == "__main__":
    # --- Configure here ---
    NDJSON_PATH = Path("dev/exports/umls_labs.ndjson")

    # (optional) Index mapping to create beforehand
    MAPPING_PATH: Optional[Path] = Path("dev/exports/umls_labs_mapping.json")
    CREATE_INDEX_IF_MISSING = True
    EXPLICIT_INDEX: Optional[str] = "umls_labs"   # or None to sniff from action lines

    # Batch & options
    LINES_PER_CHUNK = 10_000
    GZIP_ENABLED = False
    VERIFY_SSL = True

    # Credentials from environment (.env or real env)
    BASE_URL = os.getenv("OPENSEARCH_URL")
    USER = os.getenv("OPENSEARCH_USER")
    PASSWORD = os.getenv("OPENSEARCH_PASSWORD")

    if not NDJSON_PATH.exists():
        logger.error(f"NDJSON file does not exist: {NDJSON_PATH}")
        sys.exit(2)

    if MAPPING_PATH and not MAPPING_PATH.exists():
        logger.error(f"Mapping JSON does not exist: {MAPPING_PATH}")
        sys.exit(2)

    if not BASE_URL or not USER or not PASSWORD:
        logger.error("Missing OpenSearch credentials. Please set OPENSEARCH_URL, OPENSEARCH_USER, OPENSEARCH_PASSWORD.")
        sys.exit(2)

    ingest_ndjson(
        ndjson_path=NDJSON_PATH,
        base_url=BASE_URL,
        username=USER,
        password=PASSWORD,
        verify_ssl=VERIFY_SSL,
        lines_per_chunk=LINES_PER_CHUNK,
        max_retries=5,
        gzip_enabled=GZIP_ENABLED,
        mapping_path=MAPPING_PATH,
        create_index_if_missing=CREATE_INDEX_IF_MISSING,
        explicit_index=EXPLICIT_INDEX,
    )
