
"""
ingest_pubmed_docs.py

Ingest PubMed-like documents into Elasticsearch using the Bulk API.

What this script does:
1) Connects to Elasticsearch.
2) Loads a JSON corpus (a list of dict records).
3) For each record, builds a clean document with fields.
4) Sends documents in batches via helpers.bulk() for fast indexing.
5) Refreshes the index and prints the final document count.

In this step, we store the documents in searching engine for later retrieval,
analyze the data according to the fields defined in the ES mapping,
and construct inverted indices (from words to docs) for fast retrieval.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Generator, Iterable, Tuple

from elasticsearch import Elasticsearch, helpers




ES_URL = "http://localhost:9200"
INDEX = "pubdocs"
IN_PATH = Path("data/pubmed_document.json")

# Optional cap for quick experiments / debugging.
# Set LIMIT = None to ingest everything.
LIMIT = 100000

# Bulk tuning:
# - chunk_size: number of docs per bulk request
# - request_timeout: seconds to wait for ES to finish the request
CHUNK_SIZE = 500
REQUEST_TIMEOUT = 120




def safe_str(x) -> str:
    """
    Convert a value into a safe string for Elasticsearch.
    """
    if x is None:
        return ""

    # Handle float NaN (math.isnan only works reliably on floats)
    if isinstance(x, float) and math.isnan(x):
        return ""

    # Generic NaN check (NaN != NaN is True)
    try:
        if x != x:  # catches NaN-like values
            return ""
    except Exception:
        pass

    return str(x)


# Which keys to include when constructing the long "pub_text" field.
# This is useful when you want one single field to retrieve against (BM25),
# while still keeping specific fields (e.g., interventions) for targeted queries.
PUB_KEYS = [
    "study_source",
    "population",
    "interventions",
    "outcomes",
    "group_type",
    "primary_endpoint",
    "primary_endpoint_domain",
    "primary_endpoint_subdomain",
]


def build_pub_text(record: Dict) -> str:
    """
    Build a concatenated text field from selected keys.

    Output format example:
      "study_source: PubMed | population: Adult ... | interventions: ..."

    Why do this:
    - You can search a single field (pub_text) for general retrieval.
    - You preserve semantics with key labels.
    """
    parts = []
    for k in PUB_KEYS:
        parts.append(f"{k}: {safe_str(record.get(k, ''))}")
    return " | ".join(parts)



# Bulk action generator
def iter_bulk_actions(docs: Iterable[Dict]) -> Generator[Dict, None, None]:
    """
    Yield Bulk API action dicts one by one.

    helpers.bulk(...) consumes an iterator/generator of action dicts.

    Important Bulk fields:
    - _index: target index name
    - _id: document ID (we set it to doc_id so ingest is idempotent)
    - _source: the JSON document to be indexed

    Why set _id explicitly:
    - If you rerun the ingest, docs will be overwritten instead of duplicated.
    - You can update/reindex deterministically.
    """
    for d in docs:
        # NOTE: in your earlier ES mapping you used "doc_id".
        # Make sure this matches your corpus. Some datasets use "doc_id" or "pubmed_id".
        doc_id = safe_str(d.get("study_id", "")).strip()  # <- confirm your ID field
        if not doc_id:
            # Skip records without a valid ID (cannot index without _id)
            continue

        pub_text = build_pub_text(d)
        # here we get the inverted index for BM25 retrieval after tokenization, standardization, etc.
        yield {
            "_index": INDEX,
            "_id": doc_id,
            "_source": { # initial information in json format
                "doc_id": doc_id,
                "interventions": safe_str(d.get("interventions", "")),
                "primary_endpoint": safe_str(d.get("primary_endpoint", "")),
                "pub_text": pub_text,
            },
        }




def main() -> None:
    # 1) Connect to Elasticsearch
    # If your ES requires auth, you'd pass basic_auth=("user","pass") or api_key=...
    es = Elasticsearch(ES_URL)

    # Quick connectivity check (optional but helpful for beginners)
    if not es.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {ES_URL}")

    # 2) Load JSON corpus
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    with open(IN_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # Optional cap for faster debugging
    if LIMIT is not None:
        docs = docs[:LIMIT]

    print(f"[INFO] Loaded {len(docs)} publication records for ingest.")

    # 3) Bulk ingest
    # We use es.options(request_timeout=...) to avoid timeouts on large batches.
    # raise_on_error=False: don't throw immediately; collect errors for inspection.
    # raise_on_exception=False: don't stop indexing if a subset fails.
    success, errors = helpers.bulk(
        es.options(request_timeout=REQUEST_TIMEOUT),
        iter_bulk_actions(docs),
        chunk_size=CHUNK_SIZE,
        raise_on_error=False,
        raise_on_exception=False,
    )

    print(f"[INFO] Bulk ingest finished. success={success}, errors={len(errors)}")
    if errors:
        # Print only the first error to keep logs readable
        print("[INFO] First error example:")
        print(errors[0])

    # 4) Refresh so that documents become searchable immediately.
    # Without refresh, ES may still be indexing and count/search may not show latest docs yet.
    es.indices.refresh(index=INDEX)

    # 5) Final count
    count = es.count(index=INDEX)["count"]
    print(f"[INFO] Done. ES index '{INDEX}' now has {count} docs.")


if __name__ == "__main__":
    main()




'''
import json
from elasticsearch import Elasticsearch, helpers
import math

ES_URL = "http://localhost:9200"
INDEX = "pubdocs"
IN_PATH = "data/pubmed_document.json"

LIMIT = 100000

# connect to Elasticsearch
es = Elasticsearch(ES_URL)

with open(IN_PATH, "r", encoding="utf-8") as f:
    docs = json.load(f)

docs = docs[:LIMIT]
print(f"Loaded {len(docs)} publication records for ingest.")

# helper function to handle None and NaN
def safe_str(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    if x != x:
        return ""
    return str(x)

# build pub_text field
PUB_KEYS = [
    "study_source", "population", "interventions", "outcomes", "group_type",
    "primary_endpoint", "primary_endpoint_domain", "primary_endpoint_subdomain"
]

def build_pub_text(d: dict) -> str:
    # format: "study_source: PubMed | population: ... | interventions: ..."
    parts = []
    for k in PUB_KEYS: 
        parts.append(f"{k}: {safe_str(d.get(k, ''))}")
    return " | ".join(parts)


def actions():
    for d in docs:
        doc_id = safe_str(d.get("study_id", "")).strip()
        if not doc_id:
            continue

        pub_text = build_pub_text(d)
        # generate ES action
        yield {
            "_index": INDEX,
            "_id": doc_id,
            "_source": {
                "doc_id": doc_id,
                "interventions": safe_str(d.get("interventions", "")),
                "primary_endpoint": safe_str(d.get("primary_endpoint", "")),
                "pub_text": pub_text
            }
        }

#helpers.bulk(es, actions(), chunk_size=500, request_timeout=120)

success, errors = helpers.bulk(
    es.options(request_timeout=120),
    actions(),
    chunk_size=500,
    raise_on_error=False,
    raise_on_exception=False
)

print("success:", success)
print("errors:", len(errors))
print("first error example:", errors[0] if errors else None)




es.indices.refresh(index=INDEX)

count = es.count(index=INDEX)["count"]
print(f"Done. ES index '{INDEX}' now has {count} docs.")
'''