# Trial–Publication Linking Pipeline

This repository implements a simple pipeline for linking clinical trials to publications using BM25 retrieval, semantic re-ranking, and recall@K evaluation.

---

## Scripts Overview

### `prepare_pubmed_corpus.py`
Converts the raw PubMed CSV file into a JSON corpus.  
**Input:** `data/pubmed_document.csv`  
**Output:** `data/pubmed_document.json`

---

### `prepare_trials.py`
Converts the raw clinical trial CSV file into a JSON format used for retrieval.  
**Input:** trial CSV file  
**Output:** `data/nct_trial.json` (or equivalent)

---

### `create_es_index.sh`
Creates the Elasticsearch index and mapping for BM25 retrieval.  
**Input:** none  
**Output:** Elasticsearch index (`pubdocs`)

---

### `ingest_pubmed_docs.py`
Ingests the publication corpus into Elasticsearch for BM25 search.  
**Input:** `data/pubmed_document.json`  
**Output:** Indexed documents in Elasticsearch

---

### `bm25_retrieval.py`
Retrieves top-K candidate publications for each trial using BM25.  
**Input:** trial JSON + Elasticsearch index  
**Output:** `data/bm25_retrieval_results.json`

---

### `rerank_semantic_search.py`
Re-ranks BM25 candidates using sentence embeddings and cosine similarity.  
**Input:** `data/bm25_retrieval_results.json`  
**Output:** `data/final_semantic_search_results.json`

---

### `evaluate_recall.py`
Computes recall@1, recall@5, and recall@10 using labeled trial–publication links.  
**Input:** re-ranked results + ground-truth link file  
**Output:** recall@K metrics printed to console

---

