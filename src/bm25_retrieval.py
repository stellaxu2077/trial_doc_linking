
from rank_bm25 import BM25Okapi
import json

# corpus of documents
# simulated mock PubMed articles
'''
mock_pubmed = [
    {
        "pmid": "202401", 
        "title": "Efficacy of Pembrolizumab in Melanoma Patients", 
        "abstract": "This study reports the results of a phase 3 trial...",
        "authors": "Evan Pan; Kirk Roberts" 
    },
    {
        "pmid": "202402", 
        "title": "Radiation Therapy for Lung Cancer", 
        "abstract": "A comprehensive review of radiation techniques...",
        "authors": "Smith, J"
    },
    {
        "pmid": "202403", 
        "title": "New targeted therapy for Breast Cancer", 
        "abstract": "Results from the NCT000123 trial show that...",
        "authors": "Jane Doe"
    }
]
'''
'''
import random


with open("data/nct_trial.json", "r", encoding="utf-8") as f:
    trials = json.load(f)

with open("data/pubmed_document.json", "r", encoding="utf-8") as f:
    pubmed_docs = json.load(f)



corpus_texts = []
for doc in pubmed_docs:
    #full_text = f"{doc['title']} {doc.get('abstract', '')}"
    full_text = f"{doc.get('interventions', '')}"
    corpus_texts.append(full_text.lower().split())

bm25 = BM25Okapi(corpus_texts)

# 3. Generate candidates for each trial and save results
final_output = []

for trial in trials:
    # 
    trial_text = f"{trial['title']}"
    query_tokens = trial_text.lower().split()
    
    # get BM25 scores and top 5 articles
    doc_scores = bm25.get_scores(query_tokens)
    top_500_pub = bm25.get_top_n(query_tokens, pubmed_docs, n=500)
    
    # construct candidates list
    candidates = []
    for article in top_500_pub:
        
        candidates.append({
            "pmid": article['pmid'],
            "article_input_text": f"{article['title']} {article.get('abstract', '')}",
            "bm25_rank_score": float(doc_scores[mock_pubmed.index(article)])
        })
    
    #
    final_output.append({
        "study_id": trial['study_id'],
        "trial_input_text": trial_text,
        "candidates": candidates
    })

# 4. Save results to JSON for embedding-based re-ranking
with open("bm25_retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=4)

print(f"Completed. Generated candidates for {len(final_output)} trials.")'''



import json
from elasticsearch import Elasticsearch

# BM25 Retrieval from Elasticsearch
# using the indexed PubMed documents
ES_URL = "http://localhost:9200"
INDEX = "pubdocs"

#TRIAL_PATH = "data/nct_trial.json"
TRIAL_PATH = "data/linked_trials.json"
OUT_PATH = "data/bm25_retrieval_results.json"

TOPK = 500
LIMIT_TRIALS = 100000

# connect to Elasticsearch
es = Elasticsearch(ES_URL)





def safe(v):
    if v is None:
        return ""
    try:
        if v != v:  # NaN
            return ""
    except Exception:
        pass
    s = str(v).strip()
    return "" if s.lower() == "nan" else s

TRIAL_KEYS = [
    "study_source", "trial_type", "title", "abstract", "sponsor", "start_year",
    "drug_moa_id", "drug_name", "drug_description",
    "adverse_event_name", "adverse_event_description",
    "intervention_type", "group_type", "intervention_name", "intervention_description"
]

def build_trial_embed_text(trial: dict) -> str:
    # 你示例是 "field: value | field: value ..."
    parts = []
    for k in TRIAL_KEYS:  # source 固定写 ClinicalTrials.gov
        parts.append(f"{k}: {safe(trial.get(k, ''))}")
    return " | ".join(parts)





# build query from trial
def build_query(trial):
    title = trial.get("title", "") or ""
    intervention = trial.get("intervention_name", "") or ""
    query_text = f"{title} {intervention}".strip()
    return query_text

def search_topk(query_text, k=TOPK):
    body = {
        "size": k,
        "_source": ["doc_id", "interventions", "primary_endpoint", "pub_text"], # what we care about
        "query": { # search the query text in these fields
            "multi_match": {
                "query": query_text,
                "fields": ["pub_text^1"], # weight can be adjusted here
                "type": "best_fields",
                "operator": "or"
            }
        }
    }
    resp = es.search(index=INDEX, body=body) 
    hits = resp["hits"]["hits"] # list of hits including _id, _score, _source, etc.
    candidates = []
    for rank, h in enumerate(hits, start=1):
        src = h.get("_source", {}) or {}
        candidates.append({
            "pubmed_id": str(src.get("doc_id", h.get("_id"))),
            "bm25_score": float(h.get("_score", 0.0)),
            "bm25_rank": rank,
            "pub_text": src.get("pub_text", "")
        })
    return candidates

# Load trials
with open(TRIAL_PATH, "r", encoding="utf-8") as f:
    trials = json.load(f)

if LIMIT_TRIALS is not None:
    trials = trials[:LIMIT_TRIALS]

print(f"Retrieving BM25 candidates for {len(trials)} trials...")

final_output = []
for i, trial in enumerate(trials, start=1):
    query_text = build_query(trial)
    if not query_text:
        candidates = []
    else:
        candidates = search_topk(query_text, TOPK)

    final_output.append({
        "study_id": trial.get("study_id"),
        "bm25_query_text": query_text,
        "trial_text": build_trial_embed_text(trial),
        "candidates": candidates
    })

    if i % 10 == 0:
        print(f"  processed {i}/{len(trials)}")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

print(f"Done. Saved to {OUT_PATH}")
