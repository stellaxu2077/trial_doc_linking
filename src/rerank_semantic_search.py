
"""
rerank_semantic_search.py

Semantic reranking for BM25 retrieval results using SentenceTransformer embeddings.

Input (JSON):
- A list of entries, each entry contains:
  - study_id: trial identifier
  - trial_text: text query for the trial
  - candidates: list of candidate articles (each should include pubmed_id and pub_text)

Output (JSON):
- A list of entries:
  - study_id
  - reranked_candidates: list of {pubmed_id, cosine_similarity} sorted descending
"""

import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util




IN_PATH = Path("data/bm25_retrieval_results.json")
OUT_PATH = Path("data/final_semantic_search_results.json")

MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
MAX_TRIALS = 100          # set None to process all entries
BATCH_SIZE = 32
TOP_K = None            # set e.g. 50 if you only want top-N after reranking






# 1) Pick device.
# - On machines with NVIDIA GPU, torch.cuda.is_available() will be True.
# - On macOS, CUDA is usually not available; CPU will be used.
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2) Load embedding model (first run may download weights and cache them locally).
model = SentenceTransformer(MODEL_NAME, device=device)
print(f"[INFO] Loaded SentenceTransformer model: {MODEL_NAME} (device={device})")

# 3) Load BM25 retrieval results
if not IN_PATH.exists():
    raise FileNotFoundError(f"Input file not found: {IN_PATH}")

with open(IN_PATH, "r", encoding="utf-8") as f:
    retrieval_data = json.load(f)

if MAX_TRIALS is not None:
    retrieval_data = retrieval_data[:MAX_TRIALS]

print(f"[INFO] Loaded {len(retrieval_data)} retrieval entries from {IN_PATH}")

final_reranked_results = []

# 4) Process each trial entry
for idx, entry in enumerate(retrieval_data, start=1):
    study_id = entry.get("study_id")
    trial_text = entry.get("trial_text", "") or ""
    candidates = entry.get("candidates", []) or []

    if not study_id:
        print(f"[WARN] Missing study_id at entry #{idx}; skipping.")
        continue

    if not candidates:
        print(f"[WARN] No candidates for study_id={study_id}; writing empty reranked_candidates.")
        final_reranked_results.append({"study_id": study_id, "reranked_candidates": []})
        continue

    # Extract candidate texts for embedding
    candidate_texts = [c.get("pub_text", "") or "" for c in candidates]

    print(f"[INFO] ({idx}/{len(retrieval_data)}) Reranking study_id={study_id} | candidates={len(candidates)}")

    # 5) Compute embeddings and cosine similarity
    # We normalize embeddings so cosine similarity is just dot-product.
    with torch.no_grad():
        trial_emb = model.encode(
            trial_text,
            convert_to_tensor=True,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
        )
        cand_emb = model.encode(
            candidate_texts,
            convert_to_tensor=True,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
        )

    # util.cos_sim returns a matrix [1, num_candidates], so we take the first row.
    cosine_scores = util.cos_sim(trial_emb, cand_emb)[0]

    # 6) Attach scores and sort candidates by similarity
    for i, c in enumerate(candidates):
        c["cosine_similarity"] = float(cosine_scores[i])

    candidates.sort(key=lambda x: x["cosine_similarity"], reverse=True)

    # Optional: keep only top-K
    if TOP_K is not None:
        candidates = candidates[:TOP_K]

    # 7) Produce compact output (only keep the fields you want)
    cleaned_candidates = [
        {
            "pubmed_id": c.get("pubmed_id"),
            "cosine_similarity": c["cosine_similarity"],
        }
        for c in candidates
        if c.get("pubmed_id") is not None
    ]

    final_reranked_results.append(
        {
            "study_id": study_id,
            "reranked_candidates": cleaned_candidates,
        }
    )

# 8) Save final results
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_reranked_results, f, ensure_ascii=False, indent=2)

print(f"[INFO] Reranking completed. Results saved to: {OUT_PATH}")



'''
import json
import torch
from sentence_transformers import SentenceTransformer, util

IN_PATH = "data/bm25_retrieval_results.json"
OUT_PATH = "data/final_semantic_search_results.json"

# 1. load pre-trained pubmedbert embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('NeuML/pubmedbert-base-embeddings', device=device)
print("Loaded SentenceTransformer model for PubMed embeddings.")

# 2. load bm25 retrieval results
with open(IN_PATH, "r", encoding="utf-8") as f:
    retrieval_data = json.load(f)
    MAX_TRIALS = 3   # 或 1 / 5 / 10
    retrieval_data = retrieval_data[:MAX_TRIALS]

    print(f"Loaded {len(retrieval_data)} retrieval entries.")

# 3. process each trial entry for re-ranking
final_reranked_results = []

for idx, entry in enumerate(retrieval_data, start=1):
    trial_text = entry.get('trial_text', "")
    candidates = entry.get('candidates', [])
    
    # get all candidate texts for embedding
    candidate_texts = [c.get('pub_text', "") for c in candidates]
    print(f"Processing study_id: {entry['study_id']} with {len(candidates)} candidates.")
    

    if idx % 10 == 0:
        print(f"Processing {idx}/{len(retrieval_data)} | study_id={entry.get('study_id')} | candidates={len(candidates)}")



    # generate embeddings
    # calculate trial vector (v_trial) 
    with torch.no_grad():
        trial_embedding = model.encode(
            trial_text, 
            convert_to_tensor=True, 
            batch_size=32, 
            normalize_embeddings=True
            )
        article_embeddings = model.encode(
            candidate_texts, 
            convert_to_tensor=True, 
            batch_size=32, 
            normalize_embeddings=True
            )

    
    # calculate cosine similarities
    # util.cos_sim will return a matrix of size [1, num_candidates]
    cosine_scores = util.cos_sim(trial_embedding, article_embeddings)[0]
    
    # 4. Add cosine similarity scores to candidates and re-rank
    for i in range(len(candidates)):

        candidates[i]['cosine_similarity'] = float(cosine_scores[i])

    # rank candidates by cosine similarity
    candidates.sort(key=lambda x: x['cosine_similarity'], reverse=True)
    
    cleaned_candidates = [
        {
            "pubmed_id": c['pubmed_id'],
            "cosine_similarity": c['cosine_similarity']
        }
        for c in candidates
    ]

    final_reranked_results.append({
        "study_id": entry['study_id'],
        "reranked_candidates": cleaned_candidates
    })

# 5. save final re-ranked results
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_reranked_results, f, ensure_ascii=False, indent=4)

print(f"Re-ranking completed and results saved to {OUT_PATH}")
'''