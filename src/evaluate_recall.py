
"""
evaluate_recall.py

Evaluate retrieval/reranking results using Recall@K on a trial -> PubMed linking task.

Definition used in this script:
- For each trial (study_id / nct_id), we have a set of gold PubMed IDs (label == 1).
- We consider the trial a "hit" at K if ANY gold PubMed ID appears in the top-K ranked candidates.
- Recall@K = (#hit trials) / (#trials that have at least one gold PubMed)

This is sometimes called:
- "hit rate@K" per query
- "recall@K (query-level)" in information retrieval
"""

import csv
import json
from collections import defaultdict
from pathlib import Path




RESULTS_JSON = Path("data/final_semantic_search_results.json")
PAIR_CSV = Path("data/dataset_train.csv")

KS = [1, 5, 10]




with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    rerank_results = json.load(f)

print(f"[INFO] Loaded reranking results: {len(rerank_results)} trials/queries")




gt = defaultdict(set)

with open(PAIR_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row.get("label")

        # Skip rows where label is missing/empty (common in weakly labeled data)
        if label is None:
            continue
        label = label.strip()
        if label == "":
            continue

        # Only keep positive pairs (label == "1")
        if label == "1":
            trial_id = row["nct_id"].strip()
            pub_id = row["pubmed_id"].strip()
            gt[trial_id].add(pub_id)

print(f"[INFO] Built ground truth for {len(gt)} trials (label==1)")




def recall_at_k(results, ground_truth, k: int) -> float:
    """
    Compute recall@k using the per-trial hit definition:

    For each trial with at least one gold PubMed ID:
      hit(trial) = 1 if any gold PubMed ID is in top-k ranked candidates
                = 0 otherwise

    recall@k = sum(hit(trial)) / (#trials with gold)

    Notes:
    - This metric does NOT count how many gold documents are retrieved.
      It only checks whether we retrieved at least one correct doc in top-k.
    - This is the right metric if your downstream task only needs one correct link.
    """
    hits = 0
    total = 0

    for entry in results:
        trial_id = entry.get("study_id")
        if trial_id not in ground_truth:
            # No gold labels for this trial -> skip (recall would be undefined)
            continue

        gold_set = ground_truth[trial_id]
        if not gold_set:
            continue

        # Extract top-k pubmed_ids from reranked candidates.
        # Assumes your result JSON structure:
        # { "study_id": ..., "reranked_candidates": [ {"pubmed_id": ...}, ... ] }
        ranked_pubmed_ids = [
            c["pubmed_id"] for c in entry["reranked_candidates"][:k]
        ]

        hit = any(pid in gold_set for pid in ranked_pubmed_ids)
        hits += int(hit)
        total += 1

    return hits / total if total > 0 else 0.0




for k in KS:
    r = recall_at_k(rerank_results, gt, k)
    print(f"recall@{k}: {r:.4f}")




'''
import json
from collections import defaultdict

with open("data/final_semantic_search_results.json", "r", encoding="utf-8") as f:
    rerank_results = json.load(f)


import csv



import pandas as pd

df = pd.read_csv("data/dataset_train.csv")

print(df["label"].value_counts(dropna=False))




gt = defaultdict(set)  # trial_id -> set(pubmed_id)

with open("data/dataset_train.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row.get("label")

        # 跳过 label 为空 / None / 空字符串 的行
        if label is None:
            continue

        label = label.strip()
        if label == "":
            continue

        # 只保留 label == 1
        if label == "1":
            trial_id = row["nct_id"]
            pub_id = row["pubmed_id"]
            gt[trial_id].add(pub_id)


def recall_at_k(rerank_results, gt, k):
    hits = 0
    total = 0

    for entry in rerank_results:
        trial_id = entry["study_id"]
        if trial_id not in gt:
            continue  # 没有 gold，跳过（否则 recall 没意义）

        gold = gt[trial_id]
        if not gold:
            continue
        
        # get top-k ranked pubmed_ids
        ranked = [
            c["pubmed_id"]
            for c in entry["reranked_candidates"][:k]
        ]
        # check if any of the gold pubmed_ids is in the top-k ranked list
        hit = any(pid in gold for pid in ranked)
        hits += int(hit)
        total += 1

    return hits / total if total > 0 else 0.0


for k in [1, 5, 10]:
    r = recall_at_k(rerank_results, gt, k)
    print(f"recall@{k}: {r:.4f}")


'''