# compare_models_eval.py
# ------------------------------------------------------------
# Compare multiple SentenceTransformer models on the SAME pipeline:
#  - Trials from nct_trial.csv
#  - Docs from pubmed_document.csv
#  - Ground truth from dataset_test.csv (label==1)
#  - Stage 1: BM25 get_top_n (same candidates for all models)
#  - Stage 2: Bi-encoder rerank (cosine similarity)
#  - Metrics: Recall@1/5/10, MRR
#
# Usage:
# python compare_models_eval.py \
#   --trial_path data/nct_trial.csv \
#   --doc_path data/pubmed_document.csv \
#   --test_path data/dataset_test.csv \
#   --models pritamdeka/S-PubMedBert-MS-MARCO /content/drive/MyDrive/models/pubmedbert_final \
#   --bm25_top_k 50 \
#   --batch_size 64 \
#   --max_queries 0
# ------------------------------------------------------------

import argparse
import string
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

import json
import os

BM25_CACHE_PATH = "bm25_candidates_top50.json"


def build_text(df: pd.DataFrame, id_col: str):
    """Merge all columns except id_col and study_source into one text field (baseline style)."""
    cols = [c for c in df.columns if c not in {id_col, "study_source"}]
    if not cols:
        raise ValueError(f"No text columns found besides {id_col}. Columns: {df.columns.tolist()}")
    combined = df[cols[0]].astype(str)
    for col in cols[1:]:
        combined += " " + df[col].astype(str)
    return combined.fillna("")


def tokenize(text: str):
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()


def recall_hit_at_k(ranked_ids, gt_set, k):
    return any(doc_id in gt_set for doc_id in ranked_ids[:k])


def compute_metrics(ranked_map, ground_truth_map):
    ks = [1, 5, 10]
    hits = {k: 0 for k in ks}
    mrr_sum = 0.0
    total = 0

    for nct_id, ranked_ids in ranked_map.items():
        gt = ground_truth_map.get(nct_id, set())
        if not gt:
            continue
        total += 1

        for k in ks:
            if recall_hit_at_k(ranked_ids, gt, k):
                hits[k] += 1

        rr = 0.0
        for r, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in gt:
                rr = 1.0 / r
                break
        mrr_sum += rr

    if total == 0:
        return {"total": 0, "r@1": 0.0, "r@5": 0.0, "r@10": 0.0, "mrr": 0.0}

    return {
        "total": total,
        "r@1": hits[1] / total,
        "r@5": hits[5] / total,
        "r@10": hits[10] / total,
        "mrr": mrr_sum / total,
    }


def rerank(model: SentenceTransformer, query_text: str, cand_texts: list[str], cand_ids: list[str], batch_size: int):
    if not cand_ids:
        return []
    q_emb = model.encode(query_text, convert_to_tensor=True)
    d_emb = model.encode(cand_texts, batch_size=batch_size, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, d_emb)[0]  # [num_cands]
    order = torch.argsort(scores, descending=True).tolist()
    return [cand_ids[i] for i in order]











def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial_path", required=True)
    ap.add_argument("--doc_path", required=True)
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="List of model names/paths to compare")
    ap.add_argument("--bm25_top_k", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_queries", type=int, default=0, help="0 = all queries")
    ap.add_argument("--device", default="", help="cuda / cpu / empty=auto")

    ap.add_argument("--cache_path", default="", help="Path to cache BM25 candidates (pickle).")
    ap.add_argument("--rebuild_cache", action="store_true", help="Force rebuilding BM25 candidate cache.")

    args = ap.parse_args()

    print(">>> Loading data...")
    trials = pd.read_csv(args.trial_path, dtype={"study_id": str}).fillna("")
    docs = pd.read_csv(args.doc_path, dtype={"study_id": str}).fillna("")
    test = pd.read_csv(args.test_path, dtype={"nct_id": str, "pubmed_id": str}).fillna("")

    if "label" not in test.columns:
        raise ValueError("dataset_test.csv must contain column: label")
    test["label"] = test["label"].astype(float)

    print(">>> Building texts (baseline style)...")
    trials["text"] = build_text(trials, id_col="study_id")
    docs["text"] = build_text(docs, id_col="study_id")

    trial_text_map = pd.Series(trials["text"].values, index=trials["study_id"]).to_dict()
    doc_id_to_text = pd.Series(docs["text"].values, index=docs["study_id"]).to_dict()

    # Ground truth map: only label==1
    gt_pos = test[test["label"] == 1].copy()
    ground_truth_map = defaultdict(set)
    for row in gt_pos.itertuples(index=False):
        ground_truth_map[str(row.nct_id)].add(str(row.pubmed_id))

    query_ids = sorted(list(ground_truth_map.keys()))
    if args.max_queries and args.max_queries > 0:
        query_ids = query_ids[: args.max_queries]

    print(f">>> Queries (with positives): {len(query_ids)}")
    print(f">>> Doc corpus size: {len(docs)}")
    print(f">>> BM25 top_k: {args.bm25_top_k}")

    print(">>> Building BM25 index...")
    doc_ids = docs["study_id"].tolist()
    doc_texts = docs["text"].tolist()
    tokenized_corpus = [tokenize(t) for t in doc_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    print("    BM25 ready.")

    # IMPORTANT: use get_top_n to retrieve doc IDs directly
    # rank_bm25.get_top_n returns items from the 'documents' list you pass in.
    # We pass doc_ids so output is doc_id list.


    print(">>> Precomputing BM25 candidates once (shared across all models)...")
    candidates_map = {}  # nct_id -> (cand_ids, cand_texts)
    skipped = 0

    for nct_id in tqdm(query_ids, desc="BM25"):
        q_text = trial_text_map.get(nct_id, "")
        if not q_text:
            skipped += 1
            continue
        q_tok = tokenize(q_text)

        cand_ids = bm25.get_top_n(q_tok, doc_ids, n=args.bm25_top_k)
        # fetch texts (same order)
        cand_texts = [doc_id_to_text.get(cid, "") for cid in cand_ids]
        candidates_map[nct_id] = (cand_ids, cand_texts)

    if skipped:
        print(f"[WARN] Skipped {skipped} queries with missing trial text.")








    # Device
    device = args.device.strip()
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Using device: {device}")

    results = []

    for model_path in args.models:
        print(f"\n>>> Loading model: {model_path}")
        model = SentenceTransformer(model_path, device=device)

        ranked_map = {}
        for nct_id in tqdm(candidates_map.keys(), desc=f"Rerank[{model_path}]"):
            q_text = trial_text_map[nct_id]
            cand_ids, cand_texts = candidates_map[nct_id]
            ranked_ids = rerank(model, q_text, cand_texts, cand_ids, batch_size=args.batch_size)
            ranked_map[nct_id] = ranked_ids

        metrics = compute_metrics(ranked_map, ground_truth_map)
        results.append((model_path, metrics))

    # Pretty print comparison
    print("\n==================== RESULTS ====================")
    print(f"Evaluated queries: {results[0][1]['total'] if results else 0}")
    print(f"BM25 top_k: {args.bm25_top_k}")
    print("-------------------------------------------------")
    print(f"{'Model':60s}  R@1     R@5     R@10    MRR")
    print("-------------------------------------------------")
    for model_path, m in results:
        name = (model_path[:57] + "...") if len(model_path) > 60 else model_path
        print(f"{name:60s}  {m['r@1']:.4f}  {m['r@5']:.4f}  {m['r@10']:.4f}  {m['mrr']:.4f}")
    print("=================================================\n")


if __name__ == "__main__":
    main()
