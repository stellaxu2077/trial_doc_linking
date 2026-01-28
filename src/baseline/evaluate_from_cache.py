# eval_models_from_cache.py
# ------------------------------------------------------------
# Evaluate multiple SentenceTransformer models using cached BM25 candidates.
#
# Assumptions:
#   - dataset_test.csv contains: nct_id, pubmed_id, trial_info, doc_info, label
#   - pubmed_document.csv is the document database, with key column: study_id (=pubmed_id)
#   - BM25 cache is built from dataset_test.csv:trial_info as queries (recommended)
#
# Usage:
# python eval_models_from_cache.py \
#   --doc_path data/pubmed_document.csv \
#   --test_path data/dataset_test.csv \
#   --cache_path bm25_candidates_top50.pkl \
#   --models pritamdeka/S-PubMedBert-MS-MARCO /content/drive/MyDrive/models/pubmedbert_final \
#   --batch_size 64 \
#   --max_queries 0 \
#   --device ""
# ------------------------------------------------------------

import argparse
import pickle
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


def build_text(df: pd.DataFrame, id_col: str):
    """Merge all columns except id_col and study_source into one text field (baseline style)."""
    cols = [c for c in df.columns if c not in {id_col, "study_source"}]
    if not cols:
        raise ValueError(f"No text columns found besides {id_col}. Columns: {df.columns.tolist()}")
    combined = df[cols[0]].astype(str)
    for col in cols[1:]:
        combined += " " + df[col].astype(str)
    return combined.fillna("")


def build_query_map_from_test(test_df: pd.DataFrame) -> dict:
    """
    Build nct_id -> trial_info map from dataset_test.csv.
    If an nct_id appears multiple times:
      - take the first non-empty trial_info
      - warn if multiple distinct non-empty trial_info exist
    """
    if "nct_id" not in test_df.columns or "trial_info" not in test_df.columns:
        raise ValueError("dataset_test.csv must contain columns: nct_id, trial_info")

    test_df = test_df.copy()
    test_df["nct_id"] = test_df["nct_id"].astype(str)
    test_df["trial_info"] = test_df["trial_info"].fillna("").astype(str)

    query_map = {}
    inconsistent = 0
    empty = 0

    for nct_id, g in test_df.groupby("nct_id"):
        vals = [v.strip() for v in g["trial_info"].tolist() if v and v.strip()]
        if not vals:
            empty += 1
            continue

        uniq = list(dict.fromkeys(vals))  # stable unique
        if len(uniq) > 1:
            inconsistent += 1
        query_map[nct_id] = uniq[0]

    if inconsistent:
        print(f"[WARN] {inconsistent} nct_id have multiple distinct non-empty trial_info; using the first one.")
    if empty:
        print(f"[WARN] {empty} nct_id have empty trial_info; these queries may be skipped.")

    return query_map


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


@torch.inference_mode()
def rerank(model: SentenceTransformer, query_text: str, cand_texts: list[str], cand_ids: list[str], batch_size: int):
    if not cand_ids:
        return []
    if not query_text or not str(query_text).strip():
        return cand_ids  # fallback: keep BM25 order if query is empty

    q_emb = model.encode(query_text, convert_to_tensor=True)
    d_emb = model.encode(cand_texts, batch_size=batch_size, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, d_emb)[0]  # [num_cands]
    order = torch.argsort(scores, descending=True).tolist()
    return [cand_ids[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc_path", required=True)
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--cache_path", required=True, help="BM25 cache pickle from build_bm25_cache_from_test.py")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_queries", type=int, default=0, help="0 = use cache as-is; >0 = truncate")
    ap.add_argument("--device", default="", help="cuda / cpu / empty=auto")
    args = ap.parse_args()

    print(">>> Loading cache...")
    with open(args.cache_path, "rb") as f:
        cache = pickle.load(f)
    candidates_map = cache["candidates_map"]
    meta = cache.get("meta", {})
    print(f">>> Cache meta: {meta}")

    # Optional consistency warning
    if meta.get("doc_path") and meta.get("doc_path") != args.doc_path:
        print(f"[WARN] Cache doc_path != current doc_path\n"
              f"       cache:   {meta.get('doc_path')}\n"
              f"       current: {args.doc_path}")
    if meta.get("test_path") and meta.get("test_path") != args.test_path:
        print(f"[WARN] Cache test_path != current test_path\n"
              f"       cache:   {meta.get('test_path')}\n"
              f"       current: {args.test_path}")

    print(">>> Loading data...")
    docs = pd.read_csv(args.doc_path, dtype={"study_id": str}).fillna("")
    test = pd.read_csv(args.test_path, dtype={"nct_id": str, "pubmed_id": str}).fillna("")

    if "label" not in test.columns:
        raise ValueError("dataset_test.csv must contain column: label")
    test["label"] = test["label"].astype(float)

    if "trial_info" not in test.columns:
        raise ValueError("dataset_test.csv must contain column: trial_info")

    print(">>> Building doc texts (baseline style)...")
    docs["text"] = build_text(docs, id_col="study_id")
    doc_id_to_text = pd.Series(docs["text"].values, index=docs["study_id"]).to_dict()

    print(">>> Building query map from test (trial_info)...")
    query_text_map = build_query_map_from_test(test)

    print(">>> Building ground truth map (label==1)...")
    gt_pos = test[test["label"] == 1].copy()
    ground_truth_map = defaultdict(set)
    for row in gt_pos.itertuples(index=False):
        ground_truth_map[str(row.nct_id)].add(str(row.pubmed_id))

    query_ids = sorted(list(candidates_map.keys()))
    if args.max_queries and args.max_queries > 0:
        query_ids = query_ids[: args.max_queries]
    print(f">>> Queries from cache: {len(query_ids)}")

    # Device
    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    results = []

    for model_path in args.models:
        print(f"\n>>> Loading model: {model_path}")
        model = SentenceTransformer(model_path, device=device)

        ranked_map = {}
        missing_q = 0
        for nct_id in tqdm(query_ids, desc=f"Rerank[{model_path}]"):
            q_text = query_text_map.get(nct_id, "")
            if not q_text:
                missing_q += 1

            cand_ids = candidates_map.get(nct_id, [])
            cand_texts = [doc_id_to_text.get(cid, "") for cid in cand_ids]

            ranked_ids = rerank(model, q_text, cand_texts, cand_ids, batch_size=args.batch_size)
            ranked_map[nct_id] = ranked_ids

        if missing_q:
            print(f"[WARN] {missing_q} queries had empty/missing trial_info during rerank.")

        metrics = compute_metrics(ranked_map, ground_truth_map)
        results.append((model_path, metrics))

    print("\n==================== RESULTS ====================")
    print(f"Evaluated queries: {results[0][1]['total'] if results else 0}")
    print("-------------------------------------------------")
    print(f"{'Model':60s}  R@1     R@5     R@10    MRR")
    print("-------------------------------------------------")
    for model_path, m in results:
        name = (model_path[:57] + "...") if len(model_path) > 60 else model_path
        print(f"{name:60s}  {m['r@1']:.4f}  {m['r@5']:.4f}  {m['r@10']:.4f}  {m['mrr']:.4f}")
    print("=================================================\n")


if __name__ == "__main__":
    main()
