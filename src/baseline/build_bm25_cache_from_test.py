# build_bm25_cache_from_test.py
# ------------------------------------------------------------
# Build and save BM25 candidate cache using trial_info from dataset_test.csv as queries.
#
# Input:
#   - dataset_test.csv: columns include [nct_id, pubmed_id, trial_info, doc_info, label]
#   - pubmed_document.csv: document database, columns include [study_id, ...text fields...]
#
# Output:
#   - pickle cache: candidates_map {nct_id: [cand_pubmed_id1, ...]}
#
# Usage:
# python build_bm25_cache_from_test.py \
#   --doc_path data/pubmed_document.csv \
#   --test_path data/dataset_test.csv \
#   --bm25_top_k 50 \
#   --max_queries 0 \
#   --out_path bm25_candidates_top50.pkl
# ------------------------------------------------------------

import argparse
import string
import pickle
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi


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


def build_query_map_from_test(test_df: pd.DataFrame) -> dict:
    """
    Build nct_id -> trial_info map from dataset_test.csv.
    If an nct_id appears multiple times, take the first non-empty trial_info.
    Warn if multiple distinct non-empty trial_info exist for the same nct_id.
    """
    if "nct_id" not in test_df.columns or "trial_info" not in test_df.columns:
        raise ValueError("dataset_test.csv must contain columns: nct_id, trial_info")

    test_df["nct_id"] = test_df["nct_id"].astype(str)
    test_df["trial_info"] = test_df["trial_info"].fillna("").astype(str)

    query_map = {}
    inconsistent = 0
    empty = 0

    for nct_id, g in test_df.groupby("nct_id"):
        # distinct non-empty trial_info values
        vals = [v.strip() for v in g["trial_info"].tolist() if v and v.strip()]
        if not vals:
            empty += 1
            continue

        uniq = list(dict.fromkeys(vals))  # preserve order, unique
        if len(uniq) > 1:
            inconsistent += 1
        query_map[nct_id] = uniq[0]

    if inconsistent:
        print(f"[WARN] Found {inconsistent} nct_id with multiple distinct non-empty trial_info. Using the first one.")
    if empty:
        print(f"[WARN] Found {empty} nct_id with empty trial_info. These queries will be skipped.")

    return query_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc_path", required=True, help="pubmed_document.csv")
    ap.add_argument("--test_path", required=True, help="dataset_test.csv containing trial_info")
    ap.add_argument("--bm25_top_k", type=int, default=50)
    ap.add_argument("--max_queries", type=int, default=0, help="0 = all queries with positives")
    ap.add_argument("--out_path", default="bm25_candidates_top50.pkl")
    args = ap.parse_args()

    print(">>> Loading data...")
    docs = pd.read_csv(args.doc_path, dtype={"study_id": str}).fillna("")
    test = pd.read_csv(args.test_path, dtype={"nct_id": str, "pubmed_id": str}).fillna("")

    if "label" not in test.columns:
        raise ValueError("dataset_test.csv must contain column: label")
    test["label"] = test["label"].astype(float)

    if "trial_info" not in test.columns:
        raise ValueError("dataset_test.csv must contain column: trial_info")

    print(">>> Building doc texts (baseline style) ...")
    docs["text"] = build_text(docs, id_col="study_id")

    # Build ground truth map (label==1) -> decide which nct_id are evaluated / cached
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

    print(">>> Building query text map from dataset_test.csv (trial_info)...")
    query_text_map = build_query_map_from_test(test)

    print(">>> Building BM25 index...")
    doc_ids = docs["study_id"].tolist()
    doc_texts = docs["text"].tolist()
    tokenized_corpus = [tokenize(t) for t in tqdm(doc_texts, desc="Tokenize corpus")]
    bm25 = BM25Okapi(tokenized_corpus)
    print("    BM25 ready.")

    print(">>> Precomputing BM25 candidates...")
    candidates_map = {}  # nct_id -> [cand_doc_ids]
    skipped = 0

    for nct_id in tqdm(query_ids, desc="BM25 per query"):
        q_text = query_text_map.get(nct_id, "")
        if not q_text:
            skipped += 1
            continue
        q_tok = tokenize(q_text)
        cand_ids = bm25.get_top_n(q_tok, doc_ids, n=args.bm25_top_k)
        candidates_map[nct_id] = cand_ids

    if skipped:
        print(f"[WARN] Skipped {skipped} queries because trial_info was missing/empty.")

    payload = {
        "meta": {
            "doc_path": args.doc_path,
            "test_path": args.test_path,
            "bm25_top_k": args.bm25_top_k,
            "max_queries": args.max_queries,
            "num_queries_cached": len(candidates_map),
            "query_source": "dataset_test.csv:trial_info",
        },
        "candidates_map": candidates_map,
    }

    print(f">>> Saving cache to: {args.out_path}")
    with open(args.out_path, "wb") as f:
        pickle.dump(payload, f)

    print(">>> Done.")


if __name__ == "__main__":
    main()
