# train_contrastive.py
# ------------------------------------------------------------
# Contrastive fine-tuning for trial -> publication linking.
# 我们以人工标注的 (nct_id, pubmed_id, label=1) 作为正样本，并为每个正样本从人工标注的负例(若有）以及同一 trial 的 BM25 高排名候选中挖掘 hard / random 负例，从而构造出(trial, pos doc, neg doc)的对比学习训练对。
# What this script can do:
#   (A) Data preparation (optional, run once or when you change mining strategy)
#       Step1: load + text concat + split by nct_id (ONLY trials with positives)
#       Step2: BM25 cache for each split (train/val/test)  [no labels involved]
#       Step3: build training triplets.jsonl for contrastive learning
#
#   (B) Training (default behavior)
#       Step4A: triplets -> contrastive pairs (InputExample)
#       Step4B: SentenceTransformers fine-tuning with OnlineContrastiveLoss
#       Save model to disk (baseline.py can load this local folder)
#
# Usage:
#   1) First time (generate artifacts + train):
#        python train_contrastive.py --prepare_data
#
#   2) Later (re-train only, reuse artifacts_train/train_triplets.jsonl):
#        python train_contrastive.py
#
#   3) If you change mining strategy / want to rebuild triplets:
#        python train_contrastive.py --prepare_data
#
# Notes:
# - We split at the nct_id level to avoid leakage.
# - We restrict splitting to trials with >=1 positive label.
# - BM25 cache does NOT use labels and is safe to compute for all splits.
# ------------------------------------------------------------

import os
import json
import random
import argparse
import string

import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


# ============================================================
# Text preprocessing (keep consistent with baseline.py)
# ============================================================

def create_text(df: pd.DataFrame, id_col: str) -> pd.Series:
    """
    Same as baseline.py:
    Concatenate all columns except `id_col` and 'study_source' into one long text.
    """
    cols = [c for c in df.columns if c != id_col and c != "study_source"]
    if not cols:
        raise ValueError(f"No text columns found besides {id_col} and study_source.")
    combined = df[cols[0]].astype(str)
    for col in cols[1:]:
        combined += " " + df[col].astype(str)
    return combined


def tokenize(text: str):
    """
    Same as baseline.py:
    lowercase -> remove punctuation -> whitespace split.
    """
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()


# ============================================================
# BM25 mining (same logic as baseline.py)
# ============================================================

def build_bm25_index(doc_texts):
    tokenized_corpus = [tokenize(doc) for doc in doc_texts]
    return BM25Okapi(tokenized_corpus)


def bm25_retrieve(bm25: BM25Okapi, doc_ids, doc_texts, query_text: str, top_k: int = 1000):
    """
    Return BM25 top_k candidates:
      [{"doc_id": ..., "bm25_score": ...}, ...]
    """
    tokenized_query = tokenize(query_text)
    scores = bm25.get_scores(tokenized_query)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        results.append({"doc_id": doc_ids[idx], "bm25_score": float(scores[idx])})
    return results


# ============================================================
# Split utilities
# ============================================================

def split_by_nct_id(pairs: pd.DataFrame, seed: int = 42, train_ratio=0.6, val_ratio=0.2):
    """
    Split by nct_id (trial-level split) to avoid leakage.
    """
    rng = random.Random(seed)
    nct_ids = sorted(pairs["nct_id"].unique().tolist())
    rng.shuffle(nct_ids)

    n_total = len(nct_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = set(nct_ids[:n_train])
    val_ids = set(nct_ids[n_train:n_train + n_val])
    test_ids = set(nct_ids[n_train + n_val:])

    def _subset(ids_set):
        return pairs[pairs["nct_id"].isin(ids_set)].copy()

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "pairs_train": _subset(train_ids),
        "pairs_val": _subset(val_ids),
        "pairs_test": _subset(test_ids),
    }


# ============================================================
# BM25 cache I/O
# ============================================================

def load_bm25_cache(jsonl_path: str):
    """
    Load BM25 cache (jsonl, one line per nct_id).
    Returns:
      cache[nct_id] = [doc_id1, doc_id2, ...]  # ranked by BM25
    """
    cache = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            nct_id = rec["nct_id"]
            cache[nct_id] = [c["doc_id"] for c in rec["candidates"]]
    return cache


# ============================================================
# Negative sampling (strategy knobs here)
# ============================================================

def sample_negatives_for_positive(
    pos_doc_id: str,
    bm25_ranked_doc_ids: list,
    labeled_neg_doc_ids: list,
    all_pos_doc_ids_for_trial: set,
    rng: random.Random,
    hard_k: int = 10,
    n_random: int = 20,
    random_pool_k: int = 1000,
    max_labeled_neg: int = 5,
):
    """
    Build negative doc_id list for a given (trial, positive_doc).

    Negative sources (priority):
      1) human-verified negatives (label=0) -> strongest signal
      2) hard negative: sample 1 from BM25 top-hard_k
      3) random negatives: sample n_random from BM25 top-random_pool_k

    Filters:
      - must not be any positive doc for this trial
      - must not be the current pos_doc_id
    """
    negs = []

    # (A) labeled negatives (take up to max_labeled_neg for stability)
    labeled_pool = []
    for d in labeled_neg_doc_ids:
        if d != pos_doc_id and d not in all_pos_doc_ids_for_trial:
            labeled_pool.append(d)
    rng.shuffle(labeled_pool)
    negs.extend(labeled_pool[:max_labeled_neg])

    # (B) hard negative from BM25 top-hard_k
    hard_candidates = []
    for d in bm25_ranked_doc_ids[:hard_k]:
        if d != pos_doc_id and d not in all_pos_doc_ids_for_trial:
            hard_candidates.append(d)
    if hard_candidates:
        negs.append(rng.choice(hard_candidates))

    # (C) random negatives from BM25 top-random_pool_k
    pool = []
    for d in bm25_ranked_doc_ids[:random_pool_k]:
        if d != pos_doc_id and d not in all_pos_doc_ids_for_trial:
            pool.append(d)
    rng.shuffle(pool)
    negs.extend(pool[:n_random])

    # de-dup (keep order)
    seen = set()
    uniq = []
    for d in negs:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq












# ============================================================
# Step1-3: prepare data artifacts
# ============================================================

def prepare_data_artifacts(args):
    """
    Create:
      - splits.json
      - pairs_train.csv, pairs_val.csv, pairs_test.csv
      - bm25_candidates_{train,val,test}.jsonl
      - train_triplets.jsonl
    """
    os.makedirs(args.out_dir, exist_ok=True)

    print(">>> [Prepare] Loading CSVs...")
    trials = pd.read_csv(args.trial_path, dtype={"study_id": str}).fillna("")
    docs = pd.read_csv(args.doc_path, dtype={"study_id": str}).fillna("")
    pairs = pd.read_csv(args.pair_path, dtype={"nct_id": str, "pubmed_id": str})

    # normalize if needed
    pairs = pairs.rename(columns={"study_id": "nct_id"}) if "study_id" in pairs.columns else pairs

    required_cols = {"nct_id", "pubmed_id", "label"}
    missing = required_cols - set(pairs.columns)
    if missing:
        raise ValueError(f"pair file missing columns: {missing}")

    print(">>> [Prepare] Building concatenated text fields (keep consistent with baseline.py)...")
    trials["text"] = create_text(trials, "study_id")
    docs["text"] = create_text(docs, "study_id")

    trial_text = pd.Series(trials["text"].values, index=trials["study_id"]).to_dict()
    doc_text = pd.Series(docs["text"].values, index=docs["study_id"]).to_dict()

    print("\n=== Stats ===")
    print(f"Trials rows: {len(trials):,}")
    print(f"Docs rows:   {len(docs):,}")
    print(f"Pairs rows:  {len(pairs):,}")
    print(f"Unique nct_id in pairs: {pairs['nct_id'].nunique():,}")
    print(f"Pos (label=1): {int((pairs['label']==1).sum()):,}")
    print(f"Neg (label=0): {int((pairs['label']==0).sum()):,}")

    # Step1B: restrict split to trials with positives only (avoid val/test with 0 positives)
    pos_nct_ids = set(pairs.loc[pairs["label"] == 1, "nct_id"].unique().tolist())
    pairs_pos_trials = pairs[pairs["nct_id"].isin(pos_nct_ids)].copy()

    print(f"\n>>> [Prepare:Split] Trials with >=1 positive: {len(pos_nct_ids)} "
          f"(from all {pairs['nct_id'].nunique()})")

    split = split_by_nct_id(
        pairs_pos_trials,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    # save splits and split pairs
    split_json = {
        "seed": args.seed,
        "train_nct_ids": sorted(list(split["train"])),
        "val_nct_ids": sorted(list(split["val"])),
        "test_nct_ids": sorted(list(split["test"])),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "bm25_top_n": args.bm25_top_n
    }
    with open(os.path.join(args.out_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(split_json, f, ensure_ascii=False, indent=2)

    split["pairs_train"].to_csv(os.path.join(args.out_dir, "pairs_train.csv"), index=False)
    split["pairs_val"].to_csv(os.path.join(args.out_dir, "pairs_val.csv"), index=False)
    split["pairs_test"].to_csv(os.path.join(args.out_dir, "pairs_test.csv"), index=False)

    print("\n=== Split sizes ===")
    for name in ["pairs_train", "pairs_val", "pairs_test"]:
        df = split[name]
        print(f"{name}: pairs={len(df):,}, unique_nct={df['nct_id'].nunique():,}, "
              f"pos={(df['label']==1).sum():,}, neg={(df['label']==0).sum():,}")

    # Step2: BM25 cache (safe: no labels used)
    print("\n>>> [Prepare:BM25] Building BM25 index...")
    doc_ids = docs["study_id"].tolist()
    doc_texts = docs["text"].tolist()
    bm25 = build_bm25_index(doc_texts)

    def dump_candidates(split_name: str, nct_ids):
        out_path = os.path.join(args.out_dir, f"bm25_candidates_{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for nct_id in tqdm(sorted(list(nct_ids)), desc=f"BM25 cache [{split_name}]"):
                if nct_id not in trial_text:
                    continue
                qtext = trial_text[nct_id]
                cands = bm25_retrieve(bm25, doc_ids, doc_texts, qtext, top_k=args.bm25_top_n)
                f.write(json.dumps({
                    "nct_id": nct_id,
                    "top_n": args.bm25_top_n,
                    "candidates": cands
                }, ensure_ascii=False) + "\n")
        print(f"Saved BM25 candidates to: {out_path}")

    dump_candidates("train", split["train"])
    dump_candidates("val", split["val"])
    dump_candidates("test", split["test"])

    # Step3: build triplets (train only)
    print("\n>>> [Prepare:Triplets] Building training triplets...")
    pairs_train = split["pairs_train"]
    bm25_cache = load_bm25_cache(os.path.join(args.out_dir, "bm25_candidates_train.jsonl"))
    rng = random.Random(args.seed)

    # per-trial positives & negatives (from human labels)
    pos_by_trial = {}
    neg_by_trial = {}
    for nct_id, grp in pairs_train.groupby("nct_id"):
        pos_by_trial[nct_id] = grp.loc[grp["label"] == 1, "pubmed_id"].tolist()
        neg_by_trial[nct_id] = grp.loc[grp["label"] == 0, "pubmed_id"].tolist()

    triplet_out = os.path.join(args.out_dir, "train_triplets.jsonl")

    written = 0
    with open(triplet_out, "w", encoding="utf-8") as f:
        for nct_id in tqdm(sorted(list(pos_by_trial.keys())), desc="Triplet build"):
            if nct_id not in trial_text:
                continue
            if nct_id not in bm25_cache:
                continue

            pos_list = pos_by_trial.get(nct_id, [])
            if not pos_list:
                continue

            anchor = trial_text[nct_id]
            all_pos_set = set(pos_list)
            labeled_negs = neg_by_trial.get(nct_id, [])
            ranked_docs = bm25_cache[nct_id]

            for pos_doc_id in pos_list:
                if pos_doc_id not in doc_text:
                    continue

                neg_doc_ids = sample_negatives_for_positive(
                    pos_doc_id=pos_doc_id,
                    bm25_ranked_doc_ids=ranked_docs,
                    labeled_neg_doc_ids=labeled_negs,
                    all_pos_doc_ids_for_trial=all_pos_set,
                    rng=rng,
                    hard_k=args.hard_k,
                    n_random=args.n_random,
                    random_pool_k=args.random_pool_k,
                    max_labeled_neg=args.max_labeled_neg
                )

                for neg_doc_id in neg_doc_ids:
                    if neg_doc_id not in doc_text:
                        continue
                    f.write(json.dumps({
                        "nct_id": nct_id,
                        "pos_pubmed_id": pos_doc_id,
                        "neg_pubmed_id": neg_doc_id,
                        "anchor_text": anchor,
                        "pos_text": doc_text[pos_doc_id],
                        "neg_text": doc_text[neg_doc_id],
                    }, ensure_ascii=False) + "\n")
                    written += 1

    print(f">>> [Prepare] Triplets written: {written:,}")
    print(f">>> [Prepare] Saved triplets to: {triplet_out}")

    return triplet_out


# ============================================================
# Step4: triplets -> pairs -> training
# ============================================================

def build_contrastive_pairs(triplet_path: str):
    """
    Convert triplets.jsonl into SentenceTransformers InputExamples.
    Each triplet produces 2 pairs:
      (anchor, pos, label=1)
      (anchor, neg, label=0)
    """
    examples = []
    n_triplets = 0
    with open(triplet_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            anchor = rec["anchor_text"]
            examples.append(InputExample(texts=[anchor, rec["pos_text"]], label=1.0))
            examples.append(InputExample(texts=[anchor, rec["neg_text"]], label=0.0))
            n_triplets += 1

    print(f">>> [Train] Triplets loaded: {n_triplets:,}")
    print(f">>> [Train] Contrastive pairs built: {len(examples):,} (pos={n_triplets:,}, neg={n_triplets:,})")
    return examples


def train_sentence_transformer(examples, args):
    """
    Fine-tune bi-encoder using OnlineContrastiveLoss (cosine distance).
    """
    print("\n>>> [Train] Initializing model...")
    model = SentenceTransformer(args.base_model)
    model = model.to("cpu")

    train_loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=args.batch_size
    )

    train_loss = losses.OnlineContrastiveLoss(
        model=model,
        distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
        margin=args.margin
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    print("\n=== Training config ===")
    print(f"Base model: {args.base_model}")
    print(f"Epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}, margin: {args.margin}")
    print(f"Total steps: {total_steps}, warmup_steps: {warmup_steps}")

    os.makedirs(args.save_dir, exist_ok=True)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True
    )

    model.save(args.save_dir)
    print(f">>> [Train] Model saved to: {args.save_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--trial_path", default="data/nct_trial.csv")
    parser.add_argument("--doc_path", default="data/pubmed_document.csv")
    parser.add_argument("--pair_path", default="data/dataset_test.csv")  # labeled total pairs
    parser.add_argument("--out_dir", default="artifacts_train")
    parser.add_argument("--seed", type=int, default=42)

    # switches
    parser.add_argument("--prepare_data", action="store_true",
                        help="Run Step1-3 to (re)generate splits/bm25_cache/triplets.")
    parser.add_argument("--skip_if_exists", action="store_true",
                        help="If train_triplets.jsonl exists, skip regeneration when --prepare_data is set.")

    # split ratios (only on positive trials)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    # BM25 cache size for mining
    parser.add_argument("--bm25_top_n", type=int, default=2000)

    # mining strategy knobs
    parser.add_argument("--hard_k", type=int, default=10)
    parser.add_argument("--n_random", type=int, default=20)
    parser.add_argument("--random_pool_k", type=int, default=1000)
    parser.add_argument("--max_labeled_neg", type=int, default=5)

    # training
    parser.add_argument("--base_model", default="pritamdeka/S-PubMedBert-MS-MARCO")
    parser.add_argument("--save_dir", default="models/pubmedbert_contrastive_margin025")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    triplet_path = os.path.join(args.out_dir, "train_triplets.jsonl")

    if args.prepare_data:
        if args.skip_if_exists and os.path.exists(triplet_path):
            print(f">>> [Prepare] Found existing triplets: {triplet_path} (skip)")
        else:
            triplet_path = prepare_data_artifacts(args)
    else:
        if not os.path.exists(triplet_path):
            raise FileNotFoundError(
                f"Missing {triplet_path}. Run once with --prepare_data to generate artifacts."
            )

    print("\n>>> [Train] Building contrastive pairs from triplets...")
    examples = build_contrastive_pairs(triplet_path)

    print("\n>>> [Train] Fine-tuning bi-encoder...")
    train_sentence_transformer(examples, args)

    print("\n>>> Done.")


if __name__ == "__main__":
    main()

# the first time you run this script, please add the --prepare_data flag to generate data artifacts.
# e.g., python train_contrastive.py --prepare_data
# after that, you can run without the flag to just do training.
