# train_kfold.py
# ------------------------------------------------------------
# K-fold cross-validation for contrastive learning
# 
# This script extends train_contrastive.py to support K-fold CV,
# maximizing the use of small datasets.
#
# Usage:
#   1) Prepare K-fold data:
#      python train_kfold.py --prepare_data --n_folds 5
#
#   2) Train all folds:
#      python train_kfold.py --train_all_folds
#
#   3) Evaluate all folds:
#      python train_kfold.py --evaluate_all_folds
#
#   4) Or do everything in one go:
#      python train_kfold.py --prepare_data --train_all_folds --evaluate_all_folds
# ------------------------------------------------------------

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# 完全禁用 MPS
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import json
import argparse
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold

# 复用 train_contrastive.py 的函数
from train_contrastive import (
    create_text,
    tokenize,
    build_bm25_index,
    bm25_retrieve,
    load_bm25_cache,
    sample_negatives_for_positive,
    build_contrastive_pairs,
    train_sentence_transformer
)

# 复用 evaluate.py 的函数
from sentence_transformers import SentenceTransformer, util


# ============================================================
# K-fold 数据准备
# ============================================================

def split_kfold(pairs: pd.DataFrame, n_splits=5, seed=42):
    """
    K-fold split by nct_id (trial-level split to avoid leakage)
    
    Returns:
        List of fold dicts, each containing:
        {
            "fold": int,
            "train_nct_ids": set,
            "test_nct_ids": set,
            "pairs_train": DataFrame,
            "pairs_test": DataFrame
        }
    """
    # 只使用有正样本的 trials
    pos_nct_ids = sorted(list(
        pairs.loc[pairs["label"] == 1, "nct_id"].unique()
    ))
    
    print(f"Total trials with positive labels: {len(pos_nct_ids)}")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(pos_nct_ids)):
        train_ids = set([pos_nct_ids[i] for i in train_idx])
        test_ids = set([pos_nct_ids[i] for i in test_idx])
        
        # 获取对应的 pairs
        pairs_train = pairs[pairs["nct_id"].isin(train_ids)].copy()
        pairs_test = pairs[pairs["nct_id"].isin(test_ids)].copy()
        
        folds.append({
            "fold": fold_idx,
            "train_nct_ids": train_ids,
            "test_nct_ids": test_ids,
            "pairs_train": pairs_train,
            "pairs_test": pairs_test
        })
        
        print(f"  Fold {fold_idx}: train={len(train_ids)} trials, test={len(test_ids)} trials")
    
    return folds


def prepare_kfold_data(args):
    """
    准备 K-fold 的所有数据 artifacts
    
    为每个 fold 生成:
      - fold{i}/pairs_train.csv
      - fold{i}/pairs_test.csv
      - fold{i}/bm25_candidates_train.jsonl
      - fold{i}/bm25_candidates_test.jsonl
      - fold{i}/train_triplets.jsonl
    """
    print("\n" + "="*60)
    print("PREPARING K-FOLD DATA")
    print("="*60)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. 加载原始数据
    print("\n>>> [1/5] Loading data...")
    trials = pd.read_csv(args.trial_path, dtype={"study_id": str}).fillna("")
    docs = pd.read_csv(args.doc_path, dtype={"study_id": str}).fillna("")
    pairs = pd.read_csv(args.pair_path, dtype={"nct_id": str, "pubmed_id": str})
    
    # 标准化列名
    if "study_id" in pairs.columns:
        pairs = pairs.rename(columns={"study_id": "nct_id"})
    
    print(f"  Trials: {len(trials):,}")
    print(f"  Documents: {len(docs):,}")
    print(f"  Pairs: {len(pairs):,} (pos={(pairs['label']==1).sum():,}, neg={(pairs['label']==0).sum():,})")
    
    # 2. 构建文本
    print("\n>>> [2/5] Building text fields...")
    trials["text"] = create_text(trials, "study_id")
    docs["text"] = create_text(docs, "study_id")
    
    trial_text = pd.Series(trials["text"].values, index=trials["study_id"]).to_dict()
    doc_text = pd.Series(docs["text"].values, index=docs["study_id"]).to_dict()
    
    # 3. K-fold split
    print(f"\n>>> [3/5] Creating {args.n_folds}-fold split...")
    folds = split_kfold(pairs, n_splits=args.n_folds, seed=args.seed)
    
    # 保存 fold 信息
    fold_info = {
        "n_folds": args.n_folds,
        "seed": args.seed,
        "bm25_top_n": args.bm25_top_n,
        "folds": [
            {
                "fold": f["fold"],
                "train_nct_ids": sorted(list(f["train_nct_ids"])),
                "test_nct_ids": sorted(list(f["test_nct_ids"])),
                "train_pairs": len(f["pairs_train"]),
                "test_pairs": len(f["pairs_test"])
            }
            for f in folds
        ]
    }
    
    fold_info_path = os.path.join(args.out_dir, "kfold_info.json")
    with open(fold_info_path, "w", encoding="utf-8") as f:
        json.dump(fold_info, f, indent=2, ensure_ascii=False)
    print(f"  Saved fold info to: {fold_info_path}")
    
    # 4. 构建 BM25 索引 (所有 folds 共享)
    print("\n>>> [4/5] Building BM25 index (shared across folds)...")
    doc_ids = docs["study_id"].tolist()
    doc_texts = docs["text"].tolist()
    bm25 = build_bm25_index(doc_texts)
    print("  BM25 index built.")
    
    # 5. 为每个 fold 生成数据
    print(f"\n>>> [5/5] Generating data for each fold...")
    
    for fold in folds:
        fold_idx = fold["fold"]
        fold_dir = os.path.join(args.out_dir, f"fold{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        
        print(f"\n  --- Processing Fold {fold_idx} ---")
        
        # 5.1 保存 pairs
        fold["pairs_train"].to_csv(
            os.path.join(fold_dir, "pairs_train.csv"), index=False
        )
        fold["pairs_test"].to_csv(
            os.path.join(fold_dir, "pairs_test.csv"), index=False
        )
        
        # 5.2 BM25 cache for train
        train_cache_path = os.path.join(fold_dir, "bm25_candidates_train.jsonl")
        with open(train_cache_path, "w", encoding="utf-8") as f:
            for nct_id in tqdm(
                sorted(list(fold["train_nct_ids"])),
                desc=f"  BM25 cache [fold{fold_idx}/train]",
                leave=False
            ):
                if nct_id not in trial_text:
                    continue
                qtext = trial_text[nct_id]
                cands = bm25_retrieve(bm25, doc_ids, doc_texts, qtext, top_k=args.bm25_top_n)
                f.write(json.dumps({
                    "nct_id": nct_id,
                    "top_n": args.bm25_top_n,
                    "candidates": cands
                }, ensure_ascii=False) + "\n")
        
        # 5.3 BM25 cache for test
        test_cache_path = os.path.join(fold_dir, "bm25_candidates_test.jsonl")
        with open(test_cache_path, "w", encoding="utf-8") as f:
            for nct_id in tqdm(
                sorted(list(fold["test_nct_ids"])),
                desc=f"  BM25 cache [fold{fold_idx}/test]",
                leave=False
            ):
                if nct_id not in trial_text:
                    continue
                qtext = trial_text[nct_id]
                cands = bm25_retrieve(bm25, doc_ids, doc_texts, qtext, top_k=args.bm25_top_n)
                f.write(json.dumps({
                    "nct_id": nct_id,
                    "top_n": args.bm25_top_n,
                    "candidates": cands
                }, ensure_ascii=False) + "\n")
        
        # 5.4 构建训练 triplets
        triplet_path = os.path.join(fold_dir, "train_triplets.jsonl")
        pairs_train = fold["pairs_train"]
        bm25_cache = load_bm25_cache(train_cache_path)
        
        # 分离正负样本
        pos_by_trial = {}
        neg_by_trial = {}
        for nct_id, grp in pairs_train.groupby("nct_id"):
            pos_by_trial[nct_id] = grp.loc[grp["label"] == 1, "pubmed_id"].tolist()
            neg_by_trial[nct_id] = grp.loc[grp["label"] == 0, "pubmed_id"].tolist()
        
        rng = random.Random(args.seed + fold_idx)  # 每个 fold 不同的随机种子
        written = 0
        
        with open(triplet_path, "w", encoding="utf-8") as f:
            for nct_id in pos_by_trial.keys():
                if nct_id not in trial_text or nct_id not in bm25_cache:
                    continue
                
                pos_list = pos_by_trial[nct_id]
                if not pos_list:
                    continue
                
                anchor = trial_text[nct_id]
                all_pos_set = set(pos_list)
                labeled_negs = neg_by_trial.get(nct_id, [])
                ranked_docs = [c["doc_id"] for c in bm25_cache[nct_id]]
                
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
        
        print(f"  Fold {fold_idx}: {written:,} triplets generated")
    
    print("\n" + "="*60)
    print(f"K-FOLD DATA PREPARATION COMPLETE")
    print(f"Output directory: {args.out_dir}/")
    print("="*60)


# ============================================================
# K-fold 训练
# ============================================================

def train_single_fold(fold_idx, args):
    """训练单个 fold 的模型"""
    fold_dir = os.path.join(args.out_dir, f"fold{fold_idx}")
    triplet_path = os.path.join(fold_dir, "train_triplets.jsonl")
    
    if not os.path.exists(triplet_path):
        raise FileNotFoundError(f"Missing {triplet_path}. Run with --prepare_data first.")
    
    print("\n" + "="*60)
    print(f"TRAINING FOLD {fold_idx}")
    print("="*60)
    
    # 构建训练数据
    print(f"\n>>> Building contrastive pairs from triplets...")
    examples = build_contrastive_pairs(triplet_path)
    
    # 训练模型
    save_dir = os.path.join(args.model_save_dir, f"fold{fold_idx}")
    
    # 创建临时 args 用于训练
    train_args = argparse.Namespace(
        base_model=args.base_model,
        save_dir=save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        warmup_ratio=args.warmup_ratio
    )
    
    print(f"\n>>> Training model (will be saved to {save_dir})...")
    train_sentence_transformer(examples, train_args)
    
    print(f"\n>>> Fold {fold_idx} training complete!")
    return save_dir


def train_all_folds(args):
    """训练所有 folds"""
    # 读取 fold 信息
    fold_info_path = os.path.join(args.out_dir, "kfold_info.json")
    if not os.path.exists(fold_info_path):
        raise FileNotFoundError(
            f"Missing {fold_info_path}. Run with --prepare_data first."
        )
    
    with open(fold_info_path, "r", encoding="utf-8") as f:
        fold_info = json.load(f)
    
    n_folds = fold_info["n_folds"]
    
    print("\n" + "="*60)
    print(f"TRAINING ALL {n_folds} FOLDS")
    print("="*60)
    
    for fold_idx in range(n_folds):
        train_single_fold(fold_idx, args)
    
    print("\n" + "="*60)
    print("ALL FOLDS TRAINING COMPLETE")
    print(f"Models saved to: {args.model_save_dir}/fold{{0..{n_folds-1}}}/")
    print("="*60)


# ============================================================
# K-fold 评估
# ============================================================

def load_model(model_path):
    """加载模型"""
    return SentenceTransformer(model_path)


def rerank_candidates(model, trial_text, candidates, doc_texts):
    """使用 Transformer 重排序候选文档"""
    if not candidates:
        return []
    
    # 收集候选文档的文本
    cand_doc_ids = []
    cand_texts = []
    for cand in candidates:
        doc_id = cand["doc_id"]
        if doc_id in doc_texts:
            cand_doc_ids.append(doc_id)
            cand_texts.append(doc_texts[doc_id])
    
    if not cand_texts:
        return []
    
    # 编码
    trial_emb = model.encode(trial_text, convert_to_tensor=True, show_progress_bar=False)
    cand_embs = model.encode(cand_texts, convert_to_tensor=True, show_progress_bar=False)
    
    # 计算余弦相似度
    cos_scores = util.cos_sim(trial_emb, cand_embs)[0]
    
    # 构建结果
    ranked_results = []
    for i, doc_id in enumerate(cand_doc_ids):
        ranked_results.append({
            "doc_id": doc_id,
            "score": cos_scores[i].item()
        })
    
    # 按分数降序排列
    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    
    return ranked_results


def evaluate_single_fold(fold_idx, model_path, trial_texts, doc_texts, args):
    """评估单个 fold"""
    fold_dir = os.path.join(args.out_dir, f"fold{fold_idx}")
    
    # 加载测试数据
    pairs_test = pd.read_csv(
        os.path.join(fold_dir, "pairs_test.csv"),
        dtype={"nct_id": str, "pubmed_id": str}
    )
    ground_truth = pairs_test[pairs_test["label"] == 1]
    
    bm25_cache = load_bm25_cache(
        os.path.join(fold_dir, "bm25_candidates_test.jsonl")
    )
    
    # 加载模型
    model = load_model(model_path)
    
    # 评估
    test_nct_ids = ground_truth["nct_id"].unique()
    
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    total = 0
    
    for nct_id in test_nct_ids:
        if nct_id not in trial_texts or nct_id not in bm25_cache:
            continue
        
        true_doc_ids = set(
            ground_truth[ground_truth["nct_id"] == nct_id]["pubmed_id"]
        )
        
        trial_text = trial_texts[nct_id]
        candidates = bm25_cache[nct_id][:args.bm25_top_k]
        
        ranked_results = rerank_candidates(
            model, trial_text, candidates, doc_texts
        )
        
        pred_doc_ids = [r["doc_id"] for r in ranked_results]
        
        if any(pid in true_doc_ids for pid in pred_doc_ids[:1]):
            hits_at_1 += 1
        if any(pid in true_doc_ids for pid in pred_doc_ids[:5]):
            hits_at_5 += 1
        if any(pid in true_doc_ids for pid in pred_doc_ids[:10]):
            hits_at_10 += 1
        
        total += 1
    
    results = {
        "fold": fold_idx,
        "total": total,
        "recall@1": hits_at_1 / total if total > 0 else 0,
        "recall@5": hits_at_5 / total if total > 0 else 0,
        "recall@10": hits_at_10 / total if total > 0 else 0
    }
    
    return results


def evaluate_all_folds(args):
    """评估所有 folds (baseline + fine-tuned)"""
    # 读取 fold 信息
    fold_info_path = os.path.join(args.out_dir, "kfold_info.json")
    with open(fold_info_path, "r", encoding="utf-8") as f:
        fold_info = json.load(f)
    
    n_folds = fold_info["n_folds"]
    
    print("\n" + "="*60)
    print(f"EVALUATING ALL {n_folds} FOLDS")
    print("="*60)
    
    # 加载文本映射 (只需要加载一次)
    print("\n>>> Loading text mappings...")
    trials = pd.read_csv(args.trial_path, dtype={"study_id": str}).fillna("")
    docs = pd.read_csv(args.doc_path, dtype={"study_id": str}).fillna("")
    
    trials["text"] = create_text(trials, "study_id")
    docs["text"] = create_text(docs, "study_id")
    
    trial_texts = pd.Series(trials["text"].values, index=trials["study_id"]).to_dict()
    doc_texts = pd.Series(docs["text"].values, index=docs["study_id"]).to_dict()
    
    # 评估 baseline
    print("\n" + "-"*60)
    print("Evaluating BASELINE")
    print("-"*60)
    
    baseline_results = []
    for fold_idx in tqdm(range(n_folds), desc="Baseline evaluation"):
        result = evaluate_single_fold(
            fold_idx, args.baseline_model, trial_texts, doc_texts, args
        )
        baseline_results.append(result)
        print(f"  Fold {fold_idx}: R@1={result['recall@1']:.3f}, "
              f"R@5={result['recall@5']:.3f}, R@10={result['recall@10']:.3f} "
              f"(n={result['total']})")
    
    # 评估 fine-tuned
    print("\n" + "-"*60)
    print("Evaluating FINE-TUNED")
    print("-"*60)
    
    finetuned_results = []
    for fold_idx in tqdm(range(n_folds), desc="Fine-tuned evaluation"):
        model_path = os.path.join(args.model_save_dir, f"fold{fold_idx}")
        
        if not os.path.exists(model_path):
            print(f"  Warning: Model not found at {model_path}, skipping fold {fold_idx}")
            continue
        
        result = evaluate_single_fold(
            fold_idx, model_path, trial_texts, doc_texts, args
        )
        finetuned_results.append(result)
        print(f"  Fold {fold_idx}: R@1={result['recall@1']:.3f}, "
              f"R@5={result['recall@5']:.3f}, R@10={result['recall@10']:.3f} "
              f"(n={result['total']})")
    
    # 汇总结果
    print("\n" + "="*60)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*60)
    
    def print_summary(results, model_name):
        if not results:
            print(f"\n{model_name}: No results")
            return
        
        recalls_1 = [r["recall@1"] for r in results]
        recalls_5 = [r["recall@5"] for r in results]
        recalls_10 = [r["recall@10"] for r in results]
        
        print(f"\n{model_name}:")
        print(f"  Recall@1:  {np.mean(recalls_1):.4f} ± {np.std(recalls_1):.4f}")
        print(f"  Recall@5:  {np.mean(recalls_5):.4f} ± {np.std(recalls_5):.4f}")
        print(f"  Recall@10: {np.mean(recalls_10):.4f} ± {np.std(recalls_10):.4f}")
        
        print(f"\n  Per-fold details:")
        for r in results:
            print(f"    Fold {r['fold']}: R@1={r['recall@1']:.3f}, "
                  f"R@5={r['recall@5']:.3f}, R@10={r['recall@10']:.3f}")
    
    print_summary(baseline_results, "BASELINE")
    print_summary(finetuned_results, "FINE-TUNED")
    
    # 计算改进
    if baseline_results and finetuned_results:
        print("\n" + "-"*60)
        print("IMPROVEMENT")
        print("-"*60)
        
        base_r1 = np.mean([r["recall@1"] for r in baseline_results])
        fine_r1 = np.mean([r["recall@1"] for r in finetuned_results])
        
        base_r5 = np.mean([r["recall@5"] for r in baseline_results])
        fine_r5 = np.mean([r["recall@5"] for r in finetuned_results])
        
        base_r10 = np.mean([r["recall@10"] for r in baseline_results])
        fine_r10 = np.mean([r["recall@10"] for r in finetuned_results])
        
        print(f"  Recall@1:  {base_r1:.4f} → {fine_r1:.4f} ({(fine_r1-base_r1)*100:+.2f}%)")
        print(f"  Recall@5:  {base_r5:.4f} → {fine_r5:.4f} ({(fine_r5-base_r5)*100:+.2f}%)")
        print(f"  Recall@10: {base_r10:.4f} → {fine_r10:.4f} ({(fine_r10-base_r10)*100:+.2f}%)")
    
    # 保存结果
    output = {
        "baseline": baseline_results,
        "finetuned": finetuned_results,
        "summary": {}
    }
    
    if baseline_results:
        output["summary"]["baseline"] = {
            "recall@1_mean": float(np.mean([r["recall@1"] for r in baseline_results])),
            "recall@1_std": float(np.std([r["recall@1"] for r in baseline_results])),
            "recall@5_mean": float(np.mean([r["recall@5"] for r in baseline_results])),
            "recall@5_std": float(np.std([r["recall@5"] for r in baseline_results])),
            "recall@10_mean": float(np.mean([r["recall@10"] for r in baseline_results])),
            "recall@10_std": float(np.std([r["recall@10"] for r in baseline_results])),
        }
    
    if finetuned_results:
        output["summary"]["finetuned"] = {
            "recall@1_mean": float(np.mean([r["recall@1"] for r in finetuned_results])),
            "recall@1_std": float(np.std([r["recall@1"] for r in finetuned_results])),
            "recall@5_mean": float(np.mean([r["recall@5"] for r in finetuned_results])),
            "recall@5_std": float(np.std([r["recall@5"] for r in finetuned_results])),
            "recall@10_mean": float(np.mean([r["recall@10"] for r in finetuned_results])),
            "recall@10_std": float(np.std([r["recall@10"] for r in finetuned_results])),
        }
    
    result_path = os.path.join(args.out_dir, "kfold_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print(f"Results saved to: {result_path}")
    print("="*60)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="K-fold cross-validation for contrastive learning"
    )
    
    # 数据路径
    parser.add_argument("--trial_path", default="data/nct_trial.csv")
    parser.add_argument("--doc_path", default="data/pubmed_document.csv")
    parser.add_argument("--pair_path", default="data/dataset_train.csv")
    parser.add_argument("--out_dir", default="artifacts_kfold")
    parser.add_argument("--model_save_dir", default="models/kfold")
    
    # K-fold 参数
    parser.add_argument("--n_folds", type=int, default=5,
                       help="Number of folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    
    # 模型
    parser.add_argument("--base_model", default="pritamdeka/S-PubMedBert-MS-MARCO",
                       help="Base model for training")
    parser.add_argument("--baseline_model", default="pritamdeka/S-PubMedBert-MS-MARCO",
                       help="Baseline model for comparison")
    
    # BM25 参数
    parser.add_argument("--bm25_top_n", type=int, default=2000,
                       help="BM25 top-N for candidate mining")
    
    # 负样本采样
    parser.add_argument("--hard_k", type=int, default=10)
    parser.add_argument("--n_random", type=int, default=20)
    parser.add_argument("--random_pool_k", type=int, default=1000)
    parser.add_argument("--max_labeled_neg", type=int, default=5)
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    
    # 评估参数
    parser.add_argument("--bm25_top_k", type=int, default=500,
                       help="BM25 top-K for reranking in evaluation")
    
    # 操作开关
    parser.add_argument("--prepare_data", action="store_true",
                       help="Prepare K-fold data artifacts")
    parser.add_argument("--train_all_folds", action="store_true",
                       help="Train models for all folds")
    parser.add_argument("--evaluate_all_folds", action="store_true",
                       help="Evaluate all folds (baseline + fine-tuned)")
    
    args = parser.parse_args()
    
    # 执行操作
    if args.prepare_data:
        prepare_kfold_data(args)
    
    if args.train_all_folds:
        train_all_folds(args)
    
    if args.evaluate_all_folds:
        evaluate_all_folds(args)
    
    if not (args.prepare_data or args.train_all_folds or args.evaluate_all_folds):
        print("\nNo action specified. Use one of:")
        print("  --prepare_data         Prepare K-fold data")
        print("  --train_all_folds      Train all folds")
        print("  --evaluate_all_folds   Evaluate all folds")
        print("\nOr combine them:")
        print("  python train_kfold.py --prepare_data --train_all_folds --evaluate_all_folds")


if __name__ == "__main__":
    main()