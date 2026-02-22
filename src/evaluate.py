# evaluate.py (带详细进度提示版本)
import os
import json
import argparse
import pandas as pd
import numpy as np
import time

print("="*60)
print("Trial-Document Linking Evaluation")
print("="*60)

print("\n[Init] Importing libraries...")
start_time = time.time()

from tqdm import tqdm

# 延迟导入 sentence_transformers
_sentence_transformers_imported = False

def ensure_sentence_transformers():
    global _sentence_transformers_imported
    if not _sentence_transformers_imported:
        print("[Init] Loading sentence_transformers (first-time may take 10-30s)...")
        import_start = time.time()
        global SentenceTransformer, util
        from sentence_transformers import SentenceTransformer, util
        _sentence_transformers_imported = True
        print(f"       ✓ Loaded in {time.time()-import_start:.1f}s")

from train_contrastive import create_text

print(f"[Init] ✓ Import complete ({time.time()-start_time:.1f}s)\n")


def load_bm25_cache(jsonl_path):
    """
    加载 BM25 候选缓存
    返回: cache[nct_id] = [{"doc_id": ..., "bm25_score": ...}, ...]
    """
    cache = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            nct_id = rec["nct_id"]
            # 保留完整的候选信息
            cache[nct_id] = rec["candidates"]
    return cache


def load_test_data(artifacts_dir="artifacts_train"):
    """加载测试数据和预计算的 BM25 candidates"""
    print(">>> Loading test data...")
    
    pairs_test = pd.read_csv(
        os.path.join(artifacts_dir, "pairs_test.csv"),
        dtype={"nct_id": str, "pubmed_id": str}
    )
    ground_truth = pairs_test[pairs_test["label"] == 1].copy()
    
    bm25_cache = load_bm25_cache(
        os.path.join(artifacts_dir, "bm25_candidates_test.jsonl")
    )
    
    print(f"  Test pairs: {len(pairs_test):,}")
    print(f"  Ground truth (label=1): {len(ground_truth):,}")
    print(f"  Unique trials: {ground_truth['nct_id'].nunique():,}")
    print(f"  BM25 cache size: {len(bm25_cache):,}")
    
    return ground_truth, bm25_cache


def load_text_mappings(trial_path="data/nct_trial.csv", 
                       doc_path="data/pubmed_document.csv"):
    """加载并构建 id -> text 映射"""
    print(">>> Loading text mappings...")
    
    trials = pd.read_csv(trial_path, dtype={"study_id": str}).fillna("")
    docs = pd.read_csv(doc_path, dtype={"study_id": str}).fillna("")
    
    trials["text"] = create_text(trials, "study_id")
    docs["text"] = create_text(docs, "study_id")
    
    trial_texts = pd.Series(trials["text"].values, index=trials["study_id"]).to_dict()
    doc_texts = pd.Series(docs["text"].values, index=docs["study_id"]).to_dict()
    
    print(f"  Trial texts: {len(trial_texts):,}")
    print(f"  Doc texts: {len(doc_texts):,}")
    
    return trial_texts, doc_texts


def load_model(model_path):
    """加载 Transformer 模型"""
    ensure_sentence_transformers()
    
    print(f">>> Loading model: {model_path}")
    
    # 检查是否需要下载
    if not os.path.exists(model_path) and "/" in model_path:
        print("    Note: Downloading from HuggingFace (~400MB)...")
        print("    This may take 1-5 minutes on first run")
    
    load_start = time.time()
    model = SentenceTransformer(model_path)
    print(f"    ✓ Model loaded in {time.time()-load_start:.1f}s")
    
    return model


def rerank_candidates(model, trial_text, candidates, doc_texts):
    """使用 Transformer 对 BM25 candidates 重排序"""
    ensure_sentence_transformers()
    
    if not candidates:
        return []
    
    cand_doc_ids = []
    cand_texts = []
    for cand in candidates:
        doc_id = cand["doc_id"]
        if doc_id in doc_texts:
            cand_doc_ids.append(doc_id)
            cand_texts.append(doc_texts[doc_id])
    
    if not cand_texts:
        return []
    
    trial_emb = model.encode(trial_text, convert_to_tensor=True, show_progress_bar=False)
    cand_embs = model.encode(cand_texts, convert_to_tensor=True, show_progress_bar=False)
    
    cos_scores = util.cos_sim(trial_emb, cand_embs)[0]
    
    ranked_results = []
    for i, doc_id in enumerate(cand_doc_ids):
        ranked_results.append({
            "doc_id": doc_id,
            "score": cos_scores[i].item()
        })
    
    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results


def run_evaluation(model, ground_truth, bm25_cache, trial_texts, doc_texts, 
                   bm25_top_k=50):
    """对测试集运行评估，计算 Recall@1/5/10"""
    print(f"\n>>> Running evaluation (BM25 top-{bm25_top_k})...")
    
    test_nct_ids = ground_truth["nct_id"].unique()
    
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    total_queries = 0
    
    eval_start = time.time()
    
    for nct_id in tqdm(test_nct_ids, desc="Evaluating"):
        if nct_id not in trial_texts or nct_id not in bm25_cache:
            continue
        
        query_text = trial_texts[nct_id]
        candidates = bm25_cache[nct_id][:bm25_top_k]
        
        ranked_results = rerank_candidates(model, query_text, candidates, doc_texts)
        pred_doc_ids = [res["doc_id"] for res in ranked_results]
        
        true_doc_ids = set(
            ground_truth[ground_truth["nct_id"] == nct_id]["pubmed_id"].values
        )
        
        is_hit_1 = any(pid in true_doc_ids for pid in pred_doc_ids[:1])
        is_hit_5 = any(pid in true_doc_ids for pid in pred_doc_ids[:5])
        is_hit_10 = any(pid in true_doc_ids for pid in pred_doc_ids[:10])
        
        if is_hit_1:
            hits_at_1 += 1
        if is_hit_5:
            hits_at_5 += 1
        if is_hit_10:
            hits_at_10 += 1
        
        total_queries += 1
    
    eval_time = time.time() - eval_start
    
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Total Queries: {total_queries}")
    print(f"Recall@1:  {hits_at_1/total_queries:.4f}")
    print(f"Recall@5:  {hits_at_5/total_queries:.4f}")
    print(f"Recall@10: {hits_at_10/total_queries:.4f}")
    print(f"Time: {eval_time:.1f}s ({eval_time/total_queries:.2f}s per query)")
    print("="*60)
    
    return {
        "total": total_queries,
        "recall@1": hits_at_1 / total_queries,
        "recall@5": hits_at_5 / total_queries,
        "recall@10": hits_at_10 / total_queries,
        "time": eval_time
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval model on test set")
    parser.add_argument("--model_path", required=True,
                       help="Model path: HuggingFace model name or local directory")
    parser.add_argument("--artifacts_dir", default="artifacts_train")
    parser.add_argument("--bm25_top_k", type=int, default=50)
    parser.add_argument("--trial_path", default="data/nct_trial.csv")
    parser.add_argument("--doc_path", default="data/pubmed_document.csv")
    
    args = parser.parse_args()
    
    total_start = time.time()
    
    # 加载数据（只需一次）
    ground_truth, bm25_cache = load_test_data(args.artifacts_dir)
    trial_texts, doc_texts = load_text_mappings(args.trial_path, args.doc_path)
    
    # 加载模型
    model = load_model(args.model_path)
    
    # 运行评估
    results = run_evaluation(
        model=model,
        ground_truth=ground_truth,
        bm25_cache=bm25_cache,
        trial_texts=trial_texts,
        doc_texts=doc_texts,
        bm25_top_k=args.bm25_top_k
    )
    
    print(f"\n>>> Total time: {time.time()-total_start:.1f}s")
    print(">>> Done.")
    
    return results


if __name__ == "__main__":
    main()