import pandas as pd
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import string

# ==========================================
# 1. 数据加载与预处理类
# ==========================================
class TrialSearchPipeline:
    def __init__(self, trial_path, doc_path, test_path):
        print(">>> 正在加载数据...")
        # 读取数据，强制 ID 为字符串
        self.trials = pd.read_csv(trial_path, dtype={'study_id': str}).fillna('')
        self.docs = pd.read_csv(doc_path, dtype={'study_id': str}).fillna('')
        self.test_data = pd.read_csv(test_path, dtype={'nct_id': str, 'pubmed_id': str})
        
        # 只关注测试集中 label=1 的正样本（即我们需要找回的 Truth）
        self.ground_truth = self.test_data[self.test_data['label'] == 1]
        
        # 建立 ID 到 文本 的映射
        print(">>> 正在生成文本特征...")
        self.trials['text'] = self._create_text(self.trials, 'study_id')
        self.docs['text'] = self._create_text(self.docs, 'study_id')
        
        # 将文档数据转换为列表，用于构建索引
        # 注意：这里我们将 pubmed_document.csv 视为整个“搜索库”
        self.doc_ids = self.docs['study_id'].tolist()
        self.doc_texts = self.docs['text'].tolist()
        
        # 快速查找 Trial 文本的字典
        self.trial_text_map = pd.Series(
            self.trials['text'].values, index=self.trials['study_id']
        ).to_dict()

    def _create_text(self, df, id_col):
        """将多列合并为长文本，用于检索"""
        # 简单起见，合并除ID外的所有列
        cols = [c for c in df.columns if c != id_col and c != 'study_source']
        combined = df[cols[0]].astype(str)
        for col in cols[1:]:
            combined += " " + df[col].astype(str)
        return combined

    # ==========================================
    # 2. 第一阶段: BM25 (Elasticsearch 模拟)
    # ==========================================
    def build_bm25_index(self):
        print(">>> [Stage 1] 构建 BM25 索引 (模拟 Elasticsearch)...")
        # 分词 (简单的空格分词 + 去标点，实际可用 nltk/spacy)
        tokenized_corpus = [self._tokenize(doc) for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("    索引构建完成。")

    def _tokenize(self, text):
        # 简单分词：转小写 -> 去标点 -> 按空格分
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def retrieve_candidates(self, query_text, top_k=50):
        """
        使用 BM25 获取 Top-K 候选文档
        返回: List of (doc_id, doc_text)
        """
        tokenized_query = self._tokenize(query_text)
        # 获取分数
        scores = self.bm25.get_scores(tokenized_query)
        # 获取前 K 个索引
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        candidates = []
        for idx in top_n_indices:
            candidates.append({
                'doc_id': self.doc_ids[idx],
                'doc_text': self.doc_texts[idx],
                'bm25_score': scores[idx]
            })
        return candidates

    # ==========================================
    # 3. 第二阶段: Transformer Re-ranking
    # ==========================================
    def load_transformer_model(self, model_name='all-MiniLM-L6-v2'):
        print(f">>> [Stage 2] 加载 Transformer 模型: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # 如果显存足够，可以预先计算所有文档的 embedding 加速（图示是实时算的，这里为了速度优化一下）
        # 但为了严格遵守图示逻辑（只对 500 个 candidates 算），我们在 rerank 函数里实时算
        pass

    def rerank(self, trial_text, candidates):
        """
        对候选集进行语义重排序
        """
        if not candidates:
            return []
            
        # 1. 编码 Trial
        trial_emb = self.model.encode(trial_text, convert_to_tensor=True)
        
        # 2. 编码 Candidates
        cand_texts = [c['doc_text'] for c in candidates]
        cand_embs = self.model.encode(cand_texts, convert_to_tensor=True)
        
        # 3. 计算余弦相似度
        # cos_scores shape: [1, num_candidates]
        cos_scores = util.cos_sim(trial_emb, cand_embs)[0]
        
        # 4. 更新分数并排序
        for i, cand in enumerate(candidates):
            cand['transformer_score'] = cos_scores[i].item()
            
        # 按 Transformer 分数降序排列
        ranked_candidates = sorted(
            candidates, key=lambda x: x['transformer_score'], reverse=True
        )
        
        return ranked_candidates

    # ==========================================
    # 4. 运行 Pipeline 并评估
    # ==========================================
    def run_evaluation(self, bm25_top_k=50):
        """
        遍历测试集中的每个 Trial，执行 Retrieve -> Rerank，并计算 Recall
        """
        print(f">>> 开始评估 (BM25 Top-K={bm25_top_k})...")
        
        # 获取测试集中所有唯一的 NCT ID (作为查询 Query)
        test_nct_ids = self.ground_truth['nct_id'].unique()
        
        hits_at_1 = 0
        hits_at_5 = 0
        hits_at_10 = 0
        total_queries = 0
        
        test_nct_ids = test_nct_ids[:100]
        
        # 进度条
        for nct_id in tqdm(test_nct_ids):
            # 获取该 Trial 对应的真实 Document ID (可能有多个)
            true_doc_ids = set(self.ground_truth[self.ground_truth['nct_id'] == nct_id]['pubmed_id'].values)
            
            # 获取 Query 文本
            if nct_id not in self.trial_text_map:
                continue # 可能有些 ID 在 nct_trial.csv 里被过滤掉了
            
            query_text = self.trial_text_map[nct_id]
            
            # --- Step 1: BM25 Retrieval ---
            candidates = self.retrieve_candidates(query_text, top_k=bm25_top_k)
            
            # --- Step 2: Transformer Reranking ---
            # 如果候选集包含了正确答案，Transformer 有机会把它排到前面
            # 如果 BM25 没召回正确答案，Transformer 也没办法
            ranked_results = self.rerank(query_text, candidates)
            
            # --- Step 3: 计算指标 ---
            # 提取排序后的 Doc ID 列表
            pred_doc_ids = [res['doc_id'] for res in ranked_results]
            
            # 检查是否有命中
            is_hit_1 = any(pid in true_doc_ids for pid in pred_doc_ids[:1])
            is_hit_5 = any(pid in true_doc_ids for pid in pred_doc_ids[:5])
            is_hit_10 = any(pid in true_doc_ids for pid in pred_doc_ids[:10])
            
            if is_hit_1: hits_at_1 += 1
            if is_hit_5: hits_at_5 += 1
            if is_hit_10: hits_at_10 += 1
            total_queries += 1

        print("\n================ 评估结果 ================")
        print(f"Total Queries: {total_queries}")
        print(f"Recall@1:  {hits_at_1/total_queries:.4f}")
        print(f"Recall@5:  {hits_at_5/total_queries:.4f}")
        print(f"Recall@10: {hits_at_10/total_queries:.4f}")
        print("==========================================")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 路径配置
    TRIAL_PATH = 'data/nct_trial.csv'
    DOC_PATH = 'data/pubmed_document.csv'
    #TEST_PATH = 'data/dataset_test.csv' # 使用你之前生成的测试集
    TEST_PATH = 'artifacts_train/pairs_test.csv' # 使用你之前生成的测试集
    
    #model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
    model_name = "models/pubmedbert_contrastive_margin025"
    # model_name = "NeuML/pubmedbert-base-embeddings"

    # 初始化 Pipeline
    pipeline = TrialSearchPipeline(TRIAL_PATH, DOC_PATH, TEST_PATH)
    
    # 1. 构建索引
    pipeline.build_bm25_index()
    
    # 2. 加载模型
    pipeline.load_transformer_model(model_name=model_name)
    
    # 3. 运行评估
    # bm25_top_k 对应图中 "500 Candidate Articles"，因为本地数据量小，设为 50 或 100 即可演示
    pipeline.run_evaluation(bm25_top_k=50)