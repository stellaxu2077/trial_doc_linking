__author__ = "qiao"

"""
Using GPT to aggregate the scores by itself.
"""

import json
import pickle
import pandas as pd

import os
import sys
import time

from tqdm import tqdm

from TrialGPT import trial_doc_matching
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from build_bm25_cache_from_test import build_text, build_query_map_from_test
from evaluate_from_cache import recall_hit_at_k, compute_metrics

from collections import defaultdict


if __name__ == "__main__":
	'''
	corpus = sys.argv[1] 
	model = sys.argv[2]

	# the path of the matching results
	matching_results_path = sys.argv[3]
	results = json.load(open(matching_results_path))

	# loading the trial2info dict
	trial2info = json.load(open("dataset/trial_info.json"))
	
	# loading the patient info
	_, queries, _ = GenericDataLoader(data_folder=f"dataset/{corpus}/").load(split="test")

	# output file path
	output_path = f"results/aggregation_results_{corpus}_{model}.json"

	'''

	model = sys.argv[1]
	doc_path = sys.argv[2]      # pubmed_document.csv
	test_path = sys.argv[3]     # dataset_test.csv  
	cache_path = sys.argv[4]    # bm25_candidates_test.pkl
	output_path = sys.argv[5]   # results/llm_results.json


	if os.path.exists(output_path):
		output = json.load(open(output_path))
	else:
		output = {}


	# 1. 讀docs csv，建立 doc_id -> text 字典
	docs = pd.read_csv(doc_path, dtype={"study_id": str}).fillna("")
	test = pd.read_csv(test_path, dtype={"nct_id": str, "pubmed_id": str}).fillna("")

	docs["text"] = build_text(docs, id_col="study_id")
	doc_id_to_text = pd.Series(docs["text"].values, index=docs["study_id"]).to_dict()

	# 2. 建立 nct_id -> trial_info 字典
	query_text_map = build_query_map_from_test(test)

	# 3. 讀 pkl
	candidates_map = pickle.load(open(cache_path, "rb"))["candidates_map"]



	'''
	# patient-level
	for patient_id, info in results.items():
		# get the patient note
		patient = queries[patient_id]
		sents = sent_tokenize(patient)
		sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
		sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
		patient = "\n".join(sents)

		if patient_id not in output:
			output[patient_id] = {}
		
		# label-level, 3 label / patient
		for label, trials in info.items():
				
			# trial-level
			for trial_id, trial_results in trials.items():
				# already cached results
				if trial_id in output[patient_id]:
					continue

				if type(trial_results) is not dict:
					output[patient_id][trial_id] = "matching result error"

					with open(output_path, "w") as f:
						json.dump(output, f, indent=4)

					continue

				# specific trial information
				trial_info = trial2info[trial_id]	

				try:
					result = trialgpt_aggregation(patient, trial_results, trial_info, model)
					output[patient_id][trial_id] = result 

					with open(output_path, "w") as f:
						json.dump(output, f, indent=4)

				except:
					continue


	'''
	# test 5 trial-doc pairs
	#candidates_map = dict(list(candidates_map.items())[:2])

	for nct_id, cand_ids in tqdm(candidates_map.items(), desc="Trials"):
		#print(f"[INFO] Processing trial {nct_id} ({len(cand_ids)} candidates)...")

		if nct_id not in output:
			output[nct_id] = {}

		for doc_id in cand_ids:
			# already cached results
			if doc_id in output[nct_id]:
				continue

			trial_info = query_text_map[nct_id]
			doc_info = doc_id_to_text[doc_id]

			if not trial_info or not doc_info:
				continue

			try:
				result = trial_doc_matching(trial_info, doc_info, model)
				output[nct_id][doc_id] = result 
				#print(f"  [OK] {doc_id} score={result.get('relevance_score_R', 'N/A')}")

				with open(output_path, "w") as f:
					json.dump(output, f, indent=4)


			except Exception as e:
				print(f"  [ERROR] {doc_id}: {e}")
				import traceback
				traceback.print_exc()
				continue






	# 建立 ranked_map
	ranked_map = {}
	for nct_id, doc2result in output.items():
		# 按 relevance_score_R 排序
		sorted_docs = sorted(
			doc2result.items(),
			key=lambda x: x[1].get("relevance_score_R", 0) if isinstance(x[1], dict) else 0,
			reverse=True
		)
		ranked_map[nct_id] = [doc_id for doc_id, _ in sorted_docs]

	# 建立 ground truth map
	ground_truth_map = defaultdict(set)
	for row in test[test["label"] == 1].itertuples(index=False):
		ground_truth_map[str(row.nct_id)].add(str(row.pubmed_id))

	# 計算 metrics
	metrics = compute_metrics(ranked_map, ground_truth_map)
	print(metrics)