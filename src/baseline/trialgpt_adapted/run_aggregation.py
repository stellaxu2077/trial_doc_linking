__author__ = "qiao"

"""
Using GPT to aggregate the scores by itself.
"""

import tqdm

import json
import os
import sys
import time
import pandas as pd

from TrialGPT_ranking import trialgpt_aggregation
from collections import defaultdict


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from build_bm25_cache_from_test import build_query_map_from_test
from evaluate_from_cache import compute_metrics

if __name__ == "__main__":
	model = sys.argv[1]
	test_path= sys.argv[2]     # dataset_test.csv
	# the path of the matching results
	matching_results_path = sys.argv[3]
	output_path = sys.argv[4]

	matching_results = json.load(open(matching_results_path))


	test = pd.read_csv(test_path, dtype={"nct_id": str, "pubmed_id": str}).fillna("")
	query_text_map = build_query_map_from_test(test)



	if os.path.exists(output_path):
		output = json.load(open(output_path))
	else:
		output = {}

	# patient-level
	for nct_id, doc2pred in tqdm.tqdm(matching_results.items(), desc="Trials"):
		if nct_id not in output:
			output[nct_id] = {}

		trial_info = query_text_map.get(nct_id, "")
		if not trial_info:
			continue

		for doc_id, pred in doc2pred.items():
			if doc_id in output[nct_id]:
				continue
			if not isinstance(pred, dict):
				continue

			try:
				result = trialgpt_aggregation(trial_info, pred, model)
				output[nct_id][doc_id] = result 

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

		metrics = compute_metrics(ranked_map, ground_truth_map)
		print(metrics)