__author__ = "qiao"

"""
Running the TrialGPT matching for three cohorts (sigir, TREC 2021, TREC 2022).

This script compares clinical trial eligibility criteria against patient information
to determine if a patient matches each trial. It processes multiple datasets and saves
the matching results to a JSON file.
"""

# Import necessary libraries
import json
import pickle
import pandas as pd
                                  # For reading/writing JSON files

import os                                      # For checking file paths
import sys                                     # For reading command-line arguments

# Import the matching function from TrialGPT module
from TrialGPT_matching import trial_doc_matching 	
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from build_bm25_cache_from_test import build_text, build_query_map_from_test

from tqdm import tqdm



if __name__ == "__main__":


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







	