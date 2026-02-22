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
from turtle import pd

from numpy import test                                    # For reading/writing JSON files
from baseline.build_bm25_cache_from_test import build_query_map_from_test
from nltk.tokenize import sent_tokenize       # Tool to split text into sentences
import os                                      # For checking file paths
import sys                                     # For reading command-line arguments

# Import the matching function from TrialGPT module
from TrialGPT import trialgpt_matching 

from build_bm25_cache_from_test import build_text, build_query_map_from_test


if __name__ == "__main__":
	# This code only runs when the script is executed directly (not imported)
	
	# Read command-line arguments passed when running the script
	# Example: python run_matching.py sigir gpt-4
	corpus = sys.argv[1]   # Dataset name (e.g., "sigir", "trec_2021", "trec_2022")
	model = sys.argv[2]    # AI model to use for matching (e.g., "gpt-4", "gpt-3.5-turbo")
	
	# Load the dataset containing patient info and their retrieved trials
	# This file contains patient descriptions and a list of candidate trials
	#dataset = json.load(open(f"dataset/{corpus}/retrieved_trials.json"))
	dataset = pickle.load(open(f"data/bm25_candidates_test.pkl", "rb"))
	candidates_map = dataset["candidates_map"]

	doc_id_to_text = pd.Series(docs["text"].values, index=docs["study_id"]).to_dict()
    query_text_map  = build_query_map_from_test(test)  # nct_id -> trial_info

	# Define the output file path where results will be saved
	# This creates a unique filename based on the dataset and model used
	output_path = f"results/matching_results_{corpus}_{model}.json" 

	# Check if output file already exists (resume from previous run)
	# This structure stores results organized by: patient_id -> label -> trial_id -> matching_result
	# Labels represent: "0" = exclude, "1" = uncertain, "2" = include
	if os.path.exists(output_path):
		# If file exists, load the existing results to continue from where we left off
		output = json.load(open(output_path))
	else:
		# If file doesn't exist, create an empty dictionary to store new results
		output = {}

	# Loop through each patient case in the dataset
	for instance in dataset:
		# Extract basic patient information from the current case
		# Dict structure: {'patient': str(patient_description), '0': list(trials), ...}
		#patient_id = instance["patient_id"]  # Unique identifier for the patient
		#patient = instance["patient"]        # Patient's medical description/information
		trial_id = instance["trial_id"]
		trial_info = instance["trial_info"]
		# Convert patient text into individual sentences
		# This helps organize the patient information into logical chunks
		#sents = sent_tokenize(patient)
		sents = sent_tokenize(trial_info)
		
		# Add a standard sentence about patient consent
		# This is automatically assumed for all patient-trial matching scenarios
		sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
		
		# Add numbering to each sentence (e.g., "0. First sentence", "1. Second sentence")
		# This makes it easier for the AI model to reference specific parts of patient info
		sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
		
		# Combine all numbered sentences with line breaks for better formatting
		patient = "\n".join(sents)

		# Initialize storage for this patient if it's the first time processing them
		# Create empty dictionaries for each label category (0=exclude, 1=uncertain, 2=include)
		if patient_id not in output:
			output[patient_id] = {"0": {}, "1": {}, "2": {}}
		# Process trials in reverse order (label "2" first, then "1", then "0")
		# Labels represent: "2"=likely included, "1"=uncertain, "0"=likely excluded
		for label in ["2", "1", "0"]:
			# Skip this label if it doesn't exist in the current patient's data
			if label not in instance: 
				continue

			# Loop through each trial in this label category for the current patient
			for trial in instance[label]: 
				# Extract the unique trial identifier (NCT number)
				trial_id = trial["NCTID"]

				# Skip this trial if we've already computed the matching result (caching)
				# This saves time and API calls by not reprocessing the same patient-trial pair
				if trial_id in output[patient_id][label]:
					continue
				
				# Wrap in try-except to handle any errors gracefully
				# (e.g., API failures, network issues, missing data)
				try:
					# Call the TrialGPT matching function to determine if patient matches trial
					# The function returns a detailed matching result explaining the decision
					results = trialgpt_matching(trial, patient, model)
					
					# Store the matching result for this patient-trial pair
					output[patient_id][label][trial_id] = results

					# Save results to file after each successful match
					# This ensures no data is lost if the script crashes or is interrupted
					with open(output_path, "w") as f:
						json.dump(output, f, indent=4)

				# Handle any errors that occur during matching
				except Exception as e:
					# Print the error message for debugging purposes
					print(e)
					# Continue to the next trial instead of stopping the entire script
					continue
