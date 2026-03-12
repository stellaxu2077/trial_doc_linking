__author__ = "qiao"

"""
TrialGPT-Ranking main functions.
"""

import json

import time
import os

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))





def convert_aspect_to_string(
		prediction: dict
) -> str:
	aspect_names = {
		"1": "Study Population",
		"2": "Interventions",
		"3": "Reported Outcomes"
	}

	output = ""

	for aspect_idx, preds in prediction.items():
		name = aspect_names[aspect_idx]
		output += f"Aspect {aspect_idx} - {name}:\n"
		output += f"\tReasoning: {preds[0]}\n"
		output += f"\tLabel: {preds[1]}\n"

	return output



def convert_pred_to_prompt(
		trial_info: str,
		pred: dict,
) -> tuple[str, str]:
	"""Convert the prediction to a prompt string."""

	# then get the prediction strings
	pred_str = convert_aspect_to_string(pred)

	'''	
	# construct the prompt
	prompt = "You are a helpful assistant for clinical trial recruitment. You will be given a patient note, a clinical trial, and the patient eligibility predictions for each criterion.\n"
	prompt += "Your task is to output two scores, a relevance score (R) and an eligibility score (E), between the patient and the clinical trial.\n"
	prompt += "First explain the consideration for determining patient-trial relevance. Predict the relevance score R (0~100), which represents the overall relevance between the patient and the clinical trial. R=0 denotes the patient is totally irrelevant to the clinical trial, and R=100 denotes the patient is exactly relevant to the clinical trial.\n"
	prompt += "Then explain the consideration for determining patient-trial eligibility. Predict the eligibility score E (-R~R), which represents the patient's eligibility to the clinical trial. Note that -R <= E <= R (the absolute value of eligibility cannot be higher than the relevance), where E=-R denotes that the patient is ineligible (not included by any inclusion criteria, or excluded by all exclusion criteria), E=R denotes that the patient is eligible (included by all inclusion criteria, and not excluded by any exclusion criteria), E=0 denotes the patient is neutral (i.e., no relevant information for all inclusion and exclusion criteria).\n"
	prompt += 'Please output a JSON dict formatted as Dict{"relevance_explanation": Str, "relevance_score_R": Float, "eligibility_explanation": Str, "eligibility_score_E": Float}.'
	'''
	prompt = "You are a helpful assistant for linking clinical trials to their publications. "
	prompt += "You will be given a clinical trial, and the aspect-level analysis of a publication.\n"
	prompt += "Your task is to output a relevance score (R) between the clinical trial and the publication.\n"
	prompt += "First explain your reasoning for determining the overall relevance. "
	prompt += "Then predict the relevance score R (0~100), which represents the likelihood that the publication reports the results of the clinical trial. "
	prompt += "R=0 denotes the publication is totally irrelevant to the clinical trial, "
	prompt += "and R=100 denotes the publication exactly reports the results of the clinical trial.\n"
	prompt += 'Please output a JSON dict formatted as Dict{"relevance_explanation": Str, "relevance_score_R": Float}.'
	
	'''
	user_prompt = "Here is the patient note:\n"
	user_prompt += patient + "\n\n"
	user_prompt += "Here is the clinical trial description:\n"
	user_prompt += trial + "\n\n"
	user_prompt += "Here are the criterion-level eligibility prediction:\n"
	user_prompt += pred + "\n\n"
	user_prompt += "Plain JSON output:"
	'''

	user_prompt = "Here is the clinical trial:\n"
	user_prompt += trial_info + "\n\n"
	user_prompt += "Here are the aspect-level predictions:\n"
	user_prompt += pred_str + "\n\n"
	user_prompt += "Plain JSON output:"

	return prompt, user_prompt



def trialgpt_aggregation(trial_info: str, pred: dict, model: str):
	system_prompt, user_prompt = convert_pred_to_prompt(
			trial_info,
			pred
	)   

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt}
	]

	response = client.chat.completions.create(
		model=model,
		messages=messages,
		temperature=0,
	)
	result = response.choices[0].message.content.strip()



	if result.startswith("```"):
		result = result.split("```")[1]
		if result.startswith("json"):
			result = result[4:]
	result = result.strip()

	if not result:
		return {"relevance_score_R": 0, "relevance_explanation": "empty response"}

	try:
		result = json.loads(result)
	except json.JSONDecodeError:
		return {"relevance_score_R": 0, "relevance_explanation": "parse error"}

	return result