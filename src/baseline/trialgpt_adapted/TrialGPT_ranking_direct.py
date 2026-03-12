__author__ = "qiao"

"""
TrialGPT-Ranking main functions.
"""

import json

import time
import os

'''
from openai import AzureOpenAI

client = AzureOpenAI(
	api_version="2023-09-01-preview",
	azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
	api_key=os.getenv("OPENAI_API_KEY"),
)
'''

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_matching_prompt(
		#patient: str,
		#trial_info: dict,
		trial_info: str,
		doc_info: str,
#) -> str:
) -> tuple[str, str]:
	"""Convert the prediction to a prompt string."""
	# get the trial string
	#trial = f"Title: {trial_info['brief_title']}\n"
	#trial += f"Target conditions: {', '.join(trial_info['diseases_list'])}\n"
	#trial += f"Summary: {trial_info['brief_summary']}"


	# construct the prompt
	#prompt = "You are a helpful assistant for clinical trial recruitment. You will be given a patient note, a clinical trial, and the patient eligibility predictions for each criterion.\n"
	prompt = "You are a helpful assistant for linking clinical trials to their publications. You will be given a clinical trial and a publication.\n"
	#prompt += "Your task is to output two scores, a relevance score (R) and an eligibility score (E), between the patient and the clinical trial.\n"
	prompt += "Your task is to determine whether the publication reports the results of the clinical trial, and output a relevance score R between the clinical trial and the publication.\n"
	#prompt += "First explain the consideration for determining patient-trial relevance. Predict the relevance score R (0~100), which represents the overall relevance between the patient and the clinical trial. R=0 denotes the patient is totally irrelevant to the clinical trial, and R=100 denotes the patient is exactly relevant to the clinical trial.\n"
	prompt += "First explain the consideration for determining trial-publication relevance. Predict the relevance score R (0~100), which represents the overall relevance between the clinical trial and the publication. R=0 denotes the publication is totally irrelevant to the clinical trial, and R=100 denotes the publication is exactly relevant to the clinical trial.\n"
	#prompt += "Then explain the consideration for determining patient-trial eligibility. Predict the eligibility score E (-R~R), which represents the patient's eligibility to the clinical trial. Note that -R <= E <= R (the absolute value of eligibility cannot be higher than the relevance), where E=-R denotes that the patient is ineligible (not included by any inclusion criteria, or excluded by all exclusion criteria), E=R denotes that the patient is eligible (included by all inclusion criteria, and not excluded by any exclusion criteria), E=0 denotes the patient is neutral (i.e., no relevant information for all inclusion and exclusion criteria).\n"
	#prompt += 'Please output a JSON dict formatted as Dict{"relevance_explanation": Str, "relevance_score_R": Float, "eligibility_explanation": Str, "eligibility_score_E": Float}.'
	prompt += 'Please output a JSON dict formatted as Dict{"relevance_explanation": Str, "relevance_score_R": Float}.'


	#user_prompt = "Here is the patient note:\n"
	user_prompt = "Here is the clinical trial text:\n"
	#user_prompt += patient + "\n\n"
	user_prompt += trial_info + "\n\n"
	#user_prompt += "Here is the clinical trial description:\n"
	user_prompt += "Here is the publication text:\n"
	#user_prompt += trial + "\n\n"
	user_prompt += doc_info + "\n\n"
	user_prompt += "Plain JSON output:"

	return prompt, user_prompt


def trial_doc_matching(
		trial_info: str, 
		doc_info: str,
		model: str
		):
	system_prompt, user_prompt = get_matching_prompt(
		trial_info=trial_info,
		doc_info=doc_info,
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

	'''
	result = result.strip("`").strip("json")

	if not result:
		return {"relevance_score_R": 0, "relevance_explanation": "empty response"}

	result = json.loads(result)
	'''

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




