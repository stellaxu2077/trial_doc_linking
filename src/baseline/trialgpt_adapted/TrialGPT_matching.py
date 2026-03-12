__author__ = "qiao"

"""
TrialGPT-Matching main functions.

This module provides the core matching functionality to determine if a patient
matches clinical trial eligibility criteria using Azure OpenAI's GPT models.
It analyzes inclusion and exclusion criteria separately and provides detailed
reasoning for each criterion.
"""

# Import necessary libraries
#from email.mime import message
import json                           # For parsing and creating JSON responses
#from TrialGPT import results
#from nltk.tokenize import sent_tokenize  # For splitting text into sentences
#import time                           # For tracking execution timing (if needed)
import os                             # For accessing environment variables
'''
# Import Azure OpenAI API client
from openai import AzureOpenAI

# Initialize the Azure OpenAI client with API credentials from environment variables
# These credentials are stored as environment variables for security
client = AzureOpenAI(
	api_version="2023-09-01-preview",               # Azure OpenAI API version
	azure_endpoint=os.getenv("OPENAI_ENDPOINT"),   # Azure OpenAI endpoint URL
	api_key=os.getenv("OPENAI_API_KEY"),           # API key for authentication
)
'''

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))




def get_matching_prompt(
    trial_info: str,
    doc_info: str,
) -> tuple[str, str]:
    """
    Generate system and user prompts for matching a trial against a publication.
    
    Args:
        trial_info: Clinical trial information as a string
        doc_info: Publication information as a string
    
    Returns:
        Tuple of (system_prompt, user_prompt) for the OpenAI API call
    """

    """
    Generate system and user prompts for matching a patient against trial criteria.
    
    This function creates detailed prompts that instruct GPT to analyze patient eligibility
    against clinical trial criteria, providing reasoning and referencing specific sentences.
    
    Args:
        trial_info: Dictionary containing trial information
        inc_exc: "inclusion" or "exclusion" - determines which criteria to evaluate
        patient: Patient note with numbered sentences
    
    Returns:
        Tuple of (system_prompt, user_prompt) for the OpenAI API call
    """
    
    # Start with the system prompt that sets up the AI's role and task
    '''
    prompt = f"You are a helpful assistant for clinical trial recruitment. \
        Your task is to compare a given patient note and the {inc_exc} criteria \
            of a clinical trial to determine the patient's eligibility at the criterion level.\n"
    '''
    prompt = f"You are a helpful assistant for linking clinical trials to their publications. \
        Your task is to determine whether a publication reports the results of a clinical trial \
            by examining three aspects: study population, interventions, and reported outcomes.\n"


    '''
    # Add context about what inclusion/exclusion criteria are
    if inc_exc == "inclusion":
        # Explain inclusion criteria to the AI
        prompt += "The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"
    
    elif inc_exc == "exclusion":
        # Explain exclusion criteria to the AI
        prompt += "The factors that disqualify someone from participating are called exclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"

    # Instruct the AI to analyze each criterion and provide three specific outputs
    prompt += f"You should check the {inc_exc} criteria one-by-one, and output the following three elements for each criterion:\n"
    '''

    prompt += "You should analyze the following three aspects one-by-one. "
    prompt += "For each aspect, briefly generate your reasoning process, check if the document information contains direct evidence. "
    prompt += "If so, judge whether the document reports the aspect. "
    prompt += "If there is no direct evidence, try to infer from existing evidence.\n\n"

    prompt += "Aspect 1. Study Population: whether the publication studies the same patient population as the clinical trial.\n"
    prompt += "Aspect 2. Interventions: whether the publication has the same interventions as the clinical trial.\n"
    prompt += "Aspect 3. Reported Outcomes: whether the publication reports effective outcomes.\n"

    prompt += "For each aspect, output two elements:\n"
    prompt += "\tElement 1. Brief reasoning.\n"
    prompt += "\tElement 2. A label:\n"
    prompt += "\t\t- For Aspect 1 and 2, choose from: {\"matched\", \"not matched\", \"not enough information\"}\n"
    prompt += "\t\t- For Aspect 3, choose from: {\"included\", \"not included\", \"not enough information\"}\n\n"



    '''
    # Specify the exact JSON format required for the response
    prompt += "You should output only a JSON dict exactly formatted as: dict{str(criterion_number): list[str(element_1_brief_reasoning), list[int(element_2_sentence_id)], str(element_3_eligibility_label)]}."
    '''

    prompt += "You should output only a JSON dict exactly formatted as: "
    prompt += "dict{str(aspect_number): list[str(element_1_brief_reasoning), str(element_2_label)]}.\n"



    '''
    # Build the user prompt with actual patient and trial data
    # This provides the specific data to analyze
    user_prompt = f"Here is the patient note, each sentence is led by a sentence_id:\n{patient}\n\n" 
    
    # Add the formatted trial information with only the relevant criteria type
    user_prompt += f"Here is the clinical trial:\n{print_trial(trial_info, inc_exc)}\n\n"
    
    # Instruct the AI to output in plain JSON format
    user_prompt += f"Plain JSON output:"

    '''

    user_prompt = "Here is basic information of the clinical trial:\n"
    user_prompt += trial_info + "\n\n"
    user_prompt += "Here is the publication:\n"
    user_prompt += doc_info + "\n\n"
    user_prompt += "Plain JSON output:"


    # Return both prompts as a tuple
    return prompt, user_prompt






def trial_doc_matching(trial_info: str, doc_info: str, model: str) -> dict:
    """
    Match a patient against trial criteria using Azure OpenAI GPT model.
    
    This function analyzes both inclusion and exclusion criteria separately,
    calling GPT twice - once for each criterion type - to provide detailed analysis.
    
    Args:
        trial: Dictionary containing trial information (criteria, title, etc.)
        patient: Patient note with numbered sentences
        model: Name of the Azure OpenAI model to use (e.g., "gpt-4", "gpt-35-turbo")
    
    Returns:
        Dictionary with two keys: "inclusion" and "exclusion", each containing
        the AI's analysis of whether the patient meets each criterion
    """
    
    # Initialize empty dictionary to store results for both criterion types
    results = {}

    # Analyze inclusion and exclusion criteria in separate API calls
    # This allows the AI to focus on each criterion type separately

    # Generate the system and user prompts for this criterion type
    system_prompt, user_prompt = get_matching_prompt(trial_info, doc_info)
        
    # Build the message list for the OpenAI API
    # System prompt sets the AI's instructions and context
    # User prompt provides the specific data to analyze
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Call Azure OpenAI API to get GPT's analysis
    # Temperature=0 means deterministic output (no randomness)
    response = client.chat.completions.create(
        model=model,              # The GPT model to use
        messages=messages,        # The formatted prompts
        temperature=0,            # Deterministic responses
    )

    # Extract the response text from the API response
    message = response.choices[0].message.content.strip()
        
    # Remove markdown code block markers (backticks and "json" label)
    # GPT sometimes wraps JSON in ```json ... ``` which we need to remove
    #message = message.strip("`").strip("json")


    # Try to parse the response as JSON
    if message.startswith("```"):
        message = message.split("```")[1]
        if message.startswith("json"):
            message = message[4:]
    message = message.strip()

    if not message:
        return {}

    try:
        results = json.loads(message)
    except json.JSONDecodeError:
        results = {}

    return results