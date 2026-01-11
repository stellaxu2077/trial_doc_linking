"""
prepare_pubmed_corpus.py

Prepare PubMed document corpus for downstream retrieval and embedding.

This script:
- Loads publication data from a CSV file
- Converts it to a list of records
- Saves the corpus as a JSON file
"""

import json
from pathlib import Path

import pandas as pd



INPUT_CSV = Path("data/pubmed_document.csv")
OUTPUT_JSON = Path("data/pubmed_document.json")



print(f"[INFO] Loading publication data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

print(f"[INFO] Loaded {len(df)} rows.")
print("[INFO] Example row:")
print(df.iloc[0])

# Convert DataFrame to list of dictionaries
corpus = df.to_dict(orient="records")
print(f"[INFO] Converted to {len(corpus)} corpus records.")

# Ensure output directory exists
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# Save corpus to JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=2)

print(f"[INFO] Corpus saved to: {OUTPUT_JSON}")




















'''

import pandas as pd
import json



df = pd.read_csv('data/pubmed_document.csv')
print(f"Loaded publication data with {len(df)} rows.")
print("Example row:")
print(df.iloc[0])

corpus_list = df.to_dict(orient="records")
print(f"Converted to list of {len(corpus_list)} records.")

file_path = "data/pubmed_document.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(corpus_list, f, ensure_ascii=False, indent=4)
print(f"Data saved to: {file_path}")

'''


