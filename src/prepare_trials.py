
"""
prepare_linked_trials.py

Prepare trial entities that appear in the trial–publication pair dataset.

This script:
- Reads the trial–publication pair CSV
- Extracts unique NCT IDs
- Saves trials as a JSON file
"""

import json
from pathlib import Path

import pandas as pd



PAIR_CSV = Path("data/dataset_train.csv")   # nct_id, pubmed_id, ..., label
TRIAL_CSV = Path("data/nct_trial.csv")      # trial entity table
OUT_JSON  = Path("data/linked_trials.json")




# 1) Load pair data and collect unique nct_ids
pairs = pd.read_csv(PAIR_CSV, dtype={"nct_id": "string"})
nct_ids = (
    pairs["nct_id"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

print(f"[INFO] Unique nct_ids in pair file: {len(nct_ids)}")

# 2) Load trial entity table
trials = pd.read_csv(TRIAL_CSV, dtype={"study_id": "string"})
trials["study_id"] = trials["study_id"].astype(str).str.strip()

# 3) Filter trials that appear in the pair dataset
linked_trials_df = trials[trials["study_id"].isin(nct_ids)].copy()
print(f"[INFO] Matched trials in trial table: {len(linked_trials_df)}")

# 4) Convert NaN -> None for JSON compatibility
linked_trials_df = linked_trials_df.where(pd.notnull(linked_trials_df), None)

# 5) Save to JSON
records = linked_trials_df.to_dict(orient="records")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"[INFO] Saved linked trials to: {OUT_JSON}")










'''

import pandas as pd
import json

PAIR_CSV = "data/dataset_train.csv"    
TRIAL_CSV = "data/nct_trial.csv"      
OUT_JSON  = "data/linked_trials.json"
OUT_MISS  = "data/missing_nct_ids.txt"


pairs = pd.read_csv(PAIR_CSV, dtype={"nct_id": "string"})
nct_ids = pairs["nct_id"].dropna().astype(str).str.strip()
nct_ids = sorted(set(nct_ids))
print(f"Unique nct_ids in pair file: {len(nct_ids)}")


trials = pd.read_csv(TRIAL_CSV, dtype={"study_id": "string"})
trials["study_id"] = trials["study_id"].astype(str).str.strip()


linked_trials_df = trials[trials["study_id"].isin(nct_ids)].copy()
print(f"Matched trials in nct_trial.csv: {len(linked_trials_df)}")


linked_trials_df = linked_trials_df.where(pd.notnull(linked_trials_df), None)
records = linked_trials_df.to_dict(orient="records")
print(f"Prepared {len(records)} linked trial records for output.")

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)


print(f"Saved matched trials to: {OUT_JSON}")
'''