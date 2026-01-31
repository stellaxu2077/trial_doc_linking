# train_contrastive_pairs.py
# ------------------------------------------------------------
# Contrastive fine-tuning directly from labeled PAIRS (trial_info, doc_info, label).
#
# Input CSVs (required columns):
#   - dataset_train.csv: nct_id,pubmed_id, trial_info, doc_info, label
#   - dataset_valid.csv: nct_id,pubmed_id, trial_info, doc_info, label
#
# Training:
#   - SentenceTransformers bi-encoder
#   - OnlineContrastiveLoss(margin=0.25)
#
# Output:
#   - saved model folder (SentenceTransformer compatible)
#
# Usage:
#   python train_contrastive_pairs.py \
#       --train_path data/dataset_train.csv \
#       --valid_path data/dataset_valid.csv \
#       --base_model pritamdeka/S-PubMedBert-MS-MARCO \
#       --save_dir models/pubmedbert_pairs_contrastive_margin025 \
#       --epochs 2 --batch_size 16 --lr 2e-5 --margin 0.25
# ------------------------------------------------------------

import os
import json
import random
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator

'''
def force_cpu():
    """Force CPU-only execution (disable CUDA & MPS)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Monkey-patch MPS availability off (helps on macOS)
    if hasattr(torch.backends, "mps"):
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False

    print("=" * 60)
    print("FORCED CPU MODE")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends,'mps') else False}")
    print("=" * 60)
'''

def require_columns(df: pd.DataFrame, cols, name: str):
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(list(missing))}. Found: {list(df.columns)}")


def normalize_label(x):
    """
    Normalize label to float in {0.0, 1.0}.
    Accepts: 0/1, "0"/"1", True/False, etc.
    """
    try:
        v = float(x)
    except Exception:
        v = 1.0 if str(x).strip().lower() in {"true", "t", "yes", "y"} else 0.0
    return 1.0 if v >= 0.5 else 0.0


def load_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).fillna("")
    require_columns(df, ["trial_info", "doc_info", "label"], os.path.basename(csv_path))

    # Normalize types
    df["trial_info"] = df["trial_info"].astype(str)
    df["doc_info"] = df["doc_info"].astype(str)
    df["label"] = df["label"].apply(normalize_label).astype(float)

    return df


def df_to_examples(df: pd.DataFrame, max_examples: int | None = None, seed: int = 42):
    """
    Convert dataframe rows to SentenceTransformers InputExamples.
    Each row produces 1 pair example: (trial_info, doc_info, label).
    """
    if max_examples is not None and len(df) > max_examples:
        df = df.sample(n=max_examples, random_state=seed).reset_index(drop=True)

    examples = [
        InputExample(texts=[row.trial_info, row.doc_info], label=float(row.label))
        for row in df.itertuples(index=False)
    ]
    return examples


def build_binary_evaluator(df_valid: pd.DataFrame, name: str = "valid"):
    """
    Evaluator for pair classification using cosine similarity thresholding.
    SentenceTransformers will report metrics like accuracy/F1 (depends on version).
    """
    s1 = df_valid["trial_info"].tolist()
    s2 = df_valid["doc_info"].tolist()
    labels = df_valid["label"].tolist()
    return BinaryClassificationEvaluator(
        s1, s2, labels, 
        name=name,
        write_csv=True
        )


def train(args):
    #force_cpu()

    print(">>> Loading train/valid pairs...")
    train_df = load_pairs(args.train_path)
    valid_df = load_pairs(args.valid_path)

    print("\n=== Dataset stats ===")
    print(f"Train rows: {len(train_df):,} | pos={(train_df.label==1).sum():,} | neg={(train_df.label==0).sum():,}")
    print(f"Valid rows: {len(valid_df):,} | pos={(valid_df.label==1).sum():,} | neg={(valid_df.label==0).sum():,}")

    # Optional quick sanity checks for empty texts
    empty_trial = (train_df["trial_info"].str.len() == 0).sum()
    empty_doc = (train_df["doc_info"].str.len() == 0).sum()
    if empty_trial or empty_doc:
        print(f"WARNING: empty trial_info={empty_trial:,}, empty doc_info={empty_doc:,} in TRAIN.")

    print("\n>>> Building InputExamples...")
    train_examples = df_to_examples(train_df, max_examples=args.max_train_pairs, seed=args.seed)
    print(f"Train examples: {len(train_examples):,}")

    # DataLoader
    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
        #pin_memory=False,   # important for CPU-only stability
        num_workers=0       # safest cross-platform
    )

    print("\n>>> Initializing SentenceTransformer...")
    device="cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.base_model, device=device)
    print(f"Model device: {model.device}")

    # Loss: OnlineContrastiveLoss with cosine distance (typical for bi-encoders)
    train_loss = losses.OnlineContrastiveLoss(
        model=model,
        distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
        margin=args.margin
    )

    # Evaluator on valid set (optional but recommended)
    evaluator = None
    if args.do_valid_eval:
        evaluator = build_binary_evaluator(valid_df, name="valid")
        print(">>> Valid evaluator: BinaryClassificationEvaluator enabled.")

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    print("\n=== Training config ===")
    print(f"Base model: {args.base_model}")
    print(f"Epochs: {args.epochs} | batch_size: {args.batch_size} | lr: {args.lr} | margin: {args.margin}")
    print(f"Steps: {total_steps:,} | warmup_steps: {warmup_steps:,}")
    print(f"Save dir: {args.save_dir}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Save config snapshot
    with open(os.path.join(args.save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        evaluator=evaluator,
        evaluation_steps=args.evaluation_steps if evaluator is not None else 0,
        save_best_model=True,
        output_path=args.save_dir,
        show_progress_bar=True 
    )

    # If no evaluator, ensure model is saved
    if evaluator is None:
        model.save(args.save_dir)

    print(f"\n>>> Done. Model saved to: {args.save_dir}")


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train_path", default="data/dataset_train.csv")
    parser.add_argument("--valid_path", default="data/dataset_valid.csv")

    # Model / output
    parser.add_argument("--base_model", default="pritamdeka/S-PubMedBert-MS-MARCO")
    parser.add_argument("--save_dir", default="models/pubmedbert_pairs_contrastive_margin025")

    # Training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Valid eval
    parser.add_argument("--do_valid_eval", action="store_true", help="Enable valid evaluation & save_best_model.")
    parser.add_argument("--evaluation_steps", type=int, default=1000, help="Evaluate every N steps (only if do_valid_eval).")

    # Control size (useful for quick debug)
    parser.add_argument("--max_train_pairs", type=int, default=None, help="Optionally subsample train pairs.")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Ensure reproducibility-ish
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
