"""
Stage 3b: Patient-based train/test splitting.

Splits a post-processed dataset into train and test sets while ensuring
that no patient appears in both splits. This prevents data leakage,
which is critical when multiple examples can come from the same patient.

If no patient_id column is present, falls back to random splitting.

Usage:
    python train_test_split.py --input-dir /path/to/csvs --output-dir /path/to/output
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *

import argparse
import random
from typing import List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Merging multiple dataset files
# ---------------------------------------------------------------------------

def merge_datasets(
    file_configs: List[dict],
) -> pd.DataFrame:
    """Merge multiple CSV sources into a single DataFrame.

    Each entry in file_configs is a dict with:
        - 'path': str or Path to the CSV file
        - 'sample_n': (optional) int, number of rows to randomly sample
        - 'sample_frac': (optional) float, fraction of rows to sample

    At most one of sample_n / sample_frac should be set per config.

    Args:
        file_configs: List of dicts describing each CSV source.

    Returns:
        Combined DataFrame with all sources concatenated.
    """
    frames = []
    for cfg in file_configs:
        path = Path(cfg["path"])
        if not path.exists():
            print(f"Warning: {path} not found, skipping.")
            continue

        df = pd.read_csv(path)
        print(f"Loaded {path.name}: {len(df):,} rows")

        # Optional sampling
        sample_n = cfg.get("sample_n")
        sample_frac = cfg.get("sample_frac")
        if sample_n is not None:
            df = df.sample(n=min(sample_n, len(df)), random_state=RANDOM_SEED)
            print(f"  Sampled {len(df):,} rows (n={sample_n})")
        elif sample_frac is not None:
            df = df.sample(frac=sample_frac, random_state=RANDOM_SEED)
            print(f"  Sampled {len(df):,} rows (frac={sample_frac})")

        frames.append(df)

    if not frames:
        raise FileNotFoundError("No valid CSV files found to merge.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined dataset: {len(combined):,} rows")
    return combined


# ---------------------------------------------------------------------------
# Patient-based splitting
# ---------------------------------------------------------------------------

def split_train_test_by_patients(
    df: pd.DataFrame,
    test_size: float = 0.1,
    random_state: int = RANDOM_SEED,
    patient_col: str = "patient_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/test ensuring no patient overlap.

    If the patient_col is not present, falls back to a simple random split
    at the row level.

    Args:
        df: Input DataFrame.
        test_size: Fraction of *patients* (not rows) to assign to the test set.
        random_state: Seed for reproducibility.
        patient_col: Column name containing patient identifiers.

    Returns:
        Tuple of (train_df, test_df).
    """
    if patient_col not in df.columns:
        print(
            f"Warning: '{patient_col}' column not found. "
            "Falling back to random row-level split."
        )
        df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(
            drop=True
        )
        split_idx = int(len(df_shuffled) * (1 - test_size))
        return df_shuffled.iloc[:split_idx], df_shuffled.iloc[split_idx:]

    # --- Patient-based split ---
    unique_patients = df[patient_col].unique().tolist()
    n_patients = len(unique_patients)

    rng = random.Random(random_state)
    rng.shuffle(unique_patients)

    n_test = max(1, int(n_patients * test_size))
    test_patients = set(unique_patients[:n_test])
    train_patients = set(unique_patients[n_test:])

    # Sanity check: no overlap
    assert len(train_patients & test_patients) == 0, "Patient overlap detected!"

    train_df = df[~df[patient_col].isin(test_patients)].reset_index(drop=True)
    test_df = df[df[patient_col].isin(test_patients)].reset_index(drop=True)

    print(f"Patient-based split (test_size={test_size}):")
    print(f"  Unique patients: {n_patients:,}")
    print(f"  Train patients:  {len(train_patients):,}  ({len(train_df):,} rows)")
    print(f"  Test patients:   {len(test_patients):,}  ({len(test_df):,} rows)")

    return train_df, test_df


# ---------------------------------------------------------------------------
# Saving splits
# ---------------------------------------------------------------------------

def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    prefix: str = "dataset",
) -> Tuple[Path, Path]:
    """Save train and test DataFrames as CSV files.

    Args:
        train_df: Training split.
        test_df: Test split.
        output_dir: Directory to write files into (created if needed).
        prefix: Filename prefix for the output CSVs.

    Returns:
        Tuple of (train_path, test_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_train = len(train_df)
    n_test = len(test_df)

    train_path = output_dir / f"{prefix}_{n_train + n_test}_rows_train.csv"
    test_path = output_dir / f"{prefix}_{n_train + n_test}_rows_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train set: {train_path}  ({n_train:,} rows)")
    print(f"Saved test set:  {test_path}  ({n_test:,} rows)")

    return train_path, test_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Split a post-processed dataset by patient into train/test."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(GENERATED_TEXTS_DIR),
        help="Directory containing CSV file(s) to split.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(TRAINING_READY_DIR),
        help="Directory to save train/test CSV files.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of patients to hold out for testing (default: 0.1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED}).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Discover CSV files in the input directory
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    file_configs = [{"path": str(f)} for f in csv_files]

    # Merge all CSVs
    print("=" * 60)
    print("STEP 1: Merging dataset files")
    print("=" * 60)
    df = merge_datasets(file_configs)

    # Split by patients
    print("\n" + "=" * 60)
    print("STEP 2: Splitting by patient")
    print("=" * 60)
    train_df, test_df = split_train_test_by_patients(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Save
    print("\n" + "=" * 60)
    print("STEP 3: Saving splits")
    print("=" * 60)
    save_splits(train_df, test_df, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
