"""
Stage 3c: Token length analysis and filtering.

Tokenizes the training-ready dataset with the target model's tokenizer
and reports distribution statistics. Optionally filters examples to a
specified token range (e.g., 500-5000 tokens) and saves the filtered
dataset for use in fine-tuning.

Usage:
    python token_stats.py --input dataset.json --model google/medgemma-27b-text-it
    python token_stats.py --input dataset.json --min-tokens 500 --max-tokens 5000 --save
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *

import argparse
import os

import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def compute_token_lengths(dataset: Dataset, tokenizer) -> list:
    """Tokenize every example and return a list of token counts.

    Args:
        dataset: HuggingFace Dataset with a 'text' column.
        tokenizer: A HuggingFace tokenizer instance.

    Returns:
        List of integer token lengths, one per example.
    """
    lengths = []
    for example in tqdm(dataset, desc="Tokenizing"):
        tokens = tokenizer(example["text"], max_length=25000, truncation=False)
        lengths.append(len(tokens["input_ids"]))
    return lengths


def print_stats(name: str, lengths: list):
    """Print summary statistics for a list of token lengths.

    Args:
        name: Label for the dataset (e.g. "Train", "Filtered").
        lengths: List of integer token lengths.
    """
    arr = np.array(lengths)
    print(f"\n{'=' * 50}")
    print(f"Token Length Statistics: {name}")
    print(f"{'=' * 50}")
    print(f"  Count:    {len(arr):,}")
    print(f"  Min:      {arr.min():,}")
    print(f"  Max:      {arr.max():,}")
    print(f"  Mean:     {arr.mean():,.1f}")
    print(f"  Median:   {np.median(arr):,.1f}")
    print(f"  Std dev:  {arr.std():,.1f}")
    print(f"  P5:       {np.percentile(arr, 5):,.0f}")
    print(f"  P25:      {np.percentile(arr, 25):,.0f}")
    print(f"  P75:      {np.percentile(arr, 75):,.0f}")
    print(f"  P95:      {np.percentile(arr, 95):,.0f}")
    print(f"{'=' * 50}")


def filter_by_token_range(
    dataset: Dataset, lengths: list, min_tokens: int, max_tokens: int
) -> tuple:
    """Filter a dataset to keep only examples within a token range.

    Args:
        dataset: HuggingFace Dataset.
        lengths: Corresponding token lengths (same order as dataset).
        min_tokens: Minimum token count (inclusive).
        max_tokens: Maximum token count (inclusive).

    Returns:
        Tuple of (filtered_dataset, filtered_lengths).
    """
    keep_indices = [
        i for i, length in enumerate(lengths) if min_tokens <= length <= max_tokens
    ]
    filtered_dataset = dataset.select(keep_indices)
    filtered_lengths = [lengths[i] for i in keep_indices]

    n_removed = len(dataset) - len(filtered_dataset)
    print(
        f"Filtered to [{min_tokens}, {max_tokens}] tokens: "
        f"{len(filtered_dataset):,} kept, {n_removed:,} removed"
    )
    return filtered_dataset, filtered_lengths


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token lengths and optionally filter a dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/medgemma-27b-text-it",
        help="HuggingFace model name for the tokenizer.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=500,
        help="Minimum token length to keep (default: 500).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5000,
        help="Maximum token length to keep (default: 5000).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="If set, save the filtered dataset to a JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the filtered dataset. Auto-generated if not provided.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(HF_CACHE_DIR),
        help="HuggingFace cache directory for model downloads.",
    )
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        token=HF_TOKEN if HF_TOKEN else None,
    )

    # Load dataset
    print(f"Loading dataset: {args.input}")
    dataset = load_dataset("json", data_files=args.input, split="train")
    print(f"Dataset size: {len(dataset):,} examples")

    # Compute token lengths
    lengths = compute_token_lengths(dataset, tokenizer)
    print_stats("Full Dataset", lengths)

    # Filter by token range
    filtered_dataset, filtered_lengths = filter_by_token_range(
        dataset, lengths, args.min_tokens, args.max_tokens
    )
    print_stats("Filtered Dataset", filtered_lengths)

    # Optionally save the filtered dataset
    if args.save:
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input)
            output_path = str(
                input_path.parent
                / f"{input_path.stem}_filtered_{args.min_tokens}_{args.max_tokens}.json"
            )

        filtered_dataset.to_json(output_path)
        print(f"\nSaved filtered dataset to: {output_path}")


if __name__ == "__main__":
    main()
