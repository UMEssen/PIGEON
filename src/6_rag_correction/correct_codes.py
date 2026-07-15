"""
Stage 6 -- Apply RAG-based code corrections to inference results.

This script is the main entry point for Stage 6 of the PIGEON pipeline.
It reads the CSV produced by Stage 5 (inference), parses the JSON in the
``parsed_generation`` column, and recursively walks the nested structure to
correct medical codes:

- **ICD-10-GM**: ``icd10gm_code`` -> ``icd10gm_code_rag``
- **ATC**:       ``atc_code``     -> ``atc_code_rag``
- **OPS**:       ``ops_code``     -> ``ops_code_rag``

The corrected JSON is written back to the same column (or a new column,
depending on configuration) and saved to a new CSV.

Usage
-----
::

    python correct_codes.py \\
        --input  results/inference_results.csv \\
        --output results/inference_results_rag.csv \\
        --correct-icd \\
        --correct-ops \\
        --correct-atc

Checkpoints are saved every 100 rows (configurable with ``--checkpoint-every``)
so that long runs can be resumed.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR, ensure_dirs

from unified_corrector import UnifiedRAGCorrector


# ------------------------------------------------------------------
# Recursive JSON walker
# ------------------------------------------------------------------

def add_rag_corrections(
    data: Any,
    corrector: UnifiedRAGCorrector,
    correct_icd: bool = True,
    correct_ops: bool = False,
    correct_atc: bool = False,
) -> Any:
    """Recursively walk a parsed JSON structure and add RAG-corrected codes.

    For every dict that contains an ``icd10gm_code`` / ``atc_code`` /
    ``ops_code`` key, the corrected value is stored in a sibling key
    with the ``_rag`` suffix.

    Parameters
    ----------
    data : dict or list
        Parsed JSON from ``parsed_generation``.
    corrector : UnifiedRAGCorrector
        Initialised corrector instance.
    correct_icd, correct_ops, correct_atc : bool
        Toggle which code families to correct.

    Returns
    -------
    The same data structure with ``*_rag`` fields added in place.
    """
    if isinstance(data, dict):
        # --- ICD-10-GM ---
        if correct_icd and "icd10gm_code" in data and data["icd10gm_code"]:
            name = data.get("name", "") or data.get("official_name", "")
            original_code = data["icd10gm_code"]
            if name and original_code:
                data["icd10gm_code_rag"] = corrector.correct_icd_code(
                    name, original_code
                )

        # --- ATC ---
        if correct_atc and "atc_code" in data and data["atc_code"]:
            name = data.get("medication_name", "") or data.get("name", "")
            original_code = data["atc_code"]
            if name and original_code:
                data["atc_code_rag"] = corrector.correct_atc_code(
                    name, original_code
                )

        # --- OPS ---
        if correct_ops and "ops_code" in data and data["ops_code"]:
            name = data.get("procedure_name", "") or data.get("name", "")
            original_code = data["ops_code"]
            if name and original_code:
                data["ops_code_rag"] = corrector.correct_ops_code(
                    name, original_code
                )

        # Recurse into nested values
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                data[key] = add_rag_corrections(
                    value, corrector, correct_icd, correct_ops, correct_atc
                )

    elif isinstance(data, list):
        return [
            add_rag_corrections(item, corrector, correct_icd, correct_ops, correct_atc)
            for item in data
        ]

    return data


# ------------------------------------------------------------------
# Main processing loop
# ------------------------------------------------------------------

def process_inference_results(
    input_csv: str,
    output_csv: str,
    correct_icd: bool = True,
    correct_ops: bool = False,
    correct_atc: bool = False,
    checkpoint_every: int = 100,
) -> None:
    """Process an inference-results CSV and write RAG-corrected output.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV (must contain a ``parsed_generation`` column).
    output_csv : str
        Path where the corrected CSV will be written.
    correct_icd, correct_ops, correct_atc : bool
        Which code types to correct.
    checkpoint_every : int
        Save a ``.checkpoint`` file every N rows.
    """
    print(f"Loading inference results from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")

    # Initialise the unified corrector
    print("\nInitializing RAG corrector...")
    corrector = UnifiedRAGCorrector()

    corrected_generations: list = []
    checkpoint_path = output_csv + ".checkpoint"

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Correcting codes"):
        parsed_gen = row.get("parsed_generation", "")

        if pd.isna(parsed_gen) or str(parsed_gen).strip() == "":
            corrected_generations.append(parsed_gen)
            continue

        try:
            data = json.loads(str(parsed_gen))
            corrected_data = add_rag_corrections(
                data, corrector, correct_icd, correct_ops, correct_atc
            )
            corrected_generations.append(json.dumps(corrected_data, ensure_ascii=False))
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"\nWarning: JSON parse error at row {idx}: {exc}")
            corrected_generations.append(parsed_gen)
        except Exception as exc:
            print(f"\nWarning: Error processing row {idx}: {exc}")
            corrected_generations.append(parsed_gen)

        # Checkpoint
        if (idx + 1) % checkpoint_every == 0:
            print(f"\n  Checkpoint at row {idx + 1}/{len(df)}")
            df_tmp = df.copy()
            df_tmp["parsed_generation"] = corrected_generations + [None] * (
                len(df) - len(corrected_generations)
            )
            df_tmp.to_csv(checkpoint_path, index=False)

    # Write final output
    df["parsed_generation"] = corrected_generations

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Removed checkpoint file")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: Apply RAG-based code corrections to inference results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the inference-results CSV (from Stage 5).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the RAG-corrected output CSV.",
    )
    parser.add_argument(
        "--correct-icd", action="store_true", default=False,
        help="Correct ICD-10-GM diagnosis codes.",
    )
    parser.add_argument(
        "--correct-ops", action="store_true", default=False,
        help="Correct OPS procedure codes.",
    )
    parser.add_argument(
        "--correct-atc", action="store_true", default=False,
        help="Correct ATC medication codes.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=100,
        help="Save a checkpoint file every N rows (default: 100).",
    )
    args = parser.parse_args()

    # At least one correction type must be selected
    if not (args.correct_icd or args.correct_ops or args.correct_atc):
        parser.error(
            "At least one of --correct-icd, --correct-ops, --correct-atc "
            "must be specified."
        )

    ensure_dirs()

    process_inference_results(
        input_csv=args.input,
        output_csv=args.output,
        correct_icd=args.correct_icd,
        correct_ops=args.correct_ops,
        correct_atc=args.correct_atc,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
