"""
Stage 3a: Post-process generated labels and prepare for training.

This script takes the raw generated dataset (CSV with medical texts and labels)
and transforms it into a training-ready format:

1. Parses label strings (handling numpy types, None values, malformed JSON)
2. Enriches medication entries with ATC codes via fuzzy matching
3. Detects procedure code types (OPS vs SNOMED) via lookup
4. Formats each example as a Gemma chat template for SFT training
5. Outputs a HuggingFace Dataset saved as JSON

The chat template wraps each (prompt, label) pair in Gemma's turn markers
so that the SFTTrainer can mask the instruction portion during training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *

import argparse
import ast
import copy
import json
import re
from multiprocessing import Pool, cpu_count
from typing import Any, Dict

import pandas as pd
from datasets import Dataset
from rapidfuzz import fuzz, process
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Lookup tables (loaded once at module level for multiprocessing workers)
# ---------------------------------------------------------------------------

def _load_lookup_tables():
    """Load ATC and OPS lookup tables from paths defined in config.py."""
    lookups = {}
    try:
        atc_df = pd.read_csv(ATC_LOOKUP)
        lookups["atc_df"] = atc_df
        lookups["atc_column"] = "display"
        lookups["atc_code_column"] = "atc_code"
    except Exception as e:
        print(f"Warning: Could not load ATC lookup from {ATC_LOOKUP}: {e}")

    try:
        ops_df = pd.read_csv(OPS_LOOKUP)
        lookups["ops_codes_set"] = set(ops_df["code"].dropna().astype(str))
    except Exception as e:
        print(f"Warning: Could not load OPS lookup from {OPS_LOOKUP}: {e}")

    return lookups


LOOKUPS = _load_lookup_tables()


# ---------------------------------------------------------------------------
# Prompt template (German, matches the JSON schema used during generation)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """Extrahiere medizinische Informationen aus dem folgenden medizinischen Text und strukturiere sie als JSON.

REGELN:
- Antwort NUR als gültiges JSON-Objekt
- Nur explizit im Text erwähnte Informationen extrahieren
- Fehlende Informationen als leere Werte ("", [], {{}}) angeben
- Vollständige JSON-Struktur gemäß Schema verwenden

JSON-SCHEMA:
{{
  "introduction": {{
    "family_name": "",
    "given_name": "",
    "birth_date": "",
    "gender": "",
    "address_street": "",
    "address_city": "",
    "address_postal_code": "",
    "stationary_type": "",
    "encounter_start_date": "",
    "encounter_end_date": ""
  }},
  "diagnoses": [
    {{
      "type": "main_diagnosis",
      "name": "",
      "icd10gm_code": "",
      "date": ""
    }},
    {{
      "type": "side_diagnosis",
      "name": "",
      "icd10gm_code": "",
      "date": ""
    }}
  ],
  "tumor_informations": [
    {{
      "type": "pathological",
      "stage": "",
      "t": "",
      "n": "",
      "m": "",
      "date": ""
    }},
    {{
      "type": "clinical",
      "t": "",
      "n": "",
      "m": "",
      "date": ""
    }},
    {{
      "type": "histology",
      "histology": "",
      "date": ""
    }},
    {{
      "type": "overall_status",
      "status_de": "",
      "date": ""
    }},
    {{
      "type": "progression",
      "description_de": "",
      "date": ""
    }},
    {{
      "type": "tumor_marker",
      "marker": "",
      "value": 0.0,
      "unit": "",
      "date": ""
    }},
    {{
      "type": "smoking_status",
      "status": "",
      "date": ""
    }},
    {{
      "type": "ecog_performance",
      "score": 0,
      "date": ""
    }},
    {{
      "type": "comorbidities",
      "conditions": [],
      "date": ""
    }},
    {{
      "type": "operations",
      "procedures": [],
      "date": ""
    }},
    {{
      "type": "radiotherapy",
      "procedures": [],
      "date": ""
    }}
  ],
  "medication": [],
  "lab_values": [
    {{
      "lab_name": "",
      "lab_value": 0.0
    }}
  ],
  "free_text": {{
    "lab_values": [
      {{
        "name": "",
        "value": ""
      }}
    ],
    "medications": [],
    "body_values": [],
    "procedures": [
      {{
        "procedure_name": "",
        "ops_code": "",
        "code_type": ""
      }}
    ],
    "diagnoses": [
      {{
        "type": "side_diagnosis",
        "official_name": "",
        "icd10gm_code": ""
      }}
    ]
  }}
}}

MEDIZINISCHER TEXT:
{medical_text}"""


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_labels(x):
    """Parse a label value from CSV into a Python dict.

    Handles:
    - pandas NaN / None -> empty dict
    - Already a dict -> return as-is
    - String with numpy type annotations (np.int64, np.float64) -> cleaned
    - Python None literals -> replaced with empty lists
    """
    if pd.isna(x):
        return {}

    if isinstance(x, dict):
        return x

    if isinstance(x, str):
        # Strip numpy wrapper types that appear when labels were str(repr(...))
        x = re.sub(r"np\.int64\((\d+)\)", r"\1", x)
        x = re.sub(r"np\.float64\(([0-9.]+)\)", r"\1", x)
        x = re.sub(r"np\.float64\(nan\)", r'""', x)
        # Replace Python None with empty list (common in label dicts)
        x = x.replace(": None", ": []")
        return ast.literal_eval(x)

    return x


# ---------------------------------------------------------------------------
# ATC fuzzy matching
# ---------------------------------------------------------------------------

def fuzzy_match_atc_code(name: str, lookup_df: pd.DataFrame, threshold: int = 80):
    """Find the best ATC code match for a medication name via fuzzy string matching.

    Args:
        name: Medication name to look up.
        lookup_df: DataFrame with 'display' and 'atc_code' columns.
        threshold: Minimum fuzzy score (0-100) to accept a match.

    Returns:
        Tuple of (atc_code, matched_display_name) or (None, None).
    """
    if not name:
        return None, None

    atc_column = LOOKUPS.get("atc_column", "display")
    atc_code_column = LOOKUPS.get("atc_code_column", "atc_code")

    match_list = process.extract(
        name,
        lookup_df[atc_column].tolist(),
        scorer=fuzz.token_set_ratio,
        limit=1,
    )

    if match_list and len(match_list) > 0:
        best_match, score, df_index = match_list[0]
        if score >= threshold:
            matched_code = lookup_df.iloc[df_index][atc_code_column]
            return matched_code, best_match

    return None, None


# ---------------------------------------------------------------------------
# Procedure code type detection
# ---------------------------------------------------------------------------

def determine_procedure_code_type(code: str) -> str:
    """Check whether a procedure code is OPS or SNOMED by looking it up in the OPS table.

    Args:
        code: The procedure code string.

    Returns:
        'ops' if found in the OPS lookup, otherwise 'snomed'.
    """
    if not code:
        return "snomed"

    if str(code) in LOOKUPS.get("ops_codes_set", set()):
        return "ops"
    return "snomed"


# ---------------------------------------------------------------------------
# ATC enrichment
# ---------------------------------------------------------------------------

def add_atc_codes_to_medications(labels: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich medication entries with ATC codes and procedure entries with code types.

    Processes both top-level 'medication' and 'free_text.medications' sections.
    Also adds 'code_type' to 'free_text.procedures' entries.

    Args:
        labels: Parsed label dictionary for a single example.

    Returns:
        A deep copy of labels with ATC codes and code types added.
    """
    data = copy.deepcopy(labels)
    atc_df = LOOKUPS.get("atc_df")

    # --- Top-level medications ---
    if "medication" in data and isinstance(data["medication"], list):
        for medication in data["medication"]:
            if isinstance(medication, dict) and "medication_name" in medication:
                if atc_df is not None:
                    matched_code, _ = fuzzy_match_atc_code(
                        medication["medication_name"], atc_df
                    )
                    if matched_code:
                        medication["atc_code"] = matched_code

    # --- Free-text medications and procedures ---
    if "free_text" in data and isinstance(data["free_text"], dict):
        ft = data["free_text"]

        if "medications" in ft and isinstance(ft["medications"], list):
            for medication in ft["medications"]:
                if isinstance(medication, dict) and "name" in medication:
                    if atc_df is not None:
                        matched_code, _ = fuzzy_match_atc_code(
                            medication["name"], atc_df
                        )
                        if matched_code:
                            medication["atc_code"] = matched_code

        if "procedures" in ft and isinstance(ft["procedures"], list):
            for procedure in ft["procedures"]:
                if isinstance(procedure, dict) and "ops_code" in procedure:
                    procedure["code_type"] = determine_procedure_code_type(
                        procedure["ops_code"]
                    )

    return data


# ---------------------------------------------------------------------------
# Single-row processing (module-level for multiprocessing compatibility)
# ---------------------------------------------------------------------------

# The canonical key order for the output JSON label
_KEY_ORDER = [
    "introduction",
    "diagnoses",
    "tumor_informations",
    "medication",
    "lab_values",
    "free_text",
]


def process_single_row(row_data):
    """Process one (medical_text, labels) pair into a Gemma chat-template string.

    Steps:
        1. Parse the raw label string into a dict.
        2. Enrich with ATC codes and procedure code types.
        3. Re-order keys to match the canonical schema order.
        4. Format as a Gemma chat template.

    Args:
        row_data: Tuple of (medical_text: str, combined_labels: str).

    Returns:
        Dict with a single 'text' key containing the formatted chat session,
        or None if the row could not be processed.
    """
    medical_text, combined_labels = row_data

    try:
        label_dict = parse_labels(combined_labels)
    except Exception:
        return None

    # Enrich with ATC codes and procedure code types
    label_dict = add_atc_codes_to_medications(label_dict)

    # Re-order keys to canonical order
    ordered = {k: label_dict[k] for k in _KEY_ORDER if k in label_dict}
    json_str = json.dumps(ordered, ensure_ascii=False)

    # Build the Gemma chat template
    formatted_prompt = PROMPT_TEMPLATE.format(medical_text=medical_text)

    chat_session = (
        f"<bos><start_of_turn>user\n"
        f"{formatted_prompt}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{json_str}\n"
        f"<end_of_turn>"
    )

    return {"text": chat_session}


# ---------------------------------------------------------------------------
# Parallel dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset_for_training(
    data: pd.DataFrame, n_processes: int = None
) -> Dataset:
    """Transform a DataFrame into a HuggingFace Dataset of chat-template strings.

    Uses multiprocessing to parallelize the per-row processing (label parsing,
    ATC enrichment, chat template formatting).

    Args:
        data: DataFrame with 'full_text' (medical text) and 'combined_labels' columns.
        n_processes: Number of worker processes. Defaults to cpu_count().

    Returns:
        HuggingFace Dataset with a single 'text' column.
    """
    if n_processes is None:
        n_processes = cpu_count()

    print(f"Using {n_processes} processes for data transformation...")

    row_data = list(zip(data["full_text"], data["combined_labels"]))

    with Pool(processes=n_processes) as pool:
        transformed = list(
            tqdm(
                pool.imap(process_single_row, row_data),
                total=len(row_data),
                desc="Transforming data",
            )
        )

    # Filter out rows that failed to process
    transformed = [item for item in transformed if item is not None]
    n_removed = len(row_data) - len(transformed)
    print(
        f"Dataset: {len(transformed)} complete samples "
        f"(removed {n_removed} incomplete samples)"
    )

    return Dataset.from_pandas(pd.DataFrame(transformed))


# ---------------------------------------------------------------------------
# Raw dataset post-processing
# ---------------------------------------------------------------------------

# Every label dict must contain these top-level keys
_REQUIRED_KEYS = [
    "introduction",
    "diagnoses",
    "tumor_informations",
    "medication",
    "lab_values",
    "free_text",
]


def _ensure_required_keys(d: dict) -> dict:
    """Add missing required keys with sensible empty defaults."""
    for key in _REQUIRED_KEYS:
        if key not in d:
            d[key] = {} if key in ("introduction", "free_text") else []
    return d


def _convert_none_values(d):
    """Recursively replace None with '' (strings) or [] (medication lists)."""
    if isinstance(d, dict):
        for key, value in d.items():
            if value is None:
                d[key] = [] if key == "medication" else ""
            elif isinstance(value, (dict, list)):
                d[key] = _convert_none_values(value)
    elif isinstance(d, list):
        for i, item in enumerate(d):
            if item is None:
                d[i] = ""
            elif isinstance(item, (dict, list)):
                d[i] = _convert_none_values(item)
    return d


def postprocess_the_raw_dataset(input_dir: Path) -> Dataset:
    """Load raw CSVs, clean labels, and return a HuggingFace Dataset.

    This function:
    1. Reads the patient-based CSV from input_dir.
    2. Drops rows with empty generation text.
    3. Parses label strings, converts None values, ensures required keys.
    4. Returns a Dataset ready for prepare_dataset_for_training().

    Args:
        input_dir: Path to directory containing the raw CSV files.

    Returns:
        HuggingFace Dataset with columns: prompt, full_text, combined_labels,
        and optionally patient_id.
    """
    # Discover CSV files in the input directory
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    print(f"Found {len(csv_files)} CSV file(s) in {input_dir}")
    for f in csv_files:
        df_tmp = pd.read_csv(f)
        print(f"  {f.name}: {len(df_tmp):,} rows")

    # Load and combine (use last CSV if multiple, or combine as needed)
    combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    prompt_col = "prompt"
    gen_col = "full_text"
    labels_col = "combined_labels"
    patient_col = "patient_id"

    # Keep relevant columns
    cols = [prompt_col, gen_col, labels_col]
    if patient_col in combined.columns:
        cols.append(patient_col)
        print(f"Found '{patient_col}' column -- will support patient-based splitting")

    combined = combined[cols].copy()

    # Drop rows where the generated text is empty
    combined = combined.dropna(subset=[gen_col])
    print(f"After dropping empty texts: {len(combined):,} rows")

    # Parse, clean, and ensure label structure
    combined[labels_col] = combined[labels_col].apply(parse_labels)
    combined[labels_col] = combined[labels_col].apply(_convert_none_values)
    combined[labels_col] = combined[labels_col].apply(_ensure_required_keys)
    combined[labels_col] = combined[labels_col].apply(json.dumps)

    return Dataset.from_pandas(combined)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main_prepare_dataset():
    """Main entry point: parse args, post-process, transform, and save."""
    parser = argparse.ArgumentParser(
        description="Post-process raw labels and prepare dataset for training."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(GENERATED_TEXTS_DIR),
        help="Directory containing raw CSV files with medical texts and labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(TRAINING_READY_DIR),
        help="Directory to save the training-ready JSON dataset.",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=None,
        help="Number of parallel workers (default: all CPUs).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Post-process the raw dataset
    print("=" * 60)
    print("STEP 1: Post-processing raw dataset")
    print("=" * 60)
    dataset = postprocess_the_raw_dataset(input_dir)
    df = dataset.to_pandas()
    print(f"Post-processed dataset: {len(df):,} rows")

    # Step 2: Transform into chat-template format
    print("\n" + "=" * 60)
    print("STEP 2: Transforming to chat-template format")
    print("=" * 60)
    training_dataset = prepare_dataset_for_training(df, n_processes=args.n_processes)

    # Step 3: Save as JSON
    n = len(training_dataset)
    output_path = output_dir / f"postprocessed_dataset_{n}_rows.json"
    training_dataset.to_json(output_path)
    print(f"\nSaved training-ready dataset to: {output_path}")
    print(f"Total examples: {n:,}")


if __name__ == "__main__":
    main_prepare_dataset()
