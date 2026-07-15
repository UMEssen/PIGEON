"""
Stage 1 -- Unified FHIR Data Downloader
========================================

Downloads structured clinical data from a FHIR-compliant PostgreSQL database
and saves it as per-encounter or per-patient CSV files in the FHIR cache.

This is the first step in the PIGEON pipeline: before we can generate
discharge letters, we need the raw clinical facts (diagnoses, labs, vitals,
medications, procedures, demographics) as flat CSV files.

HOW IT WORKS
------------
1. You provide a CSV file that lists resource IDs grouped by encounter or
   patient.  The expected columns depend on the mode (see --mode).
2. For each row the script injects the IDs into the SQL query templates
   from ``queries.py``, executes them via ``database.MetricsMiner``, and
   writes the results to individual CSV files under the FHIR cache tree.
3. Downloads run in parallel using ThreadPoolExecutor -- each worker
   thread gets its own database connection to avoid psycopg thread-safety
   issues.

DIRECTORY STRUCTURE CREATED
---------------------------
    <FHIR_CACHE_ENCOUNTER>/          (--mode encounter)
        CONDITION/enc_id_<id>.csv
        OBSERVATION/enc_id_<id>.csv
        MEDICATION_STATEMENT/enc_id_<id>.csv
        PROCEDURE/enc_id_<id>.csv
        PATIENT/enc_id_<id>.csv
        ENCOUNTER/enc_id_<id>.csv

    <FHIR_CACHE_PATIENT>/            (--mode patient)
        CONDITION/pat_<id>.csv
        OBSERVATION/pat_<id>.csv
        MEDICATION_STATEMENT/pat_<id>.csv
        PROCEDURE/pat_<id>.csv
        PATIENT/pat_<id>.csv
        ENCOUNTER/pat_<id>.csv

USAGE
-----
    # Encounter-based download (most common for discharge-letter generation):
    python download_fhir_data.py \\
        --input data/encounter_ids.csv \\
        --mode encounter \\
        --workers 16

    # Patient-based download:
    python download_fhir_data.py \\
        --input data/patient_ids.csv \\
        --mode patient \\
        --workers 16

INPUT CSV FORMAT
----------------
Encounter mode expects columns:
    e1_id, p1_id, procedure_ids, observation_systems, medication_statement_ids, condition_ids

Patient mode expects columns:
    p1_id, procedure_ids, observation_systems, medication_statement_ids, condition_ids

The *_ids columns contain string representations of Python lists, e.g.:
    "['123', '456', '789']"

The observation_systems column contains a dict mapping system URLs to ID lists:
    "{'https://...HeartBeat': [100, 101], 'https://...Laboratory': [200]}"
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports -- config.py lives two directories above this file
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    FHIR_CACHE_ENCOUNTER,
    FHIR_CACHE_PATIENT,
    MAX_CONCURRENT_REQUESTS,
    ensure_dirs,
)

from database import MetricsMiner
from queries import (
    condition_query,
    encounter_query,
    get_observation_query,
    medication_query_optimized,
    patient_query,
    proc_query,
    enc_to_patient_query,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_id_list(id_list_str: str) -> list[str] | dict[str, list]:
    """Parse a stringified list or dict of resource IDs into a usable form.

    The input CSVs store ID collections as string representations of Python
    literals.  This function safely evaluates them and returns either:

    - A list of SQL-quoted ID strings (for simple ID lists), or
    - A dict mapping system URLs to lists of raw IDs (for observations,
      where IDs are grouped by their identifier system).

    Args:
        id_list_str: A string like "['1','2']" or "{'system_url': [1, 2]}".

    Returns:
        Parsed IDs ready for query injection, or an empty list on failure.
    """
    if pd.isna(id_list_str):
        return []

    try:
        parsed = ast.literal_eval(str(id_list_str))

        if isinstance(parsed, list):
            # Simple list of IDs -- quote each one for SQL injection
            return [f"'{item}'" for item in parsed]

        elif isinstance(parsed, dict):
            # Observation IDs grouped by system URL
            # e.g. {"https://...HeartBeat": [100, 101], ...}
            grouped: dict[str, list] = {}
            for id_val, system in parsed.items():
                grouped.setdefault(system, []).append(id_val)
            return grouped

    except (ValueError, SyntaxError):
        pass

    return []


def _quote_ids(ids: list) -> str:
    """Join a list of IDs into a comma-separated SQL-safe string.

    Already-quoted IDs (from clean_id_list) are joined directly.
    Unquoted IDs get wrapped in single quotes.
    """
    parts = []
    for item in ids:
        s = str(item)
        if s.startswith("'") and s.endswith("'"):
            parts.append(s)
        else:
            parts.append(f"'{s}'")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Per-resource download functions
# ---------------------------------------------------------------------------
# Each function takes a MetricsMiner, the relevant IDs, and an output path.
# They return True if data was written, False if the query returned nothing.

def download_conditions(miner: MetricsMiner, ids: list[str], out_path: Path) -> bool:
    """Download Condition (diagnosis) resources for the given IDs."""
    query = condition_query.replace("(batches)", f"({_quote_ids(ids)})")
    rows, cols = miner.execute_query(query)
    if not rows:
        return False
    pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
    return True


def download_procedures(miner: MetricsMiner, ids: list[str], out_path: Path) -> bool:
    """Download Procedure resources for the given IDs."""
    query = proc_query.replace("(batches)", f"({_quote_ids(ids)})")
    rows, cols = miner.execute_query(query)
    if not rows:
        return False
    pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
    return True


def download_medications(miner: MetricsMiner, ids: list[str], out_path: Path) -> bool:
    """Download MedicationStatement resources for the given IDs.

    Deduplicates on dosage text + display name because the CTE-based query
    can still produce near-duplicates when a medication has multiple codings.
    """
    query = medication_query_optimized.replace("(batches)", f"({_quote_ids(ids)})")
    rows, cols = miner.execute_query(query)
    if not rows:
        return False
    df = pd.DataFrame(rows, columns=cols)
    # Convert JSONB columns to string for deduplication
    for col in ("md0.text_list", "mcc0.display_list"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    df = df.drop_duplicates(
        subset=[c for c in ("md0.text_list", "mcc0.display_list") if c in df.columns],
        keep="first",
    )
    df.to_csv(out_path, index=False)
    return True


def download_observations(
    miner: MetricsMiner,
    systems_to_ids: dict[str, list],
    out_path: Path,
) -> bool:
    """Download Observation resources, dispatching to the right query per system.

    Observations are special: each system URL (HeartBeat, Laboratory, etc.)
    needs a different SQL query because the FHIR resource structure differs.
    We run one query per system, then concatenate all results into a single
    CSV with the union of all columns.
    """
    all_dfs: list[pd.DataFrame] = []
    all_columns: set[str] = set()

    for system_url, ids in systems_to_ids.items():
        obs_query = get_observation_query(system_url)
        if obs_query is None:
            # Unknown system -- skip silently (the hospital may have custom
            # observation types we don't handle).
            continue

        quoted = ", ".join(f"'{item}'" for item in ids)
        query = obs_query.replace("(batch_ids)", f"({quoted})")
        rows, cols = miner.execute_query(query)
        if rows:
            df = pd.DataFrame(rows, columns=cols)
            all_dfs.append(df)
            all_columns.update(cols)

    if not all_dfs:
        return False

    # Reindex so all DataFrames share the same column set before concat
    combined = pd.concat(
        [df.reindex(columns=sorted(all_columns)) for df in all_dfs],
        ignore_index=True,
    )
    combined.to_csv(out_path, index=False)
    return True


def download_patient(miner: MetricsMiner, patient_id: str, out_path: Path) -> bool:
    """Download Patient demographics for a single patient."""
    query = patient_query.replace("(batches)", f"('{patient_id}')")
    rows, cols = miner.execute_query(query)
    if not rows:
        return False
    pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
    return True


def download_encounter(miner: MetricsMiner, encounter_id: str, out_path: Path) -> bool:
    """Download Encounter metadata for a single encounter."""
    query = encounter_query.replace("(batches)", f"('{encounter_id}')")
    rows, cols = miner.execute_query(query)
    if not rows:
        return False
    pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
    return True


# ---------------------------------------------------------------------------
# Row-level processing (one invocation per encounter or patient)
# ---------------------------------------------------------------------------

def process_encounter_row(row: pd.Series, output_dir: Path) -> None:
    """Process a single encounter row: download all resource types.

    Creates its own database connection (required for thread safety)
    and closes it when done.
    """
    encounter_id = row["e1_id"]
    patient_id = row.get("p1_id")

    miner = MetricsMiner()
    try:
        # -- Conditions --
        cond_ids = clean_id_list(row.get("condition_ids", "[]"))
        if isinstance(cond_ids, list) and cond_ids:
            download_conditions(
                miner, cond_ids,
                output_dir / f"CONDITION/enc_id_{encounter_id}.csv",
            )

        # -- Procedures --
        proc_ids = clean_id_list(row.get("procedure_ids", "[]"))
        if isinstance(proc_ids, list) and proc_ids:
            download_procedures(
                miner, proc_ids,
                output_dir / f"PROCEDURE/enc_id_{encounter_id}.csv",
            )

        # -- Medications --
        med_ids = clean_id_list(row.get("medication_statement_ids", "[]"))
        if isinstance(med_ids, list) and med_ids:
            download_medications(
                miner, med_ids,
                output_dir / f"MEDICATION_STATEMENT/enc_id_{encounter_id}.csv",
            )

        # -- Observations (dict of system -> IDs) --
        obs_data = clean_id_list(row.get("observation_systems", "{}"))
        if isinstance(obs_data, dict) and obs_data:
            download_observations(
                miner, obs_data,
                output_dir / f"OBSERVATION/enc_id_{encounter_id}.csv",
            )

        # -- Patient demographics --
        if pd.notna(patient_id):
            download_patient(
                miner, str(int(patient_id)),
                output_dir / f"PATIENT/enc_id_{encounter_id}.csv",
            )

        # -- Encounter metadata --
        download_encounter(
            miner, str(encounter_id),
            output_dir / f"ENCOUNTER/enc_id_{encounter_id}.csv",
        )

    finally:
        miner.close()


def process_patient_row(row: pd.Series, output_dir: Path) -> None:
    """Process a single patient row: download all resource types.

    Same pattern as encounter mode but files are keyed by patient ID.
    """
    patient_id = row["p1_id"]

    miner = MetricsMiner()
    try:
        # -- Conditions --
        cond_ids = clean_id_list(row.get("condition_ids", "[]"))
        if isinstance(cond_ids, list) and cond_ids:
            download_conditions(
                miner, cond_ids,
                output_dir / f"CONDITION/pat_{patient_id}.csv",
            )

        # -- Procedures --
        proc_ids = clean_id_list(row.get("procedure_ids", "[]"))
        if isinstance(proc_ids, list) and proc_ids:
            download_procedures(
                miner, proc_ids,
                output_dir / f"PROCEDURE/pat_{patient_id}.csv",
            )

        # -- Medications --
        med_ids = clean_id_list(row.get("medication_statement_ids", "[]"))
        if isinstance(med_ids, list) and med_ids:
            download_medications(
                miner, med_ids,
                output_dir / f"MEDICATION_STATEMENT/pat_{patient_id}.csv",
            )

        # -- Observations (dict of system -> IDs) --
        obs_data = clean_id_list(row.get("observation_systems", "{}"))
        if isinstance(obs_data, dict) and obs_data:
            download_observations(
                miner, obs_data,
                output_dir / f"OBSERVATION/pat_{patient_id}.csv",
            )

        # -- Patient demographics --
        download_patient(
            miner, str(patient_id),
            output_dir / f"PATIENT/pat_{patient_id}.csv",
        )

        # -- Encounter metadata (if encounter IDs are available) --
        enc_ids = clean_id_list(row.get("encounter_ids", "[]"))
        if isinstance(enc_ids, list) and enc_ids:
            # Download encounter data for each encounter associated with this patient
            for enc_id_quoted in enc_ids:
                enc_id = enc_id_quoted.strip("'")
                download_encounter(
                    miner, enc_id,
                    output_dir / f"ENCOUNTER/pat_{patient_id}_enc_{enc_id}.csv",
                )

    finally:
        miner.close()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_download(
    input_csv: str,
    mode: str = "encounter",
    max_workers: int | None = None,
) -> None:
    """Orchestrate the parallel download of FHIR resources.

    Args:
        input_csv: Path to the CSV file containing resource IDs.
        mode: Either "encounter" or "patient".
        max_workers: Number of parallel threads.  Defaults to
            min(MAX_CONCURRENT_REQUESTS, cpu_count + 4).
    """
    # Ensure all output directories exist
    ensure_dirs()

    # Determine output directory and row processor based on mode
    if mode == "encounter":
        output_dir = FHIR_CACHE_ENCOUNTER
        row_processor = process_encounter_row
        id_column = "e1_id"
    elif mode == "patient":
        output_dir = FHIR_CACHE_PATIENT
        row_processor = process_patient_row
        id_column = "p1_id"
    else:
        raise ValueError(f"Unknown mode '{mode}'.  Use 'encounter' or 'patient'.")

    # Create subdirectories for each resource type
    for subdir in ("CONDITION", "OBSERVATION", "MEDICATION_STATEMENT",
                    "PROCEDURE", "PATIENT", "ENCOUNTER"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Load input CSV
    print(f"Loading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    df.reset_index(drop=True, inplace=True)

    if id_column not in df.columns:
        raise ValueError(
            f"Required column '{id_column}' not found in {input_csv}.  "
            f"Available columns: {list(df.columns)}"
        )

    print(f"  Rows to process: {len(df)}")
    print(f"  Mode: {mode}")
    print(f"  Output: {output_dir}")

    # Determine thread count
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        max_workers = min(MAX_CONCURRENT_REQUESTS, cpu_count + 4)
    print(f"  Workers: {max_workers}")

    # Submit all rows to the thread pool
    start = time.time()
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(row_processor, row, output_dir): idx
            for idx, row in df.iterrows()
        }

        with tqdm(total=len(futures), desc=f"Downloading ({mode})") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    row_idx = futures[future]
                    errors += 1
                    # Print sparingly to avoid flooding the console
                    if errors <= 10:
                        print(f"\n  Error on row {row_idx}: {exc}")
                    elif errors == 11:
                        print("\n  (suppressing further error messages)")
                pbar.update(1)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s.  "
          f"Processed {len(df)} rows, {errors} errors.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse command-line arguments and start the download."""
    parser = argparse.ArgumentParser(
        description="Download FHIR resources from the metrics database "
                    "and cache them as per-encounter or per-patient CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV file containing resource IDs.",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=("encounter", "patient"),
        default="encounter",
        help="Download mode: 'encounter' saves per-encounter CSVs, "
             "'patient' saves per-patient CSVs.  Default: encounter.",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel download threads.  "
             "Default: min(MAX_CONCURRENT_REQUESTS, cpu_count + 4).",
    )

    args = parser.parse_args()
    run_download(
        input_csv=args.input,
        mode=args.mode,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
