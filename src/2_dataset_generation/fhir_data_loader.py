"""
FHIR Data Loader -- Stage 2 of the PIGEON pipeline.

This module reads the flat-CSV FHIR cache that Stage 1 produced and organises
the data into a structured dictionary that downstream generators consume.

WHY a dedicated loader?
    The original p45_generator.py duplicated nearly identical data-loading logic
    for encounter-based and patient-based flows (lines 168-417).  Extracting it
    into a single class removes ~250 lines of copy-paste and makes it trivial to
    add a new base type (e.g. "episode") later.

Key concepts
    base        "encounter" or "patient" -- controls the file-name prefix used
                when reading CSVs from each FHIR resource directory.
    entity_id   The numeric ID of the encounter or patient whose data we load.
    FHIR resource directories
                CONDITION, ENCOUNTER, MEDICATION_STATEMENT, OBSERVATION,
                PATIENT, PROCEDURE -- each contains one CSV per entity.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Import central config -- the canonical way in this repository.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ICD-10 pattern used to keep only valid codes (letters followed by digits,
# optionally with dot, dagger, asterisk, etc.).
ICD10_PATTERN = re.compile(
    r"^\[?'?[A-Z]\d{2}(\.\d{1,2})?[\+\!\*]?'?\]?$"
)

# Column in CONDITION CSVs that holds the category display list.
CONDITION_COLUMN = "ccc0_1_displays"
CONDITION_CODE_COLUMN = "ccc0_codes"

# Terms that flag a row as a *main* (primary) diagnosis.
MAIN_DIAGNOSIS_TERMS = frozenset({
    "DRG-Hauptdiagnose",
    "Hauptdiagnose/therapie medizinisch",
    "Hauptdiagnose/therapie administrativ",
})

# Column in OBSERVATION CSVs that identifies the measurement system.
OBSERVATION_SYSTEM_COLUMN = "oi0_system"


class FHIRDataLoader:
    """Loads and organises FHIR cache data for text generation.

    Usage::

        loader = FHIRDataLoader(base="encounter")
        ids    = loader.load_ids()
        data   = loader.load_data(ids[0])
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        base: str,
        cache_path: Optional[Path] = None,
    ) -> None:
        """
        Args:
            base:       "encounter" or "patient" -- determines which file-
                        naming convention to use when reading CSVs.
            cache_path: Override for the FHIR cache root.  When *None* the
                        path is read from ``config.FHIR_CACHE_ENCOUNTER``
                        or ``config.FHIR_CACHE_PATIENT`` depending on *base*.
        """
        if base not in ("encounter", "patient"):
            raise ValueError(
                f"Invalid base type '{base}'. Use 'encounter' or 'patient'."
            )
        self.base = base

        if cache_path is not None:
            self.base_path = Path(cache_path)
        elif base == "encounter":
            self.base_path = config.FHIR_CACHE_ENCOUNTER
        else:
            self.base_path = config.FHIR_CACHE_PATIENT

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def _file_prefix(self, entity_id: str) -> str:
        """Return the CSV filename (without extension) for *entity_id*.

        Encounter-based files are named ``enc_id_{id}.csv``;
        patient-based files are named ``pat_{id}.csv``.
        """
        if self.base == "encounter":
            return f"enc_id_{entity_id}"
        return f"pat_{entity_id}"

    # ------------------------------------------------------------------
    # ID enumeration
    # ------------------------------------------------------------------

    def load_ids(self) -> List[str]:
        """List all available encounter / patient IDs.

        IDs are derived from files inside the ``ENCOUNTER`` sub-directory
        because every entity is guaranteed to have an encounter CSV.
        """
        encounter_dir = self.base_path / "ENCOUNTER"
        ids: List[str] = []
        for fname in os.listdir(encounter_dir):
            # The ID is always the last underscore-delimited token before
            # the file extension: ``enc_id_12345.csv`` -> ``12345``.
            entity_id = fname.split(".")[0].split("_")[-1]
            ids.append(entity_id)
        return ids

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, entity_id: str) -> Dict[str, Any]:
        """Load all FHIR resources for a given encounter / patient ID.

        Returns a dictionary with two top-level keys:

        * ``count`` -- a ``{resource_name: int}`` mapping that tells
          callers how many rows are available in each resource.
        * ``data``  -- a ``{resource_name: pd.DataFrame}`` mapping with
          the actual data.  Observation data is split by its
          ``oi0_system`` column into named groups like ``Laboratory``,
          ``HeartBeat``, ``BloodPressure``, etc.
        """
        prefix = self._file_prefix(entity_id)

        # -- Load raw CSVs ------------------------------------------------
        condition_data = self._load_conditions(prefix)
        encounter_data = self._load_encounters(prefix)
        medication_data = self._load_optional(
            "MEDICATION_STATEMENT", prefix, dedup=True
        )
        observation_data = self._load_observations(prefix)
        procedure_data = self._load_optional(
            "PROCEDURE", prefix, dedup_subset=["pcc0_display"]
        )
        patient_data = self._load_patient(prefix)

        # -- Organise into output dict ------------------------------------
        result: Dict[str, Any] = {"count": {}, "data": {}}

        # Conditions -> split into main / side diagnoses
        self._split_conditions(condition_data, result)

        # Medications
        result["count"]["medication"] = (
            len(medication_data) if medication_data is not None else 0
        )
        result["data"]["medication"] = (
            medication_data if medication_data is not None else pd.DataFrame()
        )

        # Observations -- split by oi0_system
        if observation_data is not None and not observation_data.empty:
            system_counts = (
                observation_data[OBSERVATION_SYSTEM_COLUMN]
                .value_counts()
                .to_dict()
            )
            for system_url, count in system_counts.items():
                # The human-readable name is the last path segment of the URL.
                system_name = system_url.split("/")[-1]
                result["count"][system_name] = count
                result["data"][system_name] = observation_data[
                    observation_data[OBSERVATION_SYSTEM_COLUMN] == system_url
                ]

        # Procedures
        result["count"]["procedure"] = (
            len(procedure_data) if procedure_data is not None else 0
        )
        result["data"]["procedure"] = (
            procedure_data if procedure_data is not None else pd.DataFrame()
        )

        # Patient
        result["count"]["patient"] = len(patient_data)
        result["data"]["patient"] = patient_data

        # Encounter
        result["count"]["encounter"] = len(encounter_data)
        result["data"]["encounter"] = encounter_data

        return result

    # ------------------------------------------------------------------
    # Private -- CSV readers
    # ------------------------------------------------------------------

    def _csv_path(self, resource_dir: str, prefix: str) -> Path:
        return self.base_path / resource_dir / f"{prefix}.csv"

    def _load_conditions(self, prefix: str) -> pd.DataFrame:
        """Load CONDITION CSV, apply ICD-10 filtering and deduplication.

        WHY the ICD-10 filter?
            The raw condition export may contain internal hospital codes
            (e.g. codes starting with "CH" or "GB") that are not valid
            ICD-10 codes.  We filter on a broad ICD-10 regex so that
            downstream jargon lookup does not fail.

        WHY the deduplication?
            A single encounter can record the same diagnosis under
            several category displays (DRG-Hauptdiagnose AND
            Hauptdiagnose/therapie medizinisch).  We keep each unique
            ``ccc0_displays`` value only once, giving priority to main-
            diagnosis rows.
        """
        path = self._csv_path("CONDITION", prefix)
        df = pd.read_csv(path)

        # --- Tag main-diagnosis rows ------------------------------------
        def _is_main_diag(row: pd.Series) -> bool:
            try:
                display_list = ast.literal_eval(row[CONDITION_COLUMN])
                return any(t in display_list for t in MAIN_DIAGNOSIS_TERMS)
            except Exception:
                return False

        df["is_main_diag"] = df.apply(_is_main_diag, axis=1)

        # --- ICD-10 pattern filter --------------------------------------
        icd10_pattern = r"^\[?'?[A-Z]\d{2}"
        mask = df[CONDITION_CODE_COLUMN].str.match(icd10_pattern, na=False)
        df = df[mask]

        # --- Deduplication ----------------------------------------------
        main_rows = df[df["is_main_diag"]].drop_duplicates(
            subset=["ccc0_displays"]
        )
        non_main_rows = df[~df["is_main_diag"]].drop_duplicates(
            subset=["ccc0_displays"]
        )

        # Remove side diagnoses that duplicate a main diagnosis display.
        main_displays = set(main_rows["ccc0_displays"])
        non_main_rows = non_main_rows[
            ~non_main_rows["ccc0_displays"].isin(main_displays)
        ]

        df = pd.concat(
            [main_rows, non_main_rows], ignore_index=True
        ).drop(columns=["is_main_diag"])
        return df

    def _load_encounters(self, prefix: str) -> pd.DataFrame:
        path = self._csv_path("ENCOUNTER", prefix)
        return pd.read_csv(path).drop_duplicates(subset=["encounter_id"])

    def _load_patient(self, prefix: str) -> pd.DataFrame:
        path = self._csv_path("PATIENT", prefix)
        return pd.read_csv(path).drop_duplicates()

    def _load_observations(self, prefix: str) -> Optional[pd.DataFrame]:
        """Load OBSERVATION CSV with column-aware deduplication.

        WHY exclude ``o0_id`` from dedup?
            Each observation row has a unique FHIR resource ID (``o0_id``),
            but two rows that differ *only* in that column are clinically
            identical.  We drop duplicates on every column except
            ``o0_id`` to remove these false-unique rows.
        """
        path = self._csv_path("OBSERVATION", prefix)
        if not path.exists():
            return None
        df = pd.read_csv(path)
        dedup_cols = [c for c in df.columns if c != "o0_id"]
        return df.drop_duplicates(subset=dedup_cols)

    def _load_optional(
        self,
        resource_dir: str,
        prefix: str,
        dedup: bool = False,
        dedup_subset: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Load a CSV that may not exist (e.g. MEDICATION_STATEMENT)."""
        path = self._csv_path(resource_dir, prefix)
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if dedup:
            df = df.drop_duplicates()
        elif dedup_subset:
            df = df.drop_duplicates(subset=dedup_subset)
        return df

    # ------------------------------------------------------------------
    # Private -- condition splitting
    # ------------------------------------------------------------------

    def _split_conditions(
        self,
        condition_data: pd.DataFrame,
        result: Dict[str, Any],
    ) -> None:
        """Split condition rows into main and side diagnosis DataFrames.

        WHY iterate rather than vectorise?
            Each row's ``ccc0_1_displays`` is a *string representation*
            of a Python list (e.g. ``"['DRG-Hauptdiagnose', 'foo']"``).
            We must ``ast.literal_eval`` every cell to inspect its
            contents, which is inherently row-level.
        """
        main_rows: List[pd.Series] = []
        side_rows: List[pd.Series] = []
        main_count = 0
        side_count = 0

        for _, row in condition_data.iterrows():
            display_str = row[CONDITION_COLUMN]
            try:
                display_list = ast.literal_eval(display_str)
            except Exception:
                side_rows.append(row)
                side_count += 1
                continue

            if any(t in display_list for t in MAIN_DIAGNOSIS_TERMS):
                main_rows.append(row)
                main_count += 1
            else:
                side_rows.append(row)
                side_count += 1

        result["count"]["main_diagnosis"] = main_count
        result["count"]["side_diagnosis"] = side_count
        result["data"]["main_diagnosis"] = (
            pd.DataFrame(main_rows) if main_rows else pd.DataFrame()
        )
        result["data"]["side_diagnosis"] = (
            pd.DataFrame(side_rows) if side_rows else pd.DataFrame()
        )
