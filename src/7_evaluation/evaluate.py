"""
Stage 7: Comprehensive evaluation of medical information extraction.

Evaluates model predictions against ground truth labels across all
extraction categories: demographics, diagnoses (ICD-10-GM), tumor staging,
medications (ATC), lab values, and free-text extracted entities.

Metrics computed:
- Per-field exact match for introduction (10 fields)
- Jaccard similarity and F1 at full code, 3-char prefix, and category levels for ICD codes
- Per-type evaluation for tumor_informations (11 types)
- Medication name, dosage, and ATC code matching
- Lab value name and value matching
- Per-subsection evaluation for free_text
- RAG correction impact analysis

Internal code handling
----------------------
Hospital-specific internal codes (long alphanumeric strings like
``B1AAYVLCKIZ4``) are detected and, for labels, resolved via the ICD lookup
table when possible.  This ensures that RAG-corrected predictions are not
unfairly penalised for not matching an internal identifier.

Usage
-----
::

    python evaluate.py \\
        --predictions results/inference_rag.csv \\
        --ground-truth results/ground_truth.csv \\
        --output-scores results/scores.csv \\
        --include-rag

All paths are configurable; nothing is hardcoded.
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ICD10_LOOKUP, RESULTS_DIR

# ---------------------------------------------------------------------------
# Optional dependency: json_repair
# ---------------------------------------------------------------------------
try:
    from json_repair import repair_json
except ImportError:
    repair_json = None  # type: ignore[assignment]


# ===================================================================
# Constants
# ===================================================================

# The 10 introduction fields that are compared via exact match.
INTRODUCTION_FIELDS: List[str] = [
    "family_name",
    "given_name",
    "birth_date",
    "gender",
    "address_street",
    "address_city",
    "address_postal_code",
    "stationary_type",
    "encounter_start_date",
    "encounter_end_date",
]

# The 11 tumor information types handled by the pipeline.
TUMOR_TYPES: List[str] = [
    "pathological",
    "clinical",
    "histology",
    "overall_status",
    "progression",
    "tumor_marker",
    "smoking_status",
    "ecog_performance",
    "comorbidities",
    "operations",
    "radiotherapy",
]

# Free-text subsections that are evaluated independently.
FREE_TEXT_SUBSECTIONS: List[str] = [
    "lab_values",
    "medications",
    "body_values",
    "procedures",
    "diagnoses",
]

# ---------------------------------------------------------------------------
# Module-level state for the ICD lookup table (loaded lazily, cached).
# ---------------------------------------------------------------------------
_icd_lookup_df: Optional[pd.DataFrame] = None
_icd_name_to_codes: Optional[Dict[str, Set[str]]] = None


# ===================================================================
# Utility functions
# ===================================================================

def normalize_code(code: str) -> str:
    """Normalize a medical code for comparison.

    Applies the following transformations so that both predictions and
    labels are compared on equal footing:

    1. Strip ICD-10-GM modifier suffixes: ``+``, ``!``, ``*``.
    2. Lower-case the result.
    3. Remove trailing ``.0`` / ``.00`` suffixes to handle equivalences
       such as ``I10.00`` == ``I10``.

    Parameters
    ----------
    code : str
        The raw medical code (ICD-10-GM, ATC, or OPS).

    Returns
    -------
    str
        The normalized code, or ``""`` if the input is empty or invalid.

    Examples
    --------
    >>> normalize_code("I10.00!")
    'i10'
    >>> normalize_code("C34.1+")
    'c34.1'
    >>> normalize_code("")
    ''
    """
    if not isinstance(code, str) or not code:
        return ""
    normalized = re.sub(r"[!+*]", "", code).strip().lower()
    # Remove trailing .00 or .0 for equivalence (I10.00 -> I10)
    if normalized.endswith(".00"):
        normalized = normalized[:-3]
    elif normalized.endswith(".0"):
        normalized = normalized[:-2]
    return normalized


def is_internal_code(code: str) -> bool:
    """Return ``True`` if *code* is a hospital-internal identifier.

    Internal codes (e.g. ``B1AAYVLCKIZ4``) do not follow the standard
    ICD-10-GM format ``[A-Z]\\d{2}(.\\d{1,2})?[!+*]?``.  When labels
    contain such codes they must be resolved via the lookup table before
    evaluation, otherwise RAG-corrected predictions would be penalised
    for not matching an unresolvable identifier.

    Parameters
    ----------
    code : str
        The code to check.

    Returns
    -------
    bool
        ``True`` if the code does **not** match a valid ICD-10-GM pattern.
    """
    if not isinstance(code, str) or not code:
        return False
    icd_pattern = r"^[A-Z]\d{2}(\.\d{1,2})?[!+*]?$"
    return not bool(re.match(icd_pattern, code.upper().strip()))


def load_icd_lookup() -> Dict[str, Set[str]]:
    """Load the ICD-10 lookup table and return a name-to-codes mapping.

    The mapping is cached globally so it is built only once per process.
    Entries with ``"existiert nicht"`` in the display name are excluded.

    Returns
    -------
    dict[str, set[str]]
        Mapping from lower-cased diagnosis name to a set of valid ICD codes.
    """
    global _icd_lookup_df, _icd_name_to_codes

    if _icd_name_to_codes is not None:
        return _icd_name_to_codes

    _icd_lookup_df = pd.read_csv(ICD10_LOOKUP)
    _icd_name_to_codes = {}

    for _, row in _icd_lookup_df.iterrows():
        code = str(row["code"]).strip()
        display = str(row["display"]).strip().lower()
        if not display or display == "nan" or "existiert nicht" in display:
            continue
        if is_internal_code(code):
            continue
        _icd_name_to_codes.setdefault(display, set()).add(code)

    return _icd_name_to_codes


def resolve_internal_code_from_name(
    name: str, lookup_df: Optional[pd.DataFrame] = None,
    internal_code: Optional[str] = None,
) -> Optional[str]:
    """Resolve an internal code by looking up the diagnosis name.

    When a label contains an internal code we attempt to find a standard
    ICD-10-GM code via an exact name match in the lookup table.  If
    multiple codes match, the most specific (longest) one is preferred.

    Parameters
    ----------
    name : str
        The diagnosis name to look up.
    lookup_df : pd.DataFrame, optional
        Ignored; kept for API compatibility.  The module-level lookup
        loaded from ``config.ICD10_LOOKUP`` is always used.
    internal_code : str, optional
        The original internal code (for logging only).

    Returns
    -------
    str or None
        A valid ICD code if found, ``None`` otherwise.

    Examples
    --------
    >>> resolve_internal_code_from_name("Essentielle Hypertonie")
    'I10'  # (if present in the lookup table)
    """
    if not name or not isinstance(name, str):
        return None

    name_to_codes = load_icd_lookup()
    normalized = name.strip().lower()

    if normalized not in name_to_codes:
        return None

    codes = list(name_to_codes[normalized])
    if len(codes) == 1:
        return codes[0]

    # Prefer longer (more specific) codes
    codes.sort(key=lambda c: (-len(c), c))
    return codes[0]


# ===================================================================
# JSON parsing helpers
# ===================================================================

def parse_json_like(value: Any) -> Optional[dict]:
    """Parse a JSON string, Python literal, or repairable JSON into a dict.

    Handles the common quirks of LLM-generated output:

    - Standard ``json.loads`` is tried first.
    - Trailing ``<end_of_turn>`` tokens are stripped.
    - ``ast.literal_eval`` is used as a fallback for single-quoted dicts.
    - ``json_repair.repair_json`` (optional dependency) is the last resort.

    Parameters
    ----------
    value : str, dict, list, or other
        The raw value to parse.

    Returns
    -------
    dict or list or None
        The parsed object, or ``None`` if parsing fails completely.
    """
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None

    # Strict JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip trailing LLM tokens
    cleaned = re.sub(r"<end_of_turn>$", "", raw).strip()
    if cleaned != raw:
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raw = cleaned

    # Python literal (single quotes)
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass

    # json_repair (optional dependency)
    if repair_json is not None:
        try:
            repaired = repair_json(raw)
            if repaired:
                return json.loads(repaired)
        except Exception:
            pass

    return None


# ===================================================================
# Schema validation
# ===================================================================

def validate_schema(prediction: Any) -> bool:
    """Check whether a parsed prediction follows the expected schema.

    The expected top-level keys are: ``introduction``, ``diagnoses``,
    ``tumor_informations``, ``medication``, ``lab_values``, ``free_text``.

    Parameters
    ----------
    prediction : str or dict
        A raw JSON string or already-parsed dict.

    Returns
    -------
    bool
        ``True`` if the schema is valid.
    """
    obj = parse_json_like(prediction)
    if not isinstance(obj, dict):
        return False

    required = [
        "introduction", "diagnoses", "tumor_informations",
        "medication", "lab_values", "free_text",
    ]
    for key in required:
        if key not in obj:
            return False

    intro = obj.get("introduction", {})
    if not isinstance(intro, dict):
        return False

    diagnoses = obj.get("diagnoses", [])
    if not isinstance(diagnoses, list):
        return False
    for d in diagnoses:
        if not isinstance(d, dict):
            return False

    if not isinstance(obj.get("tumor_informations", []), list):
        return False
    if not isinstance(obj.get("medication", []), list):
        return False
    if not isinstance(obj.get("lab_values", []), list):
        return False

    ft = obj.get("free_text", {})
    if not isinstance(ft, dict):
        return False

    return True


# ===================================================================
# Core metric functions
# ===================================================================

def _is_invalid_label_code(code: str) -> bool:
    """B99 and its variants are placeholder codes in labels."""
    if not isinstance(code, str):
        return False
    return code.upper().strip().startswith("B99")


def _filter_invalid(code_set: Set[str]) -> Set[str]:
    """Remove B99 placeholder codes from a code set."""
    return {c for c in code_set if not _is_invalid_label_code(c)}


def _codes_match_first4(c1: str, c2: str) -> bool:
    """Check if two codes match in their first 4 characters.

    This allows for minor sub-code differences (e.g. ``M54.9`` vs ``M54.99``)
    to still count as matches.
    """
    if not isinstance(c1, str) or not isinstance(c2, str):
        return False
    p1 = c1[:4] if len(c1) >= 4 else c1
    p2 = c2[:4] if len(c2) >= 4 else c2
    return p1 == p2


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity: |intersection| / |union|.

    Uses 4-character prefix matching and B99 filtering.  Two codes are
    considered matching if they are exactly equal **or** if their first
    4 characters match (e.g. ``M54.9`` and ``M54.99``).

    Parameters
    ----------
    set1 : set[str]
        First set of codes (typically predictions).
    set2 : set[str]
        Second set of codes (typically ground truth).

    Returns
    -------
    float
        Jaccard similarity in [0, 1].  Returns 1.0 when both sets are empty.
    """
    set2_filtered = _filter_invalid(set2)
    if not set1 and not set2_filtered:
        return 1.0

    matched1: set = set()
    matched2: set = set()
    for c1 in set1:
        for c2 in set2_filtered:
            if c1 == c2 or _codes_match_first4(c1, c2):
                matched1.add(c1)
                matched2.add(c2)

    intersection = len(matched1)
    union = len(set1) + len(set2_filtered) - intersection
    return intersection / union if union > 0 else 0.0


def precision_recall_f1(
    predicted: Set[str], ground_truth: Set[str]
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score.

    Uses 4-character prefix matching and B99 filtering, consistent with
    ``jaccard_similarity``.

    Parameters
    ----------
    predicted : set[str]
        Set of predicted codes/values.
    ground_truth : set[str]
        Set of ground truth codes/values.

    Returns
    -------
    tuple[float, float, float]
        ``(precision, recall, f1)``.  Returns ``(1.0, 1.0, 1.0)`` when
        both sets are empty (perfect agreement on "nothing").
    """
    gt_filtered = _filter_invalid(ground_truth)
    if not predicted and not gt_filtered:
        return 1.0, 1.0, 1.0
    if not predicted or not gt_filtered:
        return 0.0, 0.0, 0.0

    tp = 0
    for pc in predicted:
        for gc in gt_filtered:
            if pc == gc or _codes_match_first4(pc, gc):
                tp += 1
                break

    precision = tp / len(predicted)
    recall = tp / len(gt_filtered)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# --- Prefix / category variants ---

def _f3_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """Jaccard similarity using only the first 3 characters of each code."""
    set2_filtered = _filter_invalid(set2)
    s1 = {c[:3] for c in set1 if isinstance(c, str) and len(c) >= 3}
    s2 = {c[:3] for c in set2_filtered if isinstance(c, str) and len(c) >= 3}
    if not s1 and not s2:
        return 1.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def _category_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """Jaccard similarity using only the first character (ICD category)."""
    s1 = {c[:1] for c in set1 if isinstance(c, str) and len(c) >= 1}
    s2 = {c[:1] for c in set2 if isinstance(c, str) and len(c) >= 1}
    return jaccard_similarity(s1, s2)


def _f3_f1(
    predicted: Set[str], ground_truth: Set[str]
) -> Tuple[float, float, float]:
    """F1 using first 3 characters of each code."""
    gt_filtered = _filter_invalid(ground_truth)
    p3 = {c[:3] for c in predicted if isinstance(c, str) and len(c) >= 3}
    g3 = {c[:3] for c in gt_filtered if isinstance(c, str) and len(c) >= 3}
    return precision_recall_f1(p3, g3)


def _category_f1(
    predicted: Set[str], ground_truth: Set[str]
) -> Tuple[float, float, float]:
    """F1 using the first character (ICD category)."""
    gt_filtered = _filter_invalid(ground_truth)
    p1 = {c[:1] for c in predicted if isinstance(c, str) and len(c) >= 1}
    g1 = {c[:1] for c in gt_filtered if isinstance(c, str) and len(c) >= 1}
    return precision_recall_f1(p1, g1)


def _ops_category_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """Jaccard using the OPS category (everything before first ``-``)."""
    s1 = {c.split("-")[0] for c in set1 if isinstance(c, str) and "-" in c}
    s2 = {c.split("-")[0] for c in set2 if isinstance(c, str) and "-" in c}
    return jaccard_similarity(s1, s2)


def _ops_category_f1(
    predicted: Set[str], ground_truth: Set[str]
) -> Tuple[float, float, float]:
    """F1 using OPS category (everything before first ``-``)."""
    p = {c.split("-")[0] for c in predicted if isinstance(c, str) and "-" in c}
    g = {c.split("-")[0] for c in ground_truth if isinstance(c, str) and "-" in c}
    return precision_recall_f1(p, g)


def exact_match(val1: str, val2: str) -> float:
    """Return 1.0 if values are identical, else 0.0.

    Both values are compared as-is (no normalization).
    """
    return 1.0 if val1 == val2 else 0.0


# ===================================================================
# Internal helpers for dissecting parsed JSON
# ===================================================================

def _resolve_and_normalize(
    raw_code: str, name: str, is_label: bool
) -> Optional[str]:
    """Normalize a code, resolving internal codes for labels.

    For predictions, internal codes are simply dropped (return ``None``).
    For labels, we attempt resolution via the ICD lookup table first.
    """
    if not raw_code:
        return None
    if not is_internal_code(raw_code):
        return normalize_code(raw_code)
    if is_label:
        resolved = resolve_internal_code_from_name(
            name, internal_code=raw_code
        )
        if resolved:
            return normalize_code(resolved)
    return None


def _extract_intro(prediction: dict) -> dict:
    """Extract introduction fields as a flat dict."""
    intro = prediction.get("introduction", {})
    if not isinstance(intro, dict):
        intro = {}
    return {field: intro.get(field, "") for field in INTRODUCTION_FIELDS}


def _extract_diagnosis_sets(
    prediction: dict, is_label: bool
) -> Dict[str, Set[str]]:
    """Extract diagnosis code/name/date sets separated by type."""
    main_icd: Set[str] = set()
    main_names: Set[str] = set()
    main_dates: Set[str] = set()
    side_icd: Set[str] = set()
    side_names: Set[str] = set()
    side_dates: Set[str] = set()

    for diag in prediction.get("diagnoses", []):
        if not isinstance(diag, dict):
            continue
        dtype = diag.get("type", "")
        dname = diag.get("name", "")
        raw_code = diag.get("icd10gm_code_rag") or diag.get("icd10gm_code", "")
        ddate = diag.get("date", "")

        code = _resolve_and_normalize(raw_code, dname, is_label)

        if dtype == "main_diagnosis":
            main_dates.add(ddate)
            if code:
                main_icd.add(code)
            main_names.add(dname)
        elif dtype == "side_diagnosis":
            if code:
                side_icd.add(code)
            side_names.add(dname)
            side_dates.add(ddate)

    return {
        "main_diagnosis_icd_codes": main_icd,
        "main_diagnosis_names": main_names,
        "main_diagnosis_dates": main_dates,
        "side_diagnosis_icd_codes": side_icd,
        "side_diagnosis_names": side_names,
        "side_diagnosis_dates": side_dates,
    }


def _extract_tumor_sets(prediction: dict) -> Dict[str, Set[str]]:
    """Extract tumor information fields into named sets."""
    fields: Dict[str, Set[str]] = {
        "tumor_pathological_stages": set(),
        "tumor_pathological_t": set(),
        "tumor_pathological_n": set(),
        "tumor_pathological_m": set(),
        "tumor_clinical_t": set(),
        "tumor_clinical_n": set(),
        "tumor_clinical_m": set(),
        "tumor_histologies": set(),
        "tumor_overall_statuses": set(),
        "tumor_progressions": set(),
        "tumor_markers": set(),
        "tumor_marker_values": set(),
        "tumor_smoking_statuses": set(),
        "tumor_ecog_scores": set(),
        "tumor_comorbidities": set(),
        "tumor_operations": set(),
        "tumor_radiotherapies": set(),
    }

    for ti in prediction.get("tumor_informations", []):
        if not isinstance(ti, dict):
            continue
        ttype = ti.get("type", "")
        if ttype == "pathological":
            fields["tumor_pathological_stages"].add(ti.get("stage", ""))
            fields["tumor_pathological_t"].add(ti.get("t", ""))
            fields["tumor_pathological_n"].add(ti.get("n", ""))
            fields["tumor_pathological_m"].add(ti.get("m", ""))
        elif ttype == "clinical":
            fields["tumor_clinical_t"].add(ti.get("t", ""))
            fields["tumor_clinical_n"].add(ti.get("n", ""))
            fields["tumor_clinical_m"].add(ti.get("m", ""))
        elif ttype == "histology":
            fields["tumor_histologies"].add(ti.get("histology", ""))
        elif ttype == "overall_status":
            fields["tumor_overall_statuses"].add(ti.get("status_de", ""))
        elif ttype == "progression":
            fields["tumor_progressions"].add(ti.get("description_de", ""))
        elif ttype == "tumor_marker":
            fields["tumor_markers"].add(ti.get("marker", ""))
            fields["tumor_marker_values"].add(str(ti.get("value", "")))
        elif ttype == "smoking_status":
            fields["tumor_smoking_statuses"].add(ti.get("status", ""))
        elif ttype == "ecog_performance":
            fields["tumor_ecog_scores"].add(str(ti.get("score", "")))
        elif ttype == "comorbidities":
            conds = ti.get("conditions", [])
            if isinstance(conds, list):
                fields["tumor_comorbidities"].update(conds)
        elif ttype == "operations":
            procs = ti.get("procedures", [])
            if isinstance(procs, list):
                for p in procs:
                    if isinstance(p, dict):
                        fields["tumor_operations"].add(
                            p.get("procedure_name", "")
                        )
        elif ttype == "radiotherapy":
            procs = ti.get("procedures", [])
            if isinstance(procs, list):
                for p in procs:
                    if isinstance(p, dict):
                        fields["tumor_radiotherapies"].add(
                            p.get("procedure_name", "")
                        )

    return fields


def _extract_medication_sets(prediction: dict) -> Dict[str, Set[str]]:
    """Extract medication names, dosages, and ATC codes."""
    med_names: Set[str] = set()
    med_dosages: Set[str] = set()
    med_atc: Set[str] = set()

    for med in prediction.get("medication", []):
        if not isinstance(med, dict):
            continue
        mname = med.get("medication_name", "")
        if not mname:
            continue
        med_names.add(mname)
        dosage = med.get("dosage_info", {})
        med_dosages.add(
            json.dumps(dosage, sort_keys=True)
            if isinstance(dosage, dict)
            else str(dosage)
        )
        atc = normalize_code(med.get("atc_code", ""))
        med_atc.add(atc)

    return {
        "medication_names": med_names,
        "medication_dosages": med_dosages,
        "medication_atc_codes": med_atc,
    }


def _extract_lab_sets(prediction: dict) -> Dict[str, Set[str]]:
    """Extract lab value names and values."""
    lab_names: Set[str] = set()
    lab_vals: Set[str] = set()
    for lab in prediction.get("lab_values", []):
        if not isinstance(lab, dict):
            continue
        lab_names.add(lab.get("lab_name", ""))
        lab_vals.add(str(lab.get("lab_value", "")))
    return {"lab_names": lab_names, "lab_values": lab_vals}


def _extract_free_text_sets(
    prediction: dict, is_label: bool
) -> Dict[str, Set[str]]:
    """Extract free-text subsection fields into named sets."""
    ft = prediction.get("free_text", {})
    if not isinstance(ft, dict):
        ft = {}

    result: Dict[str, Set[str]] = {}

    # Lab values
    ft_lab_names: Set[str] = set()
    ft_lab_values: Set[str] = set()
    for lab in ft.get("lab_values", []):
        if isinstance(lab, dict):
            ft_lab_names.add(lab.get("name", ""))
            ft_lab_values.add(lab.get("value", ""))
    result["free_text_lab_names"] = ft_lab_names
    result["free_text_lab_values"] = ft_lab_values

    # Medications
    ft_med_names: Set[str] = set()
    ft_med_atc: Set[str] = set()
    for med in ft.get("medications", []):
        if isinstance(med, dict):
            ft_med_names.add(med.get("medication_name", ""))
            atc = normalize_code(med.get("atc_code", ""))
            if atc:
                ft_med_atc.add(atc)
    result["free_text_medication_names"] = ft_med_names
    result["free_text_medication_atc_codes"] = ft_med_atc

    # Body values
    ft_body_weights: Set[str] = set()
    ft_body_heights: Set[str] = set()
    for bv in ft.get("body_values", []):
        if isinstance(bv, dict):
            if "body_weight" in bv:
                ft_body_weights.add(bv.get("body_weight", ""))
            if "body_height" in bv:
                ft_body_heights.add(bv.get("body_height", ""))
    result["free_text_body_weights"] = ft_body_weights
    result["free_text_body_heights"] = ft_body_heights

    # Procedures
    ft_procedures: Set[str] = set()
    ft_procedure_codes: Set[str] = set()
    ft_procedure_code_types: Set[str] = set()
    for proc in ft.get("procedures", []):
        if isinstance(proc, dict):
            ft_procedures.add(proc.get("procedure_name", ""))
            ft_procedure_codes.add(
                proc.get("procedure_code", proc.get("ops_code", ""))
            )
            ft_procedure_code_types.add(proc.get("code_type", ""))
    result["free_text_procedures"] = ft_procedures
    result["free_text_procedure_codes"] = ft_procedure_codes
    result["free_text_procedure_code_types"] = ft_procedure_code_types

    # Diagnoses
    ft_side_icd: Set[str] = set()
    ft_side_names: Set[str] = set()
    for diag in ft.get("diagnoses", []):
        if not isinstance(diag, dict):
            continue
        raw_code = (
            diag.get("icd10gm_code_rag") or diag.get("icd10gm_code", "")
        )
        dname = diag.get("official_name", "")
        code = _resolve_and_normalize(raw_code, dname, is_label)
        if code:
            ft_side_icd.add(code)
        ft_side_names.add(dname)
    result["free_text_side_diagnosis_icd_codes"] = ft_side_icd
    result["free_text_side_diagnosis_names"] = ft_side_names

    return result


def _dissect_json(json_object: Any, is_label: bool = False) -> Dict[str, Any]:
    """Flatten a prediction/label JSON into sets of comparable fields.

    This is the central extraction function that delegates to the
    per-section extractors above.

    Parameters
    ----------
    json_object : str or dict
        The raw JSON string or already-parsed dict.
    is_label : bool
        When ``True``, internal codes in labels will be resolved via the
        ICD lookup table.

    Returns
    -------
    dict
        A flat dictionary where values are either strings (introduction
        fields) or sets (code / name collections).  Returns ``{}`` on
        parse failure.
    """
    prediction = parse_json_like(json_object)
    if not isinstance(prediction, dict):
        return {}

    result: Dict[str, Any] = {}
    result.update(_extract_intro(prediction))
    result.update(_extract_diagnosis_sets(prediction, is_label))
    result.update(_extract_tumor_sets(prediction))
    result.update(_extract_medication_sets(prediction))
    result.update(_extract_lab_sets(prediction))
    result.update(_extract_free_text_sets(prediction, is_label))

    return result


# ===================================================================
# Section evaluators
# ===================================================================

def evaluate_introduction(pred: dict, truth: dict) -> Dict[str, float]:
    """Compare 10 introduction fields via exact match.

    Parameters
    ----------
    pred : dict
        Parsed prediction JSON (full document).
    truth : dict
        Parsed ground truth JSON (full document).

    Returns
    -------
    dict[str, float]
        Per-field exact match (0.0 or 1.0) plus an ``introduction_f1``
        key with the mean across all 10 fields.
    """
    pred_intro = _extract_intro(pred) if isinstance(pred, dict) else {}
    truth_intro = _extract_intro(truth) if isinstance(truth, dict) else {}

    scores: Dict[str, float] = {}
    matches = 0
    for field in INTRODUCTION_FIELDS:
        match = exact_match(
            pred_intro.get(field, ""), truth_intro.get(field, "")
        )
        scores[f"intro_{field}"] = match
        matches += match

    scores["introduction_f1"] = matches / len(INTRODUCTION_FIELDS)
    return scores


def evaluate_diagnoses(
    pred: list, truth: list, level: str = "full",
    is_label_pred: bool = False, is_label_truth: bool = True,
) -> Dict[str, float]:
    """Compare ICD codes at three granularity levels.

    This function works on lists of diagnosis dicts (the ``diagnoses``
    array from the PIGEON schema).  Codes are separated into main and
    side diagnoses, and also combined into an "all" category.

    Parameters
    ----------
    pred : list[dict]
        List of predicted diagnosis dicts.
    truth : list[dict]
        List of ground truth diagnosis dicts.
    level : str
        One of ``"full"`` (exact 4+ char match), ``"prefix3"`` (first 3
        chars), or ``"category"`` (first character).
    is_label_pred : bool
        Whether to resolve internal codes in predictions.
    is_label_truth : bool
        Whether to resolve internal codes in truth (default ``True``).

    Returns
    -------
    dict[str, float]
        Jaccard and F1 scores for main, side, and all diagnoses at the
        specified level.  B99 placeholder codes are filtered out.
    """
    # Build sets
    def _build_icd_set(diag_list, dtype_filter, is_label):
        codes: Set[str] = set()
        for d in diag_list:
            if not isinstance(d, dict):
                continue
            if dtype_filter and d.get("type", "") != dtype_filter:
                continue
            raw = d.get("icd10gm_code_rag") or d.get("icd10gm_code", "")
            dname = d.get("name", "")
            code = _resolve_and_normalize(raw, dname, is_label)
            if code:
                codes.add(code)
        return codes

    results: Dict[str, float] = {}

    for prefix, dtype_filter in [
        ("main_diagnosis", "main_diagnosis"),
        ("side_diagnosis", "side_diagnosis"),
        ("all_diagnosis", None),
    ]:
        pred_codes = _build_icd_set(pred, dtype_filter, is_label_pred)
        truth_codes = _build_icd_set(truth, dtype_filter, is_label_truth)

        if level == "full":
            results[f"{prefix}_icd_codes_jaccard"] = jaccard_similarity(
                pred_codes, truth_codes
            )
            _, _, f1_val = precision_recall_f1(pred_codes, truth_codes)
            results[f"{prefix}_icd_codes_f1"] = f1_val
        elif level == "prefix3":
            results[f"{prefix}_icd_codes_f3_jaccard"] = _f3_jaccard(
                pred_codes, truth_codes
            )
            _, _, f1_val = _f3_f1(pred_codes, truth_codes)
            results[f"{prefix}_icd_codes_f3_f1"] = f1_val
        elif level == "category":
            results[f"{prefix}_icd_codes_category_jaccard"] = _category_jaccard(
                pred_codes, truth_codes
            )
            _, _, f1_val = _category_f1(pred_codes, truth_codes)
            results[f"{prefix}_icd_codes_category_f1"] = f1_val

    return results


def evaluate_tumor_informations(
    pred: list, truth: list
) -> Dict[str, float]:
    """Evaluate tumor information fields per type.

    For each of the 11 tumor types (pathological, clinical, histology,
    overall_status, progression, tumor_marker, smoking_status,
    ecog_performance, comorbidities, operations, radiotherapy), computes
    Jaccard similarity and F1 on the extracted field sets.

    Parameters
    ----------
    pred : list[dict]
        Predicted tumor_informations list.
    truth : list[dict]
        Ground truth tumor_informations list.

    Returns
    -------
    dict[str, float]
        Jaccard and F1 for each tumor sub-field.
    """
    # Build a temporary prediction-like dict for the extractor
    pred_sets = _extract_tumor_sets({"tumor_informations": pred})
    truth_sets = _extract_tumor_sets({"tumor_informations": truth})

    results: Dict[str, float] = {}
    for field in pred_sets:
        results[f"{field}_jaccard"] = jaccard_similarity(
            pred_sets[field], truth_sets[field]
        )
        _, _, f1_val = precision_recall_f1(
            pred_sets[field], truth_sets[field]
        )
        results[f"{field}_f1"] = f1_val

    return results


def evaluate_medications(pred: list, truth: list) -> Dict[str, float]:
    """Compare medication names, dosage_info, and atc_codes.

    Parameters
    ----------
    pred : list[dict]
        Predicted medication list.
    truth : list[dict]
        Ground truth medication list.

    Returns
    -------
    dict[str, float]
        Jaccard and F1 for medication names, dosages, and ATC codes.
    """
    pred_sets = _extract_medication_sets({"medication": pred})
    truth_sets = _extract_medication_sets({"medication": truth})

    results: Dict[str, float] = {}
    for key in ["medication_names", "medication_dosages", "medication_atc_codes"]:
        results[f"{key}_jaccard"] = jaccard_similarity(
            pred_sets[key], truth_sets[key]
        )
        _, _, f1_val = precision_recall_f1(pred_sets[key], truth_sets[key])
        results[f"{key}_f1"] = f1_val

    return results


def evaluate_lab_values(pred: list, truth: list) -> Dict[str, float]:
    """Compare lab_name and lab_value pairs.

    Parameters
    ----------
    pred : list[dict]
        Predicted lab_values list.
    truth : list[dict]
        Ground truth lab_values list.

    Returns
    -------
    dict[str, float]
        Jaccard and F1 for lab names and lab values.
    """
    pred_sets = _extract_lab_sets({"lab_values": pred})
    truth_sets = _extract_lab_sets({"lab_values": truth})

    results: Dict[str, float] = {}
    for key in ["lab_names", "lab_values"]:
        results[f"{key}_jaccard"] = jaccard_similarity(
            pred_sets[key], truth_sets[key]
        )
        _, _, f1_val = precision_recall_f1(pred_sets[key], truth_sets[key])
        results[f"{key}_f1"] = f1_val

    return results


def evaluate_free_text(pred: dict, truth: dict) -> Dict[str, float]:
    """Evaluate each free_text subsection independently.

    Subsections evaluated: ``lab_values``, ``medications``, ``body_values``,
    ``procedures``, ``diagnoses``.

    Parameters
    ----------
    pred : dict
        Parsed prediction JSON (full document).
    truth : dict
        Parsed ground truth JSON (full document).

    Returns
    -------
    dict[str, float]
        Jaccard and F1 for each free-text sub-field.
    """
    pred_sets = _extract_free_text_sets(
        pred if isinstance(pred, dict) else {}, is_label=False
    )
    truth_sets = _extract_free_text_sets(
        truth if isinstance(truth, dict) else {}, is_label=True
    )

    results: Dict[str, float] = {}

    # Body values
    for key in ["free_text_body_weights", "free_text_body_heights"]:
        results[f"{key}_jaccard"] = jaccard_similarity(
            pred_sets[key], truth_sets[key]
        )
        _, _, f1_val = precision_recall_f1(pred_sets[key], truth_sets[key])
        results[f"{key}_f1"] = f1_val

    # Lab values
    for key in ["free_text_lab_names", "free_text_lab_values"]:
        results[f"{key}_jaccard"] = jaccard_similarity(
            pred_sets[key], truth_sets[key]
        )
        _, _, f1_val = precision_recall_f1(pred_sets[key], truth_sets[key])
        results[f"{key}_f1"] = f1_val

    # Medications
    for key in ["free_text_medication_names", "free_text_medication_atc_codes"]:
        results[f"{key}_jaccard"] = jaccard_similarity(
            pred_sets[key], truth_sets[key]
        )
        _, _, f1_val = precision_recall_f1(pred_sets[key], truth_sets[key])
        results[f"{key}_f1"] = f1_val

    # Procedures
    results["free_text_procedures_jaccard"] = jaccard_similarity(
        pred_sets["free_text_procedures"], truth_sets["free_text_procedures"]
    )
    _, _, f1_val = precision_recall_f1(
        pred_sets["free_text_procedures"], truth_sets["free_text_procedures"]
    )
    results["free_text_procedures_f1"] = f1_val

    results["free_text_procedure_codes_jaccard"] = jaccard_similarity(
        pred_sets["free_text_procedure_codes"],
        truth_sets["free_text_procedure_codes"],
    )
    results["free_text_procedure_codes_category_jaccard"] = _ops_category_jaccard(
        pred_sets["free_text_procedure_codes"],
        truth_sets["free_text_procedure_codes"],
    )
    _, _, f1_val = precision_recall_f1(
        pred_sets["free_text_procedure_codes"],
        truth_sets["free_text_procedure_codes"],
    )
    results["free_text_procedure_codes_f1"] = f1_val
    _, _, f1_val = _ops_category_f1(
        pred_sets["free_text_procedure_codes"],
        truth_sets["free_text_procedure_codes"],
    )
    results["free_text_procedure_codes_category_f1"] = f1_val

    # Diagnoses (ICD codes at all three levels)
    ft_pred_icd = pred_sets["free_text_side_diagnosis_icd_codes"]
    ft_truth_icd = truth_sets["free_text_side_diagnosis_icd_codes"]

    results["free_text_side_icd_jaccard"] = jaccard_similarity(
        ft_pred_icd, ft_truth_icd
    )
    results["free_text_side_icd_f3_jaccard"] = _f3_jaccard(
        ft_pred_icd, ft_truth_icd
    )
    results["free_text_side_icd_category_jaccard"] = _category_jaccard(
        ft_pred_icd, ft_truth_icd
    )
    _, _, f1_val = precision_recall_f1(ft_pred_icd, ft_truth_icd)
    results["free_text_side_icd_f1"] = f1_val
    _, _, f1_val = _f3_f1(ft_pred_icd, ft_truth_icd)
    results["free_text_side_icd_f3_f1"] = f1_val
    _, _, f1_val = _category_f1(ft_pred_icd, ft_truth_icd)
    results["free_text_side_icd_category_f1"] = f1_val

    return results


# ===================================================================
# RAG impact analysis
# ===================================================================

def evaluate_rag_impact(
    pred_code: str, truth_code: str, rag_code: str
) -> str:
    """Classify the outcome of a single RAG correction.

    Compares the original prediction code and the RAG-corrected code
    against the ground truth to determine what happened.

    Parameters
    ----------
    pred_code : str
        The original (pre-RAG) predicted code.
    truth_code : str
        The ground truth code.
    rag_code : str
        The RAG-corrected code.

    Returns
    -------
    str
        One of:

        - ``"corrected"``   -- original was wrong, RAG fixed it.
        - ``"still_wrong"`` -- original was wrong, RAG did not fix it.
        - ``"made_worse"``  -- original was correct, RAG broke it.
        - ``"both_correct"``-- original was correct, RAG kept it correct.
    """
    on = normalize_code(pred_code)
    rn = normalize_code(rag_code)
    ln = normalize_code(truth_code)

    orig_ok = on == ln or _codes_match_first4(on, ln)
    rag_ok = rn == ln or _codes_match_first4(rn, ln)

    if orig_ok and rag_ok:
        return "both_correct"
    elif not orig_ok and rag_ok:
        return "corrected"
    elif orig_ok and not rag_ok:
        return "made_worse"
    else:
        return "still_wrong"


def aggregate_rag_stats(results: List[str]) -> Dict[str, Any]:
    """Aggregate a list of RAG impact classifications into counts.

    Parameters
    ----------
    results : list[str]
        List of classification strings from ``evaluate_rag_impact``.

    Returns
    -------
    dict
        Counts and percentages for each category, plus a ``total`` key.

    Examples
    --------
    >>> aggregate_rag_stats(["corrected", "both_correct", "corrected"])
    {'total': 3, 'corrected': 2, 'corrected_pct': 66.67, ...}
    """
    counts: Dict[str, int] = defaultdict(int)
    for r in results:
        counts[r] += 1

    total = len(results)
    stats: Dict[str, Any] = {"total": total}
    for category in ["both_correct", "corrected", "made_worse", "still_wrong"]:
        c = counts.get(category, 0)
        stats[category] = c
        stats[f"{category}_pct"] = round(c / total * 100, 2) if total > 0 else 0.0

    return stats


def _track_rag_impact(
    pred_parsed: dict,
    label_parsed: dict,
    records: list,
) -> None:
    """Compare original vs RAG-corrected codes and classify the outcome.

    This is an internal function used during dataset-level evaluation
    to collect per-diagnosis RAG impact records.

    Outcomes tracked (using the internal naming convention for backwards
    compatibility with existing result files):

    - ``both_correct``       -- original was correct, RAG kept it correct
    - ``rag_corrected_wrong``-- original was wrong, RAG fixed it
    - ``rag_made_worse``     -- original was correct, RAG broke it
    - ``rag_still_wrong``    -- original was wrong, RAG did not fix it
    """

    def _check(diag_list_pred, diag_list_label, name_key, section):
        for pred_diag in diag_list_pred:
            if not isinstance(pred_diag, dict):
                continue
            dname = pred_diag.get(name_key, "")
            original = pred_diag.get("icd10gm_code", "")
            rag = pred_diag.get("icd10gm_code_rag", "")
            if not rag or is_internal_code(rag) or is_internal_code(original):
                continue

            # Find matching label
            label_code = None
            for ld in diag_list_label:
                if not isinstance(ld, dict):
                    continue
                if ld.get(name_key, "") == dname:
                    raw = ld.get("icd10gm_code", "")
                    if raw and is_internal_code(raw):
                        label_code = resolve_internal_code_from_name(
                            dname, internal_code=raw
                        )
                    elif raw and not is_internal_code(raw):
                        label_code = raw
                    break

            if not label_code:
                continue

            on = normalize_code(original)
            rn = normalize_code(rag)
            ln = normalize_code(label_code)

            orig_ok = on == ln or _codes_match_first4(on, ln)
            rag_ok = rn == ln or _codes_match_first4(rn, ln)

            if orig_ok and rag_ok:
                etype = "both_correct"
            elif not orig_ok and rag_ok:
                etype = "rag_corrected_wrong"
            elif orig_ok and not rag_ok:
                etype = "rag_made_worse"
            else:
                etype = "rag_still_wrong"

            records.append({
                "diagnosis_name": dname,
                "original_code": original,
                "rag_code": rag,
                "label_code": label_code,
                "error_type": etype,
                "section": section,
            })

    # Main/side diagnoses
    _check(
        pred_parsed.get("diagnoses", []),
        label_parsed.get("diagnoses", []),
        "name",
        "diagnoses",
    )

    # Free-text diagnoses
    _check(
        pred_parsed.get("free_text", {}).get("diagnoses", []),
        label_parsed.get("free_text", {}).get("diagnoses", []),
        "official_name",
        "free_text_diagnoses",
    )


# ===================================================================
# Single-example evaluation
# ===================================================================

def evaluate_single_example(
    pred_json: Any, truth_json: Any
) -> Dict[str, float]:
    """Compute all metrics for one (prediction, label) pair.

    Calls all section evaluators and merges results into a single flat
    dictionary.  This is the primary entry point for per-example evaluation.

    Parameters
    ----------
    pred_json : str or dict
        Raw prediction JSON string or already-parsed dict.
    truth_json : str or dict
        Raw ground truth JSON string or already-parsed dict.

    Returns
    -------
    dict[str, float]
        A flat dictionary of metric names to scores.  Returns ``{}``
        if either input cannot be parsed.
    """
    dp = _dissect_json(pred_json, is_label=False)
    dl = _dissect_json(truth_json, is_label=True)

    if not dp or not dl:
        return {}

    s: Dict[str, float] = {}

    # --- Introduction (exact match for 10 fields) ---
    intro_fields_internal = [
        ("pat_family_name", "family_name"),
        ("pat_given_name", "given_name"),
        ("pat_birth_date", "birth_date"),
        ("pat_gender", "gender"),
        ("pat_adress_street", "address_street"),
        ("pat_adress_city", "address_city"),
        ("pat_adress_zip", "address_postal_code"),
        ("pat_stationary_type", "stationary_type"),
        ("pat_encounter_start_date", "encounter_start_date"),
        ("pat_encounter_end_date", "encounter_end_date"),
    ]
    intro_sum = 0.0
    for metric_key, field_key in intro_fields_internal:
        match = exact_match(dp.get(field_key, ""), dl.get(field_key, ""))
        s[metric_key] = match
        intro_sum += match
    s["introduction_f1"] = intro_sum / len(intro_fields_internal)

    # --- Main diagnosis ---
    s["main_diagnosis_dates"] = jaccard_similarity(
        dp["main_diagnosis_dates"], dl["main_diagnosis_dates"]
    )
    s["main_diagnosis_icd_codes_jaccard"] = jaccard_similarity(
        dp["main_diagnosis_icd_codes"], dl["main_diagnosis_icd_codes"]
    )
    s["main_diagnosis_icd_codes_f3_jaccard"] = _f3_jaccard(
        dp["main_diagnosis_icd_codes"], dl["main_diagnosis_icd_codes"]
    )
    s["main_diagnosis_icd_codes_category_jaccard"] = _category_jaccard(
        dp["main_diagnosis_icd_codes"], dl["main_diagnosis_icd_codes"]
    )

    _, _, f1_val = precision_recall_f1(
        dp["main_diagnosis_icd_codes"], dl["main_diagnosis_icd_codes"]
    )
    s["main_diagnosis_icd_codes_f1"] = f1_val
    _, _, f1_val = _f3_f1(
        dp["main_diagnosis_icd_codes"], dl["main_diagnosis_icd_codes"]
    )
    s["main_diagnosis_icd_codes_f3_f1"] = f1_val
    _, _, f1_val = _category_f1(
        dp["main_diagnosis_icd_codes"], dl["main_diagnosis_icd_codes"]
    )
    s["main_diagnosis_icd_codes_category_f1"] = f1_val

    # --- Side diagnosis ---
    s["side_diagnosis_icd_codes_jaccard"] = jaccard_similarity(
        dp["side_diagnosis_icd_codes"], dl["side_diagnosis_icd_codes"]
    )
    s["side_diagnosis_icd_codes_f3_jaccard"] = _f3_jaccard(
        dp["side_diagnosis_icd_codes"], dl["side_diagnosis_icd_codes"]
    )
    s["side_diagnosis_icd_codes_category_jaccard"] = _category_jaccard(
        dp["side_diagnosis_icd_codes"], dl["side_diagnosis_icd_codes"]
    )

    _, _, f1_val = precision_recall_f1(
        dp["side_diagnosis_icd_codes"], dl["side_diagnosis_icd_codes"]
    )
    s["side_diagnosis_icd_codes_f1"] = f1_val
    _, _, f1_val = _f3_f1(
        dp["side_diagnosis_icd_codes"], dl["side_diagnosis_icd_codes"]
    )
    s["side_diagnosis_icd_codes_f3_f1"] = f1_val
    _, _, f1_val = _category_f1(
        dp["side_diagnosis_icd_codes"], dl["side_diagnosis_icd_codes"]
    )
    s["side_diagnosis_icd_codes_category_f1"] = f1_val

    # --- All diagnoses combined ---
    all_pred_icd = dp["main_diagnosis_icd_codes"] | dp["side_diagnosis_icd_codes"]
    all_label_icd = dl["main_diagnosis_icd_codes"] | dl["side_diagnosis_icd_codes"]

    s["all_diagnosis_icd_codes_jaccard"] = jaccard_similarity(
        all_pred_icd, all_label_icd
    )
    s["all_diagnosis_icd_codes_f3_jaccard"] = _f3_jaccard(
        all_pred_icd, all_label_icd
    )
    s["all_diagnosis_icd_codes_category_jaccard"] = _category_jaccard(
        all_pred_icd, all_label_icd
    )
    _, _, f1_val = precision_recall_f1(all_pred_icd, all_label_icd)
    s["all_diagnosis_icd_codes_f1"] = f1_val
    _, _, f1_val = _f3_f1(all_pred_icd, all_label_icd)
    s["all_diagnosis_icd_codes_f3_f1"] = f1_val
    _, _, f1_val = _category_f1(all_pred_icd, all_label_icd)
    s["all_diagnosis_icd_codes_category_f1"] = f1_val

    # --- Tumor information ---
    tumor_set_fields = [
        "tumor_pathological_stages", "tumor_pathological_t",
        "tumor_pathological_n", "tumor_pathological_m",
        "tumor_clinical_t", "tumor_clinical_n", "tumor_clinical_m",
        "tumor_histologies", "tumor_overall_statuses", "tumor_progressions",
        "tumor_markers", "tumor_marker_values",
        "tumor_smoking_statuses", "tumor_ecog_scores",
        "tumor_comorbidities", "tumor_operations", "tumor_radiotherapies",
    ]
    for field in tumor_set_fields:
        s[f"{field}_jaccard"] = jaccard_similarity(dp[field], dl[field])
        _, _, f1_val = precision_recall_f1(dp[field], dl[field])
        s[f"{field}_f1"] = f1_val

    # --- Medications ---
    s["medication_names_jaccard"] = jaccard_similarity(
        dp["medication_names"], dl["medication_names"]
    )
    s["medication_dosages_jaccard"] = jaccard_similarity(
        dp["medication_dosages"], dl["medication_dosages"]
    )
    s["medication_atc_codes_jaccard"] = jaccard_similarity(
        dp["medication_atc_codes"], dl["medication_atc_codes"]
    )
    _, _, f1_val = precision_recall_f1(
        dp["medication_names"], dl["medication_names"]
    )
    s["medication_names_f1"] = f1_val
    _, _, f1_val = precision_recall_f1(
        dp["medication_dosages"], dl["medication_dosages"]
    )
    s["medication_dosages_f1"] = f1_val
    _, _, f1_val = precision_recall_f1(
        dp["medication_atc_codes"], dl["medication_atc_codes"]
    )
    s["medication_atc_codes_f1"] = f1_val

    # --- Lab values ---
    s["lab_names_jaccard"] = jaccard_similarity(
        dp["lab_names"], dl["lab_names"]
    )
    s["lab_values_jaccard"] = jaccard_similarity(
        dp["lab_values"], dl["lab_values"]
    )
    _, _, f1_val = precision_recall_f1(dp["lab_names"], dl["lab_names"])
    s["lab_names_f1"] = f1_val
    _, _, f1_val = precision_recall_f1(dp["lab_values"], dl["lab_values"])
    s["lab_values_f1"] = f1_val

    # --- Free text: body values ---
    s["free_text_body_weights_jaccard"] = jaccard_similarity(
        dp["free_text_body_weights"], dl["free_text_body_weights"]
    )
    s["free_text_body_heights_jaccard"] = jaccard_similarity(
        dp["free_text_body_heights"], dl["free_text_body_heights"]
    )
    _, _, f1_val = precision_recall_f1(
        dp["free_text_body_weights"], dl["free_text_body_weights"]
    )
    s["free_text_body_weights_f1"] = f1_val
    _, _, f1_val = precision_recall_f1(
        dp["free_text_body_heights"], dl["free_text_body_heights"]
    )
    s["free_text_body_heights_f1"] = f1_val

    # --- Free text: labs ---
    s["free_text_lab_names_jaccard"] = jaccard_similarity(
        dp["free_text_lab_names"], dl["free_text_lab_names"]
    )
    s["free_text_lab_values_jaccard"] = jaccard_similarity(
        dp["free_text_lab_values"], dl["free_text_lab_values"]
    )
    _, _, f1_val = precision_recall_f1(
        dp["free_text_lab_names"], dl["free_text_lab_names"]
    )
    s["free_text_lab_names_f1"] = f1_val
    _, _, f1_val = precision_recall_f1(
        dp["free_text_lab_values"], dl["free_text_lab_values"]
    )
    s["free_text_lab_values_f1"] = f1_val

    # --- Free text: medications ---
    s["free_text_medication_names_jaccard"] = jaccard_similarity(
        dp["free_text_medication_names"], dl["free_text_medication_names"]
    )
    s["free_text_medication_atc_codes_jaccard"] = jaccard_similarity(
        dp["free_text_medication_atc_codes"],
        dl["free_text_medication_atc_codes"],
    )
    _, _, f1_val = precision_recall_f1(
        dp["free_text_medication_names"], dl["free_text_medication_names"]
    )
    s["free_text_medication_names_f1"] = f1_val
    _, _, f1_val = precision_recall_f1(
        dp["free_text_medication_atc_codes"],
        dl["free_text_medication_atc_codes"],
    )
    s["free_text_medication_atc_codes_f1"] = f1_val

    # --- Free text: procedures ---
    s["free_text_procedures_jaccard"] = jaccard_similarity(
        dp["free_text_procedures"], dl["free_text_procedures"]
    )
    s["free_text_procedure_codes_jaccard"] = jaccard_similarity(
        dp["free_text_procedure_codes"], dl["free_text_procedure_codes"]
    )
    s["free_text_procedure_codes_category_jaccard"] = _ops_category_jaccard(
        dp["free_text_procedure_codes"], dl["free_text_procedure_codes"]
    )
    _, _, f1_val = precision_recall_f1(
        dp["free_text_procedures"], dl["free_text_procedures"]
    )
    s["free_text_procedures_f1"] = f1_val
    _, _, f1_val = precision_recall_f1(
        dp["free_text_procedure_codes"], dl["free_text_procedure_codes"]
    )
    s["free_text_procedure_codes_f1"] = f1_val
    _, _, f1_val = _ops_category_f1(
        dp["free_text_procedure_codes"], dl["free_text_procedure_codes"]
    )
    s["free_text_procedure_codes_category_f1"] = f1_val

    # --- Free text: side diagnoses ---
    s["free_text_side_icd_jaccard"] = jaccard_similarity(
        dp["free_text_side_diagnosis_icd_codes"],
        dl["free_text_side_diagnosis_icd_codes"],
    )
    s["free_text_side_icd_f3_jaccard"] = _f3_jaccard(
        dp["free_text_side_diagnosis_icd_codes"],
        dl["free_text_side_diagnosis_icd_codes"],
    )
    s["free_text_side_icd_category_jaccard"] = _category_jaccard(
        dp["free_text_side_diagnosis_icd_codes"],
        dl["free_text_side_diagnosis_icd_codes"],
    )
    _, _, f1_val = precision_recall_f1(
        dp["free_text_side_diagnosis_icd_codes"],
        dl["free_text_side_diagnosis_icd_codes"],
    )
    s["free_text_side_icd_f1"] = f1_val
    _, _, f1_val = _f3_f1(
        dp["free_text_side_diagnosis_icd_codes"],
        dl["free_text_side_diagnosis_icd_codes"],
    )
    s["free_text_side_icd_f3_f1"] = f1_val
    _, _, f1_val = _category_f1(
        dp["free_text_side_diagnosis_icd_codes"],
        dl["free_text_side_diagnosis_icd_codes"],
    )
    s["free_text_side_icd_category_f1"] = f1_val

    return s


# ===================================================================
# Dataset-level evaluation
# ===================================================================

def evaluate_dataset(
    predictions_path: str,
    ground_truth_path: str,
    output_path: Optional[str] = None,
    prediction_column: str = "parsed_generation",
    label_column: str = "label",
    include_rag: bool = False,
) -> pd.DataFrame:
    """Batch evaluation of an entire predictions CSV against ground truth.

    Steps:
    1. Load both CSVs.
    2. Parse JSON from the ``prediction_column`` column.
    3. Evaluate each row with ``evaluate_single_example``.
    4. Compute aggregate statistics (mean, std for each metric).
    5. Print a summary table to stdout.
    6. Optionally save detailed results CSV.

    Parameters
    ----------
    predictions_path : str
        Path to CSV containing model predictions (one row per example).
    ground_truth_path : str
        Path to CSV containing ground truth labels.  May be the same file
        if both columns are present.
    output_path : str, optional
        Path to save per-example scores CSV.  If ``None``, results are
        only printed, not saved.
    prediction_column : str
        Column name for predictions (default: ``"parsed_generation"``).
    label_column : str
        Column name for labels (default: ``"label"``).
    include_rag : bool
        When ``True``, also compute and print RAG impact analysis.

    Returns
    -------
    pd.DataFrame
        Per-example scores with columns for every metric.
    """
    print(f"Loading predictions from: {predictions_path}")
    pred_df = pd.read_csv(predictions_path)

    if predictions_path == ground_truth_path:
        gt_df = pred_df
    else:
        print(f"Loading ground truth from: {ground_truth_path}")
        gt_df = pd.read_csv(ground_truth_path)

    if len(pred_df) != len(gt_df):
        raise ValueError(
            f"Row count mismatch: predictions={len(pred_df)}, "
            f"ground_truth={len(gt_df)}"
        )

    predictions = pred_df[prediction_column].tolist()
    labels = gt_df[label_column].tolist()

    row_scores: list = []
    rag_records: list = []
    skipped = 0

    for pred_raw, label_raw in tqdm(
        zip(predictions, labels), total=len(predictions), desc="Evaluating"
    ):
        # Clean label (strip trailing LLM tokens)
        if isinstance(label_raw, str):
            label_raw = label_raw.replace("<end_of_turn>", "")

        scores = evaluate_single_example(pred_raw, label_raw)
        if not scores:
            skipped += 1
            continue
        row_scores.append(scores)

        # RAG impact tracking
        if include_rag:
            pp = parse_json_like(pred_raw)
            lp = parse_json_like(
                label_raw.replace("<end_of_turn>", "")
                if isinstance(label_raw, str)
                else label_raw
            )
            if pp and lp:
                _track_rag_impact(pp, lp, rag_records)

    print(f"\nEvaluated {len(row_scores)} examples ({skipped} skipped)")

    scores_df = pd.DataFrame(row_scores)

    # --- Print aggregate results (mean +/- std) ---
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (mean +/- std across examples)")
    print("=" * 70)
    for col in scores_df.columns:
        mean_val = scores_df[col].mean()
        std_val = scores_df[col].std()
        print(f"  {col:55s} {mean_val:.4f} +/- {std_val:.4f}")

    # --- RAG impact summary ---
    if include_rag and rag_records:
        rag_df = pd.DataFrame(rag_records)
        rag_classifications = [r["error_type"] for r in rag_records]
        # Map internal names to the public API names
        mapped = []
        _rag_name_map = {
            "both_correct": "both_correct",
            "rag_corrected_wrong": "corrected",
            "rag_made_worse": "made_worse",
            "rag_still_wrong": "still_wrong",
        }
        for r in rag_classifications:
            mapped.append(_rag_name_map.get(r, r))

        stats = aggregate_rag_stats(mapped)

        print("\n" + "=" * 70)
        print("RAG CORRECTION IMPACT ANALYSIS")
        print("=" * 70)
        print(f"  Total codes evaluated: {stats['total']}")
        for category in ["both_correct", "corrected", "made_worse", "still_wrong"]:
            count = stats.get(category, 0)
            pct = stats.get(f"{category}_pct", 0.0)
            print(f"  {category:25s}: {count:5d}  ({pct:5.1f}%)")

        # Save RAG details alongside scores if output path is given
        if output_path:
            rag_output = str(Path(output_path).with_suffix("")) + "_rag_impact.csv"
            os.makedirs(os.path.dirname(rag_output) or ".", exist_ok=True)
            rag_df.to_csv(rag_output, index=False)
            print(f"\nRAG impact details saved to: {rag_output}")
    elif include_rag:
        print("\nNo RAG correction records found (no _rag fields in predictions)")

    # --- Save scores ---
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        scores_df.to_csv(output_path, index=False)
        print(f"\nScores saved to: {output_path}")

    return scores_df


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    """Parse command-line arguments and run dataset-level evaluation.

    Arguments
    ---------
    --predictions : str (required)
        Path to predictions CSV.
    --ground-truth : str (required)
        Path to ground truth CSV (may be the same file).
    --prediction-column : str (default: ``"parsed_generation"``)
        Column name for predictions.
    --label-column : str (default: ``"label"``)
        Column name for labels.
    --output-scores : str (optional)
        Path to save per-example scores CSV.
    --include-rag : flag
        When set, include RAG correction impact analysis.
    """
    parser = argparse.ArgumentParser(
        description="Stage 7: Evaluate PIGEON predictions against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Path to predictions CSV.",
    )
    parser.add_argument(
        "--ground-truth", required=True,
        help="Path to ground truth CSV (may be the same file).",
    )
    parser.add_argument(
        "--prediction-column", default="parsed_generation",
        help="Column name for predictions (default: 'parsed_generation').",
    )
    parser.add_argument(
        "--label-column", default="label",
        help="Column name for labels (default: 'label').",
    )
    parser.add_argument(
        "--output-scores", default=None,
        help="Path to save per-example scores CSV.",
    )
    parser.add_argument(
        "--include-rag", action="store_true", default=False,
        help="Include RAG correction impact analysis.",
    )
    args = parser.parse_args()

    evaluate_dataset(
        predictions_path=args.predictions,
        ground_truth_path=args.ground_truth,
        output_path=args.output_scores,
        prediction_column=args.prediction_column,
        label_column=args.label_column,
        include_rag=args.include_rag,
    )


if __name__ == "__main__":
    main()
