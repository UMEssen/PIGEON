"""
Tumor documentation processing -- Stage 2 of the PIGEON pipeline.

This module reads the per-patient tumour documentation CSVs that are
generated during the FHIR cache build (Stage 1) and extracts structured
oncology information: TNM staging, grading, histology, tumour markers,
ECOG performance status, comorbidities, and treatment procedures.

WHY a separate module?
    Tumour documentation lives in its own sub-cache
    (``tumordoku_precise/``) and has a very different column schema from
    the standard FHIR resources.  Mixing this parsing into the main
    data loader or the section generators would make both harder to
    test and maintain.

Data flow:
    ``from_pat_get_processed_data(patient_id)``
      -> ``get_tumordoku_data()``        -- reads raw CSVs
      -> ``from_tumor_data_to_relevant_tumor_data()``  -- restructures
      -> ``extract_initial_actual_stages()``           -- staging logic
      -> ``extract_initial_actual_tnm_stages()``       -- T/N/M components
      -> returns a flat dict ready for ``generate_tumor_informations()``
"""

from __future__ import annotations

import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Import central config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402


# =========================================================================
# Low-level helpers
# =========================================================================

def _is_valid_cancer_stage(stage: Any) -> bool:
    """Check whether *stage* looks like a UICC cancer stage.

    Accepts Roman (I-IV) and Arabic (0-4) numerals, optionally
    suffixed with A/B/C.
    """
    if stage is None or pd.isna(stage):
        return False
    s = str(stage).strip()
    if not s or s == "/":
        return False
    return bool(re.match(r"^(0|[1-4][ABC]?|I{1,3}[ABC]?|IV[ABC]?)$", s))


def _is_valid_tnm_value(value: Any) -> bool:
    """Check whether a TNM component value is clinically meaningful."""
    if value is None or pd.isna(value):
        return False
    s = str(value).strip()
    return bool(s) and s not in ("/", "x")


def get_latest_data_value(
    data_entry: Dict[str, list],
    value_key: str,
) -> Optional[Any]:
    """Return the most-recent value from a list keyed by *value_key*.

    WHY not just take the last element?
        The lists are not guaranteed to be sorted.  We pair each value
        with its ``recorded_date``, sort descending, and return the
        newest non-null entry.
    """
    dates = data_entry.get("recorded_date", [])
    values = data_entry.get(value_key, [])
    if not dates:
        return None
    valid = [
        (datetime.fromisoformat(d), v)
        for d, v in zip(dates, values)
        if d is not None
    ]
    if not valid:
        return None
    valid.sort(key=lambda x: x[0], reverse=True)
    return valid[0][1]


# =========================================================================
# Stage extraction
# =========================================================================

def extract_initial_actual_stages(
    staging_list: List[Any],
    recorded_dates: List[str],
) -> Dict[str, Optional[str]]:
    """Determine initial and progression UICC stages from chronological data.

    WHY distinguish initial vs. actual?
        If a patient was first staged as "II" and later re-staged as "III"
        we need both the initial stage (for the first-diagnosis context)
        and the current stage (for the discharge summary).

    Returns dict with keys:
        initial_stage, initial_date, actual_stage, actual_date
    """
    valid = [
        (stage, date)
        for stage, date in zip(staging_list, recorded_dates)
        if _is_valid_cancer_stage(stage)
    ]
    initial_stage = initial_date = actual_stage = actual_date = None

    if valid:
        valid.sort(key=lambda x: x[1])
        unique = list(dict.fromkeys(valid))  # dedup preserving order

        if len(unique) == 1:
            actual_stage, actual_date = unique[0]
        elif len(unique) >= 2:
            stages_only = [s for s, _ in unique]
            if len(set(stages_only)) > 1:
                initial_stage, initial_date = unique[0]
                actual_stage, actual_date = unique[-1]
            else:
                actual_stage, actual_date = unique[-1]

    return {
        "initial_stage": initial_stage,
        "initial_date": initial_date,
        "actual_stage": actual_stage,
        "actual_date": actual_date,
    }


def extract_initial_actual_tnm_stages(
    system_list: List[str],
    tnm_list: List[Any],
    recorded_dates: List[str],
    tnm_type: str = "pTNM",
) -> Dict[str, Optional[str]]:
    """Extract initial and actual T, N, M components separately.

    Handles both pathological (pTNM) and clinical (cTNM) staging by
    mapping system URLs to their component letter.

    Returns dict with keys like ``initial_T_stage``, ``actual_N_stage``, etc.
    """
    if tnm_type == "pTNM":
        components_map = {
            "https://example-hospital.org/fhir/TumorDocumentation/pTNM/T": "T",
            "https://example-hospital.org/fhir/TumorDocumentation/pTNM/N": "N",
            "https://example-hospital.org/fhir/TumorDocumentation/pTNM/M": "M",
        }
    else:
        components_map = {
            "http://dktk.dkfz.de/fhir/onco/core/ValueSet/TNMTVS": "T",
            "http://dktk.dkfz.de/fhir/onco/core/ValueSet/TNMNVS": "N",
            "http://dktk.dkfz.de/fhir/onco/core/ValueSet/TNMMVS": "M",
        }

    tnm_data: Dict[str, list] = {"T": [], "N": [], "M": []}
    for system, value, date in zip(system_list, tnm_list, recorded_dates):
        if system in components_map and _is_valid_tnm_value(value):
            tnm_data[components_map[system]].append((value, date))

    result: Dict[str, Optional[str]] = {}
    for comp in ("T", "N", "M"):
        entries = tnm_data[comp]
        initial = actual = initial_date = actual_date = None
        if entries:
            entries.sort(key=lambda x: x[1])
            unique = list(dict.fromkeys(entries))
            if len(unique) == 1:
                actual, actual_date = unique[0]
            elif len(unique) >= 2:
                stages_only = [s for s, _ in unique]
                if len(set(stages_only)) > 1:
                    initial, initial_date = unique[0]
                    actual, actual_date = unique[-1]
                else:
                    actual, actual_date = unique[-1]
        result[f"initial_{comp}_stage"] = initial
        result[f"initial_{comp}_date"] = initial_date
        result[f"actual_{comp}_stage"] = actual
        result[f"actual_{comp}_date"] = actual_date
    return result


# =========================================================================
# Tumordoku CSV loading
# =========================================================================

def get_tumordoku_data(
    pat_id: str,
    tumordoku_path: Optional[Path] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Read all tumordoku cache CSVs for *pat_id* and return nested dict.

    The cache directory layout is::

        tumordoku_precise/
          <pat_id>/
            <pat_id>_observation_pTNM.csv
            <pat_id>_condition_PrimaryTumorDiagnosis.csv
            ...

    Returns::

        {
            "observation": {"pTNM": DataFrame, "cTNM": DataFrame, ...},
            "procedure":   {"Operation": DataFrame, ...},
            "condition":   {"Comorbidity": DataFrame, ...},
        }
    """
    if tumordoku_path is None:
        tumordoku_path = config.TUMORDOKU_CACHE

    pat_dir = tumordoku_path / str(pat_id)
    result: Dict[str, Dict[str, pd.DataFrame]] = {
        "observation": {},
        "procedure": {},
        "condition": {},
    }

    csv_files = glob.glob(str(pat_dir / "*.csv"))
    for csv_file in csv_files:
        fname = os.path.basename(csv_file).replace(".csv", "")
        parts = fname.split("_")
        if len(parts) >= 3:
            resource_type = parts[1]
            identifier = "_".join(parts[2:])
            if resource_type in result:
                result[resource_type][identifier] = pd.read_csv(csv_file)

    return result


# =========================================================================
# Data restructuring
# =========================================================================

# Column-name mapping per tumordoku file type.
_COLUMN_MAPPING = {
    "condition_Comorbidity": {
        "code": "ccc0.code", "name": "ccc0.display",
        "recorded_date": "c0.recordedDate",
    },
    "condition_PrimaryTumorDiagnosis": {
        "code": "ccc0.code", "name": "ccc0.display",
        "recorded_date": "c0.recordedDate",
    },
    "condition_SecondaryTumorSite": {
        "code": "ccc0.code", "name": "ccc0.display",
        "recorded_date": "c0.recordedDate",
    },
    "observation_cTNM": {
        "cancer_stage": "ovc0_display", "system": "ocvc0_system",
        "cTNM": "ocvc0_display", "recorded_date": "o0_effectiveDateTime",
    },
    "observation_DistantMetastases": {
        "location": "obc0.display",
        "recorded_date": "o0_effectiveDateTime",
    },
    "observation_ECOG": {
        "performance_status": "ovc0_display", "system": "ovc0_system",
        "recorded_date": "o0_effectiveDateTime",
    },
    "observation_Grading": {
        "grade": "ovc0_display",
        "recorded_date": "o0_effectiveDateTime",
    },
    "observation_Histology": {
        "type": "ovc0_display",
        "recorded_date": "o0_effectiveDateTime",
    },
    "observation_Oncogene": {
        "gene": "ovc0_display",
        "recorded_date": "o0_effectiveDateTime",
    },
    "observation_OtherStaging": {
        "staging": "ovc0_display",
        "recorded_date": "o0_effectiveDateTime",
    },
    "observation_Progress": {
        "progress": "ovc0_display",
        "recorded_date": "o0_effectiveDateTime",
    },
    "observation_pTNM": {
        "cancer_stage": "ovc0_display", "system": "ocvc0_system",
        "pTNM": "ocvc0_display", "recorded_date": "o0_effectiveDateTime",
    },
    "observation_SmokingStatus": {
        "value": "ocv0.smoke.value",
        "recorded_date": "o0_effectiveDateTime",
    },
    "observation_TumorMarker": {
        "marker": "occ0.display", "value": "ov0.value",
        "unit": "ov0.unit", "recorded_date": "o0_effectiveDateTime",
    },
    "observation_TumorStatusOverall": {
        "status": "ovc0_display",
        "recorded_date": "o0_effectiveDateTime",
    },
    "procedure_ExaminationPerformed": {
        "type": "pcc0.display",
        "recorded_date": "p0.performedDateTime",
    },
    "procedure_Operation": {
        "type": "pcc0.display",
        "recorded_date": "p0.performedDateTime",
    },
    "procedure_Strahlentherapie": {
        "type": "pcc0.display",
        "recorded_date": "p0.performedDateTime",
    },
}


def from_tumor_data_to_relevant_tumor_data(
    tumor_data: Dict[str, Dict[str, pd.DataFrame]],
) -> Dict[str, Any]:
    """Restructure raw tumordoku DataFrames into flat keyed lists.

    Then derive staging, TNM, histology, comorbidities, etc. into
    a single hierarchical dict suitable for text generation.

    WHY this two-pass approach?
        Pass 1 (``relevant_tumor_data``) normalises column names.
        Pass 2 extracts clinically meaningful aggregates (initial vs.
        actual staging, latest grading, etc.).
    """
    relevant: Dict[str, Dict[str, list]] = {}

    for resource_type, identifiers in tumor_data.items():
        for identifier, df in identifiers.items():
            key = f"{resource_type}_{identifier}"
            if key not in _COLUMN_MAPPING:
                continue
            col_map = _COLUMN_MAPPING[key]
            if key not in relevant:
                relevant[key] = {k: [] for k in col_map}
            for _, row in df.iterrows():
                for new_key, old_col in col_map.items():
                    val = row.get(old_col)
                    relevant[key][new_key].append(
                        None if pd.isna(val) else val
                    )

    # -- Derive staging ---------------------------------------------------
    staging = extract_initial_actual_stages(
        relevant.get("observation_pTNM", {}).get("cancer_stage", []),
        relevant.get("observation_pTNM", {}).get("recorded_date", []),
    )

    ptnm = extract_initial_actual_tnm_stages(
        relevant.get("observation_pTNM", {}).get("system", []),
        relevant.get("observation_pTNM", {}).get("pTNM", []),
        relevant.get("observation_pTNM", {}).get("recorded_date", []),
        "pTNM",
    )

    ctnm = extract_initial_actual_tnm_stages(
        relevant.get("observation_cTNM", {}).get("system", []),
        relevant.get("observation_cTNM", {}).get("cTNM", []),
        relevant.get("observation_cTNM", {}).get("recorded_date", []),
        "cTNM",
    )

    # -- Simple latest-value fields ---------------------------------------
    other_staging = get_latest_data_value(
        relevant.get("observation_OtherStaging", {"recorded_date": [], "staging": []}),
        "staging",
    )
    smoking = get_latest_data_value(
        relevant.get("observation_SmokingStatus", {"recorded_date": [], "value": []}),
        "value",
    )
    metastases = get_latest_data_value(
        relevant.get("observation_DistantMetastases", {"recorded_date": [], "location": []}),
        "location",
    )
    grading = get_latest_data_value(
        relevant.get("observation_Grading", {"recorded_date": [], "grade": []}),
        "grade",
    )
    oncogene = get_latest_data_value(
        relevant.get("observation_Oncogene", {"recorded_date": [], "gene": []}),
        "gene",
    )
    tumor_status = get_latest_data_value(
        relevant.get("observation_TumorStatusOverall", {"recorded_date": [], "status": []}),
        "status",
    )
    histology = get_latest_data_value(
        relevant.get("observation_Histology", {"recorded_date": [], "type": []}),
        "type",
    )
    progress = get_latest_data_value(
        relevant.get("observation_Progress", {"recorded_date": [], "progress": []}),
        "progress",
    )

    # -- Tumor marker (multi-value) ---------------------------------------
    tm_data = relevant.get(
        "observation_TumorMarker",
        {"recorded_date": [], "marker": [], "value": [], "unit": []},
    )
    tumor_marker = tumor_marker_value = tumor_marker_unit = None
    if tm_data["recorded_date"]:
        valid = []
        for i, d in enumerate(tm_data["recorded_date"]):
            if d:
                valid.append((
                    datetime.fromisoformat(d),
                    tm_data["marker"][i] if i < len(tm_data["marker"]) else None,
                    tm_data["value"][i] if i < len(tm_data["value"]) else None,
                    tm_data["unit"][i] if i < len(tm_data["unit"]) else None,
                ))
        if valid:
            valid.sort(key=lambda x: x[0], reverse=True)
            _, tumor_marker, tumor_marker_value, tumor_marker_unit = valid[0]

    # -- ECOG (German system only) ----------------------------------------
    ecog = None
    ecog_data = relevant.get(
        "observation_ECOG",
        {"recorded_date": [], "system": [], "performance_status": []},
    )
    if ecog_data["recorded_date"]:
        valid = []
        for i, d in enumerate(ecog_data["recorded_date"]):
            if d:
                sys_url = ecog_data["system"][i] if i < len(ecog_data["system"]) else None
                if sys_url == "https://example-hospital.org/fhir/TumorDocumentation/ECOG":
                    ps = ecog_data["performance_status"][i] if i < len(ecog_data["performance_status"]) else None
                    valid.append((datetime.fromisoformat(d), ps))
        if valid:
            valid.sort(key=lambda x: x[0], reverse=True)
            ecog = valid[0][1]

    # -- Procedures -------------------------------------------------------
    def _collect_procedures(key, type_key="type", has_date=True):
        data = relevant.get(key, {"recorded_date": [], type_key: []})
        items = []
        if data["recorded_date"]:
            for i, d in enumerate(data["recorded_date"]):
                if d and i < len(data[type_key]):
                    entry = {"type": data[type_key][i]}
                    if has_date:
                        entry["date"] = d
                    items.append(entry)
        return items

    operations = _collect_procedures("procedure_Operation")
    strahlentherapie = _collect_procedures("procedure_Strahlentherapie")
    examinations = _collect_procedures(
        "procedure_ExaminationPerformed", has_date=False
    )

    # -- Comorbidities ----------------------------------------------------
    comorbidities: List[Dict] = []
    com_data = relevant.get(
        "condition_Comorbidity",
        {"recorded_date": [], "code": [], "name": []},
    )
    if com_data["recorded_date"]:
        for i, d in enumerate(com_data["recorded_date"]):
            if d and i < len(com_data["code"]):
                name = com_data["name"][i] if i < len(com_data["name"]) else None
                if name and ("unknown" in name.lower() or "unbekannt" in name.lower()):
                    continue
                comorbidities.append({
                    "code": com_data["code"][i],
                    "name": name,
                })

    # -- Primary tumour diagnosis -----------------------------------------
    prim_code = prim_name = None
    prim_data = relevant.get(
        "condition_PrimaryTumorDiagnosis",
        {"recorded_date": [], "code": [], "name": []},
    )
    if prim_data["recorded_date"]:
        valid = []
        for i, d in enumerate(prim_data["recorded_date"]):
            if d:
                valid.append((
                    datetime.fromisoformat(d),
                    prim_data["code"][i] if i < len(prim_data["code"]) else None,
                    prim_data["name"][i] if i < len(prim_data["name"]) else None,
                ))
        if valid:
            valid.sort(key=lambda x: x[0], reverse=True)
            prim_code, prim_name = valid[0][1], valid[0][2]
        elif prim_data["code"]:
            prim_code = prim_data["code"][0]
            prim_name = prim_data["name"][0] if prim_data["name"] else None

    # -- Secondary tumour site --------------------------------------------
    sec_code = sec_name = None
    sec_data = relevant.get(
        "condition_SecondaryTumorSite",
        {"recorded_date": [], "code": [], "name": []},
    )
    if sec_data["recorded_date"]:
        valid = []
        for i, d in enumerate(sec_data["recorded_date"]):
            if d:
                valid.append((
                    datetime.fromisoformat(d),
                    sec_data["code"][i] if i < len(sec_data["code"]) else None,
                    sec_data["name"][i] if i < len(sec_data["name"]) else None,
                ))
        if valid:
            valid.sort(key=lambda x: x[0], reverse=True)
            sec_code, sec_name = valid[0][1], valid[0][2]

    # -- Build combined staging strings -----------------------------------
    def _join_staging(stage_dict, tnm_dict, prefix_key):
        parts = []
        s = stage_dict.get(f"{prefix_key}_stage")
        if s:
            parts.append(s)
        for comp in ("T", "N", "M"):
            v = tnm_dict.get(f"{prefix_key}_{comp}_stage")
            if v:
                parts.append(f"{comp}{v}")
        return " ".join(parts) if parts else None

    initial_staging = _join_staging(staging, ptnm, "initial")
    actual_staging = _join_staging(staging, ptnm, "actual")

    clinical_parts = []
    for comp in ("T", "N", "M"):
        v = ctnm.get(f"actual_{comp}_stage")
        if v:
            clinical_parts.append(f"{comp}{v}")
    clinical_tnm = " ".join(clinical_parts) if clinical_parts else None

    return {
        "staging": {
            "initial_staging": initial_staging,
            "actual_staging": actual_staging,
            "clinical_tnm": clinical_tnm,
        },
        "other_staging": other_staging,
        "grading": grading,
        "primary_diagnosis": {"code": prim_code, "name": prim_name},
        "secondary_tumor_site": {"code": sec_code, "name": sec_name},
        "histology": histology,
        "oncogene": oncogene,
        "tumor_status_overall": tumor_status,
        "progress": progress,
        "distant_metastases": metastases,
        "tumor_markers": {
            "marker": tumor_marker,
            "value": tumor_marker_value,
            "unit": tumor_marker_unit,
        },
        "smoking_status": smoking,
        "ecog_performance_status": ecog,
        "comorbidities": comorbidities,
        "procedures": {
            "operations": operations,
            "strahlentherapie": strahlentherapie,
            "examinations": examinations,
        },
    }


# =========================================================================
# Public entry points
# =========================================================================

def from_pat_get_processed_data(
    patient_id: str,
    tumordoku_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Main entry: load tumordoku CSVs for *patient_id* and return
    a processed dict with staging, histology, comorbidities, etc.
    """
    raw = get_tumordoku_data(patient_id, tumordoku_path)
    return from_tumor_data_to_relevant_tumor_data(raw)


def generate_tumor_informations(
    patient_id: str,
    tumordoku_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate the 'Tumorstatus' text section and structured labels.

    This is called by the orchestrator for patient-level reports that
    include oncology data.

    Returns:
        ``{"text": str, "labels": list[dict]}``
    """
    tumor_info = from_pat_get_processed_data(patient_id, tumordoku_path)
    if not tumor_info or not isinstance(tumor_info, dict):
        return {
            "text": "Tumorstatus: Keine verwertbaren Tumorinformationen",
            "labels": [],
        }

    lines = ["Tumorstatus:"]
    labels: List[Dict] = []

    # -- Staging ----------------------------------------------------------
    st = tumor_info.get("staging", {})
    if isinstance(st, dict):
        initial = st.get("initial_staging")
        actual = st.get("actual_staging")
        ctnm = st.get("clinical_tnm")

        if initial:
            lines.append(f"Initiales Stadium (pTNM): {initial}")
            _add_tnm_label(labels, initial, "pathological")
        if actual:
            lines.append(f"Aktuelles Stadium (pTNM): {actual}")
            _add_tnm_label(labels, actual, "pathological")
        if ctnm:
            lines.append(f"Klinisches cTNM: {ctnm}")
            t = re.search(r"T\S+", ctnm)
            n = re.search(r"N\S+", ctnm)
            m = re.search(r"M\S+", ctnm)
            labels.append({
                "type": "clinical",
                "t": t.group(0) if t else "",
                "n": n.group(0) if n else "",
                "m": m.group(0) if m else "",
                "date": "",
            })

    # -- Other staging ----------------------------------------------------
    other = tumor_info.get("other_staging")
    if other:
        lines.append(f"Weitere Stadieneinstufung: {other}")

    # -- Grading ----------------------------------------------------------
    grading = tumor_info.get("grading")
    if grading and grading != "unbekannt" and str(grading).strip() not in ("-", "?", "/"):
        lines.append(f"Grading: {str(grading).strip()}")

    # -- Histology --------------------------------------------------------
    hist = tumor_info.get("histology")
    if hist and hist.strip() and hist.strip() not in ("-", "?", "/"):
        lines.append(f"Histologie: {hist.strip()}")
        labels.append({"type": "histology", "histology": hist.strip(), "date": ""})

    # -- Oncogene ---------------------------------------------------------
    onc = tumor_info.get("oncogene")
    if onc:
        lines.append(f"Molekulare Marker / Onkogene: {onc}")

    # -- Tumour status overall --------------------------------------------
    tso = tumor_info.get("tumor_status_overall")
    if tso:
        lines.append(f"Gesamt-Tumorstatus: {tso}")
        labels.append({"type": "overall_status", "status_de": tso, "date": ""})

    # -- Progress ---------------------------------------------------------
    prog = tumor_info.get("progress")
    if prog:
        lines.append(f"Verlauf / Progression: {prog}")
        labels.append({"type": "progression", "description_de": prog, "date": ""})

    # -- Distant metastases -----------------------------------------------
    dm = tumor_info.get("distant_metastases")
    if dm:
        lines.append(f"Fernmetastasen: {dm}")
    elif dm == "":
        lines.append("Fernmetastasen: keine Angabe")

    # -- Tumour markers ---------------------------------------------------
    tm = tumor_info.get("tumor_markers", {})
    if isinstance(tm, dict):
        marker = tm.get("marker")
        value = tm.get("value")
        unit = tm.get("unit")
        if marker or value is not None:
            parts = []
            if marker:
                parts.append(str(marker))
            if value is not None:
                val_str = f"{value}{(' ' + unit) if unit else ''}"
                parts.append(val_str)
            lines.append(f"Tumormarker: {' - '.join(parts)}")
            labels.append({
                "type": "tumor_marker",
                "marker": marker,
                "value": value,
                "unit": unit,
                "date": "",
            })

    # -- Smoking status ---------------------------------------------------
    smoking = tumor_info.get("smoking_status")
    if smoking:
        lines.append(f"Raucherstatus: {smoking}")
        labels.append({"type": "smoking_status", "status": smoking, "date": ""})

    # -- ECOG -------------------------------------------------------------
    ecog = tumor_info.get("ecog_performance_status")
    if ecog is not None:
        lines.append(f"ECOG-Performance-Status: {ecog}")
        labels.append({"type": "ecog_performance", "score": ecog, "date": ""})

    # -- Comorbidities ----------------------------------------------------
    comorbs = tumor_info.get("comorbidities", [])
    if isinstance(comorbs, list) and comorbs:
        names = [c["name"] for c in comorbs if isinstance(c, dict) and c.get("name")][:3]
        if names:
            lines.append("Relevante Komorbiditaeten: " + "; ".join(names))
            labels.append({"type": "comorbidities", "conditions": names, "date": ""})

    # -- Procedures -------------------------------------------------------
    procs = tumor_info.get("procedures", {})
    if isinstance(procs, dict):
        ops = procs.get("operations", [])
        if ops:
            op_types = list(dict.fromkeys(
                o["type"] for o in ops if isinstance(o, dict) and o.get("type")
            ))
            if op_types:
                lines.append("Operationen: " + "; ".join(op_types))
                labels.append({"type": "operations", "procedures": op_types, "date": ""})

        rads = procs.get("strahlentherapie", [])
        if rads:
            rad_types = list(dict.fromkeys(
                r["type"] for r in rads if isinstance(r, dict) and r.get("type")
            ))
            if rad_types:
                lines.append("Strahlentherapie: " + "; ".join(rad_types))
                labels.append({"type": "radiotherapy", "procedures": rad_types, "date": ""})

    return {"text": "\n".join(lines), "labels": labels}


def _add_tnm_label(labels: list, tnm_string: str, label_type: str) -> None:
    """Parse a staging string like '2 T1 N0 M0' into a structured label."""
    stage = re.search(r"\(Stage\s+([^\)]+)\)", tnm_string)
    t = re.search(r"T\S+", tnm_string)
    n = re.search(r"N\S+", tnm_string)
    m = re.search(r"M\S+", tnm_string)
    labels.append({
        "type": label_type,
        "stage": stage.group(1).strip() if stage else "",
        "t": t.group(0) if t else "",
        "n": n.group(0) if n else "",
        "m": m.group(0) if m else "",
        "date": "",
    })
