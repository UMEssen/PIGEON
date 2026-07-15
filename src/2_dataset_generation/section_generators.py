"""
Non-LLM section generators -- Stage 2 of the PIGEON pipeline.

Every public function in this module creates one section of a synthetic
German medical report ("Arztbrief") from structured FHIR data **without**
calling an LLM.  Each function returns::

    {"text": str, "labels": dict | list}

WHY separate from the orchestrator?
    These are pure-data transforms: given a DataFrame, emit text + labels.
    Keeping them in their own module lets us unit-test each template in
    isolation and swap templates without touching the async orchestrator.

WHY keep the German strings?
    The templates (headers, greeting variants, medication table layouts)
    are domain artefacts that reproduce the style of real German discharge
    letters.  They must be preserved verbatim.
"""

from __future__ import annotations

import ast
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dateutil import parser as dateutil_parser

# ---------------------------------------------------------------------------
# Import central config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# =========================================================================
# Module-level helpers
# =========================================================================

def _load_jargon_lookups() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load the ICD-10 jargon lookup CSV and return two dicts.

    Returns:
        (jargon_lookup, official_lookup)
        *jargon_lookup*  maps ``icd10gm_code`` -> stringified list of jargon
        *official_lookup* maps ``icd10gm_code`` -> official display name

    WHY not cache globally?
        The CSV is ~3 MB and loading it takes <50 ms.  Caching would
        save time in batch runs; we leave the door open by reading from
        ``config.ICD10_JARGON_LOOKUP`` so callers can memoise easily.
    """
    df = pd.read_csv(config.ICD10_JARGON_LOOKUP)
    jargon = df.set_index("icd10gm_code")["doctor_jargon"].to_dict()
    official = df.set_index("icd10gm_code")["display"].to_dict()
    return jargon, official


def _resolve_icd_code(icd_code: str, lookup: dict) -> Optional[str]:
    """Try the raw code first, then strip ``+``, ``!``, ``*`` in turn.

    WHY?
        ICD-10-GM codes sometimes carry dagger/asterisk suffixes that the
        jargon lookup table does not include.  Stripping them is the
        standard fallback used throughout the original generator.
    """
    for candidate in [
        icd_code,
        icd_code.replace("+", ""),
        icd_code.replace("!", ""),
        icd_code.replace("*", ""),
    ]:
        if candidate in lookup:
            return candidate
    return None


def _format_date_randomly(
    date_str: str,
    chance: float = 0.4,
) -> Tuple[str, str]:
    """Format a raw date string with weighted-random style choices.

    Returns:
        (line_suffix, label_date)
        *line_suffix* is the text to append, e.g. ``" (ED: 03.2021)"``
        *label_date* is the clean date portion, e.g. ``"03.2021"``

    WHY randomise?
        Real German discharge letters show dates in a wide variety of
        formats (month.year, year only) and with different prefixes
        ("ED:", "seit", "bek.").  Weighted random selection reproduces
        this natural variance in the synthetic corpus.
    """
    if not date_str or random.random() >= chance:
        return "", ""

    # Parse date robustly
    iso_str = date_str
    if len(date_str) > 10 and date_str[10] == " ":
        iso_str = date_str[:10] + "T" + date_str[11:]
    try:
        dt = datetime.fromisoformat(iso_str)
    except ValueError:
        dt = dateutil_parser.parse(date_str, dayfirst=True)

    # Detail level: 85 % partial (mm.yyyy or mm/yyyy), 15 % year only
    detail = random.choices(["partial", "year"], weights=[85, 15], k=1)[0]
    month = dt.strftime("%m")
    year = dt.strftime("%Y")

    if detail == "partial":
        fmt = random.choices(["mm.yyyy", "mm/yyyy"], weights=[5, 2], k=1)[0]
        formatted = f"{month}.{year}" if fmt == "mm.yyyy" else f"{month}/{year}"
    else:
        formatted = year

    # Prefix
    prefix_opts = {"": 30, "am ": 10, "vom ": 10, "seit ": 10, "ED: ": 35, "bek. ": 5}
    prefix = random.choices(
        list(prefix_opts.keys()), weights=list(prefix_opts.values()), k=1
    )[0]

    return f" ({prefix}{formatted})", formatted


def _filter_internal_codes(code_str_list: List[str]) -> List[str]:
    """Remove entries whose first ICD code starts with 'CH' or 'GB'.

    WHY?
        These are internal hospital classification codes that must not
        appear in the synthetic letter or be sent to the jargon lookup.
    """
    result = []
    for code_str in code_str_list:
        try:
            codes = ast.literal_eval(code_str)
            if codes and isinstance(codes[0], str) and not codes[0].startswith(("CH", "GB")):
                result.append(code_str)
        except Exception:
            continue
    return result


# =========================================================================
# Section generators
# =========================================================================


def generate_introduction(
    data: Dict[str, Any],
    stationary_type_selector: int = 0,
    include_address: bool = False,
) -> Dict[str, Any]:
    """Generate the opening paragraph of a discharge letter.

    The introduction names the patient, states the treatment period,
    and optionally includes a postal address block.

    Args:
        data:                       Structured dict from ``FHIRDataLoader.load_data``.
        stationary_type_selector:   Index into the list of ward types
                                    (stationaer, intensivstationaer, ...).
        include_address:            Whether to prepend a postal address block.

    Returns:
        ``{"text": str, "labels": dict}``
    """
    df = data["data"]
    p = df["patient"]
    e = df["encounter"]

    # -- Patient demographics ---------------------------------------------
    first = p["png0_value"].iat[0]
    last = p["pn0.family"].iat[0]
    full_name = f"{first} {last}"
    bd = str(p["p0.birthDate"].iat[0])[:10].split("-")
    birth = f"{bd[2]}.{bd[1]}.{bd[0]}"
    gen = p["p0.gender"].iat[0] or ""

    # -- Address block (conditional) --------------------------------------
    street = p["pa0.line"].iat[0]
    city = p["pa0.city"].iat[0]
    zipc = p["pa0.postalCode"].iat[0]
    address_block = (
        f"Pat:{full_name}\n{street}\n{zipc} {city}\n\n"
        if include_address
        else ""
    )

    # -- Patient reference ------------------------------------------------
    if not include_address:
        title = "Patientin" if gen.lower() == "female" else "Patient"
        ref = f"{title} {full_name}, geb. {birth}"
    else:
        title = "o.g. Patientin" if gen.lower() == "female" else "o.g. Patient"
        ref = f"{title} geb. {birth}"

    # -- Ward type --------------------------------------------------------
    types = ["stationaer", "intensivstationaer", "halbintensivstationaer"]
    st_type = types[stationary_type_selector % len(types)]

    # -- Date formatting --------------------------------------------------
    def _fmt(col: str) -> str:
        d = str(e[col].iat[0])[:10].split("-")
        return f"{d[2]}.{d[1]}.{d[0]}"

    dt_start = _fmt("ep0_start_max")
    dt_end = _fmt("ep0_end_max")

    # -- Greeting ---------------------------------------------------------
    greetings = [
        "Sehr geehrte Damen und Herren,",
        "Liebe Kolleginnen und Kollegen,",
        "Guten Tag,",
    ]
    greet = greetings[random.randint(0, len(greetings) - 1)]

    # -- Summary templates ------------------------------------------------
    if include_address:
        summaries = [
            f"{ref} wurde vom {dt_start} bis {dt_end} {st_type} behandelt.",
            f"Hiermit erhalten Sie den Entlassbrief fuer {ref}, aufgenommen {st_type} im genannten Zeitraum.",
            f"{ref} verblieb {st_type} vom {dt_start} bis {dt_end} in unserer Klinik.",
            f"Wir berichten ueber {ref}, Behandlung {st_type} in der Zeit {dt_start}-{dt_end}.",
            f"Behandlungszeitraum {dt_start} bis {dt_end} ({st_type}) fuer {ref}.",
            f"{ref} war {st_type} vom {dt_start} bis {dt_end} stationaer bei uns.",
        ]
    else:
        summaries = [
            f"{ref} wurde vom {dt_start} bis {dt_end} {st_type} behandelt.",
            f"Wir berichten ueber die {st_type} Behandlung {ref} im Zeitraum {dt_start} bis {dt_end}.",
            f"{ref} war {st_type} vom {dt_start} bis {dt_end} stationaer bei uns.",
            f"Im Zeitraum {dt_start}-{dt_end} befand sich {ref} zur {st_type} Behandlung.",
            f"Die {st_type} Aufnahme erfolgte {dt_start} bis {dt_end} fuer {ref}.",
        ]

    summary = summaries[random.randint(0, len(summaries) - 1)]

    labels = {
        "family_name": last,
        "given_name": first,
        "birth_date": birth,
        "gender": gen.lower(),
        "address_street": street if include_address else "",
        "address_city": city if include_address else "",
        "address_postal_code": zipc if include_address else "",
        "stationary_type": st_type,
        "encounter_start_date": dt_start,
        "encounter_end_date": dt_end,
    }

    intro = f"{address_block}{greet}\n\n{summary}"
    return {"text": intro, "labels": labels}


# -------------------------------------------------------------------------
# Diagnosis sections
# -------------------------------------------------------------------------


def generate_diagnose(
    data: Dict[str, Any],
    chance_for_date: float = 0.4,
    jargon_variant: int = 0,
    patient_id: str = "",
) -> Dict[str, Any]:
    """Combined main + side diagnoses with ICD-10 jargon lookup.

    This function merges both diagnosis categories into a single
    "Diagnosen:" section, optionally prepending tumour staging info
    when ``data["data"]`` contains a ``histologie`` key.

    Args:
        data:             Structured data dict.
        chance_for_date:  Probability [0..1] that a date is appended.
        jargon_variant:   Index 0-2 selecting which jargon synonym to use.
        patient_id:       Patient ID for tumour-data lookup.

    Returns:
        ``{"text": str, "labels": list, "locations": list}``
    """
    jargon_lookup, official_lookup = _load_jargon_lookups()

    main_codes = _filter_internal_codes(
        data["data"]["main_diagnosis"]["ccc0_codes"].tolist()
        if not data["data"]["main_diagnosis"].empty
        else []
    )
    main_dates = (
        data["data"]["main_diagnosis"]["c0_recordedDate"].tolist()
        if not data["data"]["main_diagnosis"].empty
        else []
    )
    main_names = (
        data["data"]["main_diagnosis"]["ccc0_displays"].tolist()
        if not data["data"]["main_diagnosis"].empty
        else []
    )

    side_codes = _filter_internal_codes(
        data["data"]["side_diagnosis"]["ccc0_codes"].tolist()
        if not data["data"]["side_diagnosis"].empty
        else []
    )
    side_dates = (
        data["data"]["side_diagnosis"]["c0_recordedDate"].tolist()
        if not data["data"]["side_diagnosis"].empty
        else []
    )
    side_names = (
        data["data"]["side_diagnosis"]["ccc0_displays"].tolist()
        if not data["data"]["side_diagnosis"].empty
        else []
    )

    # -- Optionally inject primary tumour diagnosis from tumordoku --------
    if patient_id:
        try:
            from tumor_extraction import from_pat_get_processed_data
            tumor_info = from_pat_get_processed_data(patient_id)
            prim = tumor_info.get("primary_diagnosis", {})
            if prim and isinstance(prim, dict) and prim.get("code"):
                main_codes.append(f"['{prim['code']}']")
                main_names.append(f"['{prim.get('name', '')}']")
                main_dates.append("")
        except Exception:
            pass  # tumordoku data may not be available for every patient

    # -- Build jargon entries ---------------------------------------------
    all_diagnoses: List[Dict[str, Any]] = []
    seen_codes: set = set()

    def _process_codes(codes, dates, names, diag_type):
        for idx, code_str in enumerate(codes):
            try:
                icd_code = ast.literal_eval(code_str)[0]
            except Exception:
                continue
            if icd_code in seen_codes:
                continue
            candidate = _resolve_icd_code(icd_code, jargon_lookup)
            if candidate is None:
                continue
            icd_code = candidate
            jargon_list = ast.literal_eval(jargon_lookup[icd_code])
            pick = jargon_list[jargon_variant % len(jargon_list)]
            official = official_lookup.get(icd_code, "")

            raw_date = ""
            if idx < len(dates) and dates[idx] and not pd.isnull(dates[idx]):
                try:
                    raw_date = datetime.fromisoformat(
                        str(dates[idx])
                    ).strftime("%d/%m/%Y")
                except Exception:
                    raw_date = ""

            all_diagnoses.append({
                "picked_jargon": pick,
                "official_name": official,
                "icd_code": icd_code,
                "date": raw_date,
                "type": diag_type,
            })
            seen_codes.add(icd_code)

    _process_codes(main_codes, main_dates, main_names, "main_diagnosis")
    _process_codes(side_codes, side_dates, side_names, "side_diagnosis")

    if not all_diagnoses:
        return {"text": "Diagnosen", "labels": [], "locations": []}

    # -- Format output text -----------------------------------------------
    text_parts: List[str] = []
    labels: List[Dict[str, Any]] = []

    for diag in all_diagnoses:
        line = diag["picked_jargon"]
        suffix, label_date = _format_date_randomly(diag["date"], chance_for_date)
        line += suffix

        label_entry = {
            "type": diag["type"],
            "name": diag["official_name"],
            "icd10gm_code": diag["icd_code"],
            "date": label_date,
        }
        text_parts.append(line)
        labels.append(label_entry)

    # -- Optional histologie prepend --------------------------------------
    tumor_info_data = data["data"].get("histologie", {})
    if tumor_info_data:
        hist = generate_histologie(tumor_info_data)
        hist_text = hist.get("text", "")
        hist_labels = hist.get("labels", {})
        labels.append({
            "main_diagnosis_official_name": hist_labels.get("condition_display", ""),
            "stage": hist_labels.get("stage", ""),
            "t_stage": hist_labels.get("t_stage", ""),
            "n_stage": hist_labels.get("n_stage", ""),
            "m_stage": hist_labels.get("m_stage", ""),
            "lymph_vessel_invasion": hist_labels.get("lymph_vessel_invasion", ""),
            "venous_invasion": hist_labels.get("venous_invasion", ""),
            "perineural_invasion": hist_labels.get("perineural_invasion", ""),
        })
        text = f"Diagnosen:\n{hist_text}\n" + "\n".join(text_parts)
    else:
        text = "Diagnosen:\n" + "\n".join(text_parts)

    return {"text": text, "labels": labels, "locations": []}


def generate_hauptdiagnose(
    data: Dict[str, Any],
    chance_for_date: float = 0.4,
    jargon_variant: int = 0,
    patient_id: str = "",
) -> Dict[str, Any]:
    """Generate the 'Hauptdiagnose:' section (main diagnoses only).

    Follows the same pattern as ``generate_diagnose`` but only emits
    main-diagnosis rows and uses a "Hauptdiagnose:" header.
    """
    jargon_lookup, official_lookup = _load_jargon_lookups()
    main_data = data["data"]["main_diagnosis"]

    main_codes = (
        _filter_internal_codes(main_data["ccc0_codes"].tolist())
        if not main_data.empty
        else []
    )
    main_dates = main_data["c0_recordedDate"].tolist() if not main_data.empty else []

    # Inject primary tumour diagnosis
    if patient_id:
        try:
            from tumor_extraction import from_pat_get_processed_data
            tumor_info = from_pat_get_processed_data(patient_id)
            prim = tumor_info.get("primary_diagnosis", {})
            if prim and isinstance(prim, dict) and prim.get("code"):
                main_codes.append(f"['{prim['code']}']")
                main_dates.append("")
        except Exception:
            pass

    labels: List[Dict] = []
    lines: List[str] = []
    seen: set = set()

    for idx, code_str in enumerate(main_codes):
        if code_str in seen:
            continue
        seen.add(code_str)
        try:
            icd_code = ast.literal_eval(code_str)[0]
        except Exception:
            continue
        candidate = _resolve_icd_code(icd_code, jargon_lookup)
        if candidate is None:
            continue
        icd_code = candidate
        jargon_list = ast.literal_eval(jargon_lookup[icd_code])
        pick = jargon_list[jargon_variant % len(jargon_list)]
        official = official_lookup.get(icd_code, "")

        raw_date = ""
        if idx < len(main_dates) and main_dates[idx]:
            try:
                raw_date = str(main_dates[idx])
            except Exception:
                raw_date = ""

        line = pick
        suffix, label_date = _format_date_randomly(raw_date, chance_for_date)
        line += suffix
        lines.append(line)
        labels.append({
            "type": "main_diagnosis",
            "name": official,
            "icd10gm_code": icd_code,
            "date": label_date,
        })

    # Histologie
    tumor_info_data = data["data"].get("histologie", {})
    if tumor_info_data:
        hist = generate_histologie(tumor_info_data)
        hist_text = hist.get("text", "")
        hist_labels = hist.get("labels", {})
        labels.append({
            "main_diagnosis_official_name": hist_labels.get("condition_display", ""),
            "stage": hist_labels.get("stage", ""),
            "t_stage": hist_labels.get("t_stage", ""),
            "n_stage": hist_labels.get("n_stage", ""),
            "m_stage": hist_labels.get("m_stage", ""),
            "lymph_vessel_invasion": hist_labels.get("lymph_vessel_invasion", ""),
            "venous_invasion": hist_labels.get("venous_invasion", ""),
            "perineural_invasion": hist_labels.get("perineural_invasion", ""),
        })
        text = f"Hauptdiagnose:\n{hist_text}\n" + "\n".join(lines)
    else:
        text = "Hauptdiagnose:" + "\n".join(lines)

    return {"text": text, "labels": labels, "locations": []}


def generate_nebendiagnose(
    data: Dict[str, Any],
    chance_for_date: float = 0.4,
    jargon_variant: int = 0,
) -> Dict[str, Any]:
    """Generate the 'Nebendiagnosen:' section (side diagnoses only).

    Side diagnoses that also appear as main diagnoses (by ICD code) are
    excluded to prevent duplication across sections.
    """
    jargon_lookup, official_lookup = _load_jargon_lookups()

    # Collect main ICD codes for exclusion
    main_codes_set: set = set()
    if not data["data"]["main_diagnosis"].empty:
        for code_str in data["data"]["main_diagnosis"]["ccc0_codes"]:
            try:
                main_codes_set.add(ast.literal_eval(code_str)[0])
            except Exception:
                continue

    side_data = data["data"]["side_diagnosis"]
    side_codes = (
        _filter_internal_codes(side_data["ccc0_codes"].tolist())
        if not side_data.empty
        else []
    )
    side_dates = side_data["c0_recordedDate"].tolist() if not side_data.empty else []
    side_names = side_data["ccc0_displays"].tolist() if not side_data.empty else []

    labels: List[Dict] = []
    lines: List[str] = ["Nebendiagnosen:"]
    processed: set = set()

    for idx, code_str in enumerate(side_codes):
        try:
            icd_code = ast.literal_eval(code_str)[0]
        except Exception:
            continue
        if icd_code in main_codes_set or icd_code in processed:
            continue
        candidate = _resolve_icd_code(icd_code, jargon_lookup)
        if candidate is None:
            continue
        icd_code = candidate
        jargon_list = ast.literal_eval(jargon_lookup[icd_code])
        pick = jargon_list[jargon_variant % len(jargon_list)]
        official = official_lookup.get(icd_code, "")

        raw_date = ""
        if idx < len(side_dates) and not pd.isnull(side_dates[idx]):
            try:
                raw_date = datetime.fromisoformat(
                    str(side_dates[idx])
                ).strftime("%d/%m/%Y")
            except Exception:
                raw_date = ""

        line = pick
        suffix, label_date = _format_date_randomly(raw_date, chance_for_date)
        line += suffix

        lines.append(line)
        labels.append({
            "type": "side_diagnosis",
            "name": official,
            "icd10gm_code": icd_code,
            "date": label_date,
        })
        processed.add(icd_code)

    if not labels:
        return {"text": "Nebendiagnosen", "labels": [], "locations": []}
    return {"text": "\n".join(lines), "labels": labels, "locations": []}


# -------------------------------------------------------------------------
# Medication section
# -------------------------------------------------------------------------


def generate_medication_section(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the 'Medikation' section with one of 6 random table layouts.

    WHY 6 templates?
        Real hospital information systems produce medication lists in
        various column layouts.  Random template selection teaches the
        downstream model to parse all of them.

    Dosage parsing:
        Each dosage entry is pipe-delimited ``"HH:MM|dose|form"``.  The
        hour component maps to one of four time slots:
          06-10 -> Morgens, 10-14 -> Mittags, 14-20 -> Abends, else -> Nacht
    """
    df_medication = data["data"]["medication"]
    if df_medication.empty:
        return {"text": "", "labels": [], "locations": []}

    medication_name_col = "mcc0.display_list"
    medication_dosage_col = "md0.text_list"

    medication_data: List[Dict] = []
    processed: set = set()

    for _, row in df_medication.iterrows():
        # -- Choose best name (prefer one containing "mg") ----------------
        raw_name = row.get(medication_name_col, "")
        try:
            name_list = ast.literal_eval(raw_name)
        except Exception:
            continue
        med_name = None
        for n in name_list:
            if n and "mg" in n:
                med_name = n
                break
        if med_name is None and name_list:
            med_name = name_list[0]
        if not med_name or med_name in processed:
            continue
        processed.add(med_name)

        # -- Parse dosages ------------------------------------------------
        dosing_counts = {"Morgens": 0, "Mittags": 0, "Abends": 0, "Nacht": 0}
        dosage_text = row.get(medication_dosage_col)
        dosage_list: List[str] = []
        if isinstance(dosage_text, str):
            try:
                dosage_list = ast.literal_eval(dosage_text)
            except (ValueError, SyntaxError):
                dosage_list = []
        elif isinstance(dosage_text, list):
            dosage_list = dosage_text

        for dosage in dosage_list:
            parts = dosage.split("|")
            if len(parts) == 3:
                time_str, dose_str, _ = parts
                try:
                    dose = abs(float(dose_str.replace(",", ".")))
                except ValueError:
                    continue
                hour = int(time_str.split(":")[0])
                if 6 <= hour < 10:
                    dosing_counts["Morgens"] += dose
                elif 10 <= hour < 14:
                    dosing_counts["Mittags"] += dose
                elif 14 <= hour < 20:
                    dosing_counts["Abends"] += dose
                else:
                    dosing_counts["Nacht"] += dose

        has_night = dosing_counts["Nacht"] > 0
        label_data = {"medication_name": med_name, "dosage_info": dosing_counts}
        medication_data.append({
            "medication_name": med_name,
            "dosage_info": {"dosing_counts": dosing_counts},
            "is_nachts": has_night,
            "medication_json": label_data,
        })

    if not medication_data:
        return {"text": "", "labels": [], "locations": []}

    # -- Template functions -----------------------------------------------
    def _doses_str(med: Dict) -> str:
        dc = med["dosage_info"]["dosing_counts"]
        return "-".join(
            f"{dc.get(p, 0):g}" for p in ["Morgens", "Mittags", "Abends"]
        )

    def _template_one(meds):
        parts = ["\nMedikation:\n"]
        labels, locs = [], []
        ci = len(parts[0])
        for m in meds:
            line = f"\t{m['medication_name']}\t\t{_doses_str(m)}\n"
            parts.append(line)
            labels.append(m["medication_json"])
            locs.append((ci, ci + len(line)))
            ci += len(line)
        return "".join(parts), labels, locs

    def _template_two(meds):
        parts = ["\nNr.\n\tMedikament\n\tWirkstoff\n\tHaeufigkeit\n\tDosierschema\n\tDF\n\tKommentar\n\n"]
        labels, locs = [], []
        ci = len(parts[0])
        for i, m in enumerate(meds, 1):
            line = (
                f" {i}.\n\t{m['medication_name']}\n\t-\n\t\n"
                f"\t{_doses_str(m)}\n\t-\n\t-\n"
            )
            parts.append(line)
            labels.append(m["medication_json"])
            locs.append((ci, ci + len(line)))
            ci += len(line)
        return "".join(parts), labels, locs

    def _template_three(meds):
        parts = ["\nMedikation bei Entlassung:\n"]
        labels, locs = [], []
        ci = len(parts[0])
        for m in meds:
            line = f"\t{m['medication_name']}\t\t{_doses_str(m)}\n"
            parts.append(line)
            labels.append(m["medication_json"])
            locs.append((ci, ci + len(line)))
            ci += len(line)
        parts.append("\n\tDie haeusliche Medikation kann unveraendert eingenommen werden:\n")
        return "".join(parts), labels, locs

    def _template_four(meds):
        parts = ["\nTherapieempfehlung:\n\tWirkstoff\n\tStaerke\n\tForm\n\tMorgens\n\tMittags\n\tAbends\n\tEinheit\n\tHinweise\n\n"]
        labels, locs = [], []
        ci = len(parts[0])
        for m in meds:
            dc = m["dosage_info"]["dosing_counts"]
            line = (
                f"{m['medication_name']}\n\n\n"
                f"{dc.get('Morgens', 0):g}\n{dc.get('Mittags', 0):g}\n"
                f"{dc.get('Abends', 0):g}\n\n\n\n"
            )
            parts.append(line)
            labels.append(m["medication_json"])
            locs.append((ci, ci + len(line)))
            ci += len(line)
        return "".join(parts), labels, locs

    def _template_five(meds):
        parts = ["\nDauermedikation\n\n\tHandelsname\n\tWirkstoff\n\tStaerke\n\tForm\n\tMorgens\n\tMittags\n\tAbends\n\tZur Nacht\n\tHinweise\n\tVerordnungsgrund\n\n"]
        labels, locs = [], []
        ci = len(parts[0])
        for m in meds:
            dc = m["dosage_info"]["dosing_counts"]
            line = (
                f"{m['medication_name']}\n\n\n\n"
                f"{dc.get('Morgens', 0):g}\n{dc.get('Mittags', 0):g}\n"
                f"{dc.get('Abends', 0):g}\n{dc.get('Nacht', 0):g}\n\n\n\n"
            )
            parts.append(line)
            labels.append(m["medication_json"])
            locs.append((ci, ci + len(line)))
            ci += len(line)
        return "".join(parts), labels, locs

    def _template_six(meds):
        parts = ["\nMedikationsplan:\n\n"]
        labels, locs = [], []
        ci = len(parts[0])
        for m in meds:
            line = f"\t{m['medication_name']} \t\t{_doses_str(m)}\n"
            parts.append(line)
            labels.append(m["medication_json"])
            locs.append((ci, ci + len(line)))
            ci += len(line)
        return "".join(parts), labels, locs

    templates = [
        _template_one, _template_two, _template_three,
        _template_four, _template_five, _template_six,
    ]

    # If any medication has a night dose, force template 5 (has Nacht column)
    if any(m["is_nachts"] for m in medication_data):
        template_func = _template_five
    else:
        template_func = random.choice(templates)

    text, labels_out, locations = template_func(medication_data)
    return {"text": text, "labels": labels_out, "locations": locations}


# -------------------------------------------------------------------------
# Lab values section
# -------------------------------------------------------------------------


def generate_lab_values(
    data: Dict[str, Any],
    max_values: int = 5,
) -> Dict[str, Any]:
    """Generate a lab-report section using one of 5 random templates.

    Only the first *max_values* lab rows are included to keep the
    section length realistic.

    WHY different templates?
        Hospital LIS exports vary by admission vs. discharge context
        and by column ordering.  Five templates cover the most common
        real-world layouts.
    """
    df_lab = data["data"].get("Laboratory", pd.DataFrame())
    if df_lab.empty:
        return {"text": "", "labels": [], "locations": []}

    lab_data: List[Dict] = []
    for _, row in df_lab.head(max_values).iterrows():
        lab_name = row.get("occ0.display", "")
        lab_value = row.get("ov0_value", "")
        ref_low = row.get("orl0_value")
        ref_high = row.get("orh0_value")
        lab_unit = row.get("ov0_unit", "")

        ref_range = (
            f"{ref_low} - {ref_high}"
            if pd.notnull(ref_low) and pd.notnull(ref_high)
            else ""
        )

        # Flag
        flag = ""
        try:
            val = float(lab_value)
            lo = float(ref_low) if pd.notnull(ref_low) else None
            hi = float(ref_high) if pd.notnull(ref_high) else None
            if lo is not None and val < lo:
                flag = "L"
            elif hi is not None and val > hi:
                flag = "H"
        except (ValueError, TypeError):
            pass

        status = "F" if str(row.get("o0_status", "")).lower() == "final" else ""

        lab_data.append({
            "lab_name": lab_name,
            "lab_value": lab_value,
            "lab_unit": lab_unit,
            "lab_ref_range": ref_range,
            "Flag": flag,
            "Status": status,
            "Vorwert": "",
            "label_data": {"lab_name": lab_name, "lab_value": lab_value},
        })

    def _lab_template(header: str, lab_items: List[Dict]):
        parts = [header]
        labels, locs = [], []
        ci = len(header)
        for lab in lab_items:
            line = (
                f"{lab['lab_name']}\n{lab['lab_value']}\n"
                f"{lab['lab_unit']}\n{lab['Flag']}\n"
                f"{lab['lab_ref_range']}\n{lab['Status']}\n\n"
            )
            parts.append(line)
            labels.append(lab["label_data"])
            locs.append((ci, ci + len(line)))
            ci += len(line)
        return "".join(parts), labels, locs

    headers = [
        "Laborparameter:\nBestimmung\nWert\n\nFlag\nStatus\nVorwert\nvom\nEinheit\nReferenz\n\n",
        "Laborparameter bei Aufnahme:\nBestimmung\nWert\nEinheit\nFlag\nReferenz\nvom\nStatus\n\n",
        "Labor:\nBestimmung\nWert\nEinheit\nFlag\nReferenz\nvom\nStatus\n\n",
        "Laborwerte bei Entlassung:\nBestimmung\nWert\nEinheit\nFlag\nReferenz\nvom\nStatus\n\n",
        "Laborparameter bei Entlassung:\nBestimmung\nWert\nEinheit\nFlag\nReferenz\nvom\nStatus\n\n",
    ]

    chosen_header = random.choice(headers)
    text, labels_out, locations = _lab_template(chosen_header, lab_data)
    return {"text": text, "labels": labels_out, "locations": locations}


# -------------------------------------------------------------------------
# Vital signs
# -------------------------------------------------------------------------


def generate_vitalzeichen(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a one-line vital-signs summary.

    Aggregates blood pressure, heart rate, body temperature, and SpO2
    from the observation sub-DataFrames.  For blood pressure the first
    two values per observation ID are joined with ``/`` (systolic/diastolic).
    """
    d = data.get("data", {})
    bp = d.get("BloodPressure", pd.DataFrame(columns=["o0_id", "ov0_value"]))
    temp = d.get("BodyTemperature", pd.DataFrame(columns=["o0_id", "ov0_value"]))
    spo2 = d.get("OxygenSaturation", pd.DataFrame(columns=["o0_id", "ov0_value"]))
    hr = d.get("HeartBeat", pd.DataFrame(columns=["o0_id", "ov0_value"]))

    if bp.empty or temp.empty or spo2.empty or hr.empty:
        return {"text": "", "labels": {}}

    # BP: group by observation ID, join first 2 values as sys/dia
    bp_s = (
        bp.groupby("o0_id")["ov0_value"]
        .agg(lambda x: "/".join(map(str, x.iloc[:2])) if len(x) >= 2 else None)
        .dropna()
    )
    temp_s = temp.set_index("o0_id")["ov0_value"]
    spo2_s = spo2.set_index("o0_id")["ov0_value"]
    hr_s = hr.set_index("o0_id")["ov0_value"]

    bp_val = bp_s.get(random.choice(bp_s.index.tolist()), "N/A")
    temp_val = temp_s.get(random.choice(temp_s.index.tolist()), "N/A")
    spo2_val = spo2_s.get(random.choice(spo2_s.index.tolist()), "N/A")
    hr_val = hr_s.get(random.choice(hr_s.index.tolist()), "N/A")

    temp_str = (
        f"{float(temp_val):.1f}"
        if isinstance(temp_val, (int, float))
        else str(temp_val)
    )

    text = (
        f"Vitalzeichen: NiBP: {bp_val} mmHg, "
        f"{hr_val} bpm, {temp_str} Grad C, SpO2 {spo2_val}%"
    )
    labels = {
        "blood_pressure": bp_val,
        "heart_rate": hr_val,
        "body_temperature": temp_str,
        "oxygen_saturation": spo2_val,
    }
    return {"text": text, "labels": labels}


# -------------------------------------------------------------------------
# Histologie (tumour staging)
# -------------------------------------------------------------------------


def generate_histologie(tumor_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a TNM / UICC staging line from tumour documentation data.

    WHY a separate function?
        Histologie information comes from the tumour documentation
        system, not from the regular FHIR condition resource.  It is
        injected into the diagnosis section when available.
    """
    jargon_lookup, _ = _load_jargon_lookups()

    tumor_icd = tumor_info.get("condition_code", "")
    jargon_list_str = jargon_lookup.get(tumor_icd, "")
    jargon_list = ast.literal_eval(jargon_list_str) if jargon_list_str else []
    tumor_jargon = (
        jargon_list[random.randint(0, len(jargon_list) - 1)]
        if jargon_list
        else tumor_info.get("condition_display", "")
    )

    # TNM components
    tnm_parts: List[str] = []
    for key, prefix in [
        ("t_stage", "T"), ("n_stage", "N"), ("m_stage", "M"),
    ]:
        val = tumor_info.get(key, "")
        if val and str(val).lower() not in ("x", "/"):
            tnm_parts.append(f"{prefix}{val}")

    for key in ("lymph_vessel_invasion", "venous_invasion", "perineural_invasion"):
        val = tumor_info.get(key, "")
        if val and str(val).lower() not in ("x", "/"):
            tnm_parts.append(str(val))

    stage = tumor_info.get("stage", "")
    tnm_str = " ".join(tnm_parts)

    if tnm_str:
        text = f"{tumor_jargon}\nInitial: {tnm_str}, UICC-Stadium: {stage}"
    else:
        text = f"{tumor_jargon}\nUICC-Stadium: {stage}"

    labels = {
        "condition_display": tumor_info.get("condition_display", ""),
        "stage": stage,
        "t_stage": tumor_info.get("t_stage", ""),
        "n_stage": tumor_info.get("n_stage", ""),
        "m_stage": tumor_info.get("m_stage", ""),
        "lymph_vessel_invasion": tumor_info.get("lymph_vessel_invasion", ""),
        "venous_invasion": tumor_info.get("venous_invasion", ""),
        "perineural_invasion": tumor_info.get("perineural_invasion", ""),
    }
    return {"text": text, "labels": labels}
