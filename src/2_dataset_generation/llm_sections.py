"""
LLM-based section generators -- Stage 2 of the PIGEON pipeline.

These async functions generate the *narrative* sections of a synthetic
German discharge letter (Anamnese and Epikrise) by calling an LLM.

WHY LLM-generated?
    Anamnese and Epikrise sections are free-text narratives that weave
    diagnoses, medications, lab values, and procedures into a coherent
    clinical story.  Rule-based templates cannot produce the stylistic
    variance found in real letters, so we prompt an LLM with structured
    data and real examples.

Both functions follow the same 5-step pattern:
    1. Extract diagnoses, medications (name only), labs (abnormal only),
       procedures, and body measurements from the input data.
    2. Build a structured-data block with confirmed / negated items.
    3. Construct a German prompt that includes 3 real examples.
    4. Call the LLM (Qwen model via secondary endpoint).
    5. Parse the JSON response and build labels.
"""

from __future__ import annotations

import ast
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Import central config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402


# =========================================================================
# Utility helpers
# =========================================================================

def _extract_json_string_by_brace_balance(text: str) -> str:
    """Extract the outermost JSON object from *text* by brace balancing.

    WHY not just ``json.loads``?
        LLMs often wrap JSON in markdown fences, preamble text, or
        trailing commentary.  This function finds the first ``{`` and
        tracks brace depth (respecting string literals) to locate the
        matching ``}``, returning only the JSON substring.
    """
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return text

    balance = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape_next:
                escape_next = False
            elif ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                balance += 1
            elif ch == "}":
                balance -= 1
                if balance == 0:
                    return text[start : i + 1]
    return text  # fallback: return as-is if unbalanced


def _load_examples(
    examples_path: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """Load real Anamnese/Epikrise example texts from a pickle file.

    The pickle stores a dict with keys ``"anamnese"`` and ``"epikrise"``,
    each mapping to a list of strings.

    WHY pickle?
        The original pipeline used a pickle cache of pre-extracted real
        report sections.  We keep this format for backward compatibility
        but make the path configurable.

    Returns:
        (anamnese_list, epikrise_list)  --  lists of example strings.
        Returns empty lists if the file does not exist.
    """
    if examples_path is None:
        # Default path -- configurable via environment or override
        examples_path = config.DATASETS_DIR / "anamnese_epikrise.pkl"

    if not examples_path.exists():
        return [], []

    with open(examples_path, "rb") as fh:
        data = pickle.load(fh)
    return data.get("anamnese", []), data.get("epikrise", [])


def _load_jargon_lookups() -> Tuple[dict, dict]:
    """Thin wrapper -- identical to section_generators but avoids cross-import."""
    df = pd.read_csv(config.ICD10_JARGON_LOOKUP)
    jargon = df.set_index("icd10gm_code")["doctor_jargon"].to_dict()
    official = df.set_index("icd10gm_code")["display"].to_dict()
    return jargon, official


# =========================================================================
# Shared data-extraction logic
# =========================================================================

def _extract_llm_section_data(
    input_data: Dict[str, Any],
    jargon_variant: int,
    conditions_to_not_use: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract and limit structured data for Anamnese / Epikrise prompts.

    Both LLM sections need the same data extraction logic:
      - Side diagnoses (with jargon lookup, excluding main diag codes)
      - Medications (name only)
      - Lab values (abnormal only for Anamnese, all for Epikrise)
      - Procedures (name + OPS code)
      - Body measurements (weight, height)

    WHY shared?
        The original generator duplicated ~150 lines between
        ``generate_anamnese`` and ``generate_epikrise``.  Extracting
        into a helper eliminates that duplication.
    """
    jargon_lookup, official_lookup = _load_jargon_lookups()
    input_df = input_data.get("data", {})

    main_df = input_df.get("main_diagnosis", pd.DataFrame())
    side_df = input_df.get("side_diagnosis", pd.DataFrame())
    med_df = input_df.get("medication", pd.DataFrame())
    lab_df = input_df.get("Laboratory", pd.DataFrame())
    bw_df = input_df.get("BodyWeight", pd.DataFrame())
    bh_df = input_df.get("BodyHeight", pd.DataFrame())
    proc_df = input_df.get("procedure", pd.DataFrame())

    # -- Main ICD codes (for exclusion) -----------------------------------
    main_icd_codes: set = set()
    if not main_df.empty and "ccc0_codes" in main_df.columns:
        for code_str in main_df["ccc0_codes"]:
            if isinstance(code_str, str):
                try:
                    codes = ast.literal_eval(code_str)
                    if codes and isinstance(codes[0], str) and not codes[0].startswith(("CH", "GB")):
                        main_icd_codes.add(codes[0])
                except Exception:
                    continue

    # -- Side diagnoses ---------------------------------------------------
    side_list: List[Dict[str, str]] = []
    processed_side: set = set()
    if not side_df.empty and "ccc0_codes" in side_df.columns:
        for _, row in side_df.iterrows():
            code_str = row.get("ccc0_codes")
            if not isinstance(code_str, str):
                continue
            try:
                codes = ast.literal_eval(code_str)
            except Exception:
                continue
            if not codes or not isinstance(codes[0], str):
                continue
            icd = codes[0]
            if icd.startswith(("CH", "GB")) or icd in main_icd_codes or icd in processed_side:
                continue
            jargon_str = jargon_lookup.get(icd)
            if jargon_str is None and "+" in icd:
                jargon_str = jargon_lookup.get(icd.replace("+", ""))
            if not isinstance(jargon_str, str):
                continue
            jargon_list = ast.literal_eval(jargon_str)
            if not isinstance(jargon_list, list) or len(jargon_list) <= jargon_variant:
                continue
            pick = jargon_list[jargon_variant]
            official = official_lookup.get(icd, f"Official name for {icd}") or f"Official name for {icd}"
            if isinstance(pick, str) and isinstance(official, str):
                side_list.append({
                    "picked_jargon": pick,
                    "icd_code": icd,
                    "official_name": official,
                })
                processed_side.add(icd)

    # -- Medications (name only) ------------------------------------------
    med_dict: Dict[str, str] = {}
    if not med_df.empty:
        name_col = "mcc0.display_list"
        dosage_col = "md0.text_list"
        processed_names: set = set()
        if name_col in med_df.columns and dosage_col in med_df.columns:
            for _, row in med_df.iterrows():
                raw = row.get(name_col)
                if not isinstance(raw, str):
                    continue
                try:
                    nl = ast.literal_eval(raw)
                except Exception:
                    continue
                name = None
                for n in nl:
                    if n and isinstance(n, str) and "mg" in n:
                        name = n
                        break
                if name is None and nl and isinstance(nl[0], str):
                    name = nl[0]
                if not name or name in processed_names:
                    continue
                processed_names.add(name)
                # Read dosage but only store name for labels
                dosage_raw = row.get(dosage_col, "")
                dosage = ""
                if isinstance(dosage_raw, str):
                    try:
                        dl = ast.literal_eval(dosage_raw)
                        if dl and isinstance(dl[0], str):
                            dosage = dl[0]
                    except Exception:
                        pass
                med_dict[name] = dosage

    # -- Lab values (abnormal only) ----------------------------------------
    lab_dict: Dict[str, str] = {}
    if not lab_df.empty:
        disp_col = "occ0.display"
        val_col = "ov0_value"
        ref_lo_col = "orl0_value"
        ref_hi_col = "orh0_value"
        if disp_col in lab_df.columns and val_col in lab_df.columns:
            for _, row in lab_df.iterrows():
                name = row.get(disp_col)
                val = row.get(val_col)
                if pd.isna(name) or pd.isna(val):
                    continue
                ref_lo = row.get(ref_lo_col)
                ref_hi = row.get(ref_hi_col)
                if pd.notna(ref_lo) and pd.notna(ref_hi):
                    try:
                        fval = float(val)
                        if fval < float(ref_lo):
                            lab_dict[str(name)] = f"{val} (niedrig)"
                        elif fval > float(ref_hi):
                            lab_dict[str(name)] = f"{val} (hoch)"
                    except ValueError:
                        pass

    # -- Body measurements -------------------------------------------------
    bw_val = None
    if not bw_df.empty and "ov0_value" in bw_df.columns:
        valid = bw_df["ov0_value"].dropna()
        if not valid.empty:
            bw_val = str(valid.iloc[0])

    bh_val = None
    if not bh_df.empty and "ov0_value" in bh_df.columns:
        valid = bh_df["ov0_value"].dropna()
        if not valid.empty:
            bh_val = str(valid.iloc[0])

    # -- Procedures --------------------------------------------------------
    proc_list: List[Tuple[str, str]] = []
    if not proc_df.empty:
        name_col = "pcc0_display"
        code_col = "pcc0_code"
        if name_col in proc_df.columns:
            names = proc_df[name_col].dropna().astype(str).tolist()
            codes = (
                proc_df[code_col].dropna().astype(str).tolist()
                if code_col in proc_df.columns
                else [""] * len(names)
            )
            proc_list = list(zip(names, codes))

    return {
        "side_list": side_list,
        "med_dict": med_dict,
        "lab_dict": lab_dict,
        "body_weight": bw_val,
        "body_height": bh_val,
        "procedures": proc_list,
    }


def _build_labels(
    limited_side: List[Dict],
    limited_meds: List[Tuple[str, str]],
    limited_labs: List[Tuple[str, str]],
    limited_procs: List[Tuple[str, str]],
    body_weight: Optional[str],
    body_height: Optional[str],
) -> Dict[str, List]:
    """Construct the labels dict shared by Anamnese and Epikrise."""
    side_labels = [
        {
            "type": "side_diagnosis",
            "official_name": e["official_name"],
            "icd10gm_code": e["icd_code"],
        }
        for e in limited_side
    ]
    med_labels = [{"name": n, "dosage": d} for n, d in limited_meds]
    lab_labels = [{"name": n, "value": v} for n, v in limited_labs]
    proc_labels = [{"procedure_name": p, "ops_code": c} for p, c in limited_procs]

    body_labels: List[Dict] = []
    body_entry: Dict[str, str] = {}
    if body_weight:
        body_entry["body_weight"] = f"{body_weight} kg"
    if body_height:
        body_entry["body_height"] = f"{body_height} cm"
    if len(body_entry) > 1:
        body_labels.append(body_entry)

    return {
        "diagnoses": side_labels,
        "medications": med_labels,
        "lab_values": lab_labels,
        "procedures": proc_labels,
        "body_values": body_labels,
    }


# =========================================================================
# Anamnese generator
# =========================================================================

async def generate_anamnese(
    data: Dict[str, Any],
    conditions_to_not_use: Optional[List[str]],
    jargon_variant: int,
    llm_client: AsyncOpenAI,
    llm_model: str = None,
    examples_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate the Anamnese section via LLM.

    The prompt instructs the model to weave structured clinical data
    into a coherent German free-text paragraph, using three real
    Anamnese examples for style guidance.

    Args:
        data:                 Allocated input data dict.
        conditions_to_not_use: ICD codes to exclude (reserved).
        jargon_variant:       Index 0-2 for jargon synonym selection.
        llm_client:           Pre-configured ``AsyncOpenAI`` client.
        llm_model:            Model name override (defaults to config).
        examples_path:        Path to anamnese/epikrise pickle.

    Returns:
        ``{"text": str, "labels": dict}``
    """
    if llm_model is None:
        llm_model = config.LLM_MODEL_SECONDARY

    # -- Load examples ----------------------------------------------------
    anamnese_examples, _ = _load_examples(examples_path)
    ex1 = random.choice(anamnese_examples) if anamnese_examples else ""
    ex2 = random.choice(anamnese_examples) if anamnese_examples else ""
    ex3 = random.choice(anamnese_examples) if anamnese_examples else ""

    # -- Extract structured data ------------------------------------------
    extracted = _extract_llm_section_data(data, jargon_variant, conditions_to_not_use)
    limited_side = extracted["side_list"][:6]
    limited_meds = list(extracted["med_dict"].items())[:3]
    limited_labs = list(extracted["lab_dict"].items())[:3]
    limited_procs = extracted["procedures"][:3]

    # -- Build structured data block --------------------------------------
    parts: List[str] = []
    if limited_side:
        confirmed = [f"- {d['picked_jargon']}" for d in limited_side[:4]]
        negated = [f"- Kein Hinweis auf: {d['picked_jargon']}" for d in limited_side[4:]]
        if confirmed:
            parts.append("Bekannte Nebendiagnosen:\n" + "\n".join(confirmed))
        if negated:
            parts.append("Ausgeschlossene Diagnosen:\n" + "\n".join(negated))

    if limited_meds:
        confirmed_meds = []
        negated_meds = []
        negate_last = len(limited_meds) == 3
        for i, (name, _) in enumerate(limited_meds):
            if negate_last and i == len(limited_meds) - 1:
                negated_meds.append(f"- Nicht (mehr) verordnet: {name}")
            else:
                confirmed_meds.append(f"- {name}")
        if confirmed_meds:
            parts.append("Aktuelle Medikation:\n" + "\n".join(confirmed_meds))
        if negated_meds:
            parts.append("Nicht verordnete Medikation:\n" + "\n".join(negated_meds))

    if limited_labs:
        formatted = "\n".join(
            f"- Name: {n}\n  Wert: {v}" for n, v in limited_labs
        )
        parts.append(f"Relevante Laborwerte:\n{formatted}")

    body_lines: List[str] = []
    if extracted["body_weight"]:
        body_lines.append(f"- Koerpergewicht: {extracted['body_weight']} kg")
    if extracted["body_height"]:
        body_lines.append(f"- Koerperhoehe: {extracted['body_height']} cm")
    if body_lines:
        parts.append("Messwerte:\n" + "\n".join(body_lines))

    if limited_procs:
        formatted = "\n".join(f"- {p[0]}" for p in limited_procs)
        parts.append(f"Prozeduren:\n{formatted}")

    data_heading = "Strukturierte Daten:\n" if parts else ""
    data_block = "\n\n".join(parts)

    # -- Build prompt (German, preserved verbatim) ------------------------
    prompt = f"""
        Generiere einen kohaerenten und medizinisch relevanten Abschnitt 'Anamnese' fuer einen Patienten. Nutze dafuer die bereitgestellten strukturierten Daten und Beispiele realer Anamnesen.

        Verwende die folgenden strukturierten Daten des Patienten. Integriere diese Informationen natuerlich in den Bericht. Achte darauf, Informationen ueber bekannte und ausgeschlossene Diagnosen sowie aktuelle und nicht verordnete Medikation (nur Namen) korrekt zu verarbeiten.

        {data_heading}{data_block}

        Hier sind drei Beispiele realer Anamnesen. Nutze diese Beispiele, um den typischen Stil, die Formulierungen und die Art und Weise zu verstehen, wie medizinisch relevante Routineinformationen oder "Fuelltexte" eingebunden werden, die helfen, eine Geschichte ueber die Anamnese und den aktuellen Zustand des Patienten zu erzaehlen. Uebernimm aehnliche narrative Elemente, waehrend du die oben genannten strukturierten Daten integrierst.

        Beispiel 1:
        ---
        {ex1}
        ---

        Beispiel 2:
        ---
        {ex2}
        ---

        Beispiel 3:
        ---
        {ex3}
        ---

        Erstelle den Abschnitt Anamnese. Dieser sollte fluessig lesbar sein, die bereitgestellten strukturierten Informationen natuerlich integrieren und narrative Elemente aehnlich den Beispielen enthalten.

        Gib die Ausgabe im JSON-Format zurueck. Das JSON-Objekt MUSS folgende Schluessel enthalten:
        1.  `anamnese_section`: Der vollstaendige Text des Anamnese-Abschnitts.

        Beispiel fuer das JSON-Format:
        {{
        "anamnese_section": "Patient wurde mit Diabetes mellitus aufgenommen. Metformin wurde verordnet. Das Koerpergewicht betraegt 80 kg. Die Koerperhoehe betraegt 175 cm. Labor: HbA1c 7.2%. Es wurde eine 3D-CT-Tomographie durchgefuehrt.",
        }}

        Achte darauf, dass die Indizes korrekt und konsistent mit dem generierten Text sind. Gib KEINE weiteren Erklaerungen oder Kommentare aus, sondern nur das JSON-Objekt.
        """

    # -- Call LLM ---------------------------------------------------------
    completion = await llm_client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.8,
    )
    raw = completion.choices[0].message.content

    # -- Parse response ---------------------------------------------------
    json_str = _extract_json_string_by_brace_balance(raw)
    parsed = json.loads(json_str)
    text = parsed.get("anamnese_section", "")

    # -- Build labels -----------------------------------------------------
    labels = _build_labels(
        limited_side,
        limited_meds,
        limited_labs,
        limited_procs,
        extracted["body_weight"],
        extracted["body_height"],
    )

    return {"text": text, "labels": labels}


# =========================================================================
# Epikrise generator
# =========================================================================

async def generate_epikrise(
    data: Dict[str, Any],
    conditions_to_not_use: Optional[List[str]],
    jargon_variant: int,
    llm_client: AsyncOpenAI,
    llm_model: str = None,
    examples_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate the Epikrise (discharge summary) section via LLM.

    Structurally identical to ``generate_anamnese`` but uses Epikrise
    examples and a discharge-oriented prompt.  Unlike Anamnese, no
    diagnoses or medications are negated.

    Returns:
        ``{"text": str, "labels": dict}``
    """
    if llm_model is None:
        llm_model = config.LLM_MODEL_SECONDARY

    # -- Load examples ----------------------------------------------------
    _, epikrise_examples = _load_examples(examples_path)
    ex1 = random.choice(epikrise_examples) if epikrise_examples else ""
    ex2 = random.choice(epikrise_examples) if epikrise_examples else ""
    ex3 = random.choice(epikrise_examples) if epikrise_examples else ""

    # -- Extract structured data ------------------------------------------
    extracted = _extract_llm_section_data(data, jargon_variant, conditions_to_not_use)
    limited_side = extracted["side_list"][:6]
    limited_meds = list(extracted["med_dict"].items())[:3]
    limited_labs = list(extracted["lab_dict"].items())[:3]
    limited_procs = extracted["procedures"][:3]

    # -- Build structured data block (NO negation for Epikrise) -----------
    parts: List[str] = []
    if limited_side:
        formatted = "\n".join(f"- {d['picked_jargon']}" for d in limited_side)
        parts.append(f"Relevante Nebendiagnosen:\n{formatted}")

    if limited_meds:
        formatted = "\n".join(f"- {name}" for name, _ in limited_meds)
        parts.append(f"Empfohlene Entlassmedikation:\n{formatted}")

    if limited_labs:
        formatted = "\n".join(
            f"- Name: {n}\n  Wert: {v}" for n, v in limited_labs
        )
        parts.append(f"Relevante Laborwerte bei Entlassung:\n{formatted}")

    body_lines: List[str] = []
    if extracted["body_weight"]:
        body_lines.append(f"- Koerpergewicht: {extracted['body_weight']} kg")
    if extracted["body_height"]:
        body_lines.append(f"- Koerperhoehe: {extracted['body_height']} cm")
    if body_lines:
        parts.append("Messwerte bei Entlassung:\n" + "\n".join(body_lines))

    if limited_procs:
        formatted = "\n".join(f"- {p[0]}" for p in limited_procs)
        parts.append(f"Durchgefuehrte relevante Prozeduren:\n{formatted}")

    data_heading = "Strukturierte Daten fuer Epikrise:\n" if parts else ""
    data_block = "\n\n".join(parts)

    # -- Build prompt (German, preserved verbatim) ------------------------
    prompt = f"""
        Generiere einen kohaerenten und medizinisch relevanten Abschnitt 'Epikrise' (Entlassungsbericht) fuer einen Patienten. Nutze dafuer die bereitgestellten strukturierten Daten und Beispiele realer Epikrisen.

        Verwende die folgenden strukturierten Daten des Patienten. Integriere diese Informationen natuerlich in den Bericht.

        {data_heading}{data_block}

        Hier sind drei Beispiele realer Epikrisen. Nutze diese Beispiele, um den typischen Stil, die Zusammenfassungen und die abschliessenden Bemerkungen zu verstehen, die in einer Epikrise enthalten sind.

        Beispiel 1:
        ---
        {ex1}
        ---

        Beispiel 2:
        ---
        {ex2}
        ---

        Beispiel 3:
        ---
        {ex3}
        ---

        Generiere die Ausgabe im JSON-Format. Das JSON-Objekt MUSS folgende Schluessel enthalten:
        1.  `epikrise_section`: Der vollstaendige Text des Epikrise-Abschnitts.

        Gib nur das JSON-Objekt zurueck, ohne weitere Erklaerungen oder Kommentare.
        """

    # -- Call LLM ---------------------------------------------------------
    completion = await llm_client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.8,
    )
    raw = completion.choices[0].message.content

    # -- Parse response ---------------------------------------------------
    json_str = _extract_json_string_by_brace_balance(raw)
    parsed = json.loads(json_str)
    text = parsed.get("epikrise_section", "")

    # -- Build labels -----------------------------------------------------
    labels = _build_labels(
        limited_side,
        limited_meds,
        limited_labs,
        limited_procs,
        extracted["body_weight"],
        extracted["body_height"],
    )

    return {"text": text, "labels": labels}
