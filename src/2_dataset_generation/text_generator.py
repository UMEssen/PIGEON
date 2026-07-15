"""
Text Generator orchestrator -- Stage 2 of the PIGEON pipeline.

This module is the cleaned-up ``TextGenerator`` class that replaces the
3,686-line monolith.  It delegates data loading to ``FHIRDataLoader``,
rule-based sections to ``section_generators``, LLM sections to
``llm_sections``, and tumour data to ``tumor_extraction``.

Orchestration flow:
    1. Load raw FHIR data for the entity (encounter or patient).
    2. Use the ``recipe_dict`` to allocate data rows across sections,
       ensuring no data item is used by two sections ("double-use").
    3. Run synchronous generators sequentially.
    4. Run async (LLM) generators concurrently via ``asyncio.gather``.
    5. Assemble the final text from section outputs.
    6. Post-process labels: merge anamnese/epikrise into "free_text",
       rename diagnosen to "diagnoses".
    7. Return ``{"text": ..., "labels": ..., "synthetic_text": ...}``.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Import central config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Sibling modules
# ---------------------------------------------------------------------------
from fhir_data_loader import FHIRDataLoader
import section_generators as sg
import llm_sections
import tumor_extraction
from llm_sections import _extract_json_string_by_brace_balance


# =========================================================================
# Priority ordering for data allocation
# =========================================================================

_ALLOCATION_PRIORITY = [
    "introduction",
    "diagnosen",
    "hauptdiagnose",
    "nebendiagnose",
    "medication",
    "lab_values",
    "vitalzeichen",
    "histologie",
    "anamnese",
    "epikrise",
]

# Sections whose generators are async (LLM-based)
_ASYNC_SECTIONS = {"anamnese", "epikrise"}


class TextGenerator:
    """Orchestrates synthetic medical text generation from FHIR data.

    Usage::

        gen = TextGenerator(base="patient")
        result = await gen.generate_arztbrief_patient(
            patient_id="12345",
            sections=["introduction", "hauptdiagnose", "anamnese", "epikrise"],
            jargon_variant=1,
            recipe_dict={...},
        )
    """

    def __init__(
        self,
        base: str,
        llm_endpoint_primary: Optional[str] = None,
        llm_endpoint_secondary: Optional[str] = None,
    ) -> None:
        self.base = base
        self.loader = FHIRDataLoader(base=base)

        primary_url = llm_endpoint_primary or config.LLM_ENDPOINT_PRIMARY
        secondary_url = llm_endpoint_secondary or config.LLM_ENDPOINT_SECONDARY

        self.client_primary = AsyncOpenAI(base_url=primary_url, api_key="dummy")
        self.client_secondary = AsyncOpenAI(base_url=secondary_url, api_key="dummy")

    # ------------------------------------------------------------------
    # High-level generators
    # ------------------------------------------------------------------

    async def generate_arztbrief(
        self,
        encounter_id: str,
        sections: List[str],
        jargon_variant: int,
        recipe_dict: Dict[str, int],
        real_ab: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate an encounter-level medical report."""
        raw = self.loader.load_data(encounter_id)
        if not raw or not raw.get("data"):
            raise ValueError(f"No data found for encounter {encounter_id}")
        return await self._assemble(
            raw, sections, jargon_variant, recipe_dict, real_ab
        )

    async def generate_arztbrief_patient(
        self,
        patient_id: str,
        sections: List[str],
        jargon_variant: int,
        recipe_dict: Dict[str, int],
        real_ab: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a patient-level medical report."""
        raw = self.loader.load_data(patient_id)
        if not raw or not raw.get("data"):
            raise ValueError(f"No data found for patient {patient_id}")

        result = await self._assemble(
            raw, sections, jargon_variant, recipe_dict, real_ab,
            patient_id=patient_id,
        )
        return result

    async def generate_free_text(
        self,
        input_data: Dict[str, Any],
        jargon_variant: int,
    ) -> Dict[str, Any]:
        """Generate free-form medical text (uses primary LLM).

        Randomly selects a subset of rows from each data frame,
        then asks the LLM to produce a clinical note.
        """
        rng = np.random.default_rng()
        chosen = {"data": {}}
        for key, df in input_data.get("data", {}).items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                count = rng.integers(0, len(df) + 1)
                if count > 0:
                    chosen["data"][key] = df.sample(
                        n=count, random_state=None
                    ).reset_index(drop=True)
                else:
                    chosen["data"][key] = df.iloc[0:0]
            else:
                chosen["data"][key] = df

        # Build a simple prompt requesting JSON output
        data_block = self._build_free_text_data_block(chosen)
        task = random.choice([
            "ein kurzes Fragment einer Verlaufsdokumentation",
            "ein Abschnitt fuer eine Aufnahmenotiz",
            "ein Absatz fuer einen Entlassbrief",
            "die Begruendung fuer eine Konsilanforderung",
            "ein praegnantes Visiten-Update",
        ])
        style = random.choice([
            "sehr formell und detailliert, in vollstaendigen Saetzen",
            "im ueblichen klinischen Stil",
            "starker medizinischer Jargon und Stichpunktstil",
        ])

        prompt = (
            f"Stell dir vor, du bist medizinisches Fachpersonal in einem deutschen Krankenhaus.\n"
            f"Aufgabe: Verfasse {task}.\n"
            f"Stil: {style}.\n\n"
            f"Patientendaten:\n{data_block}\n\n"
            f"Gib die Ausgabe als JSON-Objekt mit dem Schluessel 'text' zurueck."
        )

        completion = await self.client_primary.chat.completions.create(
            model=config.LLM_MODEL_PRIMARY,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=6000,
            temperature=0.8,
            top_p=0.9,
        )
        raw_text = completion.choices[0].message.content
        json_str = _extract_json_string_by_brace_balance(raw_text)
        parsed = json.loads(json_str)

        return {
            "text": parsed.get("text", ""),
            "prompt": prompt,
            "labels": {"free_text": self._collect_free_text_labels(chosen)},
        }

    # ------------------------------------------------------------------
    # Internal orchestration
    # ------------------------------------------------------------------

    async def _assemble(
        self,
        raw_data: Dict[str, Any],
        sections: List[str],
        jargon_variant: int,
        recipe_dict: Dict[str, int],
        real_ab: Optional[str],
        patient_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Core assembly: allocate data, run generators, combine results."""

        raw_frames = raw_data["data"]

        # -- Validate recipe against sections -----------------------------
        sections_set = set(sections)
        for key, count in recipe_dict.items():
            if count > 0 and "_in_" in key:
                parts = key.split("_in_")
                if len(parts) > 1:
                    target = parts[-1]
                    if target.endswith("_count"):
                        target = target[: -len("_count")]
                    target = target.replace("_section", "")
                    if target and target not in sections_set:
                        raise ValueError(
                            f"Recipe key '{key}' targets section '{target}' "
                            f"which is not in sections_to_include: {sections}"
                        )

        # -- Initialize data pools ----------------------------------------
        available: Dict[str, Any] = {}
        used_idx: Dict[str, set] = {}
        for k, v in raw_frames.items():
            if isinstance(v, pd.DataFrame):
                available[k] = v.copy().reset_index(drop=True)
                used_idx[k] = set()
            else:
                available[k] = v

        # -- Allocate data and schedule calls -----------------------------
        sync_calls: List[Dict] = []
        async_tasks: Dict[str, Any] = {}
        results: Dict[str, Any] = {}

        for section in _ALLOCATION_PRIORITY:
            if section not in sections:
                continue

            sec_data = {"data": {}, "count": {}}
            chance_for_date = 0.4

            # Allocation logic per section
            if section == "introduction":
                sec_data["data"]["patient"] = available.get("patient", pd.DataFrame())
                sec_data["data"]["encounter"] = available.get("encounter", pd.DataFrame())

            elif section == "diagnosen":
                count = recipe_dict.get("diagnosis_in_diagnosen_count", 0)
                self._allocate_main_side(available, used_idx, sec_data, count)

            elif section == "hauptdiagnose":
                count = recipe_dict.get("main_diagnosis_in_hauptdiagnose_count",
                                        len(available.get("main_diagnosis", pd.DataFrame())))
                sec_data["data"]["main_diagnosis"] = self._take(
                    available, used_idx, "main_diagnosis", count
                )

            elif section == "nebendiagnose":
                sec_data["data"]["main_diagnosis"] = available.get("main_diagnosis", pd.DataFrame())
                count = recipe_dict.get("side_diagnosis_in_nebendiagnose_count",
                                        len(available.get("side_diagnosis", pd.DataFrame())))
                sec_data["data"]["side_diagnosis"] = self._take(
                    available, used_idx, "side_diagnosis", count
                )

            elif section == "medication":
                count = recipe_dict.get("medication_in_medication_count",
                                        len(available.get("medication", pd.DataFrame())))
                sec_data["data"]["medication"] = self._take(
                    available, used_idx, "medication", count
                )

            elif section == "lab_values":
                count = recipe_dict.get("lab_values_in_lab_values_section_count",
                                        len(available.get("Laboratory", pd.DataFrame())))
                sec_data["data"]["Laboratory"] = self._take(
                    available, used_idx, "Laboratory", count
                )

            elif section == "vitalzeichen":
                for vt in ("BloodPressure", "BodyTemperature", "OxygenSaturation", "HeartBeat"):
                    sec_data["data"][vt] = available.get(vt, pd.DataFrame())

            elif section in ("anamnese", "epikrise"):
                self._allocate_llm_section(
                    available, used_idx, sec_data, recipe_dict, section
                )

            # -- Prepare generator call -----------------------------------
            func = None
            args = [sec_data]
            kwargs: Dict[str, Any] = {}

            if section == "introduction":
                func = sg.generate_introduction
                kwargs["include_address"] = random.random() < 0.5
            elif section == "diagnosen":
                func = sg.generate_diagnose
                kwargs.update(chance_for_date=chance_for_date, jargon_variant=jargon_variant, patient_id=patient_id or "")
            elif section == "hauptdiagnose":
                func = sg.generate_hauptdiagnose
                kwargs.update(chance_for_date=chance_for_date, jargon_variant=jargon_variant, patient_id=patient_id or "")
            elif section == "nebendiagnose":
                func = sg.generate_nebendiagnose
                kwargs.update(chance_for_date=chance_for_date, jargon_variant=jargon_variant)
            elif section == "medication":
                func = sg.generate_medication_section
            elif section == "lab_values":
                func = sg.generate_lab_values
                kwargs["max_values"] = recipe_dict.get("lab_values_in_lab_values_section_count", 10)
            elif section == "vitalzeichen":
                func = sg.generate_vitalzeichen
            elif section == "anamnese":
                func = llm_sections.generate_anamnese
                kwargs.update(
                    conditions_to_not_use=[],
                    jargon_variant=jargon_variant,
                    llm_client=self.client_secondary,
                )
            elif section == "epikrise":
                func = llm_sections.generate_epikrise
                kwargs.update(
                    conditions_to_not_use=[],
                    jargon_variant=jargon_variant,
                    llm_client=self.client_secondary,
                )

            if func is None:
                continue

            if section in _ASYNC_SECTIONS:
                async_tasks[section] = func(*args, **kwargs)
            else:
                sync_calls.append({
                    "name": section, "func": func,
                    "args": args, "kwargs": kwargs,
                })

        # -- Execute sync tasks -------------------------------------------
        for call in sync_calls:
            ret = call["func"](*call["args"], **call["kwargs"])
            if ret:
                results[call["name"]] = ret

        # -- Execute async tasks concurrently -----------------------------
        if async_tasks:
            names = list(async_tasks.keys())
            outs = await asyncio.gather(*(async_tasks[n] for n in names))
            for i, n in enumerate(names):
                results[n] = outs[i]

        # -- Assemble full_text and combined_labels -----------------------
        full_text = ""
        combined_labels: Dict[str, Any] = {}
        postprocessed: List[Dict] = []
        chosen_headers: List[str] = []

        # Optionally generate tumour section for patient-level reports
        tumor_section = None
        if patient_id:
            try:
                tumor_section = tumor_extraction.generate_tumor_informations(patient_id)
            except Exception:
                pass

        for sec_name in sections:
            if sec_name not in results:
                continue

            result = results[sec_name]
            sec_text = result.get("text", "")
            sec_labels = result.get("labels", {})

            if sec_name in ("medication", "lab_values"):
                postprocessed.append(result)
            else:
                header = ""
                if sec_name == "anamnese":
                    header = random.choice(
                        ["Anamnese", "Aufnahmebefund / Verlauf", "Verlauf", "Vorgeschichte"]
                    )
                elif sec_name == "epikrise":
                    header = random.choice(
                        ["Epikrise", "Epikritische Zusammenfassung", "Zusammenfassung"]
                    )

                if header:
                    chosen_headers.append(header)

                sep = "\n\n" if full_text else ""
                if sec_text:
                    if sec_name in ("anamnese", "epikrise"):
                        content = f"{header}\n{sec_text}"
                    else:
                        content = sec_text
                    full_text += sep + content

            combined_labels[sec_name] = sec_labels

            # Inject tumour section after diagnoses
            if tumor_section and sec_name in ("nebendiagnose", "diagnosen"):
                tsep = "\n\n" if full_text else ""
                full_text += tsep + tumor_section["text"]
                combined_labels["tumor_informations"] = tumor_section["labels"]

        # Append medication/lab sections
        for pp in postprocessed:
            full_text += "\n\n" + pp.get("text", "")

        # -- Post-process labels ------------------------------------------
        combined_labels = self._postprocess_labels(combined_labels)

        return {
            "text": None,
            "labels": combined_labels,
            "synthetic_text": full_text,
            "real_text": real_ab,
        }

    # ------------------------------------------------------------------
    # Data allocation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _take(
        available: Dict[str, pd.DataFrame],
        used: Dict[str, set],
        key: str,
        count: int,
    ) -> pd.DataFrame:
        """Take up to *count* unused rows from *available[key]*."""
        df = available.get(key, pd.DataFrame())
        if df.empty:
            return pd.DataFrame()
        unused = df[~df.index.isin(used.setdefault(key, set()))]
        allocated = unused.head(count)
        if not allocated.empty:
            used[key].update(allocated.index)
        return allocated

    def _allocate_main_side(self, available, used, sec_data, total_count):
        """Allocate main diagnoses first, then fill remainder with side."""
        allocated_main = self._take(available, used, "main_diagnosis", total_count)
        sec_data["data"]["main_diagnosis"] = allocated_main
        remaining = total_count - len(allocated_main)
        if remaining > 0:
            sec_data["data"]["side_diagnosis"] = self._take(
                available, used, "side_diagnosis", remaining
            )
        else:
            sec_data["data"]["side_diagnosis"] = pd.DataFrame()

    def _allocate_llm_section(self, available, used, sec_data, recipe, section):
        """Allocate data for an LLM section (anamnese or epikrise)."""
        prefix = section  # "anamnese" or "epikrise"
        sec_data["data"]["side_diagnosis"] = self._take(
            available, used, "side_diagnosis",
            recipe.get(f"side_diagnosis_in_{prefix}_count", 0),
        )
        sec_data["data"]["medication"] = self._take(
            available, used, "medication",
            recipe.get(f"medication_in_{prefix}_count", 0),
        )
        sec_data["data"]["Laboratory"] = self._take(
            available, used, "Laboratory",
            recipe.get(f"lab_values_in_{prefix}_count", 0),
        )
        sec_data["data"]["procedure"] = self._take(
            available, used, "procedure",
            recipe.get(f"procedures_in_{prefix}_count", 0),
        )
        if recipe.get(f"body_weight_in_{prefix}_count", 0) > 0:
            sec_data["data"]["BodyWeight"] = self._take(
                available, used, "BodyWeight", 1
            )
        if recipe.get(f"body_height_in_{prefix}_count", 0) > 0:
            sec_data["data"]["BodyHeight"] = self._take(
                available, used, "BodyHeight", 1
            )

    # ------------------------------------------------------------------
    # Label post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess_labels(labels: Dict[str, Any]) -> Dict[str, Any]:
        """Merge anamnese+epikrise into 'free_text', rename diagnosen -> diagnoses.

        WHY merge?
            Downstream training expects a single 'free_text' label key
            containing all narrative-section labels, deduplicated across
            the two sections.

        WHY rename?
            The German key 'diagnosen' is normalised to 'diagnoses' for
            consistency with the English-centric evaluation pipeline.
        """
        # -- Merge anamnese & epikrise ------------------------------------
        ana = labels.pop("anamnese", {})
        epi = labels.pop("epikrise", {})
        all_keys = set(ana.keys()) | set(epi.keys())
        merged: Dict[str, list] = {}
        for k in all_keys:
            merged[k] = (ana.get(k, []) or []) + (epi.get(k, []) or [])
        labels["free_text"] = merged

        # -- Rename diagnosen -> diagnoses --------------------------------
        if "diagnosen" in labels:
            labels["diagnoses"] = labels.pop("diagnosen")
        else:
            labels["diagnoses"] = (
                labels.pop("hauptdiagnose", []) + labels.pop("nebendiagnose", [])
            )

        # -- Recursive None -> "" -----------------------------------------
        def _fix(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if v is None:
                        obj[k] = ""
                    elif isinstance(v, (dict, list)):
                        _fix(v)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        _fix(item)

        _fix(labels)
        return labels

    # ------------------------------------------------------------------
    # Free-text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_free_text_data_block(data: Dict) -> str:
        """Build a simple text summary of chosen data for the free-text prompt."""
        parts: List[str] = []
        df_dict = data.get("data", {})

        side = df_dict.get("side_diagnosis", pd.DataFrame())
        if not side.empty and "ccc0_display" in side.columns:
            items = side["ccc0_display"].dropna().astype(str).unique()[:5]
            if len(items):
                parts.append("Diagnosen:\n" + "\n".join(f"- {d}" for d in items))

        med = df_dict.get("medication", pd.DataFrame())
        if not med.empty and "mcc0.display_list" in med.columns:
            names = []
            for raw in med["mcc0.display_list"].dropna():
                try:
                    nl = eval(raw)
                    if nl:
                        names.append(nl[0])
                except Exception:
                    pass
            if names:
                parts.append("Medikation:\n" + "\n".join(f"- {n}" for n in names[:4]))

        lab = df_dict.get("Laboratory", pd.DataFrame())
        if not lab.empty:
            items = []
            for _, r in lab.head(4).iterrows():
                n = r.get("occ0.display")
                v = r.get("ov0_value")
                if pd.notna(n) and pd.notna(v):
                    items.append(f"- {n}: {v}")
            if items:
                parts.append("Labor:\n" + "\n".join(items))

        return "\n\n".join(parts) if parts else "(keine Daten)"

    @staticmethod
    def _collect_free_text_labels(data: Dict) -> Dict[str, list]:
        """Collect simple label lists from the chosen data subset."""
        labels: Dict[str, list] = {
            "side_diagnosis_labels": [],
            "medication_labels": [],
            "lab_value_labels": [],
            "procedure_labels": [],
            "body_value_labels": [],
            "vital_sign_labels": [],
        }
        df_dict = data.get("data", {})

        side = df_dict.get("side_diagnosis", pd.DataFrame())
        if not side.empty and "ccc0_display" in side.columns:
            labels["side_diagnosis_labels"] = (
                side["ccc0_display"].dropna().astype(str).unique()[:5].tolist()
            )

        med = df_dict.get("medication", pd.DataFrame())
        if not med.empty and "mcc0.display_list" in med.columns:
            for raw in med["mcc0.display_list"].dropna():
                try:
                    nl = eval(raw)
                    if nl:
                        labels["medication_labels"].append(nl[0])
                except Exception:
                    pass
            labels["medication_labels"] = labels["medication_labels"][:4]

        lab = df_dict.get("Laboratory", pd.DataFrame())
        if not lab.empty:
            for _, r in lab.head(4).iterrows():
                n = r.get("occ0.display")
                v = r.get("ov0_value")
                if pd.notna(n) and pd.notna(v):
                    labels["lab_value_labels"].append(f"{n}: {v}")

        return labels
