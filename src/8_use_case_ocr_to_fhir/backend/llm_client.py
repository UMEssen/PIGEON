"""
Stage 8: LLM client for medical information extraction.

This module provides ``LLMClient``, a thin wrapper around the OpenAI-
compatible chat/completions API exposed by vLLM.  It talks to two
different models:

- **PIGEON model** (``LLM_ENDPOINT_INFERENCE``): the fine-tuned medical
  extraction model that converts raw clinical text into the structured
  JSON schema defined in ``prompts.py``.
- **Strong LLM** (``LLM_ENDPOINT_SECONDARY``): a large general-purpose
  model (e.g. Qwen3-235B) used for document classification and summary
  generation -- tasks that benefit from broad world knowledge rather
  than domain-specific fine-tuning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (
    LLM_ENDPOINT_SECONDARY,
    LLM_ENDPOINT_INFERENCE,
    LLM_MODEL_SECONDARY,
    LLM_MODEL_INFERENCE,
)

import ast
import json
import logging
import re
from typing import Any, Dict, Optional

from openai import OpenAI

from prompts import (
    ARZTBRIEF_EXTRACTION_PROMPT,
    GERMAN_DOCUMENT_CLASSIFICATION_PROMPT,
    GERMAN_SUMMARY_EXTRACTION_PROMPT,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for the two LLM endpoints used by the pipeline.

    Parameters
    ----------
    inference_endpoint : str
        Base URL of the PIGEON inference model (fine-tuned).
    inference_model : str
        Model name at the inference endpoint.
    secondary_endpoint : str
        Base URL of the strong general-purpose LLM.
    secondary_model : str
        Model name at the secondary endpoint.
    """

    def __init__(
        self,
        inference_endpoint: str = LLM_ENDPOINT_INFERENCE,
        inference_model: str = LLM_MODEL_INFERENCE,
        secondary_endpoint: str = LLM_ENDPOINT_SECONDARY,
        secondary_model: str = LLM_MODEL_SECONDARY,
    ):
        self.inference_endpoint = inference_endpoint
        self.inference_model = inference_model
        self.secondary_endpoint = secondary_endpoint
        self.secondary_model = secondary_model

        # OpenAI-compatible clients -- vLLM does not require a real API key
        self.inference_client = OpenAI(
            base_url=inference_endpoint,
            api_key="EMPTY",
        )
        self.secondary_client = OpenAI(
            base_url=secondary_endpoint,
            api_key="EMPTY",
        )

    # ------------------------------------------------------------------
    # 1. Medical extraction (PIGEON model)
    # ------------------------------------------------------------------

    def extract_medical_info_german(self, text: str) -> Dict[str, Any]:
        """Run the PIGEON model to extract structured medical data.

        Parameters
        ----------
        text : str
            Plain German clinical text (e.g. from a discharge letter).

        Returns
        -------
        dict
            The structured extraction result.  On failure an empty dict
            is returned.
        """
        prompt = ARZTBRIEF_EXTRACTION_PROMPT.format(text=text)

        try:
            response = self.inference_client.chat.completions.create(
                model=self.inference_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=0.3,
                top_p=0.95,
            )

            raw_content = response.choices[0].message.content.strip()
            logger.debug("PIGEON raw response length: %d", len(raw_content))

            # Parse and sanitise the JSON payload
            data = self._extract_json_payload(raw_content)
            if data:
                return self._sanitize_extraction(data)

            logger.warning("Could not parse PIGEON response as JSON")
            return {}

        except Exception as exc:
            logger.error("PIGEON extraction failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # 2. Document classification (strong LLM)
    # ------------------------------------------------------------------

    def classify_document(self, text: str) -> Dict[str, Any]:
        """Classify a clinical document using the strong LLM.

        Parameters
        ----------
        text : str
            First ~2000 characters of the document text (enough for
            classification without sending the entire document).

        Returns
        -------
        dict
            Keys: ``category``, ``confidence``, ``reasoning``.
        """
        # Truncate to keep the prompt manageable
        truncated = text[:2000]
        prompt = GERMAN_DOCUMENT_CLASSIFICATION_PROMPT.format(text=truncated)

        try:
            response = self.secondary_client.chat.completions.create(
                model=self.secondary_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.1,
                top_p=0.95,
            )

            raw = response.choices[0].message.content.strip()
            result = self._extract_json_payload(raw)
            if result:
                return result

            # Fallback: if the model returned a plain category name
            return {
                "category": raw.split("\n")[0].strip(),
                "confidence": 0.5,
                "reasoning": "Could not parse structured response",
            }

        except Exception as exc:
            logger.error("Document classification failed: %s", exc)
            return {
                "category": "Sonstiges",
                "confidence": 0.0,
                "reasoning": f"Error: {exc}",
            }

    # ------------------------------------------------------------------
    # 3. Summary generation (strong LLM)
    # ------------------------------------------------------------------

    def call_strong_llm_summary(
        self,
        extraction: Dict[str, Any],
        category_context: str = "",
    ) -> str:
        """Generate a natural-language summary of the extraction.

        Parameters
        ----------
        extraction : dict
            The structured extraction result from PIGEON.
        category_context : str
            The document category (e.g. "Arztbrief") for context.

        Returns
        -------
        str
            German-language clinical summary text.
        """
        extraction_json = json.dumps(extraction, ensure_ascii=False, indent=2)
        prompt = GERMAN_SUMMARY_EXTRACTION_PROMPT.format(
            extraction_json=extraction_json,
            category=category_context or "Unbekannt",
        )

        try:
            response = self.secondary_client.chat.completions.create(
                model=self.secondary_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.3,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            logger.error("Summary generation failed: %s", exc)
            return f"Zusammenfassung konnte nicht erstellt werden: {exc}"

    # ------------------------------------------------------------------
    # JSON parsing and sanitisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_payload(content: str) -> Optional[Dict[str, Any]]:
        """Robustly extract a JSON object from LLM output.

        The model may wrap its JSON in markdown code fences, prepend
        explanatory text, or include trailing commentary.  This method
        handles all those cases.
        """
        if not content:
            return None

        # Strategy 1: strip markdown code fences
        cleaned = content.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
            if "```" in cleaned:
                cleaned = cleaned.split("```", 1)[0]

        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1]

        # Strategy 2: find the outermost { ... } pair
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = cleaned[start : end + 1]

            # Try json.loads first
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

            # Try ast.literal_eval as a fallback (handles single quotes)
            try:
                result = ast.literal_eval(json_str)
                if isinstance(result, dict):
                    return result
            except (ValueError, SyntaxError):
                pass

            # Strategy 3: aggressive cleanup -- remove trailing commas,
            # fix common LLM JSON mistakes
            try:
                fixed = re.sub(r",\s*([}\]])", r"\1", json_str)  # trailing commas
                fixed = fixed.replace("'", '"')  # single -> double quotes
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _sanitize_extraction(data: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce the raw extraction into the expected schema.

        The model occasionally omits keys or uses unexpected types.
        This method ensures all top-level keys exist and list fields
        are actually lists.

        Returns the cleaned dict (mutated in-place for performance).
        """
        # Ensure expected top-level keys exist
        defaults = {
            "patient": {},
            "encounter": {},
            "diagnoses": [],
            "procedures": [],
            "medications": [],
            "vital_signs": [],
            "laboratory_results": [],
            "tumor_markers": [],
            "free_text": {},
        }

        for key, default in defaults.items():
            if key not in data:
                data[key] = default
            # Coerce None to the expected type
            elif data[key] is None:
                data[key] = default

        # Ensure list fields are lists (model sometimes returns a single dict)
        list_keys = [
            "diagnoses",
            "procedures",
            "medications",
            "vital_signs",
            "laboratory_results",
            "tumor_markers",
        ]
        for key in list_keys:
            val = data.get(key)
            if isinstance(val, dict):
                data[key] = [val]
            elif not isinstance(val, list):
                data[key] = []

        # Ensure patient and encounter are dicts
        if not isinstance(data.get("patient"), dict):
            data["patient"] = {}
        if not isinstance(data.get("encounter"), dict):
            data["encounter"] = {}

        return data
