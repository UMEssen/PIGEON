"""
Stage 8: ICD-10-GM RAG code corrector for the Flying Pigeon pipeline.

This module provides a lightweight ICD-10 code corrector that reuses
the RAG infrastructure from Stage 6.  It is designed to be used as a
singleton within the FastAPI application -- the FAISS vector store is
loaded once at startup and shared across requests.

The correction workflow:
1. Receive a diagnosis name + the code predicted by PIGEON.
2. Retrieve the top-k semantically similar entries from the ICD-10-GM
   lookup table (FAISS vector store).
3. Ask the inference LLM to pick the best code from the candidates.
4. Return the corrected code (or the original if no better match).

This module also provides ``process_extraction_result()``, a convenience
function that walks the entire PIGEON extraction dict and corrects every
``icd10gm_code`` field in-place.
"""

import sys
from pathlib import Path

# Insert the project root so we can import config and Stage 6 modules
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))
from config import (
    ICD10_LOOKUP,
    VECTORSTORE_CACHE_ICD,
    LLM_ENDPOINT_INFERENCE,
    LLM_MODEL_INFERENCE,
)

# Also add Stage 6 to the path so we can reuse the corrector classes
_STAGE6_DIR = _PROJECT_ROOT / "src" / "6_rag_correction"
sys.path.insert(0, str(_STAGE6_DIR))

import copy
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton instance -- initialised on first use
# ---------------------------------------------------------------------------
_SINGLETON_INSTANCE: Optional["ICDCodeRAGCorrector"] = None


class ICDCodeRAGCorrector:
    """Lightweight ICD-10-GM code corrector using RAG.

    Wraps the Stage 6 ``ICDCodeRAGCorrector`` (or provides a standalone
    implementation when Stage 6 is not available) and exposes a simple
    ``correct_icd_code(name, code) -> code`` interface.

    Parameters
    ----------
    lookup_path : str or Path
        Path to the ICD-10-GM lookup CSV.  Defaults to ``config.ICD10_LOOKUP``.
    vectorstore_cache : str or Path
        Path to the cached FAISS index.  Defaults to ``config.VECTORSTORE_CACHE_ICD``.
    llm_endpoint : str
        vLLM base URL for the re-ranking LLM.
    llm_model : str
        Model name served at the endpoint.
    top_k : int
        Number of candidate codes to retrieve per query.
    """

    def __init__(
        self,
        lookup_path: str = str(ICD10_LOOKUP),
        vectorstore_cache: str = str(VECTORSTORE_CACHE_ICD),
        llm_endpoint: str = LLM_ENDPOINT_INFERENCE,
        llm_model: str = LLM_MODEL_INFERENCE,
        top_k: int = 10,
    ):
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.top_k = top_k
        self._corrector = None

        # Try to import the Stage 6 corrector for full RAG functionality.
        # If Stage 6 dependencies are not installed we fall back to a
        # simple regex-based validator.
        try:
            from icd_corrector import ICDCodeRAGCorrector as _Stage6ICD
            self._corrector = _Stage6ICD(
                lookup_csv_path=lookup_path,
                vectorstore_cache_dir=vectorstore_cache,
                llm_endpoint=llm_endpoint,
                llm_model=llm_model,
                top_k=top_k,
                use_official_names=False,
            )
            logger.info("ICD RAG corrector initialised (Stage 6 backend)")
        except Exception as exc:
            logger.warning(
                "Could not initialise Stage 6 ICD corrector (%s). "
                "RAG correction will be a no-op passthrough.",
                exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct_icd_code(
        self,
        diagnosis_name: str,
        original_code: str,
    ) -> str:
        """Correct a single ICD-10-GM code using RAG.

        Parameters
        ----------
        diagnosis_name : str
            Human-readable diagnosis name (German).
        original_code : str
            The code predicted by the PIGEON model.

        Returns
        -------
        str
            The corrected code.  Returns ``original_code`` unchanged
            when RAG cannot find a better alternative or the backend
            is not available.
        """
        if not diagnosis_name or not original_code:
            return original_code or ""

        if self._corrector is None:
            # No RAG backend -- passthrough
            return original_code

        try:
            corrected = self._corrector.correct_code(
                diagnosis_name, original_code
            )
            return corrected if corrected else original_code
        except Exception as exc:
            logger.warning(
                "RAG correction failed for '%s' / '%s': %s",
                diagnosis_name,
                original_code,
                exc,
            )
            return original_code

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_extraction_result(
        self,
        extraction_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Walk the PIGEON extraction and correct all ICD-10 codes.

        The method recursively finds every dict that has both a ``name``
        (or ``official_name``) key and an ``icd10gm_code`` key, and adds
        an ``icd10gm_code_rag`` field with the corrected value.

        A deep copy is made so the original dict is not mutated.

        Parameters
        ----------
        extraction_result : dict
            The full extraction dict from the PIGEON model.

        Returns
        -------
        dict
            A new dict with ``icd10gm_code_rag`` fields added wherever
            an ICD code was found.
        """
        data = copy.deepcopy(extraction_result)
        self._correct_icd_codes_recursive(data)
        return data

    def _correct_icd_codes_recursive(self, data: Any) -> None:
        """In-place recursive ICD code correction."""
        if isinstance(data, dict):
            # Check if this dict has an ICD code to correct
            if "icd10gm_code" in data and data["icd10gm_code"]:
                name = data.get("name") or data.get("official_name", "")
                original = data["icd10gm_code"]
                if name and original:
                    corrected = self.correct_icd_code(name, original)
                    data["icd10gm_code_rag"] = corrected

            # Recurse into nested values
            for value in data.values():
                if isinstance(value, (dict, list)):
                    self._correct_icd_codes_recursive(value)

        elif isinstance(data, list):
            for item in data:
                self._correct_icd_codes_recursive(item)


# ---------------------------------------------------------------------------
# Singleton accessor and convenience function
# ---------------------------------------------------------------------------

def get_corrector(**kwargs) -> ICDCodeRAGCorrector:
    """Return the singleton ``ICDCodeRAGCorrector`` instance.

    Creates the instance on first call.  All keyword arguments are
    forwarded to the constructor only on the first invocation.
    """
    global _SINGLETON_INSTANCE
    if _SINGLETON_INSTANCE is None:
        _SINGLETON_INSTANCE = ICDCodeRAGCorrector(**kwargs)
    return _SINGLETON_INSTANCE


def correct_icd_codes(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function: correct all ICD codes in an extraction dict.

    This is the simplest way to use the RAG corrector from the FastAPI
    application::

        corrected = correct_icd_codes(pigeon_output)
    """
    corrector = get_corrector()
    return corrector.process_extraction_result(extraction_result)
