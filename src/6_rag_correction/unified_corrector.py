"""
Unified interface for RAG-based medical code correction.

This module provides ``UnifiedRAGCorrector``, a convenience wrapper that
initialises and exposes all three correctors (ICD-10-GM, OPS, ATC) behind a
single object.  It is the recommended entry point for scripts that need to
correct multiple code types in a single pass.

Example
-------
::

    from unified_corrector import UnifiedRAGCorrector

    corrector = UnifiedRAGCorrector()

    icd = corrector.correct_icd_code("Brustkrebs", "C50.9")
    ops = corrector.correct_ops_code("Herzkatheteruntersuchung", "1-277.0")
    atc = corrector.correct_atc_code("Ibuprofen", "M01AE01")
"""

from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    ICD10_LOOKUP,
    OPS_LOOKUP,
    ATC_LOOKUP,
    VECTORSTORE_CACHE_ICD,
    VECTORSTORE_CACHE_OPS,
    VECTORSTORE_CACHE_ATC,
    LLM_ENDPOINT_INFERENCE,
    LLM_MODEL_INFERENCE,
)

from icd_corrector import ICDCodeRAGCorrector
from ops_corrector import OPSCodeRAGCorrector
from atc_corrector import ATCCodeRAGCorrector


class UnifiedRAGCorrector:
    """Unified interface for correcting ICD-10, OPS, and ATC codes.

    All three sub-correctors share the same embedding model (loaded once)
    and talk to the same vLLM endpoint for re-ranking.

    Parameters
    ----------
    llm_endpoint : str or None
        OpenAI-compatible vLLM base URL.  Defaults to
        ``config.LLM_ENDPOINT_INFERENCE``.
    llm_model : str or None
        Model name served at the endpoint.
    use_official_names : bool
        Whether to fetch official names from gesund.bund.de for ICD/OPS
        codes.  Default ``False``.
    """

    def __init__(
        self,
        llm_endpoint: Optional[str] = LLM_ENDPOINT_INFERENCE,
        llm_model: Optional[str] = LLM_MODEL_INFERENCE,
        use_official_names: bool = False,
    ):
        print("Initializing ICD-10 RAG Corrector...")
        self.icd_corrector = ICDCodeRAGCorrector(
            llm_endpoint=llm_endpoint,
            llm_model=llm_model,
            use_official_names=use_official_names,
        )

        print("\nInitializing OPS RAG Corrector...")
        self.ops_corrector = OPSCodeRAGCorrector(
            llm_endpoint=llm_endpoint,
            llm_model=llm_model,
            use_official_names=use_official_names,
        )

        print("\nInitializing ATC RAG Corrector...")
        self.atc_corrector = ATCCodeRAGCorrector(
            llm_endpoint=llm_endpoint,
            llm_model=llm_model,
        )

        print("\nAll correctors initialized successfully!")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct_icd_code(self, diagnosis_name: str, original_code: str) -> str:
        """Correct an ICD-10-GM code.

        Returns the corrected code, or ``original_code`` if no better
        alternative was found.
        """
        return self.icd_corrector.correct_code(diagnosis_name, original_code)

    def correct_ops_code(self, procedure_name: str, original_code: str) -> str:
        """Correct an OPS procedure code."""
        return self.ops_corrector.correct_code(procedure_name, original_code)

    def correct_atc_code(self, medication_name: str, original_code: str) -> str:
        """Correct an ATC medication code."""
        return self.atc_corrector.correct_code(medication_name, original_code)


if __name__ == "__main__":
    corrector = UnifiedRAGCorrector()

    print("\n" + "=" * 70)
    print("Example: ICD-10 Correction")
    print("=" * 70)
    result = corrector.correct_icd_code("Brustkrebs", "C50.9")
    print(f"  Diagnosis: Brustkrebs | Original: C50.9 | Corrected: {result}")

    print("\n" + "=" * 70)
    print("Example: OPS Correction")
    print("=" * 70)
    result = corrector.correct_ops_code("Herzkatheteruntersuchung", "1-277.0")
    print(f"  Procedure: Herzkatheteruntersuchung | Original: 1-277.0 | Corrected: {result}")

    print("\n" + "=" * 70)
    print("Example: ATC Correction")
    print("=" * 70)
    result = corrector.correct_atc_code("Ibuprofen", "M01AE01")
    print(f"  Medication: Ibuprofen | Original: M01AE01 | Corrected: {result}")
