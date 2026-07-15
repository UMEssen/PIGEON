"""
ATC medication code corrector using RAG.

This module corrects ATC (Anatomical Therapeutic Chemical Classification)
medication codes that were predicted by the language model.  It inherits
the generic retrieval and LLM re-ranking logic from ``BaseRAGCorrector``.

Compared to the ICD and OPS correctors this one is simpler: there is no
external official-name lookup (the ATC catalog definitions are already
self-contained).

ATC code format examples
-------------------------
- ``A01AA01`` -- Sodium fluoride
- ``C09AA02`` -- Enalapril
- ``N02BE01`` -- Paracetamol

Structure: 1 letter + 2 digits + 2 letters + 2 digits (7 characters).
Shorter prefixes (e.g. ``A01``) represent higher-level categories.
"""

import re
import logging
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from config import (
    ATC_LOOKUP,
    VECTORSTORE_CACHE_ATC,
    LLM_ENDPOINT_INFERENCE,
    LLM_MODEL_INFERENCE,
)

from base_corrector import BaseRAGCorrector

logger = logging.getLogger(__name__)


class ATCCodeRAGCorrector(BaseRAGCorrector):
    """RAG corrector specialised for ATC medication codes.

    Parameters
    ----------
    lookup_csv_path : str or Path
        Defaults to ``config.ATC_LOOKUP``.
    vectorstore_cache_dir : str or Path
        Defaults to ``config.VECTORSTORE_CACHE_ATC``.
    llm_endpoint, llm_model : str or None
        vLLM endpoint / model.
    """

    # ATC-7 pattern: Letter + 2 digits + 2 letters + 2 digits
    _CODE_RE = re.compile(r"\b([A-Z]\d{2}[A-Z]{2}\d{2})\b")

    def __init__(
        self,
        lookup_csv_path: str = str(ATC_LOOKUP),
        vectorstore_cache_dir: str = str(VECTORSTORE_CACHE_ATC),
        llm_endpoint: Optional[str] = LLM_ENDPOINT_INFERENCE,
        llm_model: Optional[str] = LLM_MODEL_INFERENCE,
        top_k: int = 10,
        **kwargs,
    ):
        super().__init__(
            lookup_csv_path=lookup_csv_path,
            code_column="atc_code",
            display_column="display",
            vectorstore_cache_dir=vectorstore_cache_dir,
            llm_endpoint=llm_endpoint,
            llm_model=llm_model,
            top_k=top_k,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract an ATC code from the LLM response."""
        if not response or response.upper() in ("NONE", "ERROR", ""):
            return None
        matches = self._CODE_RE.findall(response.upper())
        if matches:
            return matches[0]
        first_word = (
            response.split("\n")[0].strip().split()[0].upper()
            if response.strip()
            else None
        )
        if first_word and len(first_word) >= 7:
            return first_word
        return None

    def _format_prompt(
        self,
        name: str,
        original_code: str,
        candidates: List[Dict[str, Any]],
    ) -> str:
        """Build a German-language prompt for ATC code selection."""
        # Original code context
        original_text = f"\nORIGINAL VORHERGESAGTER CODE: {original_code}\n"
        if original_code in self.code_to_definitions:
            defs = self.code_to_definitions[original_code]
            original_text += f"Definitionen: {', '.join(defs[:3])}\n"
        else:
            original_text += (
                "(Code nicht in Lookup-Tabelle gefunden"
                " - moeglicherweise ungueltiger Code)\n"
            )

        # Candidate list
        candidates_text = ""
        for i, cand in enumerate(candidates, 1):
            code = cand["code"]
            candidates_text += f"\n{i}. ATC Code: {code}\n"
            candidates_text += "   Definitionen:\n"
            for j, defn in enumerate(cand["definitions"][:3], 1):
                candidates_text += f"   {j}) {defn}\n"

        prompt = (
            "Du bist ein medizinischer Experte fuer ATC-Kodierung "
            "(Anatomisch-Therapeutisch-Chemische Klassifikation). "
            "Waehle den korrekten ATC-Code fuer das gegebene Medikament aus.\n\n"
            f"MEDIKAMENT: {name}\n"
            f"{original_text}\n"
            "ALTERNATIVE KANDIDATEN:\n"
            f"{candidates_text}\n\n"
            "WICHTIG:\n"
            "- Antworte NUR mit dem ATC-Code "
            "(Format: Buchstabe + 2 Ziffern + Buchstabe + 2 Ziffern + optional 2 Ziffern)\n"
            "- Beispiele: A01AA01, C09AA02, N02BE01, J01CR02\n"
            "- Du kannst den ORIGINAL CODE behalten, wenn er bereits korrekt ist\n"
            "- Achte auf den Wirkstoff und die therapeutische Anwendung\n"
            "- Keine Erklaerungen, keine zusaetzlichen Woerter, nur der Code!\n\n"
            "Code:"
        )
        return prompt


if __name__ == "__main__":
    corrector = ATCCodeRAGCorrector()
    candidates = corrector._retrieve_candidates("Ibuprofen")
    print("Top candidates for 'Ibuprofen':")
    for c in candidates[:5]:
        print(f"  {c['code']}: {c['definitions'][0]}")
