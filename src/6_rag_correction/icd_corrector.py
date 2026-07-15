"""
ICD-10-GM code corrector using RAG.

This module corrects ICD-10-GM (German Modification) diagnosis codes that were
predicted by the language model.  It inherits the generic retrieval and LLM
re-ranking logic from ``BaseRAGCorrector`` and adds:

- ICD-10-GM-specific regex extraction (``A00``-``Z99``, optional dot + digits,
  optional ``!``, ``+``, ``*`` suffixes).
- A German-language prompt tailored to ICD-10 coding conventions.
- An **optional** external name cache that fetches the official ICD name from
  `gesund.bund.de <https://gesund.bund.de/icd-code-suche/>`_.  This feature is
  entirely optional and can be disabled (``use_official_names=False``).

Code format examples
--------------------
- ``C34.1`` -- Malignant neoplasm of upper lobe, bronchus or lung
- ``I10.00`` -- Essential (primary) hypertension
- ``F32.0!`` -- Mild depressive episode (mandatory secondary code marker)
"""

import re
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from config import (
    ICD10_LOOKUP,
    VECTORSTORE_CACHE_ICD,
    LLM_ENDPOINT_INFERENCE,
    LLM_MODEL_INFERENCE,
)

from base_corrector import BaseRAGCorrector

logger = logging.getLogger(__name__)


class ICDCodeRAGCorrector(BaseRAGCorrector):
    """RAG corrector specialised for ICD-10-GM diagnosis codes.

    Parameters
    ----------
    lookup_csv_path : str or Path
        Defaults to ``config.ICD10_LOOKUP``.
    vectorstore_cache_dir : str or Path
        Defaults to ``config.VECTORSTORE_CACHE_ICD``.
    llm_endpoint, llm_model : str or None
        vLLM endpoint / model.  Defaults to the inference endpoint from
        ``config``.
    use_official_names : bool
        When ``True``, attempt to enrich prompts with the *official* ICD name
        fetched from ``gesund.bund.de``.  Requires network access.  Results
        are cached in a persistent pickle file so subsequent runs are fast.
    """

    # ICD-10 pattern: Letter + 2 digits, optional .digits, optional suffix
    _CODE_RE = re.compile(r"\b([A-Z]\d{2}(?:\.\d{1,2})?[+*!]?)\b")

    def __init__(
        self,
        lookup_csv_path: str = str(ICD10_LOOKUP),
        vectorstore_cache_dir: str = str(VECTORSTORE_CACHE_ICD),
        llm_endpoint: Optional[str] = LLM_ENDPOINT_INFERENCE,
        llm_model: Optional[str] = LLM_MODEL_INFERENCE,
        top_k: int = 10,
        use_official_names: bool = False,
        **kwargs,
    ):
        super().__init__(
            lookup_csv_path=lookup_csv_path,
            code_column="code",
            display_column="display",
            vectorstore_cache_dir=vectorstore_cache_dir,
            llm_endpoint=llm_endpoint,
            llm_model=llm_model,
            top_k=top_k,
            **kwargs,
        )
        # Optional official-name cache (gesund.bund.de)
        self.use_official_names = use_official_names
        self._official_names_cache: Dict[str, Optional[str]] = {}
        self._official_names_cache_path = (
            Path(vectorstore_cache_dir) / "official_names_cache.pkl"
        )
        self._cache_updates_since_save = 0
        if use_official_names:
            self._load_official_names_cache()

    # ------------------------------------------------------------------
    # Official name cache (optional, for gesund.bund.de)
    # ------------------------------------------------------------------

    def _load_official_names_cache(self) -> None:
        """Load the persistent pickle cache of official ICD names."""
        if self._official_names_cache_path.exists():
            try:
                with open(self._official_names_cache_path, "rb") as fh:
                    self._official_names_cache = pickle.load(fh)
                print(
                    f"Loaded {len(self._official_names_cache)} cached "
                    "official ICD names from disk"
                )
            except Exception as exc:
                logger.warning("Could not load official names cache: %s", exc)

    def _save_official_names_cache(self, force: bool = False) -> None:
        """Persist the official-name cache (batched writes)."""
        self._cache_updates_since_save += 1
        if not force and self._cache_updates_since_save < 50:
            return
        try:
            self._official_names_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._official_names_cache_path, "wb") as fh:
                pickle.dump(self._official_names_cache, fh)
            self._cache_updates_since_save = 0
        except Exception as exc:
            logger.warning("Could not save official names cache: %s", exc)

    def _fetch_official_icd_name(self, icd_code: str) -> Optional[str]:
        """Fetch the official ICD name from gesund.bund.de (with caching).

        This method is **only** called when ``use_official_names=True``.
        It makes HTTP requests to ``https://gesund.bund.de/icd-code-suche/``
        and scrapes the ``<h1>`` tag for the official name.

        .. note::

           The external service may be unavailable.  Failures are silently
           cached as ``None`` so they are not retried.
        """
        if not icd_code:
            return None
        if icd_code in self._official_names_cache:
            return self._official_names_cache[icd_code]

        try:
            import requests
            from bs4 import BeautifulSoup

            url_code = (
                icd_code.lower()
                .replace(".", "-")
                .replace("+", "")
                .replace("*", "")
                .replace("!", "")
            )
            url = f"https://gesund.bund.de/icd-code-suche/{url_code}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                h1 = soup.find("h1")
                if h1:
                    text = h1.get_text().strip()
                    match = re.search(r"ICD-Code\s+[^:]+:\s*(.+)", text)
                    if match:
                        official_name = match.group(1).strip()
                        self._official_names_cache[icd_code] = official_name
                        self._save_official_names_cache()
                        return official_name
        except Exception:
            pass

        # Cache "not found" to avoid repeated requests
        self._official_names_cache[icd_code] = None
        self._save_official_names_cache()
        return None

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract an ICD-10-GM code from the LLM response."""
        if not response or response.upper() in ("NONE", "ERROR", ""):
            return None
        matches = self._CODE_RE.findall(response.upper())
        if matches:
            return matches[0]
        # Fallback: first word of first line
        first_word = response.split("\n")[0].strip().split()[0] if response.strip() else None
        if first_word and len(first_word) >= 3:
            return first_word.upper()
        return None

    def _format_prompt(
        self,
        name: str,
        original_code: str,
        candidates: List[Dict[str, Any]],
    ) -> str:
        """Build a German-language prompt for ICD-10-GM code selection."""
        # Original code context
        original_text = f"\nORIGINAL VORHERGESAGTER CODE: {original_code}\n"
        if self.use_official_names:
            official = self._fetch_official_icd_name(original_code)
            if official:
                original_text += f"OFFIZIELLE BEZEICHNUNG: {official}\n"
            else:
                original_text += (
                    "(Offizielle Bezeichnung nicht gefunden"
                    " - moeglicherweise ungueltiger Code)\n"
                )

        # Candidate list
        candidates_text = ""
        for i, cand in enumerate(candidates, 1):
            code = cand["code"]
            candidates_text += f"\n{i}. ICD-10 Code: {code}\n"
            if self.use_official_names:
                official = self._fetch_official_icd_name(code)
                if official:
                    candidates_text += f"   OFFIZIELLE BEZEICHNUNG: {official}\n"
            candidates_text += "   Definitionen aus Lookup-Tabelle:\n"
            for j, defn in enumerate(cand["definitions"][:3], 1):
                candidates_text += f"   {j}) {defn}\n"

        prompt = (
            "Du bist ein medizinischer Experte fuer ICD-10-Kodierung. "
            "Waehle den korrekten ICD-10-Code fuer die gegebene Diagnose aus.\n\n"
            f"DIAGNOSE: {name}\n"
            f"{original_text}\n"
            "ALTERNATIVE KANDIDATEN:\n"
            f"{candidates_text}\n\n"
            "WICHTIG:\n"
            "- Antworte NUR mit dem ICD-10-Code "
            "(Format: Buchstabe + 2 Ziffern + optional .Ziffern)\n"
            "- Beispiele: A01.0, C44.5, I25.1\n"
            "- Achte darauf wenn keine Lokalisation erkennbar ist, "
            "den allgemeineren Code zu waehlen (meistens nicht naeher bezeichnet)\n"
            "- Du kannst den ORIGINAL CODE behalten, wenn er bereits korrekt ist\n"
            "- Keine Erklaerungen, keine zusaetzlichen Woerter, nur der Code!\n\n"
            "Code:"
        )
        return prompt


if __name__ == "__main__":
    # Quick smoke test
    corrector = ICDCodeRAGCorrector(use_official_names=False)
    candidates = corrector._retrieve_candidates("Brustkrebs")
    print("Top candidates for 'Brustkrebs':")
    for c in candidates[:5]:
        print(f"  {c['code']}: {c['definitions'][0]}")
