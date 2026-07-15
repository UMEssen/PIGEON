"""
OPS procedure code corrector using RAG.

This module corrects OPS (Operationen- und Prozedurenschluessel) procedure
codes that were predicted by the language model.  It inherits the generic
retrieval and LLM re-ranking logic from ``BaseRAGCorrector``.

OPS code format examples
------------------------
- ``5-399.5``  -- Other operations on blood vessels
- ``8-800.c0`` -- Transfusion of blood components
- ``1-440.9``  -- Endoscopic biopsy

The category prefix is everything before the first hyphen (e.g. ``5`` in
``5-399.5``).
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
    OPS_LOOKUP,
    VECTORSTORE_CACHE_OPS,
    LLM_ENDPOINT_INFERENCE,
    LLM_MODEL_INFERENCE,
)

from base_corrector import BaseRAGCorrector

logger = logging.getLogger(__name__)


class OPSCodeRAGCorrector(BaseRAGCorrector):
    """RAG corrector specialised for OPS procedure codes.

    Parameters
    ----------
    lookup_csv_path : str or Path
        Defaults to ``config.OPS_LOOKUP``.
    vectorstore_cache_dir : str or Path
        Defaults to ``config.VECTORSTORE_CACHE_OPS``.
    llm_endpoint, llm_model : str or None
        vLLM endpoint / model.
    use_official_names : bool
        When ``True``, attempt to enrich prompts with the official OPS name
        fetched from ``gesund.bund.de``.  Requires network access.
    """

    # OPS pattern: digit(s)-digits.optional_alphanumeric
    _CODE_RE = re.compile(r"\b(\d{1,2}-\d{2,3}(?:\.\w+)?)\b")

    def __init__(
        self,
        lookup_csv_path: str = str(OPS_LOOKUP),
        vectorstore_cache_dir: str = str(VECTORSTORE_CACHE_OPS),
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
        self.use_official_names = use_official_names
        self._official_names_cache: Dict[str, Optional[str]] = {}
        self._official_names_cache_path = (
            Path(vectorstore_cache_dir) / "official_names_cache_ops.pkl"
        )
        self._cache_updates_since_save = 0
        if use_official_names:
            self._load_official_names_cache()

    # ------------------------------------------------------------------
    # Official name cache (optional, for gesund.bund.de)
    # ------------------------------------------------------------------

    def _load_official_names_cache(self) -> None:
        if self._official_names_cache_path.exists():
            try:
                with open(self._official_names_cache_path, "rb") as fh:
                    self._official_names_cache = pickle.load(fh)
                print(
                    f"Loaded {len(self._official_names_cache)} cached "
                    "official OPS names from disk"
                )
            except Exception as exc:
                logger.warning("Could not load official names cache: %s", exc)

    def _save_official_names_cache(self, force: bool = False) -> None:
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

    def _fetch_official_ops_name(self, ops_code: str) -> Optional[str]:
        """Fetch the official OPS name from gesund.bund.de (with caching)."""
        if not ops_code:
            return None
        if ops_code in self._official_names_cache:
            return self._official_names_cache[ops_code]

        try:
            import requests
            from bs4 import BeautifulSoup

            url_code = ops_code.replace(".", "-")
            url = f"https://gesund.bund.de/en/ops-code-search/{url_code}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                h1 = soup.find("h1")
                if h1:
                    text = h1.get_text().strip()
                    if text:
                        self._official_names_cache[ops_code] = text
                        self._save_official_names_cache()
                        return text
        except Exception:
            pass

        self._official_names_cache[ops_code] = None
        self._save_official_names_cache()
        return None

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract an OPS code from the LLM response."""
        if not response or response.upper() in ("NONE", "ERROR", ""):
            return None
        matches = self._CODE_RE.findall(response)
        if matches:
            return matches[0]
        first_word = response.split("\n")[0].strip().split()[0] if response.strip() else None
        if first_word and len(first_word) >= 3:
            return first_word
        return None

    def _format_prompt(
        self,
        name: str,
        original_code: str,
        candidates: List[Dict[str, Any]],
    ) -> str:
        """Build a German-language prompt for OPS code selection."""
        original_text = f"\nORIGINAL VORHERGESAGTER CODE: {original_code}\n"
        if self.use_official_names:
            official = self._fetch_official_ops_name(original_code)
            if official:
                original_text += f"OFFIZIELLE BEZEICHNUNG: {official}\n"
            else:
                original_text += (
                    "(Offizielle Bezeichnung nicht gefunden"
                    " - moeglicherweise ungueltiger Code)\n"
                )

        candidates_text = ""
        for i, cand in enumerate(candidates, 1):
            code = cand["code"]
            candidates_text += f"\n{i}. OPS Code: {code}\n"
            if self.use_official_names:
                official = self._fetch_official_ops_name(code)
                if official:
                    candidates_text += f"   OFFIZIELLE BEZEICHNUNG: {official}\n"
            candidates_text += "   Definitionen aus Lookup-Tabelle:\n"
            for j, defn in enumerate(cand["definitions"][:3], 1):
                candidates_text += f"   {j}) {defn}\n"

        prompt = (
            "Du bist ein medizinischer Experte fuer OPS-Kodierung "
            "(Operationen- und Prozedurenschluessel). "
            "Waehle den korrekten OPS-Code fuer die gegebene Prozedur aus.\n\n"
            f"PROZEDUR: {name}\n"
            f"{original_text}\n"
            "ALTERNATIVE KANDIDATEN:\n"
            f"{candidates_text}\n\n"
            "WICHTIG:\n"
            "- Antworte NUR mit dem OPS-Code "
            "(Format: variabel, z.B. 5-399.5, 8-800.c0, 3-200)\n"
            "- Du kannst den ORIGINAL CODE behalten, wenn er bereits korrekt ist\n"
            "- Achte auf die genaue Beschreibung der Prozedur\n"
            "- Keine Erklaerungen, keine zusaetzlichen Woerter, nur der Code!\n\n"
            "Code:"
        )
        return prompt


if __name__ == "__main__":
    corrector = OPSCodeRAGCorrector(use_official_names=False)
    candidates = corrector._retrieve_candidates("Herzkatheteruntersuchung")
    print("Top candidates for 'Herzkatheteruntersuchung':")
    for c in candidates[:5]:
        print(f"  {c['code']}: {c['definitions'][0]}")
