"""
Stage 6 -- RAG-based medical code correction.

After the language model generates structured clinical documents (Stage 5),
the predicted medical codes (ICD-10-GM, OPS, ATC) may contain errors.
This stage uses Retrieval-Augmented Generation to post-correct those codes
by searching official catalogs with semantic similarity and re-ranking
candidates with an LLM.

Modules
-------
- ``base_corrector``     -- Abstract base class with shared FAISS / LLM logic
- ``icd_corrector``      -- ICD-10-GM diagnosis code corrector
- ``ops_corrector``      -- OPS procedure code corrector
- ``atc_corrector``      -- ATC medication code corrector
- ``unified_corrector``  -- Convenience wrapper exposing all three correctors
- ``correct_codes``      -- CLI script to apply corrections to a CSV
"""
