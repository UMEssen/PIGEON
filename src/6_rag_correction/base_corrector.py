"""
Base class for RAG-based medical code correction.

Stage 6 of the PIGEON pipeline uses Retrieval-Augmented Generation (RAG) to
post-correct medical codes (ICD-10-GM, OPS, ATC) that were predicted by the
fine-tuned language model during Stage 5 (inference).

The RAG correction pipeline works as follows:

1. **Embed** the diagnosis / procedure / medication *name* produced by the model.
2. **Retrieve** the top-k semantically similar entries from an official code
   catalog that has been pre-indexed in a FAISS vector store.
3. **Re-rank** the candidates with an LLM that selects the single best
   matching code given the clinical context.

This module provides ``BaseRAGCorrector``, an abstract-ish base class that
encapsulates the shared logic (embedding, FAISS index management, LLM
querying).  Concrete correctors for ICD-10-GM, OPS, and ATC inherit from it.

Dependencies
------------
- ``langchain-community`` (FAISS vector store, HuggingFace embeddings)
- ``sentence-transformers`` (embedding model)
- ``openai`` (OpenAI-compatible client for vLLM)
- ``faiss-cpu`` or ``faiss-gpu``
"""

import os
import json
import logging
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# Langchain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Suppress noisy tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class BaseRAGCorrector(ABC):
    """Base class shared by all RAG-based medical-code correctors.

    Subclasses must implement:
    - ``_extract_code``  -- regex extraction of a code from raw LLM output
    - ``_format_prompt`` -- build the LLM prompt for candidate re-ranking
    - ``_get_category``  -- return the broad category prefix of a code

    Parameters
    ----------
    lookup_csv_path : str or Path
        Path to the official code catalog CSV file.
    code_column : str
        Name of the column that contains the code (e.g. ``"code"``).
    display_column : str
        Name of the column that contains the human-readable name.
    vectorstore_cache_dir : str or Path
        Directory where the FAISS index is cached on disk.
    embedding_model : str
        HuggingFace model identifier used for semantic search.
    llm_endpoint : str or None
        Base URL of an OpenAI-compatible vLLM server.  When ``None`` the
        corrector can still *retrieve* candidates but not re-rank them.
    llm_model : str or None
        Model name served at ``llm_endpoint``.
    top_k : int
        Number of candidate codes to present to the LLM.
    """

    # Invalid sentinel codes that should never appear in the index.
    INVALID_CODES: set = {"***.**", ".", "???.?", "-", ""}

    def __init__(
        self,
        lookup_csv_path: str,
        code_column: str,
        display_column: str,
        vectorstore_cache_dir: str,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        llm_endpoint: Optional[str] = None,
        llm_model: Optional[str] = None,
        top_k: int = 10,
    ):
        self.lookup_csv_path = Path(lookup_csv_path)
        self.code_column = code_column
        self.display_column = display_column
        self.vectorstore_cache_dir = Path(vectorstore_cache_dir)
        self.top_k = top_k

        # ------- LLM client (OpenAI-compatible) -------
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.client: Optional[OpenAI] = None
        if llm_endpoint:
            self.client = OpenAI(
                base_url=llm_endpoint,
                api_key="EMPTY",  # vLLM does not require an API key
            )

        # ------- Embeddings -------
        print(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ------- Lookup table & vector store -------
        self.lookup_df: Optional[pd.DataFrame] = None
        self.code_to_definitions: Dict[str, List[str]] = defaultdict(list)
        self.vectorstore: Optional[FAISS] = None

        self._load_and_index_lookup_table()

    # ------------------------------------------------------------------
    # Vector store construction / loading
    # ------------------------------------------------------------------

    def _load_and_index_lookup_table(self) -> None:
        """Load the code catalog CSV and build (or load) the FAISS index."""
        print(f"Loading lookup table from {self.lookup_csv_path}")
        self.lookup_df = pd.read_csv(self.lookup_csv_path)
        print(f"Loaded {len(self.lookup_df)} lookup entries")

        # Build code -> definitions mapping
        for _, row in self.lookup_df.iterrows():
            code = str(row[self.code_column]).strip()
            display = str(row[self.display_column]).strip()
            if (
                display
                and display != "nan"
                and "existiert nicht" not in display
                and code not in self.INVALID_CODES
            ):
                self.code_to_definitions[code].append(display)

        print(f"Found {len(self.code_to_definitions)} unique codes")
        self.vectorstore = self._build_or_load_vectorstore()

    def _build_or_load_vectorstore(self) -> FAISS:
        """Build a FAISS index from the lookup CSV, or load from cache."""
        cache_path = self.vectorstore_cache_dir
        if cache_path.exists() and (cache_path / "index.faiss").exists():
            print(f"Loading cached vector store from {cache_path}")
            vs = FAISS.load_local(
                str(cache_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print("Vector store loaded from cache")
            return vs

        # Create documents -- one per (code, definition) pair
        print("Creating vector store (this may take a while)...")
        documents: list[Document] = []
        for code, definitions in tqdm(
            self.code_to_definitions.items(), desc="Creating documents"
        ):
            if code in self.INVALID_CODES:
                continue
            for definition in definitions:
                documents.append(
                    Document(
                        page_content=definition,
                        metadata={"code": code, "definition": definition},
                    )
                )

        print(f"Creating FAISS index with {len(documents)} documents...")
        vs = FAISS.from_documents(documents, self.embeddings)

        # Persist to disk
        cache_path.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(cache_path))
        print(f"Vector store cached to {cache_path}")
        return vs

    # ------------------------------------------------------------------
    # Candidate retrieval
    # ------------------------------------------------------------------

    def _retrieve_candidates(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Semantic search for the top-k candidate codes.

        Parameters
        ----------
        query : str
            Free-text name of the diagnosis / procedure / medication.
        k : int, optional
            Override the default ``self.top_k``.

        Returns
        -------
        list[dict]
            Each dict has keys ``code``, ``definitions``, ``best_score``,
            ``avg_score``.  Sorted by ascending FAISS distance (lower = better).
        """
        if not query or not query.strip():
            return []

        k = k or self.top_k

        # Retrieve more results than needed so we can group by code
        results = self.vectorstore.similarity_search_with_score(
            query, k=k * 3
        )

        # Group by code
        code_groups: Dict[str, Dict] = defaultdict(
            lambda: {"definitions": [], "scores": []}
        )
        for doc, score in results:
            code = doc.metadata["code"]
            if code in self.INVALID_CODES:
                continue
            code_groups[code]["definitions"].append(doc.metadata["definition"])
            code_groups[code]["scores"].append(float(score))

        # Build candidate list sorted by best (lowest) FAISS distance.
        # Tie-breaker: prefer longer (more specific) codes, then lexicographic.
        candidates = []
        for code, data in code_groups.items():
            candidates.append(
                {
                    "code": code,
                    "definitions": data["definitions"],
                    "best_score": min(data["scores"]),
                    "avg_score": float(np.mean(data["scores"])),
                }
            )

        candidates.sort(key=lambda x: (x["best_score"], -len(x["code"]), x["code"]))
        return candidates[:k]

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _llm_select_best(
        self,
        name: str,
        original_code: str,
        candidates: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Use the LLM to pick the single best code from candidates.

        Returns the extracted code string, or ``None`` when the LLM is
        unavailable or does not return a parseable answer.
        """
        if self.client is None:
            return None

        prompt = self._format_prompt(name, original_code, candidates)
        response = self._query_model(prompt)
        return self._extract_code(response)

    def _query_model(self, prompt: str, max_retries: int = 3) -> str:
        """Send a prompt to the vLLM endpoint and return the raw answer."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=50,
                    top_p=0.95,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                print(f"Warning: LLM attempt {attempt + 1} failed: {exc}")
                if attempt == max_retries - 1:
                    return "ERROR"
        return "ERROR"

    # ------------------------------------------------------------------
    # Public correction API
    # ------------------------------------------------------------------

    def correct_code(self, name: str, original_code: str) -> str:
        """Correct a single medical code using RAG.

        Parameters
        ----------
        name : str
            Human-readable name of the diagnosis / procedure / medication.
        original_code : str
            The code predicted by the model.

        Returns
        -------
        str
            The corrected code.  Falls back to ``original_code`` when the
            RAG pipeline cannot produce a better alternative.
        """
        if not name or not name.strip():
            return original_code

        candidates = self._retrieve_candidates(name)
        if not candidates:
            return original_code

        extracted = self._llm_select_best(name, original_code, candidates)
        if not extracted:
            return original_code

        # Validate: the selected code must be present in our catalog
        extracted_upper = extracted.upper()

        # Check candidates first
        for c in candidates:
            if c["code"].upper() == extracted_upper:
                return c["code"]

        # Check full lookup table
        for code in self.code_to_definitions:
            if code.upper() == extracted_upper:
                return code

        # Could not validate -- keep original
        return original_code

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract a well-formed code from the raw LLM response."""
        ...

    @abstractmethod
    def _format_prompt(
        self, name: str, original_code: str, candidates: List[Dict[str, Any]]
    ) -> str:
        """Build the German-language prompt for code selection."""
        ...
