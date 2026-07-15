"""
Stage 8: Pydantic models for the Flying Pigeon pipeline.

These models define the data contracts for the OCR-to-FHIR pipeline:
- Processing status tracking
- Patient and document metadata
- Extraction results from the PIGEON model
- Provenance records for audit trails
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import *

from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Processing status enum -- tracks the stage of each document
# ---------------------------------------------------------------------------

class ProcessingStatus(str, Enum):
    """Pipeline processing status for a single document."""

    PENDING = "pending"             # Uploaded but not yet processed
    OCR_IN_PROGRESS = "ocr"        # Docling / VLM OCR is running
    OCR_COMPLETE = "ocr_complete"   # OCR finished, text available
    EXTRACTING = "extracting"       # PIGEON model extraction running
    EXTRACTED = "extracted"         # Extraction JSON available
    RAG_CORRECTING = "rag"         # RAG ICD code correction in progress
    RAG_COMPLETE = "rag_complete"   # RAG correction finished
    CONVERTING = "converting"       # FHIR R4 conversion in progress
    COMPLETE = "complete"           # Full pipeline done
    ERROR = "error"                 # Something went wrong


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

class Patient(BaseModel):
    """Minimal patient demographics extracted from the document."""

    name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    patient_id: Optional[str] = None


class Document(BaseModel):
    """Metadata about an uploaded clinical document."""

    filename: str
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    file_size_bytes: int = 0
    page_count: int = 0
    has_text_layer: bool = False
    document_category: Optional[str] = None  # e.g. "Arztbrief", "Befund"


class ExtractionResult(BaseModel):
    """The structured extraction produced by the PIGEON model.

    The ``data`` field holds the raw JSON dict that the model returned.
    It typically contains keys like ``diagnoses``, ``procedures``,
    ``medications``, ``vital_signs``, ``free_text``, etc.
    """

    data: Dict[str, Any] = Field(default_factory=dict)
    model_used: str = ""
    extraction_time_seconds: float = 0.0
    raw_text_length: int = 0


class ProcessingResult(BaseModel):
    """End-to-end result of the full pipeline for one document."""

    status: ProcessingStatus = ProcessingStatus.PENDING
    patient: Optional[Patient] = None
    document: Optional[Document] = None
    medical_text: str = ""                    # Plain text after OCR
    extraction: Optional[ExtractionResult] = None
    rag_corrected: Optional[Dict[str, Any]] = None
    fhir_bundle: Optional[Dict[str, Any]] = None
    fhir_statistics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    total_time_seconds: float = 0.0


class ProvenanceRecord(BaseModel):
    """Audit trail entry for a pipeline run.

    Records *who* ran the pipeline, *when*, and *what* happened at each step.
    Useful for clinical governance and reproducibility.
    """

    run_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_filename: str = ""
    pipeline_version: str = "1.0.0"
    steps_completed: List[str] = Field(default_factory=list)
    models_used: Dict[str, str] = Field(default_factory=dict)
    rag_corrections_made: int = 0
    fhir_resources_generated: int = 0
    errors: List[str] = Field(default_factory=list)
