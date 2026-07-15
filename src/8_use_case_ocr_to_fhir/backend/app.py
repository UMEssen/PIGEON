"""
Stage 8: Flying Pigeon -- OCR to FHIR R4 Pipeline

A real-world application demonstrating the PIGEON model for:
1. PDF document processing (OCR via Docling + Qwen VLM)
2. Medical information extraction (PIGEON fine-tuned model)
3. RAG-based ICD-10 code correction
4. FHIR R4 resource generation

Endpoints
---------
GET  /             Health check (HTML welcome page)
POST /api/extract  Upload a PDF, get the extraction JSON
POST /api/fhir/convert  Convert an extraction dict to a FHIR R4 Bundle
POST /api/pipeline  Full pipeline with streaming NDJSON events
GET  /api/health   Check LLM endpoint connectivity

Run with::

    cd src/8_use_case_ocr_to_fhir/backend
    uvicorn app:app --host 0.0.0.0 --port 8080 --reload
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so all config imports work
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_BACKEND_DIR))

from config import (
    LLM_ENDPOINT_INFERENCE,
    LLM_ENDPOINT_SECONDARY,
    LLM_ENDPOINT_VLM,
)

import json
import logging
import tempfile
import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# Local modules
from docling_processor import DoclingProcessor
from llm_client import LLMClient
from icd_rag_corrector import correct_icd_codes
from fhir_converter import FHIRConverter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("flying_pigeon")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Flying Pigeon -- OCR to FHIR R4",
    description=(
        "A real-world use case of the PIGEON pipeline: "
        "PDF OCR -> Medical Extraction -> RAG Correction -> FHIR R4"
    ),
    version="1.0.0",
)

# CORS -- allow localhost development (Vite, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-initialised singletons (heavy objects loaded on first request)
# ---------------------------------------------------------------------------
_docling_processor: DoclingProcessor | None = None
_llm_client: LLMClient | None = None
_fhir_converter: FHIRConverter | None = None


def get_docling_processor() -> DoclingProcessor:
    global _docling_processor
    if _docling_processor is None:
        _docling_processor = DoclingProcessor()
    return _docling_processor


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def get_fhir_converter() -> FHIRConverter:
    global _fhir_converter
    if _fhir_converter is None:
        _fhir_converter = FHIRConverter()
    return _fhir_converter


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class FHIRConvertRequest(BaseModel):
    """Request body for the /api/fhir/convert endpoint."""
    extraction_data: Dict[str, Any]


# ---------------------------------------------------------------------------
# GET / -- Health check / welcome page
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Return a simple HTML page confirming the service is running."""
    return """
    <html>
    <head><title>Flying Pigeon</title></head>
    <body style="font-family: system-ui; max-width: 600px; margin: 4rem auto;">
        <h1>Flying Pigeon</h1>
        <p>OCR to FHIR R4 pipeline is running.</p>
        <ul>
            <li><code>POST /api/extract</code> -- Upload PDF, get extraction</li>
            <li><code>POST /api/fhir/convert</code> -- Extraction to FHIR R4</li>
            <li><code>POST /api/pipeline</code> -- Full pipeline (streaming)</li>
            <li><code>GET  /api/health</code> -- Endpoint connectivity</li>
        </ul>
    </body>
    </html>
    """


# ---------------------------------------------------------------------------
# POST /api/extract -- Upload PDF, get extraction JSON
# ---------------------------------------------------------------------------

@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    """Process a PDF upload and return the PIGEON extraction.

    Steps:
    1. Save the uploaded file to a temp directory.
    2. Run Docling to extract text (with VLM fallback for scanned docs).
    3. Run the PIGEON model to produce structured medical data.
    4. Return the extraction JSON.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    start = time.time()

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # OCR
        processor = get_docling_processor()
        html, images, has_text, metadata = processor.parse_pdf(tmp_path)
        medical_text = processor.extract_plain_text(html)

        if not medical_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract any text from the PDF.",
            )

        # Extraction
        client = get_llm_client()
        extraction = client.extract_medical_info_german(medical_text)

        elapsed = time.time() - start
        return {
            "extraction": extraction,
            "medical_text": medical_text[:500],  # Preview only
            "metadata": metadata,
            "processing_time_seconds": round(elapsed, 2),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# POST /api/fhir/convert -- Extraction dict to FHIR R4 Bundle
# ---------------------------------------------------------------------------

@app.post("/api/fhir/convert")
async def fhir_convert(request: FHIRConvertRequest):
    """Convert a PIGEON extraction dict into a FHIR R4 Bundle.

    This endpoint does NOT run OCR or extraction -- it only converts
    an already-extracted dict into FHIR resources.  Useful when the
    frontend has already obtained the extraction from /api/extract
    and wants to convert it separately.
    """
    try:
        converter = get_fhir_converter()
        result = converter.convert(request.extraction_data)
        return {
            "bundle": result["bundle"],
            "statistics": result["statistics"],
        }
    except Exception as exc:
        logger.exception("FHIR conversion failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /api/pipeline -- Full pipeline with streaming NDJSON
# ---------------------------------------------------------------------------

@app.post("/api/pipeline")
async def pipeline(file: UploadFile = File(...)):
    """Run the full OCR -> Extract -> RAG -> FHIR pipeline.

    Returns a streaming NDJSON response where each line is a JSON
    object with an ``event`` field indicating the pipeline stage::

        {"event": "upload_received", ...}
        {"event": "ocr_start", ...}
        {"event": "ocr_complete", "medical_text": "...", ...}
        {"event": "extraction_start", ...}
        {"event": "extraction_complete", "extraction_result": {...}, ...}
        {"event": "rag_start", ...}
        {"event": "rag_complete", "corrected_result": {...}, ...}
        {"event": "fhir_start", ...}
        {"event": "fhir_complete", "bundle": {...}, "statistics": {...}, ...}
        {"event": "complete", ...}
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save upload to temp file before starting the stream
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    async def event_stream():
        """Generator that yields NDJSON events for each pipeline stage."""
        run_id = str(uuid.uuid4())
        pipeline_start = time.time()

        def emit(event: str, **kwargs) -> str:
            """Format a single NDJSON line."""
            payload = {"event": event, "run_id": run_id, **kwargs}
            return json.dumps(payload, ensure_ascii=False, default=str) + "\n"

        try:
            # -- Upload received --
            yield emit(
                "upload_received",
                filename=file.filename,
                file_size_bytes=len(content),
            )

            # -- OCR --
            yield emit("ocr_start")
            ocr_start = time.time()

            processor = get_docling_processor()
            html, images, has_text, metadata = processor.parse_pdf(tmp_path)
            medical_text = processor.extract_plain_text(html)

            ocr_elapsed = time.time() - ocr_start
            yield emit(
                "ocr_complete",
                medical_text=medical_text[:2000],  # Truncated preview
                has_text_layer=has_text,
                page_count=metadata.get("page_count", 0),
                converter_used=metadata.get("converter_used", "unknown"),
                ocr_time_seconds=round(ocr_elapsed, 2),
            )

            if not medical_text.strip():
                yield emit("error", message="No text could be extracted from the PDF.")
                return

            # -- Extraction --
            yield emit("extraction_start")
            ext_start = time.time()

            client = get_llm_client()
            extraction_result = client.extract_medical_info_german(medical_text)

            ext_elapsed = time.time() - ext_start
            yield emit(
                "extraction_complete",
                extraction_result=extraction_result,
                extraction_time_seconds=round(ext_elapsed, 2),
            )

            if not extraction_result:
                yield emit("error", message="Extraction returned empty result.")
                return

            # -- RAG correction --
            yield emit("rag_start")
            rag_start = time.time()

            corrected_result = correct_icd_codes(extraction_result)

            rag_elapsed = time.time() - rag_start
            yield emit(
                "rag_complete",
                corrected_result=corrected_result,
                rag_time_seconds=round(rag_elapsed, 2),
            )

            # -- FHIR R4 conversion --
            yield emit("fhir_start")
            fhir_start = time.time()

            converter = get_fhir_converter()
            fhir_result = converter.convert(corrected_result)

            fhir_elapsed = time.time() - fhir_start
            yield emit(
                "fhir_complete",
                bundle=fhir_result["bundle"],
                statistics=fhir_result["statistics"],
                fhir_time_seconds=round(fhir_elapsed, 2),
            )

            # -- Done --
            total_elapsed = time.time() - pipeline_start
            yield emit(
                "complete",
                total_time_seconds=round(total_elapsed, 2),
            )

        except Exception as exc:
            logger.exception("Pipeline error")
            yield emit("error", message=str(exc))

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
    )


# ---------------------------------------------------------------------------
# GET /api/health -- Check LLM endpoint connectivity
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Check connectivity to the configured LLM endpoints.

    Returns a dict with the status of each endpoint (reachable or not).
    """
    import httpx

    endpoints = {
        "inference": LLM_ENDPOINT_INFERENCE,
        "secondary": LLM_ENDPOINT_SECONDARY,
        "vlm": LLM_ENDPOINT_VLM,
    }

    results = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in endpoints.items():
            try:
                # Try to hit the /models endpoint (standard for vLLM)
                models_url = f"{url}/models"
                resp = await client.get(models_url)
                results[name] = {
                    "url": url,
                    "status": "reachable",
                    "http_code": resp.status_code,
                }
            except Exception as exc:
                results[name] = {
                    "url": url,
                    "status": "unreachable",
                    "error": str(exc),
                }

    all_ok = all(r["status"] == "reachable" for r in results.values())
    return {
        "healthy": all_ok,
        "endpoints": results,
    }
