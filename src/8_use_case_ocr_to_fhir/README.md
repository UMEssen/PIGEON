# Stage 8: Use Case — OCR to FHIR R4 Pipeline

**Flying Pigeon**: A real-world application demonstrating the PIGEON model for end-to-end clinical document processing.

## Overview

This use case shows how the fine-tuned PIGEON model can be deployed in a production-like setting to process scanned or digital German medical documents (Arztbriefe) and convert them to structured FHIR R4 resources.

```
PDF Document ──> OCR/Text Extraction ──> PIGEON Extraction ──> RAG Correction ──> FHIR R4 Bundle
  (Docling)      (Qwen VLM fallback)    (Fine-tuned Gemma)   (ICD-10 codes)    (Structured output)
```

## Pipeline Steps

### 1. PDF Processing (Docling)
- **Dual-mode**: Fast text extraction for digital PDFs, VLM-based OCR for scanned documents
- Uses [Docling](https://github.com/DS4SD/docling) for document parsing
- Falls back to Qwen VLM (Vision-Language Model) when no text layer is detected
- Extracts clean medical text from HTML/markdown output

### 2. Medical Information Extraction (PIGEON)
- Sends extracted text to the fine-tuned PIGEON model (MedGemma-27B)
- Extracts structured JSON matching the PIGEON schema:
  - Patient demographics (introduction)
  - Diagnoses with ICD-10-GM codes
  - Tumor staging information
  - Medications with dosage
  - Lab values
  - Procedures with OPS codes

### 3. RAG-Based ICD-10 Code Correction
- Validates and corrects ICD-10-GM codes using semantic search
- FAISS vector store over the complete ICD-10-GM catalog
- MedGemma-4B as reranker for candidate selection
- Adds `icd10gm_code_rag` field alongside original codes

### 4. FHIR R4 Bundle Generation
- Converts structured extraction to FHIR R4 resources:
  - **Patient** — Demographics, address
  - **Encounter** — Hospital stay period, type (inpatient/ambulatory)
  - **Condition** — Diagnoses with ICD-10-GM coding (uses RAG-corrected codes)
  - **Procedure** — Clinical procedures with OPS codes
  - **MedicationStatement** — Medications with dosage
  - **Observation** — Lab values, tumor staging (TNM components)
- Wraps all resources in a FHIR R4 Bundle (type: collection)
- Tracks RAG corrections in Condition.note

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    React Frontend                         │
│   Upload PDF → View Progress → Inspect FHIR Resources    │
└──────────────────────┬───────────────────────────────────┘
                       │ REST API + NDJSON Streaming
┌──────────────────────▼───────────────────────────────────┐
│                  FastAPI Backend (app.py)                  │
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │   Docling    │  │  LLM Client  │  │ FHIR Converter  │ │
│  │  Processor   │  │   (PIGEON)   │  │     (R4)        │ │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘ │
│         │                │                     │          │
│  ┌──────▼──────┐  ┌──────▼───────┐  ┌────────▼────────┐ │
│  │  Qwen VLM   │  │ MedGemma-27B │  │  ICD RAG        │ │
│  │  (OCR)      │  │ (Extraction) │  │  Corrector      │ │
│  └─────────────┘  └──────────────┘  └─────────────────┘ │
└──────────────────────────────────────────────────────────┘
         │                │                    │
         ▼                ▼                    ▼
   vLLM Endpoint    vLLM Endpoint        FAISS + vLLM
   (Qwen VLM)      (PIGEON model)       (MedGemma-4B)
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- vLLM serving the PIGEON model + optional VLM endpoint

### Backend
```bash
cd src/8_use_case_ocr_to_fhir/backend

# Install dependencies (uses main project's pyproject.toml)
pip install fastapi uvicorn httpx docling docling-ibm-models tenacity pyyaml

# Configure endpoints in .env (project root)
# LLM_ENDPOINT_INFERENCE=http://localhost:8005/v1  (PIGEON model)
# LLM_ENDPOINT_SECONDARY=http://localhost:8001/v1  (Strong LLM for summaries)

# Start backend
uvicorn app:app --host 0.0.0.0 --port 8003 --reload
```

### Frontend
```bash
cd src/8_use_case_ocr_to_fhir/frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/health` | Check LLM endpoint connectivity |
| POST | `/api/extract` | Upload PDF, get extraction JSON |
| POST | `/api/fhir/convert` | Convert extraction JSON to FHIR R4 Bundle |
| POST | `/api/pipeline` | Full pipeline with NDJSON streaming |

### Example: Full Pipeline (cURL)
```bash
curl -X POST http://localhost:8003/api/pipeline \
  -F "file=@arztbrief.pdf" \
  --no-buffer
```

Response (NDJSON stream):
```json
{"event": "upload_received", "filename": "arztbrief.pdf", "size": 12345}
{"event": "ocr_complete", "medical_text": "Patient Hans Müller..."}
{"event": "extraction_complete", "extraction_result": {"introduction": {...}, "diagnoses": [...]}}
{"event": "rag_complete", "corrected_diagnoses": 5}
{"event": "fhir_complete", "statistics": {"Patient": 1, "Encounter": 1, "Condition": 5, ...}}
{"event": "complete", "bundle": {"resourceType": "Bundle", "type": "collection", ...}}
```

## FHIR R4 Output Example

```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "name": [{"family": "Müller", "given": ["Hans"]}],
        "gender": "male",
        "birthDate": "1958-07-12"
      }
    },
    {
      "resource": {
        "resourceType": "Condition",
        "code": {
          "coding": [{
            "system": "http://fhir.de/CodeSystem/bfarm/icd-10-gm",
            "code": "C34.1",
            "display": "Bösartige Neubildung: Oberlappen (-Bronchus)"
          }]
        },
        "note": [{"text": "ICD code corrected by RAG: C34.0 → C34.1"}]
      }
    }
  ]
}
```

## Configuration

All endpoints and paths are configured via the project's `.env` file:

| Variable | Used For | Default |
|----------|----------|---------|
| `LLM_ENDPOINT_INFERENCE` | PIGEON model endpoint | `http://localhost:8002/v1` |
| `LLM_ENDPOINT_SECONDARY` | Strong LLM (summaries) | `http://localhost:8001/v1` |
| `LLM_ENDPOINT_VLM` | VLM for OCR fallback | `http://localhost:8000/v1/chat/completions` |
| `ICD10_LOOKUP` | ICD-10 catalog CSV path | `lookups/icd10gm_lookup_merged.csv` |

## Key Dependencies (additional to main project)

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `docling` | PDF document parsing |
| `httpx` | Async HTTP client |
| `tenacity` | Retry logic |
| `pyyaml` | Prompt configuration |
