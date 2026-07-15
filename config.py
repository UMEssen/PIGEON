"""
Central configuration for the PIGEON pipeline.
All paths and endpoints are configured here — no hardcoded paths elsewhere.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Project root (auto-detected)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────
# FHIR Cache paths
# ──────────────────────────────────────────────
FHIR_CACHE_BASE = Path(os.getenv("FHIR_CACHE_BASE", PROJECT_ROOT / "data" / "fhir_cache"))
FHIR_CACHE_ENCOUNTER = FHIR_CACHE_BASE / "fhir_cache_encounter"
FHIR_CACHE_PATIENT = FHIR_CACHE_BASE / "fhir_cache_patient"
TUMORDOKU_CACHE = FHIR_CACHE_PATIENT / "tumordoku_precise"

# ──────────────────────────────────────────────
# Lookup tables  (ICD-10-GM, OPS, ATC)
# ──────────────────────────────────────────────
LOOKUPS_DIR = Path(os.getenv("LOOKUPS_DIR", PROJECT_ROOT / "lookups"))
ICD10_LOOKUP = LOOKUPS_DIR / "icd10gm_lookup_merged.csv"
ICD10_JARGON_LOOKUP = LOOKUPS_DIR / "icd10gm_lookup_with_jargon.csv"
OPS_LOOKUP = LOOKUPS_DIR / "ops_lookup_merged.csv"
ATC_LOOKUP = LOOKUPS_DIR / "atc_lookup_merged.csv"

# ──────────────────────────────────────────────
# Dataset output paths
# ──────────────────────────────────────────────
DATASETS_DIR = Path(os.getenv("DATASETS_DIR", PROJECT_ROOT / "data" / "datasets"))
GENERATED_TEXTS_DIR = DATASETS_DIR / "generated_texts"
TRAINING_READY_DIR = DATASETS_DIR / "training_ready"

# ──────────────────────────────────────────────
# Model & training paths
# ──────────────────────────────────────────────
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "data" / "models"))
FINETUNED_DIR = MODELS_DIR / "finetuned"
HF_CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", MODELS_DIR / "cache"))

# ──────────────────────────────────────────────
# Results & evaluation
# ──────────────────────────────────────────────
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "results"))

# ──────────────────────────────────────────────
# vLLM / LLM API endpoints
# ──────────────────────────────────────────────
LLM_ENDPOINT_PRIMARY = os.getenv("LLM_ENDPOINT_PRIMARY", "http://localhost:8000/v1")
LLM_ENDPOINT_SECONDARY = os.getenv("LLM_ENDPOINT_SECONDARY", "http://localhost:8001/v1")
LLM_ENDPOINT_INFERENCE = os.getenv("LLM_ENDPOINT_INFERENCE", "http://localhost:8002/v1")
LLM_ENDPOINT_VLM = os.getenv("LLM_ENDPOINT_VLM", "http://localhost:8003/v1")

# Model names served by vLLM
LLM_MODEL_PRIMARY = os.getenv("LLM_MODEL_PRIMARY", "meta-llama/Llama-3.3-70B-Instruct")
LLM_MODEL_SECONDARY = os.getenv("LLM_MODEL_SECONDARY", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
LLM_MODEL_INFERENCE = os.getenv("LLM_MODEL_INFERENCE", "google/medgemma-27b-text-it")
LLM_MODEL_VLM = os.getenv("LLM_MODEL_VLM", "Qwen/Qwen2.5-VL-72B-Instruct")

# ──────────────────────────────────────────────
# Database (for FHIR cache generation)
# ──────────────────────────────────────────────
METRICS_DB = os.getenv("METRICS_DB", "fhir_metrics")
METRICS_USER = os.getenv("METRICS_USER", "")
METRICS_PASSWORD = os.getenv("METRICS_PASSWORD", "")
METRICS_HOST = os.getenv("METRICS_HOST", "localhost")
METRICS_PORT = int(os.getenv("METRICS_PORT", "5432"))

# ──────────────────────────────────────────────
# HuggingFace
# ──────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Set via env var, never hardcode

# ──────────────────────────────────────────────
# RAG vector store caches
# ──────────────────────────────────────────────
VECTORSTORE_CACHE_ICD = LOOKUPS_DIR / "vectorstore_cache"
VECTORSTORE_CACHE_ATC = LOOKUPS_DIR / "vectorstore_cache_atc"
VECTORSTORE_CACHE_OPS = LOOKUPS_DIR / "vectorstore_cache_ops"

# ──────────────────────────────────────────────
# Processing parameters
# ──────────────────────────────────────────────
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
RANDOM_SEED = 42


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [
        FHIR_CACHE_ENCOUNTER, FHIR_CACHE_PATIENT, TUMORDOKU_CACHE,
        GENERATED_TEXTS_DIR, TRAINING_READY_DIR,
        FINETUNED_DIR, HF_CACHE_DIR,
        RESULTS_DIR,
        VECTORSTORE_CACHE_ICD, VECTORSTORE_CACHE_ATC, VECTORSTORE_CACHE_OPS,
    ]:
        d.mkdir(parents=True, exist_ok=True)
