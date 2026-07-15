"""
Stage 2 -- Dataset Generation.

Generates synthetic German medical texts (Arztbriefe, progress notes,
free-form clinical text) from structured FHIR cache data using a
combination of rule-based templates and LLM prompts.

Public API:
    TextGenerator        -- Main orchestrator class.
    FHIRDataLoader       -- Loads FHIR cache CSVs into structured dicts.
    create_dataset.main  -- CLI entry point for batch generation.
"""
