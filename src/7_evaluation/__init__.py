"""
Stage 7 -- Evaluation of model predictions against ground truth.

This stage computes comprehensive metrics across all sections of the
PIGEON structured clinical document format:

- Introduction fields (exact match)
- Diagnosis ICD codes (Jaccard at full / 3-char / category levels, F1)
- Tumor information (per-type Jaccard and F1)
- Medications (name, dosage, ATC code matching)
- Lab values (name and value matching)
- Free text sections (per-subsection metrics)
- RAG correction impact analysis
"""
