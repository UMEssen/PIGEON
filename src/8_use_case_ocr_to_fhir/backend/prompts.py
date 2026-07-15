"""
Stage 8: Prompt templates for the Flying Pigeon pipeline.

All prompts are written in German because the clinical documents processed
by this pipeline are German-language discharge letters (Arztbriefe),
laboratory reports, and similar clinical documents from German hospitals.

Four prompts are defined:

1. **VLM_OCR_PROMPT** -- instructs the vision-language model to transcribe
   scanned / image-only PDF pages into clean text.
2. **GERMAN_DOCUMENT_CLASSIFICATION_PROMPT** -- classifies the document into
   a clinical category (Arztbrief, Befund, OP-Bericht, etc.).
3. **ARZTBRIEF_EXTRACTION_PROMPT** -- THE core PIGEON extraction prompt that
   tells the fine-tuned model exactly which structured fields to produce.
4. **GERMAN_SUMMARY_EXTRACTION_PROMPT** -- asks a strong LLM to summarise
   the extraction in natural language (used for the UI summary card).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import *


# ---------------------------------------------------------------------------
# 1. VLM OCR prompt -- used when a PDF page has no text layer
# ---------------------------------------------------------------------------

VLM_OCR_PROMPT = """Du bist ein medizinischer Dokumenten-Scanner. Deine Aufgabe ist es, den gesamten Text aus diesem gescannten medizinischen Dokument zu extrahieren.

WICHTIGE REGELN:
- Extrahiere ALLEN sichtbaren Text, Zeile fuer Zeile
- Behalte die urspruengliche Struktur und Formatierung bei
- Medizinische Fachbegriffe muessen exakt uebernommen werden
- ICD-Codes, OPS-Codes und Medikamentennamen EXAKT abschreiben
- Tabellen als strukturierten Text wiedergeben
- Unleserliche Stellen mit [unleserlich] markieren
- KEINE Interpretation oder Zusammenfassung -- nur den reinen Text

Bitte extrahiere jetzt den vollstaendigen Text aus dem Dokument."""


# ---------------------------------------------------------------------------
# 2. Document classification prompt
# ---------------------------------------------------------------------------

GERMAN_DOCUMENT_CLASSIFICATION_PROMPT = """Du bist ein Experte fuer medizinische Dokumentenklassifikation in deutschen Krankenhaeusern.

Klassifiziere das folgende Dokument in GENAU EINE der folgenden Kategorien:

- Arztbrief (Entlassbrief / Discharge Letter)
- Befundbericht (Diagnostic Report / Findings)
- OP-Bericht (Surgical Report)
- Laborbericht (Laboratory Report)
- Pathologiebericht (Pathology Report)
- Radiologiebericht (Radiology Report)
- Konsiliarbericht (Consultation Report)
- Pflegebericht (Nursing Report)
- Verlegungsbericht (Transfer Report)
- Sonstiges (Other)

Antworte im folgenden JSON-Format:
{{
    "category": "<Kategorie>",
    "confidence": <0.0-1.0>,
    "reasoning": "<Kurze Begruendung>"
}}

DOKUMENT:
{text}

KLASSIFIKATION:"""


# ---------------------------------------------------------------------------
# 3. PIGEON extraction prompt -- the core of the pipeline
#
# This is the prompt that the fine-tuned PIGEON model was trained on.
# It defines the exact JSON schema the model should produce.  The schema
# mirrors the FHIR-inspired structure used during training data generation
# (Stage 2) and is consumed downstream by the FHIR R4 converter (Stage 8).
# ---------------------------------------------------------------------------

ARZTBRIEF_EXTRACTION_PROMPT = """Du bist ein medizinischer Experte. Extrahiere alle strukturierten medizinischen Informationen aus dem folgenden deutschen Arztbrief.

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt im folgenden Format. Verwende null fuer fehlende Informationen:

{{
    "patient": {{
        "name": "<Patientenname>",
        "date_of_birth": "<Geburtsdatum DD.MM.YYYY>",
        "gender": "<maennlich/weiblich/divers>",
        "patient_id": "<Patienten-ID falls vorhanden>"
    }},
    "encounter": {{
        "admission_date": "<Aufnahmedatum DD.MM.YYYY>",
        "discharge_date": "<Entlassdatum DD.MM.YYYY>",
        "ward": "<Station>",
        "attending_physician": "<Behandelnder Arzt>",
        "encounter_type": "<stationaer/ambulant/teilstationaer>"
    }},
    "diagnoses": [
        {{
            "name": "<Diagnosename>",
            "icd10gm_code": "<ICD-10-GM Code>",
            "type": "<Hauptdiagnose/Nebendiagnose>",
            "body_site": "<Koerperregion falls angegeben>",
            "certainty": "<gesichert/Verdacht auf/Zustand nach/ausgeschlossen>"
        }}
    ],
    "procedures": [
        {{
            "name": "<Prozedurname>",
            "ops_code": "<OPS Code>",
            "date": "<Datum DD.MM.YYYY>",
            "body_site": "<Koerperregion>"
        }}
    ],
    "medications": [
        {{
            "medication_name": "<Medikamentenname>",
            "atc_code": "<ATC Code>",
            "dosage": "<Dosierung>",
            "frequency": "<Haeufigkeit>",
            "route": "<Verabreichungsweg>",
            "status": "<aktiv/abgesetzt/pausiert>"
        }}
    ],
    "vital_signs": [
        {{
            "type": "<Vitalzeichen-Typ>",
            "value": "<Wert>",
            "unit": "<Einheit>",
            "date": "<Datum DD.MM.YYYY>"
        }}
    ],
    "laboratory_results": [
        {{
            "test_name": "<Labortest>",
            "value": "<Wert>",
            "unit": "<Einheit>",
            "reference_range": "<Referenzbereich>",
            "date": "<Datum DD.MM.YYYY>",
            "interpretation": "<normal/erhoeht/erniedrigt/kritisch>"
        }}
    ],
    "tumor_markers": [
        {{
            "marker_name": "<Tumormarker-Name>",
            "value": "<Wert>",
            "unit": "<Einheit>",
            "date": "<Datum DD.MM.YYYY>",
            "interpretation": "<normal/erhoeht/erniedrigt>"
        }}
    ],
    "free_text": {{
        "anamnesis": "<Anamnese-Freitext>",
        "findings": "<Befunde-Freitext>",
        "therapy": "<Therapie-Freitext>",
        "epicrisis": "<Epikrise-Freitext>",
        "recommendations": "<Empfehlungen-Freitext>",
        "diagnoses": [
            {{
                "name": "<Freitext-Diagnose>",
                "icd10gm_code": "<ICD-10-GM Code>"
            }}
        ]
    }}
}}

ARZTBRIEF:
{text}

EXTRAKTION:"""


# ---------------------------------------------------------------------------
# 4. Summary extraction prompt -- natural-language summary for the UI
# ---------------------------------------------------------------------------

GERMAN_SUMMARY_EXTRACTION_PROMPT = """Du bist ein medizinischer Dokumentationsexperte. Erstelle eine praeegnante, strukturierte Zusammenfassung der folgenden medizinischen Extraktion.

Die Zusammenfassung soll fuer Aerzte verstaendlich sein und die wichtigsten klinischen Informationen hervorheben.

EXTRAHIERTE DATEN:
{extraction_json}

DOKUMENTKATEGORIE: {category}

Erstelle eine Zusammenfassung mit folgender Struktur:

1. **Patienteninformation**: Name, Alter, Geschlecht
2. **Aufenthalt**: Zeitraum, Station, Art
3. **Hauptdiagnosen**: Die wichtigsten Diagnosen mit ICD-Codes
4. **Durchgefuehrte Prozeduren**: Wesentliche Eingriffe
5. **Aktuelle Medikation**: Relevante Medikamente
6. **Relevante Befunde**: Auffaellige Labor- oder Vitalwerte
7. **Empfehlungen**: Wesentliche Empfehlungen bei Entlassung

Halte die Zusammenfassung praegnant aber vollstaendig. Maximal 500 Woerter.

ZUSAMMENFASSUNG:"""
