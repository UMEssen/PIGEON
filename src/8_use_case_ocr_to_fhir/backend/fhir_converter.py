"""
Stage 8: FHIR R4 resource converter.

This module converts the structured extraction produced by the PIGEON
model into a valid **FHIR R4** Bundle containing interlinked clinical
resources.

Key FHIR R4 specifics (vs R5)
------------------------------
- **Encounter.class** is a single ``Coding`` object (not an array of
  ``CodeableConcept`` as in R5).
- **MedicationStatement** uses ``medicationCodeableConcept`` for inline
  medication references and ``context`` for encounter links (not
  ``encounter`` as in R5).
- Profile URLs use the R4 base ``http://hl7.org/fhir/StructureDefinition/``.

The converter produces these FHIR R4 resource types:
- Patient
- Encounter
- Condition (diagnoses)
- Procedure
- MedicationStatement
- Observation (vital signs, laboratory results, tumor markers)

All resources are wrapped in a ``Bundle`` of type ``collection``.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import *

import re
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FHIRConverter:
    """Convert a PIGEON extraction dict into FHIR R4 resources.

    Usage::

        converter = FHIRConverter()
        result = converter.convert(extraction_data)

        bundle = result["bundle"]       # Full FHIR R4 Bundle
        stats  = result["statistics"]   # Summary counts
        resources = result["resources"] # Flat list of resources
    """

    # FHIR R4 base URL for standard profiles
    PROFILE_BASE = "http://hl7.org/fhir/StructureDefinition"

    # V2 encounter class value set (used in R4 Encounter.class)
    ENCOUNTER_CLASS_SYSTEM = "http://terminology.hl7.org/CodeSystem/v3-ActCode"

    # ICD-10-GM system URL
    ICD10GM_SYSTEM = "http://fhir.de/CodeSystem/bfarm/icd-10-gm"

    # OPS system URL
    OPS_SYSTEM = "http://fhir.de/CodeSystem/bfarm/ops"

    # ATC system URL
    ATC_SYSTEM = "http://fhir.de/CodeSystem/bfarm/atc"

    def __init__(self):
        """Initialise the converter.  No external dependencies needed."""
        self._patient_id: Optional[str] = None
        self._encounter_id: Optional[str] = None
        self._resources: List[Dict[str, Any]] = []
        self._statistics: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------

    def convert(
        self,
        extraction_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert a PIGEON extraction into FHIR R4 resources.

        Parameters
        ----------
        extraction_data : dict
            The structured extraction from the PIGEON model (or the
            RAG-corrected version thereof).

        Returns
        -------
        dict
            Keys:
            - ``resources``: flat list of FHIR resource dicts
            - ``statistics``: counts by resource type
            - ``bundle``: the complete FHIR R4 Bundle
        """
        # Reset state for this conversion
        self._resources = []
        self._statistics = {
            "Patient": 0,
            "Encounter": 0,
            "Condition": 0,
            "Procedure": 0,
            "MedicationStatement": 0,
            "Observation": 0,
        }

        # Step 1: Patient (anchor for all other references)
        patient = self._create_patient(extraction_data.get("patient", {}))
        if patient:
            self._resources.append(patient)
            self._patient_id = patient["id"]
            self._statistics["Patient"] += 1

        # Step 2: Encounter
        encounter = self._create_encounter(extraction_data.get("encounter", {}))
        if encounter:
            self._resources.append(encounter)
            self._encounter_id = encounter["id"]
            self._statistics["Encounter"] += 1

        # Step 3: Conditions (diagnoses)
        conditions = self._create_conditions(extraction_data.get("diagnoses", []))
        self._resources.extend(conditions)
        self._statistics["Condition"] += len(conditions)

        # Also process free-text diagnoses if present
        free_text = extraction_data.get("free_text", {})
        if isinstance(free_text, dict):
            ft_diagnoses = free_text.get("diagnoses", [])
            if ft_diagnoses:
                ft_conditions = self._create_conditions(ft_diagnoses, source="free_text")
                self._resources.extend(ft_conditions)
                self._statistics["Condition"] += len(ft_conditions)

        # Step 4: Procedures
        procedures = self._create_procedures(extraction_data.get("procedures", []))
        self._resources.extend(procedures)
        self._statistics["Procedure"] += len(procedures)

        # Step 5: MedicationStatements
        med_statements = self._create_medication_statements(
            extraction_data.get("medications", [])
        )
        self._resources.extend(med_statements)
        self._statistics["MedicationStatement"] += len(med_statements)

        # Step 6: Observations (vital signs)
        vitals = self._create_observations(
            extraction_data.get("vital_signs", []),
            category_code="vital-signs",
            category_display="Vital Signs",
        )
        self._resources.extend(vitals)
        self._statistics["Observation"] += len(vitals)

        # Step 7: Observations (laboratory results)
        labs = self._create_observations(
            extraction_data.get("laboratory_results", []),
            category_code="laboratory",
            category_display="Laboratory",
        )
        self._resources.extend(labs)
        self._statistics["Observation"] += len(labs)

        # Step 8: Observations (tumor markers)
        tumors = self._create_tumor_observations(
            extraction_data.get("tumor_markers", [])
        )
        self._resources.extend(tumors)
        self._statistics["Observation"] += len(tumors)

        # Build the Bundle
        bundle = self.to_bundle()

        return {
            "resources": self._resources,
            "statistics": self._statistics,
            "bundle": bundle,
        }

    def to_bundle(self) -> Dict[str, Any]:
        """Wrap all generated resources in a FHIR R4 Bundle.

        Returns a ``collection``-type Bundle::

            {
                "resourceType": "Bundle",
                "id": "<uuid>",
                "type": "collection",
                "timestamp": "<ISO 8601>",
                "entry": [{"resource": {...}}, ...]
            }
        """
        return {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "type": "collection",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "entry": [
                {"resource": resource} for resource in self._resources
            ],
        }

    # ------------------------------------------------------------------
    # Resource builders
    # ------------------------------------------------------------------

    def _create_patient(self, patient_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a FHIR R4 Patient resource."""
        if not patient_data:
            # Generate a minimal patient even without data
            return {
                "resourceType": "Patient",
                "id": str(uuid.uuid4()),
                "meta": {
                    "profile": [f"{self.PROFILE_BASE}/Patient"],
                },
            }

        patient_id = str(uuid.uuid4())
        resource: Dict[str, Any] = {
            "resourceType": "Patient",
            "id": patient_id,
            "meta": {
                "profile": [f"{self.PROFILE_BASE}/Patient"],
            },
        }

        # Name
        name = patient_data.get("name")
        if name and name.lower() not in ("null", "none", ""):
            parts = name.strip().split()
            family = parts[-1] if parts else name
            given = parts[:-1] if len(parts) > 1 else []
            resource["name"] = [
                {
                    "use": "official",
                    "family": family,
                    "given": given if given else [name],
                }
            ]

        # Date of birth
        dob = patient_data.get("date_of_birth")
        if dob:
            parsed = self._parse_date(dob)
            if parsed:
                resource["birthDate"] = parsed

        # Gender
        gender = patient_data.get("gender", "")
        if gender:
            gender_map = {
                "maennlich": "male",
                "männlich": "male",
                "male": "male",
                "m": "male",
                "weiblich": "female",
                "female": "female",
                "w": "female",
                "f": "female",
                "divers": "other",
                "other": "other",
                "d": "other",
            }
            resource["gender"] = gender_map.get(gender.lower().strip(), "unknown")

        # Patient ID as identifier
        pid = patient_data.get("patient_id")
        if pid and str(pid).lower() not in ("null", "none", ""):
            resource["identifier"] = [
                {
                    "system": "http://hospital.example.org/patient-id",
                    "value": str(pid),
                }
            ]

        return resource

    def _create_encounter(self, encounter_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a FHIR R4 Encounter resource.

        FHIR R4 key difference: ``Encounter.class`` is a single Coding
        object, NOT an array of CodeableConcept (that is R5).

        R4 format::

            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "IMP",
                "display": "inpatient encounter"
            }
        """
        if not encounter_data:
            return None

        encounter_id = str(uuid.uuid4())
        resource: Dict[str, Any] = {
            "resourceType": "Encounter",
            "id": encounter_id,
            "meta": {
                "profile": [f"{self.PROFILE_BASE}/Encounter"],
            },
            "status": "finished",
        }

        # Class -- FHIR R4: single Coding (NOT array, NOT CodeableConcept)
        encounter_type = encounter_data.get("encounter_type", "stationaer")
        class_map = {
            "stationaer": ("IMP", "inpatient encounter"),
            "stationär": ("IMP", "inpatient encounter"),
            "ambulant": ("AMB", "ambulatory"),
            "teilstationaer": ("SS", "short stay"),
            "teilstationär": ("SS", "short stay"),
        }
        code, display = class_map.get(
            (encounter_type or "").lower().strip(),
            ("IMP", "inpatient encounter"),
        )
        resource["class"] = {
            "system": self.ENCOUNTER_CLASS_SYSTEM,
            "code": code,
            "display": display,
        }

        # Subject reference (Patient)
        if self._patient_id:
            resource["subject"] = {
                "reference": f"Patient/{self._patient_id}",
            }

        # Period (admission -> discharge)
        period: Dict[str, str] = {}
        admission = encounter_data.get("admission_date")
        if admission:
            parsed = self._parse_date(admission)
            if parsed:
                period["start"] = parsed

        discharge = encounter_data.get("discharge_date")
        if discharge:
            parsed = self._parse_date(discharge)
            if parsed:
                period["end"] = parsed

        if period:
            resource["period"] = period

        # Location (ward)
        ward = encounter_data.get("ward")
        if ward and str(ward).lower() not in ("null", "none", ""):
            resource["location"] = [
                {
                    "location": {
                        "display": str(ward),
                    },
                    "status": "active",
                }
            ]

        # Participant (attending physician)
        physician = encounter_data.get("attending_physician")
        if physician and str(physician).lower() not in ("null", "none", ""):
            resource["participant"] = [
                {
                    "individual": {
                        "display": str(physician),
                    },
                }
            ]

        return resource

    def _create_conditions(
        self,
        diagnoses: List[Dict[str, Any]],
        source: str = "structured",
    ) -> List[Dict[str, Any]]:
        """Create FHIR R4 Condition resources from diagnoses.

        If the extraction contains RAG-corrected codes (``icd10gm_code_rag``),
        the corrected code is used as the primary code and a note is added
        documenting the correction.
        """
        conditions = []
        if not isinstance(diagnoses, list):
            return conditions

        for dx in diagnoses:
            if not isinstance(dx, dict):
                continue

            name = dx.get("name") or dx.get("official_name", "")
            if not name or str(name).lower() in ("null", "none", ""):
                continue

            condition_id = str(uuid.uuid4())
            resource: Dict[str, Any] = {
                "resourceType": "Condition",
                "id": condition_id,
                "meta": {
                    "profile": [f"{self.PROFILE_BASE}/Condition"],
                },
            }

            # Code -- prefer RAG-corrected code if available
            original_code = dx.get("icd10gm_code", "")
            rag_code = dx.get("icd10gm_code_rag", "")
            effective_code = rag_code if rag_code else original_code

            coding: Dict[str, Any] = {"text": str(name)}
            if effective_code and str(effective_code).lower() not in ("null", "none", ""):
                coding["coding"] = [
                    {
                        "system": self.ICD10GM_SYSTEM,
                        "code": str(effective_code),
                        "display": str(name),
                    }
                ]
            resource["code"] = coding

            # Track RAG correction in a note
            if rag_code and original_code and rag_code != original_code:
                resource["note"] = [
                    {
                        "text": (
                            f"ICD code corrected by RAG: "
                            f"{original_code} -> {rag_code}"
                        ),
                    }
                ]

            # Category (encounter-diagnosis for both main and secondary)
            dx_type = dx.get("type", "")
            category_code = "encounter-diagnosis"
            resource["category"] = [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                            "code": category_code,
                            "display": "Encounter Diagnosis",
                        }
                    ]
                }
            ]

            # Clinical status
            certainty = dx.get("certainty", "gesichert")
            if certainty and "ausgeschlossen" in str(certainty).lower():
                resource["verificationStatus"] = {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                            "code": "refuted",
                        }
                    ]
                }
            elif certainty and "verdacht" in str(certainty).lower():
                resource["verificationStatus"] = {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                            "code": "provisional",
                        }
                    ]
                }
            else:
                resource["verificationStatus"] = {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                            "code": "confirmed",
                        }
                    ]
                }
                resource["clinicalStatus"] = {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
                        }
                    ]
                }

            # Body site
            body_site = dx.get("body_site")
            if body_site and str(body_site).lower() not in ("null", "none", ""):
                resource["bodySite"] = [{"text": str(body_site)}]

            # Subject and encounter references
            if self._patient_id:
                resource["subject"] = {"reference": f"Patient/{self._patient_id}"}
            if self._encounter_id:
                resource["encounter"] = {"reference": f"Encounter/{self._encounter_id}"}

            conditions.append(resource)

        return conditions

    def _create_procedures(
        self,
        procedures: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Create FHIR R4 Procedure resources."""
        results = []
        if not isinstance(procedures, list):
            return results

        for proc in procedures:
            if not isinstance(proc, dict):
                continue

            name = proc.get("name") or proc.get("procedure_name", "")
            if not name or str(name).lower() in ("null", "none", ""):
                continue

            procedure_id = str(uuid.uuid4())
            resource: Dict[str, Any] = {
                "resourceType": "Procedure",
                "id": procedure_id,
                "meta": {
                    "profile": [f"{self.PROFILE_BASE}/Procedure"],
                },
                "status": "completed",
            }

            # Code
            ops_code = proc.get("ops_code", "")
            coding: Dict[str, Any] = {"text": str(name)}
            if ops_code and str(ops_code).lower() not in ("null", "none", ""):
                coding["coding"] = [
                    {
                        "system": self.OPS_SYSTEM,
                        "code": str(ops_code),
                        "display": str(name),
                    }
                ]
            resource["code"] = coding

            # Performed date
            date = proc.get("date")
            if date:
                parsed = self._parse_date(date)
                if parsed:
                    resource["performedDateTime"] = parsed

            # Body site
            body_site = proc.get("body_site")
            if body_site and str(body_site).lower() not in ("null", "none", ""):
                resource["bodySite"] = [{"text": str(body_site)}]

            # References
            if self._patient_id:
                resource["subject"] = {"reference": f"Patient/{self._patient_id}"}
            if self._encounter_id:
                resource["encounter"] = {"reference": f"Encounter/{self._encounter_id}"}

            results.append(resource)

        return results

    def _create_medication_statements(
        self,
        medications: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Create FHIR R4 MedicationStatement resources.

        FHIR R4 key differences from R5:
        - Use ``medicationCodeableConcept`` (not ``medicationReference``
          or the R5 ``medication[x]`` pattern with CodeableReference).
        - Use ``context`` for the encounter reference (not ``encounter``
          which was introduced in R5).
        """
        results = []
        if not isinstance(medications, list):
            return results

        for med in medications:
            if not isinstance(med, dict):
                continue

            name = med.get("medication_name") or med.get("name", "")
            if not name or str(name).lower() in ("null", "none", ""):
                continue

            med_id = str(uuid.uuid4())
            resource: Dict[str, Any] = {
                "resourceType": "MedicationStatement",
                "id": med_id,
                "meta": {
                    "profile": [f"{self.PROFILE_BASE}/MedicationStatement"],
                },
                "status": "active",
            }

            # Medication -- FHIR R4: medicationCodeableConcept
            medication_cc: Dict[str, Any] = {"text": str(name)}
            atc_code = med.get("atc_code", "")
            if atc_code and str(atc_code).lower() not in ("null", "none", ""):
                medication_cc["coding"] = [
                    {
                        "system": self.ATC_SYSTEM,
                        "code": str(atc_code),
                        "display": str(name),
                    }
                ]
            resource["medicationCodeableConcept"] = medication_cc

            # Status mapping
            status = med.get("status", "aktiv")
            status_map = {
                "aktiv": "active",
                "active": "active",
                "abgesetzt": "stopped",
                "stopped": "stopped",
                "pausiert": "on-hold",
                "on-hold": "on-hold",
            }
            resource["status"] = status_map.get(
                (status or "").lower().strip(), "active"
            )

            # Dosage
            dosage_parts = []
            dosage_text = med.get("dosage", "")
            frequency = med.get("frequency", "")
            route = med.get("route", "")

            if dosage_text and str(dosage_text).lower() not in ("null", "none", ""):
                dosage_parts.append(str(dosage_text))
            if frequency and str(frequency).lower() not in ("null", "none", ""):
                dosage_parts.append(str(frequency))
            if route and str(route).lower() not in ("null", "none", ""):
                dosage_parts.append(str(route))

            if dosage_parts:
                resource["dosage"] = [{"text": ", ".join(dosage_parts)}]

            # Subject reference
            if self._patient_id:
                resource["subject"] = {"reference": f"Patient/{self._patient_id}"}

            # Context reference (FHIR R4 uses "context", not "encounter")
            if self._encounter_id:
                resource["context"] = {"reference": f"Encounter/{self._encounter_id}"}

            results.append(resource)

        return results

    def _create_observations(
        self,
        items: List[Dict[str, Any]],
        category_code: str,
        category_display: str,
    ) -> List[Dict[str, Any]]:
        """Create FHIR R4 Observation resources (vital signs or lab results)."""
        results = []
        if not isinstance(items, list):
            return results

        for item in items:
            if not isinstance(item, dict):
                continue

            # Get the observation name / type
            name = (
                item.get("test_name")
                or item.get("type")
                or item.get("marker_name")
                or ""
            )
            if not name or str(name).lower() in ("null", "none", ""):
                continue

            obs_id = str(uuid.uuid4())
            resource: Dict[str, Any] = {
                "resourceType": "Observation",
                "id": obs_id,
                "meta": {
                    "profile": [f"{self.PROFILE_BASE}/Observation"],
                },
                "status": "final",
            }

            # Category
            resource["category"] = [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": category_code,
                            "display": category_display,
                        }
                    ]
                }
            ]

            # Code
            resource["code"] = {"text": str(name)}

            # Value
            value = item.get("value", "")
            unit = item.get("unit", "")
            if value and str(value).lower() not in ("null", "none", ""):
                # Try to parse as a number for valueQuantity
                try:
                    numeric = float(str(value).replace(",", "."))
                    resource["valueQuantity"] = {
                        "value": numeric,
                        "unit": str(unit) if unit else "",
                    }
                except (ValueError, TypeError):
                    # Non-numeric value -> valueString
                    resource["valueString"] = str(value)

            # Reference range (lab results)
            ref_range = item.get("reference_range", "")
            if ref_range and str(ref_range).lower() not in ("null", "none", ""):
                resource["referenceRange"] = [{"text": str(ref_range)}]

            # Interpretation
            interpretation = item.get("interpretation", "")
            if interpretation and str(interpretation).lower() not in ("null", "none", ""):
                interp_map = {
                    "normal": ("N", "Normal"),
                    "erhoeht": ("H", "High"),
                    "erhoht": ("H", "High"),
                    "erhöht": ("H", "High"),
                    "erniedrigt": ("L", "Low"),
                    "kritisch": ("AA", "Critical abnormal"),
                }
                code_val, display_val = interp_map.get(
                    str(interpretation).lower().strip(),
                    ("N", "Normal"),
                )
                resource["interpretation"] = [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                "code": code_val,
                                "display": display_val,
                            }
                        ]
                    }
                ]

            # Effective date
            date = item.get("date")
            if date:
                parsed = self._parse_date(date)
                if parsed:
                    resource["effectiveDateTime"] = parsed

            # References
            if self._patient_id:
                resource["subject"] = {"reference": f"Patient/{self._patient_id}"}
            if self._encounter_id:
                resource["encounter"] = {"reference": f"Encounter/{self._encounter_id}"}

            results.append(resource)

        return results

    def _create_tumor_observations(
        self,
        tumor_markers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Create FHIR R4 Observation resources for tumor markers.

        Tumor markers get their own category
        (``http://snomed.info/sct|116221001 = Tumor marker``).
        """
        if not isinstance(tumor_markers, list) or not tumor_markers:
            return []

        observations = self._create_observations(
            tumor_markers,
            category_code="laboratory",
            category_display="Laboratory",
        )

        # Add SNOMED tumor marker coding to each observation's code
        for obs in observations:
            if "code" in obs:
                existing_text = obs["code"].get("text", "")
                obs["code"]["coding"] = [
                    {
                        "system": "http://snomed.info/sct",
                        "code": "116221001",
                        "display": f"Tumor marker: {existing_text}",
                    }
                ]

        return observations

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(date_str: str) -> Optional[str]:
        """Parse a German DD.MM.YYYY date string into FHIR yyyy-MM-dd format.

        Also handles ISO dates (yyyy-MM-dd) as a passthrough.

        Returns ``None`` if the string cannot be parsed.
        """
        if not date_str or str(date_str).lower() in ("null", "none", ""):
            return None

        date_str = str(date_str).strip()

        # DD.MM.YYYY (German format)
        match = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", date_str)
        if match:
            day, month, year = match.groups()
            try:
                dt = datetime(int(year), int(month), int(day))
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                return None

        # yyyy-MM-dd (ISO passthrough)
        match = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", date_str)
        if match:
            year, month, day = match.groups()
            try:
                dt = datetime(int(year), int(month), int(day))
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                return None

        # MM/YYYY or YYYY (partial dates)
        match = re.match(r"(\d{1,2})/(\d{4})", date_str)
        if match:
            month, year = match.groups()
            return f"{year}-{int(month):02d}"

        match = re.match(r"^(\d{4})$", date_str)
        if match:
            return match.group(1)

        return None
