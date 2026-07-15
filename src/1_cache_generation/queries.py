"""
Stage 1 -- SQL Queries for FHIR Resource Extraction
====================================================

This module contains every SQL query used by the PIGEON pipeline to extract
structured clinical data from a FHIR-compliant PostgreSQL database (fhirmetrics /
fhir-ql schema).

WHY THIS EXISTS
---------------
Hospital FHIR servers store data as deeply-nested resources.  To build a
discharge-letter dataset we need to *flatten* those resources into tabular
CSV files -- one per encounter or per patient.  Each query below does that
flattening for a specific FHIR resource type.

DATABASE CONVENTIONS
--------------------
The target database uses two custom helper functions that appear throughout:

    fmx_code(column)    -- Extracts the coded value from an internal FHIRmetrics
                           representation.  Think of it as "unwrap the system URL".
    fhirql_code(column) -- Similar extraction used by the FHIR-QL query layer,
                           typically for coded fields like ICD-10 or OPS codes.

Both are server-side PostgreSQL functions provided by the FHIRmetrics engine.
If you are adapting these queries to a different FHIR store you will need to
replace them with equivalent expressions for your schema.

PLACEHOLDER TOKENS
------------------
    (batch_ids)  -- Used in observation queries.  Replace with a comma-separated
                    list of quoted internal _id values for the observation batch.
    (batches)    -- Used in all other queries.  Same idea, different token name
                    kept for backward compatibility with the original pipeline.

USAGE
-----
    from queries import observation_queries, condition_query, ...

    # Inject real IDs before executing:
    safe_query = condition_query.replace("(batches)", "('id1','id2','id3')")
"""

# ---------------------------------------------------------------------------
# Observation queries -- keyed by the FHIR identifier system URL
# ---------------------------------------------------------------------------
# Each observation type lives under a different system URL in the hospital's
# FHIR server.  The dictionary key IS that URL so the downloader can look up
# the right query for each batch of observation IDs grouped by system.

observation_queries: dict[str, str] = {

    # ------------------------------------------------------------------
    # SchmerzenAenderung  (Pain Assessment -- nursing documentation)
    # ------------------------------------------------------------------
    # Captures pain-change observations recorded by nursing staff.
    # Unlike vital signs this query joins bodySite to record WHERE the
    # pain was assessed (e.g. "left knee").
    #
    # Output columns:
    #   o0_id                 -- FHIR Observation.id
    #   oi0_system            -- identifier system URL
    #   occ0_display          -- code display (pain assessment type)
    #   obc0_display          -- bodySite display (anatomical location)
    #   o0_effectiveDateTime  -- when the observation was recorded
    #   p1_id                 -- FHIR Patient.id
    "https://example-hospital.org/fhir/NursingDocumentation/SchmerzenAenderung": """
        SELECT
            o0.id "o0_id",
            fmx_code (oi0.system) "oi0_system",
            occ0.display "occ0_display",
            obc0.display "obc0_display",
            lower(o0."effectiveDateTime") "o0_effectiveDateTime",
            p1.id "p1_id"
        FROM
            observation o0
            JOIN observation_identifier oi0 ON (oi0._resource = o0._id)
            JOIN observation_code oc0 ON (oc0._resource = o0._id)
            JOIN observation_code_coding occ0 ON (occ0._resource = o0._id)
            JOIN "observation_bodySite" ob0 ON (ob0._resource = o0._id)
            JOIN "observation_bodySite_coding" obc0 ON (obc0._resource = o0._id)
            JOIN observation_subject os0 ON (os0._resource = o0._id)
            JOIN patient p1 ON (
                p1._id = os0._reference_id
                and os0._reference_type = 'Patient'
            )
        WHERE
            o0._id IN (batch_ids)
            AND fmx_code (oi0.system) = 'https://example-hospital.org/fhir/NursingDocumentation/SchmerzenAenderung';
    """,

    # ------------------------------------------------------------------
    # HeartBeat  (Vital Sign)
    # ------------------------------------------------------------------
    # Standard vital-sign pattern: value + unit + timestamp.
    # All vital signs (HeartBeat, SpO2, RR, Weight, Height, Temp, BMI)
    # share the same JOIN structure -- only the system filter changes.
    #
    # Output columns:
    #   o0_id, oi0_system, o0_effectiveDateTime, ov0_value, ov0_unit, p1_id
    "https://example-hospital.org/fhir/Observation/HeartBeat": """
        SELECT
            o0.id "o0_id",
            fmx_code (oi0.system) "oi0_system",
            lower(o0."effectiveDateTime") "o0_effectiveDateTime",
            ov0.value "ov0_value",
            ov0.unit "ov0_unit",
            p1.id "p1_id"
        FROM
            observation o0
            JOIN observation_identifier oi0 ON (oi0._resource = o0._id)
            JOIN "observation_valueQuantity" ov0 ON (ov0._resource = o0._id)
            JOIN observation_subject os0 ON (os0._resource = o0._id)
            JOIN patient p1 ON (
                p1._id = os0._reference_id
                and os0._reference_type = 'Patient'
            )
        WHERE
            o0._id IN (batch_ids)
            AND fmx_code (oi0.system) = 'https://example-hospital.org/fhir/Observation/HeartBeat';
    """,

    # ------------------------------------------------------------------
    # OxygenSaturation  (SpO2 -- Vital Sign)
    # ------------------------------------------------------------------
    # Output columns:
    #   o0_id, oi0_system, o0_effectiveDateTime, ov0_value, ov0_unit, p1_id
    "https://example-hospital.org/fhir/Observation/OxygenSaturation": """
        SELECT
            o0.id "o0_id",
            fmx_code (oi0.system) "oi0_system",
            lower(o0."effectiveDateTime") "o0_effectiveDateTime",
            ov0.value "ov0_value",
            ov0.unit "ov0_unit",
            p1.id "p1_id"
        FROM
            observation o0
            JOIN observation_identifier oi0 ON (oi0._resource = o0._id)
            JOIN "observation_valueQuantity" ov0 ON (ov0._resource = o0._id)
            JOIN observation_subject os0 ON (os0._resource = o0._id)
            JOIN patient p1 ON (
                p1._id = os0._reference_id
                and os0._reference_type = 'Patient'
            )
        WHERE
            o0._id IN (batch_ids)
            AND fmx_code (oi0.system) = 'https://example-hospital.org/fhir/Observation/OxygenSaturation';
    """,

    # ------------------------------------------------------------------
    # RespiratoryRate  (Vital Sign)
    # ------------------------------------------------------------------
    # Output columns:
    #   o0_id, oi0_system, o0_effectiveDateTime, ov0_value, ov0_unit, p1_id
    "https://example-hospital.org/fhir/Observation/RespiratoryRate": """
        SELECT
            o0.id "o0_id",
            fmx_code (oi0.system) "oi0_system",
            lower(o0."effectiveDateTime") "o0_effectiveDateTime",
            ov0.value "ov0_value",
            ov0.unit "ov0_unit",
            p1.id "p1_id"
        FROM
            observation o0
            JOIN observation_identifier oi0 ON (oi0._resource = o0._id)
            JOIN "observation_valueQuantity" ov0 ON (ov0._resource = o0._id)
            JOIN observation_subject os0 ON (os0._resource = o0._id)
            JOIN patient p1 ON (
                p1._id = os0._reference_id
                and os0._reference_type = 'Patient'
            )
        WHERE
            o0._id IN (batch_ids)
            AND fmx_code (oi0.system) = 'https://example-hospital.org/fhir/Observation/RespiratoryRate';
    """,

    # ------------------------------------------------------------------
    # BodyWeight  (Vital Sign)
    # ------------------------------------------------------------------
    # Output columns:
    #   o0_id, oi0_system, o0_effectiveDateTime, ov0_value, ov0_unit, p1_id
    "https://example-hospital.org/fhir/Observation/BodyWeight": """
        SELECT
            o0.id "o0_id",
            fmx_code (oi0.system) "oi0_system",
            lower(o0."effectiveDateTime") "o0_effectiveDateTime",
            ov0.value "ov0_value",
            ov0.unit "ov0_unit",
            p1.id "p1_id"
        FROM
            observation o0
            JOIN observation_identifier oi0 ON (oi0._resource = o0._id)
            JOIN "observation_valueQuantity" ov0 ON (ov0._resource = o0._id)
            JOIN observation_subject os0 ON (os0._resource = o0._id)
            JOIN patient p1 ON (
                p1._id = os0._reference_id
                and os0._reference_type = 'Patient'
            )
        WHERE
            o0._id IN (batch_ids)
            AND fmx_code (oi0.system) = 'https://example-hospital.org/fhir/Observation/BodyWeight';
    """,

    # ------------------------------------------------------------------
    # BodyHeight  (Vital Sign)
    # ------------------------------------------------------------------
    # Output columns:
    #   o0_id, oi0_system, o0_effectiveDateTime, ov0_value, ov0_unit, p1_id
    "https://example-hospital.org/fhir/Observation/BodyHeight": """
        SELECT
            o0.id "o0_id",
            fmx_code (oi0.system) "oi0_system",
            lower(o0."effectiveDateTime") "o0_effectiveDateTime",
            ov0.value "ov0_value",
            ov0.unit "ov0_unit",
            p1.id "p1_id"
        FROM
            observation o0
            JOIN observation_identifier oi0 ON (oi0._resource = o0._id)
            JOIN "observation_valueQuantity" ov0 ON (ov0._resource = o0._id)
            JOIN observation_subject os0 ON (os0._resource = o0._id)
            JOIN patient p1 ON (
                p1._id = os0._reference_id
                and os0._reference_type = 'Patient'
            )
        WHERE
            o0._id IN (batch_ids)
            AND fmx_code (oi0.system) = 'https://example-hospital.org/fhir/Observation/BodyHeight';
    """,

    # ------------------------------------------------------------------
    # BodyTemperature  (Vital Sign)
    # ------------------------------------------------------------------
    # Output columns:
    #   o0_id, oi0_system, o0_effectiveDateTime, ov0_value, ov0_unit, p1_id
    "https://example-hospital.org/fhir/Observation/BodyTemperature": """
        SELECT
            o0.id "o0_id",
            fmx_code (oi0.system) "oi0_system",
            lower(o0."effectiveDateTime") "o0_effectiveDateTime",
            ov0.value "ov0_value",
            ov0.unit "ov0_unit",
            p1.id "p1_id"
        FROM
            observation o0
            JOIN observation_identifier oi0 ON (oi0._resource = o0._id)
            JOIN "observation_valueQuantity" ov0 ON (ov0._resource = o0._id)
            JOIN observation_subject os0 ON (os0._resource = o0._id)
            JOIN patient p1 ON (
                p1._id = os0._reference_id
                and os0._reference_type = 'Patient'
            )
        WHERE
            o0._id IN (batch_ids)
            AND fmx_code (oi0.system) = 'https://example-hospital.org/fhir/Observation/BodyTemperature';
    """,

    # ------------------------------------------------------------------
    # BodyMassIndex  (Vital Sign)
    # ------------------------------------------------------------------
    # Output columns:
    #   o0_id, oi0_system, o0_effectiveDateTime, ov0_value, ov0_unit, p1_id
    "https://example-hospital.org/fhir/Observation/BodyMassIndex": """
        SELECT
            o0.id "o0_id",
            fmx_code (oi0.system) "oi0_system",
            lower(o0."effectiveDateTime") "o0_effectiveDateTime",
            ov0.value "ov0_value",
            ov0.unit "ov0_unit",
            p1.id "p1_id"
        FROM
            observation o0
            JOIN observation_identifier oi0 ON (oi0._resource = o0._id)
            JOIN "observation_valueQuantity" ov0 ON (ov0._resource = o0._id)
            JOIN observation_subject os0 ON (os0._resource = o0._id)
            JOIN patient p1 ON (
                p1._id = os0._reference_id
                and os0._reference_type = 'Patient'
            )
        WHERE
            o0._id IN (batch_ids)
            AND fmx_code (oi0.system) = 'https://example-hospital.org/fhir/Observation/BodyMassIndex';
    """,

    # ------------------------------------------------------------------
    # BloodPressure  (Complex Component Observation)
    # ------------------------------------------------------------------
    # Blood pressure is fundamentally different from the other vital signs
    # because it uses FHIR *components* (systolic + diastolic live inside
    # a single Observation resource).  The standard relational JOINs cannot
    # unpack those components, so we fall back to LATERAL jsonb_path_query
    # to extract each component's value, unit, and LOINC coding directly
    # from the raw _json column.
    #
    # Output columns:
    #   o0_id                  -- FHIR Observation.id
    #   oi0_system             -- identifier system URL
    #   p1_id                  -- FHIR Patient.id (extracted from subject ref)
    #   o0_effectiveDateTime   -- timestamp
    #   status                 -- observation status
    #   issued                 -- when the resource was issued
    #   component_display      -- LOINC display (e.g. "Systolic blood pressure")
    #   component_code_code    -- LOINC code (e.g. "8480-6")
    #   component_code_system  -- always "http://loinc.org"
    #   ov0_value              -- numeric value (mmHg)
    #   ov0_unit               -- unit string
    #   component_value_code   -- coded unit
    #   component_value_system -- unit system
    "https://example-hospital.org/fhir/Observation/BloodPressure": """
SELECT
    o0.id AS "o0_id",
    o0._json -> 'identifier' -> 0 ->> 'system' AS "oi0_system",
    substring(o0._json -> 'subject' ->> 'reference' from 'Patient/(.*)') AS "p1_id",
    o0._json ->> 'effectiveDateTime' AS "o0_effectiveDateTime",
    o0._json ->> 'status' AS "status",
    o0._json ->> 'issued' AS "issued",
    loinc_coding_json ->> 'display' AS "component_display",
    loinc_coding_json ->> 'code' AS "component_code_code",
    loinc_coding_json ->> 'system' AS "component_code_system",
    (comp_value_json ->> 0)::numeric AS "ov0_value",
    comp_unit_json ->> 0 AS "ov0_unit",
    comp_value_code_json ->> 0 AS "component_value_code",
    comp_value_system_json ->> 0 AS "component_value_system"
FROM
    observation o0,
    LATERAL jsonb_path_query(o0._json, '$.component[*]') AS component_element,
    LATERAL jsonb_path_query(component_element, '$.valueQuantity.value') AS comp_value_json,
    LATERAL jsonb_path_query(component_element, '$.valueQuantity.unit') AS comp_unit_json,
    LATERAL jsonb_path_query(component_element, '$.valueQuantity.code') AS comp_value_code_json,
    LATERAL jsonb_path_query(component_element, '$.valueQuantity.system') AS comp_value_system_json,
    LATERAL jsonb_path_query_first(component_element, '$.code.coding[*] ? (@.system == "http://loinc.org")') AS loinc_coding_json
WHERE
    o0._id IN (batch_ids)
    AND o0._json -> 'identifier' -> 0 ->> 'system' = 'https://example-hospital.org/fhir/Observation/BloodPressure';
    """,

    # ------------------------------------------------------------------
    # pTNM  (Tumor Staging -- Tumor Documentation)
    # ------------------------------------------------------------------
    # Pathological TNM staging from the tumor board.  Uses
    # valueCodeableConcept (overall stage) AND component-level
    # valueCodeableConcept (T/N/M sub-stages).
    #
    # Output columns:
    #   o0_id                 -- FHIR Observation.id
    #   oi0_system            -- identifier system URL
    #   o0_effectiveDateTime  -- when staging was recorded
    #   ovc0_system           -- value coding system (overall stage)
    #   ovc0_display          -- value coding display (overall stage)
    #   ocvc0_system          -- component value coding system (T/N/M)
    #   ocvc0_display         -- component value coding display (T/N/M)
    "https://example-hospital.org/fhir/TumorDocumentation/pTNM": """
        SELECT
            o0.id "o0_id",
            fmx_code (oi0.system) "oi0_system",
            lower(o0."effectiveDateTime") "o0_effectiveDateTime",
            ovc0.system "ovc0_system",
            ovc0.display "ovc0_display",
            ocvc0.system "ocvc0_system",
            ocvc0.display "ocvc0_display"
        FROM
            observation o0
            JOIN observation_subject os0 ON os0._resource = o0._id
            JOIN patient p0 ON (
                p0._id = os0._reference_id
                AND os0._reference_type = 'Patient'
            )
            JOIN observation_identifier oi0 ON oi0._resource = o0._id
            JOIN "observation_valueCodeableConcept_coding" ovc0 ON ovc0._resource = o0._id
            JOIN "observation_component_valueCodeableConcept_coding" ocvc0 ON ocvc0._resource = o0._id
        WHERE
            o0._id IN (batch_ids)
            AND fmx_code (oi0.system) = 'https://example-hospital.org/fhir/TumorDocumentation/pTNM';
    """,

    # ------------------------------------------------------------------
    # Laboratory Values
    # ------------------------------------------------------------------
    # Lab results from the laboratory system.  Includes the measured value
    # AND reference ranges (low/high) so downstream stages can flag
    # abnormal results in the generated discharge letter.
    #
    # Output columns:
    #   o0_id                 -- FHIR Observation.id
    #   oi0_system            -- identifier system (laboratory URL prefix)
    #   o0_effectiveDateTime  -- when the sample was analyzed
    #   occ0.display          -- lab test name (e.g. "Hemoglobin")
    #   ov0_value             -- measured numeric value
    #   ov0_unit              -- unit (e.g. "g/dL")
    #   orl0_value            -- reference range LOW
    #   orh0_value            -- reference range HIGH
    "https://example-hospital.org/fhir/Laboratory": """
        SELECT
            o0.id "o0_id",
            fmx_code (oi0.system) "oi0_system",
            lower(o0."effectiveDateTime") "o0_effectiveDateTime",
            occ0.display "occ0.display",
            ov0.value "ov0_value",
            ov0.unit "ov0_unit",
            orl0.value "orl0_value",
            orh0.value "orh0_value"
        FROM
            observation o0
            JOIN observation_subject os0 ON os0._resource = o0._id
            JOIN patient p0 ON (
                p0._id = os0._reference_id
                AND os0._reference_type = 'Patient'
            )
            JOIN observation_identifier oi0 ON oi0._resource = o0._id
            JOIN "observation_valueQuantity" ov0 ON ov0._resource = o0._id
            JOIN "observation_referenceRange_low" orl0 ON orl0._resource = o0._id
            JOIN "observation_referenceRange_high" orh0 ON orh0._resource = o0._id
            JOIN observation_code_coding occ0 ON occ0._resource = o0._id
        WHERE
            o0._id IN (batch_ids)
            AND fmx_code (oi0.system) = 'https://example-hospital.org/fhir/Laboratory';""",
}


def get_observation_query(system_url: str) -> str | None:
    """Look up the observation query for a given identifier system URL.

    Args:
        system_url: The FHIR identifier system URL (dict key).

    Returns:
        The SQL query string, or None if the system is not recognized.
    """
    return observation_queries.get(system_url)


# ---------------------------------------------------------------------------
# Condition query  (ICD-10 diagnoses)
# ---------------------------------------------------------------------------
# Extracts diagnosis codes (ICD-10-GM), their display names, category codes
# and displays, the recorded date, and the owning patient.
#
# WHY ARRAY_AGG + GROUP BY?
# A single Condition resource can carry multiple codings (e.g. ICD-10 + alpha-ID).
# Aggregating them avoids row duplication while keeping all codes accessible.
#
# Output columns:
#   c0_id            -- FHIR Condition.id
#   ccc0_codes       -- array of diagnosis codes (ICD-10-GM)
#   ccc0_displays    -- array of diagnosis display names
#   ccc0_1_codes     -- array of category codes
#   ccc0_1_displays  -- array of category display names
#   c0_recordedDate  -- earliest recorded date
#   patient_id       -- FHIR Patient.id
#
# Placeholder: (batches) -- replace with condition internal _id values
condition_query = """
SELECT
    c0.id AS "c0_id",
    ARRAY_AGG(DISTINCT fhirql_code(ccc0.code)) AS "ccc0_codes",
    ARRAY_AGG(DISTINCT ccc0.display) AS "ccc0_displays",
    ARRAY_AGG(DISTINCT fhirql_code(ccc0_1.code)) AS "ccc0_1_codes",
    ARRAY_AGG(DISTINCT ccc0_1.display) AS "ccc0_1_displays",
    MIN(lower(c0."recordedDate")) AS "c0_recordedDate",
    p1.id as "patient_id"
FROM
    condition c0
    JOIN condition_code cc0 ON (cc0._resource = c0._id)
    JOIN condition_code_coding ccc0 ON (ccc0._resource = c0._id)
    JOIN condition_category cc0_1 ON (cc0_1._resource = c0._id)
    JOIN condition_category_coding ccc0_1 ON (ccc0_1._resource = c0._id)
    JOIN condition_subject cs0 ON (cs0._resource = c0._id)
    JOIN patient p1 ON (
        p1._id = cs0._reference_id
        AND cs0._reference_type = 'Patient'
    )
WHERE
    c0._id IN (batches)
GROUP BY
    c0.id, p1.id;
"""


# ---------------------------------------------------------------------------
# Medication query  (MedicationStatement + Medication -- optimized)
# ---------------------------------------------------------------------------
# This query uses CTEs to pre-aggregate dosage texts and medication code
# displays BEFORE joining, which dramatically reduces row explosion when a
# single MedicationStatement has multiple dosage instructions or a Medication
# carries multiple codings (e.g. ATC + PZN).
#
# Output columns:
#   m1.id               -- MedicationStatement internal _id
#   m0.id               -- Medication internal _id
#   p0.id               -- Patient internal _id
#   md0.text_list        -- JSONB array of dosage instruction texts
#   mcc0.display_list    -- JSONB array of medication name displays
#
# Placeholder: (batches) -- replace with medicationstatement internal _id values
medication_query_optimized = """
WITH dosage_aggregated AS (
  SELECT
    md0._resource AS medicationstatement_id,
    jsonb_agg(md0.text) AS dosage_text_list
  FROM
    medicationstatement_dosage md0
  WHERE md0._resource IN (batches)
  GROUP BY
    md0._resource
), coding_aggregated AS (
  SELECT
    mcc0._resource AS medication_id,
    jsonb_agg(mcc0.display) AS code_display_list
  FROM
    medication_code_coding mcc0
  GROUP BY
    mcc0._resource
)
SELECT
  m1._id AS "m1.id",
  m0._id AS "m0.id",
  p0._id AS "p0.id",
  da.dosage_text_list AS "md0.text_list",
  ca.code_display_list AS "mcc0.display_list"
FROM
  medicationstatement m1
  JOIN "medicationstatement_medicationReference" mm0 ON mm0._resource = m1._id
  JOIN medication m0 ON (
    m0._id = mm0._reference_id AND mm0._reference_type = 'Medication'
  )
  JOIN medicationstatement_subject ms0 ON ms0._resource = m1._id
  JOIN patient p0 ON (
    p0._id = ms0._reference_id AND ms0._reference_type = 'Patient'
  )
  LEFT JOIN dosage_aggregated da ON da.medicationstatement_id = m1._id
  LEFT JOIN coding_aggregated ca ON ca.medication_id = m0._id
WHERE
  m1._id IN (batches);
"""


# ---------------------------------------------------------------------------
# Procedure query  (OPS / SNOMED procedure codes)
# ---------------------------------------------------------------------------
# Extracts performed procedures with their coded representation.
# fhirql_code() unwraps the OPS or SNOMED-CT code from the FHIR coding.
#
# Output columns:
#   p0_id                -- FHIR Procedure.id
#   pcc0_display         -- procedure display name
#   pcc0_code            -- procedure code (OPS or SNOMED)
#   p0_performedDateTime -- when the procedure was performed
#   patient_id           -- FHIR Patient.id
#
# Placeholder: (batches) -- replace with procedure internal _id values
proc_query = """
SELECT
    p0.id as "p0_id",
    pcc0.display "pcc0_display",
    fhirql_code (pcc0.code) "pcc0_code",
    lower(p0."performedDateTime") "p0_performedDateTime",
    p1.id as "patient_id"
FROM
    procedure p0
    JOIN procedure_code pc0 ON (pc0._resource = p0._id)
    JOIN procedure_code_coding pcc0 ON (pcc0._resource = p0._id)
    JOIN procedure_subject ps0 ON (ps0._resource = p0._id)
    JOIN patient p1 ON (
        p1._id = ps0._reference_id
        and ps0._reference_type = 'Patient'
    )
WHERE
    p0._id IN (batches);
"""


# ---------------------------------------------------------------------------
# Patient query  (Demographics)
# ---------------------------------------------------------------------------
# Extracts patient demographics needed for the discharge letter header:
# name, birth date, gender, and full address.
#
# Output columns:
#   p0.id          -- FHIR Patient.id
#   pn0.family     -- family (last) name
#   png0_value     -- given (first) name
#   p0.birthDate   -- date of birth
#   p0.gender      -- administrative gender
#   pa0.city       -- city
#   pa0.state      -- state / Bundesland
#   pa0.country    -- country
#   pa0.line       -- street address line
#   pa0.postalCode -- postal / ZIP code
#
# Placeholder: (batches) -- replace with patient internal _id values
patient_query = """
SELECT
    p0.id "p0.id",
    pn0.family "pn0.family",
    png0.value "png0_value",
    lower(p0."birthDate") "p0.birthDate",
    fmx_code (p0.gender) "p0.gender",
    pa0.city "pa0.city",
    pa0.state "pa0.state",
    pa0.country "pa0.country",
    pal0.value "pa0.line",
    pa0."postalCode" "pa0.postalCode"
FROM
    patient p0
    JOIN patient_name pn0 ON pn0._resource = p0._id
    JOIN patient_address pa0 ON pa0._resource = p0._id
    JOIN patient_address_line pal0 ON pal0._resource = p0._id
    JOIN patient_name_given png0 ON (png0._resource = p0._id)
WHERE
    p0._id IN (batches);
"""


# ---------------------------------------------------------------------------
# Encounter query  (Hospital stay metadata)
# ---------------------------------------------------------------------------
# Extracts encounter period (admission/discharge dates), type of encounter,
# and location (ward/department) by joining through to the Location resource.
#
# WHY GROUP BY + STRING_AGG?
# An encounter can span multiple locations (transfers between wards).
# We aggregate all distinct location aliases into a single comma-separated
# string so each encounter produces exactly one output row.
#
# Output columns:
#   encounter_id          -- encounter internal _id
#   ep0_start_min         -- earliest admission timestamp
#   ep0_start_max         -- latest admission timestamp (for multi-period)
#   ep0_end_min           -- earliest discharge timestamp
#   ep0_end_max           -- latest discharge timestamp
#   distinct_etc0_display -- comma-separated encounter type displays
#   distinct_la1_value    -- comma-separated location aliases (ward names)
#
# Placeholder: (batches) -- replace with encounter internal _id values
encounter_query = """
SELECT
    e0._id AS encounter_id,
    MIN(lower(ep0.start)) AS ep0_start_min,
    MAX(lower(ep0.start)) AS ep0_start_max,
    MIN(lower(ep0.end)) AS ep0_end_min,
    MAX(lower(ep0.end)) AS ep0_end_max,
    STRING_AGG(DISTINCT etc0.display, ', ') AS distinct_etc0_display,
    STRING_AGG(DISTINCT la1.value, ', ') AS distinct_la1_value
FROM
    encounter e0
    JOIN encounter_period ep0 ON (ep0._resource = e0._id)
    JOIN encounter_type et0 ON (et0._resource = e0._id)
    JOIN encounter_type_coding etc0 ON (etc0._resource = e0._id)
    JOIN encounter_location el0 ON (el0._resource = e0._id)
    JOIN encounter_location_location ell0 ON (ell0._resource = e0._id)
    JOIN location l1 ON (
        l1._id = ell0._reference_id
        AND ell0._reference_type = 'Location'
    )
    JOIN location_alias la1 ON (la1._resource = l1._id)
WHERE
    e0._id IN (batches)
GROUP BY
    e0._id;
"""


# ---------------------------------------------------------------------------
# Encounter-to-Patient mapping query
# ---------------------------------------------------------------------------
# Simple lookup: given encounter _id(s), return the associated patient _id.
# Used to resolve which patient belongs to which encounter before running
# the patient-level queries.
#
# Output columns:
#   e0_id  -- encounter internal _id
#   p1_id  -- patient internal _id
#
# Placeholder: (batches) -- replace with encounter internal _id values
enc_to_patient_query = """
SELECT
  e0._id "e0_id",
  p1._id "p1_id"
FROM
  encounter e0
  JOIN encounter_subject es0 ON (es0._resource = e0._id)
  JOIN patient p1 ON (
    p1._id = es0._reference_id
    and es0._reference_type = 'Patient'
  )
  WHERE e0._id IN (batches);
"""


# ---------------------------------------------------------------------------
# TNM detailed query  (patient-based, for tumor documentation cache)
# ---------------------------------------------------------------------------
# A more detailed TNM query that filters by specific pTNM sub-stage systems
# (T, N, M) and uses patient _id as the batch key instead of observation _id.
# Used in the patient-based download mode for oncology-specific caching.
#
# Output columns:
#   p1_id           -- patient internal _id
#   o0_id           -- FHIR Observation.id
#   occc0_display   -- component code display (e.g. "Primary tumor.pathology")
#   occc0_code      -- component code
#   ocvc0_system    -- component value system (pTNM/T, pTNM/N, or pTNM/M)
#   ocvc0_display   -- component value display (the actual staging value)
#
# Placeholder: (batches) -- replace with patient internal _id values
TNM_query = """
SELECT DISTINCT
  p1._id "p1_id",
  o0.id "o0_id",
  occc0.display "occc0_display",
  fhirql_code (occc0.code) "occc0_code",
  fhirql_code (ocvc0.system) "ocvc0_system",
  ocvc0.display "ocvc0_display"
FROM
  observation o0
  LEFT JOIN observation_component oc0 ON (oc0._resource = o0._id)
  LEFT JOIN observation_component_code occ0 ON (occ0._resource = o0._id)
  LEFT JOIN observation_component_code_coding occc0 ON (occc0._resource = o0._id)
  JOIN "observation_component_valueCodeableConcept" ocv0 ON (ocv0._resource = o0._id)
  JOIN "observation_component_valueCodeableConcept_coding" ocvc0 ON (ocvc0._resource = o0._id)
  JOIN observation_subject os0 ON (os0._resource = o0._id)
  JOIN patient p1 ON (
    p1._id = os0._reference_id
    and os0._reference_type = 'Patient'
  )
  WHERE occc0.display = 'Primary tumor.pathology'
  AND fhirql_code(ocvc0.system) IN (
    'https://example-hospital.org/fhir/TumorDocumentation/pTNM/T',
    'https://example-hospital.org/fhir/TumorDocumentation/pTNM/N',
    'https://example-hospital.org/fhir/TumorDocumentation/pTNM/M'
  )
  AND p1._id IN (batches);
"""


# ---------------------------------------------------------------------------
# Encounter FHIR-ID conversion query
# ---------------------------------------------------------------------------
# Maps encounter internal _id to the FHIR-visible id.  Useful when your
# input CSV contains internal IDs but you need the FHIR resource IDs for
# downstream processing.
#
# Output columns:
#   fhir_id               -- FHIR Encounter.id (the public identifier)
#   encounter_internal_id  -- database internal _id
#
# Placeholder: (batches) -- replace with encounter internal _id values
encounter_convert_query = """
SELECT
    e0.id "fhir_id",
    e0._id "encounter_internal_id"
FROM
    encounter e0
WHERE
    e0._id IN (batches);
"""
