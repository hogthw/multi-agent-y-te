"""
Controlled HL7 FHIR R4 Benchmark Dataset Generator (International + Vietnam FHIR Profile).

Generates 32 balanced, realistic multi-resource FHIR R4 Bundles (8 per category):
  - CLEAN             : Valid records (international & Vietnam SI units / QĐ 130/QĐ-BYT)
  - STRUCTURAL        : Missing required R4/VN fields, broken references, invalid dates, bad CCCD/BHYT ID
  - TERMINOLOGY       : LOINC/ICD-10 code-display mismatch (incl. Vietnamese abbreviations),
                        UCUM & SI unit confusion (mmol/L vs mg/dL), impossible physiological values
  - CLINICAL_SEMANTIC : Cross-resource clinical contradictions (drug-allergy, renal Metformin,
                        sex-diagnosis, pediatric Aspirin, hypoglycemia-Insulin, INR-Warfarin,
                        Vietnam Dengue fever + NSAID contraindication, beta-blocker in severe asthma)

All case titles use clean, icon-free academic Vietnamese suitable for thesis slides & reports.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional


DEFAULT_OUTPUT_DIR = os.path.join("my_datasets", "processed", "fhir_audit")


def _base_bundle(
    case_id: str,
    patient_id: str,
    name: str,
    gender: str,
    birth_date: str,
    enc_start: str,
    enc_end: str,
    conditions: List[Dict[str, Any]],
    observations: List[Dict[str, Any]],
    medications: List[Dict[str, Any]],
    allergies: List[Dict[str, Any]],
    cccd: Optional[str] = "079085012345",
    bhyt: Optional[str] = "DN4792012345678",
) -> Dict[str, Any]:
    identifiers = []
    if cccd is not None:
        identifiers.append({"system": "https://moh.gov.vn/fhir/sid/cccd", "value": cccd})
    if bhyt is not None:
        identifiers.append({"system": "https://moh.gov.vn/fhir/sid/bhyt", "value": bhyt})

    patient_res: Dict[str, Any] = {
        "resourceType": "Patient",
        "id": patient_id,
        "name": [{"text": name}],
        "gender": gender,
        "birthDate": birth_date,
    }
    if identifiers:
        patient_res["identifier"] = identifiers

    entries: List[Dict[str, Any]] = [
        {"resource": patient_res},
        {
            "resource": {
                "resourceType": "Encounter",
                "id": f"enc-{patient_id}",
                "status": "finished",
                "class": {"code": "IMP", "display": "inpatient encounter"},
                "subject": {"reference": f"Patient/{patient_id}"},
                "period": {"start": enc_start, "end": enc_end},
            }
        },
    ]
    for cond in conditions:
        entries.append({"resource": cond})
    for obs in observations:
        entries.append({"resource": obs})
    for med in medications:
        entries.append({"resource": med})
    for alg in allergies:
        entries.append({"resource": alg})

    return {
        "resourceType": "Bundle",
        "id": case_id,
        "type": "collection",
        "entry": entries,
    }


def _cond(cid: str, pid: str, code: str, display: str, system: str = "http://hl7.org/fhir/sid/icd-10") -> Dict[str, Any]:
    return {
        "resourceType": "Condition",
        "id": cid,
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "code": {"coding": [{"system": system, "code": code, "display": display}]},
        "subject": {"reference": f"Patient/{pid}"},
    }


def _obs(oid: str, pid: str, loinc: str, display: str, value: float, unit: str, status: str = "final") -> Dict[str, Any]:
    return {
        "resourceType": "Observation",
        "id": oid,
        "status": status,
        "code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": display}]},
        "subject": {"reference": f"Patient/{pid}"},
        "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org"},
    }


def _med(mid: str, pid: str, rx_code: str, display: str, dosage: str, status: str = "active") -> Dict[str, Any]:
    return {
        "resourceType": "MedicationRequest",
        "id": mid,
        "status": status,
        "intent": "order",
        "medicationCodeableConcept": {
            "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": rx_code, "display": display}]
        },
        "subject": {"reference": f"Patient/{pid}"},
        "dosageInstruction": [{"text": dosage}],
    }


def _allergy(aid: str, pid: str, substance: str, criticality: str = "high", reaction: str = "Anaphylaxis") -> Dict[str, Any]:
    return {
        "resourceType": "AllergyIntolerance",
        "id": aid,
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "criticality": criticality,
        "code": {"text": substance},
        "patient": {"reference": f"Patient/{pid}"},
        "reaction": [{"manifestation": [{"text": reaction}]}],
    }


def build_benchmark_cases() -> List[Dict[str, Any]]:
    """Create 32 balanced HL7 FHIR R4 audit cases (8 per category), including Vietnam clinical cases."""
    cases: List[Dict[str, Any]] = []

    # =========================================================================
    # GROUP A: 8 CLEAN VALID FHIR BUNDLES (Including Vietnam SI & trap cases)
    # =========================================================================
    cases.append(
        {
            "case_id": "fhir_clean_01",
            "title": "[Hợp lệ] Đái tháo đường type 2 ổn định — Hồ sơ chuẩn",
            "primary_category": "CLEAN",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_clean_01", "pat-101", "Nguyen Van An", "male", "1968-04-12",
                "2025-10-01T08:00:00Z", "2025-10-03T14:00:00Z",
                [_cond("c-101", "pat-101", "E11.9", "Type 2 diabetes mellitus without complications")],
                [
                    _obs("o-101a", "pat-101", "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", 7.4, "%"),
                    _obs("o-101b", "pat-101", "33914-3", "Glomerular filtration rate (eGFR)", 78.0, "mL/min/1.73m2"),
                ],
                [_med("m-101", "pat-101", "860975", "Metformin 500 MG Oral Tablet", "500 mg uống 2 lần/ngày sau ăn")],
                [],
            ),
            "ground_truth_issues": [],
        }
    )

    cases.append(
        {
            "case_id": "fhir_clean_02",
            "title": "[Hợp lệ] Tăng huyết áp + dị ứng Penicillin nhưng kê Amlodipine",
            "primary_category": "CLEAN",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_clean_02", "pat-102", "Tran Thi Binh", "female", "1959-09-20",
                "2025-10-05T09:15:00Z", "2025-10-05T11:00:00Z",
                [_cond("c-102", "pat-102", "I10", "Essential (primary) hypertension")],
                [
                    _obs("o-102a", "pat-102", "8480-6", "Systolic blood pressure", 148.0, "mmHg"),
                    _obs("o-102b", "pat-102", "8462-4", "Diastolic blood pressure", 92.0, "mmHg"),
                ],
                [_med("m-102", "pat-102", "197361", "Amlodipine 5 MG Oral Tablet", "5 mg uống 1 lần/ngày")],
                [_allergy("a-102", "pat-102", "Penicillin", "high", "Urticaria")],
            ),
            "ground_truth_issues": [],
        }
    )

    cases.append(
        {
            "case_id": "fhir_clean_03",
            "title": "[Hợp lệ] Viêm phổi cộng đồng — Dị ứng Penicillin, đổi sang Azithromycin",
            "primary_category": "CLEAN",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_clean_03", "pat-103", "Le Minh Cuong", "male", "1975-01-15",
                "2025-10-07T07:30:00Z", "2025-10-10T10:00:00Z",
                [_cond("c-103", "pat-103", "J18.9", "Pneumonia, unspecified organism")],
                [
                    _obs("o-103a", "pat-103", "8310-5", "Body temperature", 38.4, "Cel"),
                    _obs("o-103b", "pat-103", "2708-6", "Oxygen saturation in Arterial blood", 95.0, "%"),
                ],
                [_med("m-103", "pat-103", "308460", "Azithromycin 500 MG Oral Tablet", "500 mg uống 1 lần/ngày trong 3 ngày")],
                [_allergy("a-103", "pat-103", "Amoxicillin / Penicillin", "high", "Anaphylaxis")],
            ),
            "ground_truth_issues": [],
        }
    )

    cases.append(
        {
            "case_id": "fhir_clean_04",
            "title": "[Hợp lệ] Thai phụ khám thai định kỳ — Hồ sơ sản khoa chuẩn",
            "primary_category": "CLEAN",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_clean_04", "pat-104", "Pham Thu Dung", "female", "1996-06-11",
                "2025-10-08T10:00:00Z", "2025-10-08T11:30:00Z",
                [_cond("c-104", "pat-104", "Z34.0", "Encounter for supervision of normal first pregnancy")],
                [
                    _obs("o-104a", "pat-104", "718-7", "Hemoglobin [Mass/volume] in Blood", 11.8, "g/dL"),
                    _obs("o-104b", "pat-104", "8480-6", "Systolic blood pressure", 115.0, "mmHg"),
                ],
                [_med("m-104", "pat-104", "315966", "Folic Acid 0.4 MG Oral Tablet", "400 mcg mỗi ngày")],
                [],
            ),
            "ground_truth_issues": [],
        }
    )

    cases.append(
        {
            "case_id": "fhir_clean_05",
            "title": "[Hợp lệ] Suy thận mạn CKD độ 4 — Dùng Insulin thay Metformin",
            "primary_category": "CLEAN",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_clean_05", "pat-105", "Vo Hoang Em", "male", "1952-11-02",
                "2025-10-09T08:00:00Z", "2025-10-12T15:00:00Z",
                [
                    _cond("c-105a", "pat-105", "N18.4", "Chronic kidney disease, stage 4"),
                    _cond("c-105b", "pat-105", "E11.22", "Type 2 diabetes mellitus with diabetic chronic kidney disease"),
                ],
                [
                    _obs("o-105a", "pat-105", "33914-3", "Glomerular filtration rate (eGFR)", 22.0, "mL/min/1.73m2"),
                    _obs("o-105b", "pat-105", "2339-0", "Glucose [Mass/volume] in Blood", 192.0, "mg/dL"),
                ],
                [_med("m-105", "pat-105", "261551", "Insulin Glargine 100 UNT/ML Injectable Solution", "12 đơn vị tiêm dưới da buổi tối")],
                [],
            ),
            "ground_truth_issues": [],
        }
    )

    cases.append(
        {
            "case_id": "fhir_clean_06",
            "title": "[Hợp lệ] Bệnh nhi 5 tuổi sốt siêu vi — Dùng Paracetamol đúng chỉ định",
            "primary_category": "CLEAN",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_clean_06", "pat-106", "Nguyen Gia Huy", "male", "2020-05-14",
                "2025-10-11T14:00:00Z", "2025-10-11T15:30:00Z",
                [_cond("c-106", "pat-106", "B34.9", "Viral infection, unspecified")],
                [_obs("o-106", "pat-106", "8310-5", "Body temperature", 38.7, "Cel")],
                [_med("m-106", "pat-106", "161", "Acetaminophen (Paracetamol) Oral Suspension", "250 mg mỗi 6 giờ khi sốt")],
                [],
            ),
            "ground_truth_issues": [],
        }
    )

    # Vietnam-specific Clean Cases (SI units mmol/L + Dengue treated safely with Paracetamol)
    cases.append(
        {
            "case_id": "fhir_clean_07_vn",
            "title": "[Hợp lệ - VN] Đái tháo đường dùng đơn vị SI Việt Nam (Glucose 7.8 mmol/L)",
            "primary_category": "CLEAN",
            "profile": "VN-FHIR",
            "bundle": _base_bundle(
                "fhir_clean_07_vn", "pat-107", "Tran Quoc Khanh", "male", "1966-08-19",
                "2025-10-12T08:00:00Z", "2025-10-12T10:30:00Z",
                [_cond("c-107", "pat-107", "E11.9", "ĐTĐ type 2 không biến chứng (Type 2 diabetes mellitus)")],
                [
                    _obs("o-107a", "pat-107", "2339-0", "Glucose máu lúc đói", 7.8, "mmol/L"),
                    _obs("o-107b", "pat-107", "4548-4", "HbA1c", 7.1, "%"),
                ],
                [_med("m-107", "pat-107", "860975", "Metformin 500 MG", "500 mg uống 2 lần/ngày")],
                [],
            ),
            "ground_truth_issues": [],
        }
    )

    cases.append(
        {
            "case_id": "fhir_clean_08_vn",
            "title": "[Hợp lệ - VN] Sốt xuất huyết Dengue — Hạ sốt an toàn bằng Paracetamol",
            "primary_category": "CLEAN",
            "profile": "VN-FHIR",
            "bundle": _base_bundle(
                "fhir_clean_08_vn", "pat-108", "Le Thi Hoa", "female", "1998-03-11",
                "2025-10-14T09:00:00Z", "2025-10-16T10:00:00Z",
                [_cond("c-108", "pat-108", "A97.0", "Sốt xuất huyết Dengue không có dấu hiệu cảnh báo")],
                [
                    _obs("o-108a", "pat-108", "8310-5", "Thân nhiệt", 39.0, "Cel"),
                    _obs("o-108b", "pat-108", "777-3", "Tiểu cầu (Platelets)", 112.0, "10*9/L"),
                ],
                [_med("m-108", "pat-108", "161", "Paracetamol 500 MG", "500 mg uống khi sốt >= 38.5 độ C, cách 6 giờ")],
                [],
            ),
            "ground_truth_issues": [],
        }
    )

    # =========================================================================
    # GROUP B: 8 STRUCTURAL ERRORS (Schema, References, Dates, VN QĐ 130 ID)
    # =========================================================================
    b_s1 = _base_bundle(
        "fhir_struct_01", "pat-201", "Hoang Van Phuc", "male", "1971-03-19",
        "2025-10-02T09:00:00Z", "2025-10-04T10:00:00Z",
        [_cond("c-201", "pat-201", "I25.10", "Atherosclerotic heart disease")],
        [_obs("o-201", "pat-999", "8480-6", "Systolic blood pressure", 136.0, "mmHg")],
        [_med("m-201", "pat-201", "617314", "Atorvastatin 20 MG Oral Tablet", "20 mg buổi tối")],
        [],
    )
    cases.append(
        {
            "case_id": "fhir_struct_01",
            "title": "[Cấu trúc] Gãy liên kết tham chiếu (Observation trỏ Patient/pat-999)",
            "primary_category": "STRUCTURAL",
            "profile": "INTL",
            "bundle": b_s1,
            "ground_truth_issues": [
                {
                    "issue_id": "gt_s1",
                    "category": "STRUCTURAL",
                    "error_code": "BROKEN_REFERENCE",
                    "severity": "high",
                    "resources": ["Observation/o-201", "Patient/pat-201"],
                    "description": "Observation/o-201 tham chiếu Patient/pat-999 không tồn tại trong Bundle (bệnh nhân thực tế là Patient/pat-201).",
                }
            ],
        }
    )

    b_s2 = _base_bundle(
        "fhir_struct_02", "pat-202", "Do Thi Giang", "female", "1984-07-22",
        "2025-10-15T14:00:00Z", "2025-10-12T08:00:00Z",
        [_cond("c-202", "pat-202", "J45.909", "Unspecified asthma, uncomplicated")],
        [_obs("o-202", "pat-202", "2708-6", "Oxygen saturation in Arterial blood", 97.0, "%")],
        [_med("m-202", "pat-202", "745679", "Albuterol Inhaler", "2 nhát xịt khi khó thở")],
        [],
    )
    cases.append(
        {
            "case_id": "fhir_struct_02",
            "title": "[Cấu trúc] Sai thời gian Encounter (Ngày ra viện trước ngày nhập viện)",
            "primary_category": "STRUCTURAL",
            "profile": "INTL",
            "bundle": b_s2,
            "ground_truth_issues": [
                {
                    "issue_id": "gt_s2",
                    "category": "STRUCTURAL",
                    "error_code": "INVALID_PERIOD",
                    "severity": "medium",
                    "resources": ["Encounter/enc-pat-202"],
                    "description": "Encounter.period.end (2025-10-12) sớm hơn Encounter.period.start (2025-10-15).",
                }
            ],
        }
    )

    b_s3 = _base_bundle(
        "fhir_struct_03", "pat-203", "Bui Quang Hai", "male", "1964-12-01",
        "2025-10-03T08:00:00Z", "2025-10-05T16:00:00Z",
        [_cond("c-203", "pat-203", "E78.5", "Hyperlipidemia, unspecified")],
        [_obs("o-203", "pat-203", "2093-3", "Cholesterol [Mass/volume] in Serum", 242.0, "mg/dL")],
        [_med("m-203", "pat-203", "859751", "Rosuvastatin 10 MG Oral Tablet", "10 mg mỗi ngày")],
        [],
    )
    del b_s3["entry"][3]["resource"]["status"]
    cases.append(
        {
            "case_id": "fhir_struct_03",
            "title": "[Cấu trúc] Thiếu trường bắt buộc FHIR R4 (Observation.status)",
            "primary_category": "STRUCTURAL",
            "profile": "INTL",
            "bundle": b_s3,
            "ground_truth_issues": [
                {
                    "issue_id": "gt_s3",
                    "category": "STRUCTURAL",
                    "error_code": "MISSING_REQUIRED_FIELD",
                    "severity": "medium",
                    "resources": ["Observation/o-203"],
                    "description": "Observation/o-203 thiếu trường bắt buộc 'status' theo chuẩn HL7 FHIR R4.",
                }
            ],
        }
    )

    b_s4 = _base_bundle(
        "fhir_struct_04", "pat-204", "Dang Thi Khanh", "female", "1990-02-10",
        "2025-10-06T09:00:00Z", "2025-10-06T12:00:00Z",
        [_cond("c-204", "pat-204", "K21.9", "Gastro-esophageal reflux disease")],
        [_obs("o-204", "pat-204", "8310-5", "Body temperature", 36.8, "Cel")],
        [_med("m-204", "pat-204", "198051", "Omeprazole 20 MG Oral Capsule", "20 mg trước bữa sáng")],
        [],
    )
    del b_s4["entry"][4]["resource"]["intent"]
    del b_s4["entry"][4]["resource"]["subject"]
    cases.append(
        {
            "case_id": "fhir_struct_04",
            "title": "[Cấu trúc] MedicationRequest thiếu đồng thời intent và subject",
            "primary_category": "STRUCTURAL",
            "profile": "INTL",
            "bundle": b_s4,
            "ground_truth_issues": [
                {
                    "issue_id": "gt_s4",
                    "category": "STRUCTURAL",
                    "error_code": "MISSING_REQUIRED_FIELD",
                    "severity": "high",
                    "resources": ["MedicationRequest/m-204"],
                    "description": "MedicationRequest/m-204 thiếu 2 trường bắt buộc 'intent' và 'subject'.",
                }
            ],
        }
    )

    b_s5 = _base_bundle(
        "fhir_struct_05", "pat-205", "Ngo Duc Long", "male", "2029-08-15",
        "2025-10-04T08:00:00Z", "2025-10-04T11:00:00Z",
        [_cond("c-205", "pat-205", "I10", "Essential hypertension")],
        [_obs("o-205", "pat-205", "8480-6", "Systolic blood pressure", 140.0, "mmHg")],
        [_med("m-205", "pat-205", "197361", "Amlodipine 5 MG Oral Tablet", "5 mg mỗi ngày")],
        [],
    )
    cases.append(
        {
            "case_id": "fhir_struct_05",
            "title": "[Cấu trúc] Ngày sinh Patient ở tương lai (2029 > Encounter 2025)",
            "primary_category": "STRUCTURAL",
            "profile": "INTL",
            "bundle": b_s5,
            "ground_truth_issues": [
                {
                    "issue_id": "gt_s5",
                    "category": "STRUCTURAL",
                    "error_code": "INVALID_BIRTHDATE",
                    "severity": "high",
                    "resources": ["Patient/pat-205", "Encounter/enc-pat-205"],
                    "description": "Patient.birthDate (2029-08-15) lớn hơn ngày khám Encounter.period.start (2025-10-04).",
                }
            ],
        }
    )

    b_s6 = _base_bundle(
        "fhir_struct_06", "pat-206", "Ly Thi Mai", "female", "1978-05-05",
        "2025-10-08T08:00:00Z", "2025-10-09T10:00:00Z",
        [_cond("c-206", "pat-206", "E03.9", "Hypothyroidism, unspecified")],
        [_obs("o-206", "pat-206", "3016-3", "Thyrotropin (TSH) in Serum", 9.5, "mIU/L")],
        [_med("m-206", "pat-206", "966222", "Levothyroxine 50 MCG Oral Tablet", "50 mcg mỗi ngày")],
        [_allergy("a-206", "pat-000", "Sulfonamides", "high", "Rash")],
    )
    cases.append(
        {
            "case_id": "fhir_struct_06",
            "title": "[Cấu trúc] AllergyIntolerance trỏ nhầm Patient/pat-000 không tồn tại",
            "primary_category": "STRUCTURAL",
            "profile": "INTL",
            "bundle": b_s6,
            "ground_truth_issues": [
                {
                    "issue_id": "gt_s6",
                    "category": "STRUCTURAL",
                    "error_code": "BROKEN_REFERENCE",
                    "severity": "high",
                    "resources": ["AllergyIntolerance/a-206", "Patient/pat-206"],
                    "description": "AllergyIntolerance/a-206 tham chiếu Patient/pat-000 không tồn tại trong Bundle.",
                }
            ],
        }
    )

    # Vietnam-specific Structural Errors (QĐ 130/QĐ-BYT identifier & missing gender)
    b_s7 = _base_bundle(
        "fhir_struct_07_vn", "pat-207", "Nguyen Thanh Phong", "male", "1981-04-02",
        "2025-10-12T08:00:00Z", "2025-10-12T10:00:00Z",
        [_cond("c-207", "pat-207", "I10", "Tăng huyết áp nguyên phát")],
        [_obs("o-207", "pat-207", "8480-6", "Huyết áp tâm thu", 142.0, "mmHg")],
        [_med("m-207", "pat-207", "197361", "Amlodipine 5 MG", "5 mg mỗi ngày")],
        [],
        cccd="079081",  # Invalid CCCD: only 6 digits instead of 12 digits
        bhyt="DN47920",  # Invalid BHYT: only 7 chars instead of 10 or 15 chars
    )
    cases.append(
        {
            "case_id": "fhir_struct_07_vn",
            "title": "[Cấu trúc - VN] Sai định dạng CCCD/BHYT theo QĐ 130/QĐ-BYT",
            "primary_category": "STRUCTURAL",
            "profile": "VN-FHIR",
            "bundle": b_s7,
            "ground_truth_issues": [
                {
                    "issue_id": "gt_s7",
                    "category": "STRUCTURAL",
                    "error_code": "INVALID_IDENTIFIER",
                    "severity": "medium",
                    "resources": ["Patient/pat-207"],
                    "description": "Định danh bệnh nhân sai chuẩn Việt Nam (QĐ 130/QĐ-BYT): CCCD '079081' không đủ 12 chữ số và mã BHYT 'DN47920' không đúng quy cách 10 hoặc 15 ký tự.",
                }
            ],
        }
    )

    b_s8 = _base_bundle(
        "fhir_struct_08_vn", "pat-208", "Vo Thi Thu Ha", "female", "1974-10-20",
        "2025-10-13T07:30:00Z", "2025-10-13T09:00:00Z",
        [_cond("c-208", "pat-208", "E11.9", "Đái tháo đường type 2")],
        [_obs("o-208", "pat-208", "2339-0", "Glucose máu", 7.4, "mmol/L")],
        [_med("m-208", "pat-208", "860975", "Metformin 500 MG", "500 mg mỗi ngày")],
        [],
    )
    del b_s8["entry"][0]["resource"]["gender"]
    cases.append(
        {
            "case_id": "fhir_struct_08_vn",
            "title": "[Cấu trúc - VN] Thiếu trường bắt buộc Patient.gender trong hồ sơ EMR",
            "primary_category": "STRUCTURAL",
            "profile": "VN-FHIR",
            "bundle": b_s8,
            "ground_truth_issues": [
                {
                    "issue_id": "gt_s8",
                    "category": "STRUCTURAL",
                    "error_code": "MISSING_REQUIRED_FIELD",
                    "severity": "medium",
                    "resources": ["Patient/pat-208"],
                    "description": "Patient/pat-208 thiếu trường bắt buộc 'gender' theo hồ sơ chuẩn HL7 FHIR R4 và VN-EMR.",
                }
            ],
        }
    )

    # =========================================================================
    # GROUP C: 8 TERMINOLOGY & UNIT ERRORS (Incl. VN SI mmol/L vs mg/dL & VN abbr)
    # =========================================================================
    cases.append(
        {
            "case_id": "fhir_term_01",
            "title": "[Mã hóa] HbA1c ghi sai đơn vị UCUM (mmHg thay vì %)",
            "primary_category": "TERMINOLOGY",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_term_01", "pat-301", "Truong Van Nam", "male", "1965-03-10",
                "2025-10-01T08:00:00Z", "2025-10-01T10:00:00Z",
                [_cond("c-301", "pat-301", "E11.9", "Type 2 diabetes mellitus")],
                [_obs("o-301", "pat-301", "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", 8.1, "mmHg")],
                [_med("m-301", "pat-301", "860975", "Metformin 500 MG Oral Tablet", "500 mg 2 lần/ngày")],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_t1",
                    "category": "TERMINOLOGY",
                    "error_code": "INVALID_UNIT",
                    "severity": "medium",
                    "resources": ["Observation/o-301"],
                    "description": "Observation/o-301 (HbA1c, LOINC 4548-4) ghi sai đơn vị áp suất 'mmHg' thay vì '%'.",
                }
            ],
        }
    )

    cases.append(
        {
            "case_id": "fhir_term_02",
            "title": "[Mã hóa] Mã ICD-10 lệch tên bệnh (J45.9 Hen nhưng ghi Đái tháo đường)",
            "primary_category": "TERMINOLOGY",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_term_02", "pat-302", "Vu Thi Oanh", "female", "1973-08-25",
                "2025-10-02T09:00:00Z", "2025-10-02T11:00:00Z",
                [_cond("c-302", "pat-302", "J45.9", "Type 2 diabetes mellitus without complications")],
                [_obs("o-302", "pat-302", "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", 7.8, "%")],
                [_med("m-302", "pat-302", "860975", "Metformin 500 MG Oral Tablet", "500 mg 2 lần/ngày")],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_t2",
                    "category": "TERMINOLOGY",
                    "error_code": "CODE_DISPLAY_MISMATCH",
                    "severity": "high",
                    "resources": ["Condition/c-302"],
                    "description": "Condition/c-302 gắn mã ICD-10 J45.9 (Hen phế quản) với mô tả 'Type 2 diabetes mellitus' (đúng phải là E11.9).",
                }
            ],
        }
    )

    cases.append(
        {
            "case_id": "fhir_term_03",
            "title": "[Mã hóa] Chỉ số sinh tồn bất khả thi (SpO2 = 148%, Thân nhiệt 52 độ C)",
            "primary_category": "TERMINOLOGY",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_term_03", "pat-303", "Mai Van Quan", "male", "1980-11-12",
                "2025-10-04T07:00:00Z", "2025-10-05T10:00:00Z",
                [_cond("c-303", "pat-303", "J18.9", "Pneumonia, unspecified organism")],
                [
                    _obs("o-303a", "pat-303", "2708-6", "Oxygen saturation in Arterial blood", 148.0, "%"),
                    _obs("o-303b", "pat-303", "8310-5", "Body temperature", 52.0, "Cel"),
                ],
                [_med("m-303", "pat-303", "308460", "Azithromycin 500 MG Oral Tablet", "500 mg mỗi ngày")],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_t3",
                    "category": "TERMINOLOGY",
                    "error_code": "IMPOSSIBLE_VALUE",
                    "severity": "high",
                    "resources": ["Observation/o-303a", "Observation/o-303b"],
                    "description": "Giá trị sinh tồn bất khả thi sinh lý: SpO2 = 148% (> 100%) và thân nhiệt = 52 Cel.",
                }
            ],
        }
    )

    cases.append(
        {
            "case_id": "fhir_term_04",
            "title": "[Mã hóa] Mã LOINC không khớp xét nghiệm (8480-6 Huyết áp gắn cho Glucose)",
            "primary_category": "TERMINOLOGY",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_term_04", "pat-304", "Cao Thi Quynh", "female", "1969-04-30",
                "2025-10-05T08:30:00Z", "2025-10-05T10:00:00Z",
                [_cond("c-304", "pat-304", "E11.9", "Type 2 diabetes mellitus")],
                [_obs("o-304", "pat-304", "8480-6", "Glucose [Mass/volume] in Blood", 165.0, "mg/dL")],
                [_med("m-304", "pat-304", "860975", "Metformin 500 MG Oral Tablet", "500 mg mỗi ngày")],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_t4",
                    "category": "TERMINOLOGY",
                    "error_code": "CODE_DISPLAY_MISMATCH",
                    "severity": "medium",
                    "resources": ["Observation/o-304"],
                    "description": "Observation/o-304 gắn mã LOINC 8480-6 (Huyết áp tâm thu) cho xét nghiệm Glucose máu (đúng phải là 2339-0).",
                }
            ],
        }
    )

    cases.append(
        {
            "case_id": "fhir_term_05",
            "title": "[Mã hóa] Huyết áp tâm thu nhỏ hơn tâm trương (SBP 75 / DBP 130 mmHg)",
            "primary_category": "TERMINOLOGY",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_term_05", "pat-305", "Phan Minh Son", "male", "1957-09-14",
                "2025-10-06T09:00:00Z", "2025-10-06T10:00:00Z",
                [_cond("c-305", "pat-305", "I10", "Essential hypertension")],
                [
                    _obs("o-305a", "pat-305", "8480-6", "Systolic blood pressure", 75.0, "mmHg"),
                    _obs("o-305b", "pat-305", "8462-4", "Diastolic blood pressure", 130.0, "mmHg"),
                ],
                [_med("m-305", "pat-305", "197361", "Amlodipine 5 MG Oral Tablet", "5 mg mỗi ngày")],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_t5",
                    "category": "TERMINOLOGY",
                    "error_code": "IMPOSSIBLE_VALUE",
                    "severity": "high",
                    "resources": ["Observation/o-305a", "Observation/o-305b"],
                    "description": "Huyết áp tâm thu (75 mmHg) nhỏ hơn huyết áp tâm trương (130 mmHg), khả năng đảo ngược số liệu.",
                }
            ],
        }
    )

    cases.append(
        {
            "case_id": "fhir_term_06",
            "title": "[Mã hóa] Đường huyết ghi sai đơn vị (Cel thay vì mg/dL hoặc mmol/L)",
            "primary_category": "TERMINOLOGY",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_term_06", "pat-306", "Le Thi Trang", "female", "1977-01-20",
                "2025-10-09T09:00:00Z", "2025-10-09T11:00:00Z",
                [_cond("c-306", "pat-306", "E11.9", "Type 2 diabetes mellitus")],
                [_obs("o-306", "pat-306", "2339-0", "Glucose [Mass/volume] in Blood", 184.0, "Cel")],
                [_med("m-306", "pat-306", "860975", "Metformin 500 MG Oral Tablet", "500 mg mỗi ngày")],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_t6",
                    "category": "TERMINOLOGY",
                    "error_code": "INVALID_UNIT",
                    "severity": "medium",
                    "resources": ["Observation/o-306"],
                    "description": "Observation/o-306 (Glucose máu, LOINC 2339-0) dùng đơn vị nhiệt độ 'Cel' thay vì 'mg/dL' hoặc 'mmol/L'.",
                }
            ],
        }
    )

    # Vietnam-specific Terminology Errors: mmol/L <-> mg/dL confusion & Vietnamese abbr ICD mismatch
    cases.append(
        {
            "case_id": "fhir_term_07_vn",
            "title": "[Mã hóa - VN] Nhầm đơn vị SI Việt Nam (Glucose 185 mmol/L thay vì mg/dL)",
            "primary_category": "TERMINOLOGY",
            "profile": "VN-FHIR",
            "bundle": _base_bundle(
                "fhir_term_07_vn", "pat-307", "Hoang Minh Tri", "male", "1962-05-18",
                "2025-10-12T08:00:00Z", "2025-10-12T09:30:00Z",
                [_cond("c-307", "pat-307", "E11.9", "ĐTĐ type 2")],
                [_obs("o-307", "pat-307", "2339-0", "Glucose máu lúc đói", 185.0, "mmol/L")],
                [_med("m-307", "pat-307", "860975", "Metformin 500 MG", "500 mg 2 lần/ngày")],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_t7",
                    "category": "TERMINOLOGY",
                    "error_code": "IMPOSSIBLE_VALUE",
                    "severity": "high",
                    "resources": ["Observation/o-307"],
                    "description": "Lỗi nhầm đơn vị phổ biến tại VN: Glucose máu = 185 mmol/L (~3330 mg/dL) bất khả thi sinh lý; thực chất giá trị 185 thuộc đơn vị mg/dL (~10.3 mmol/L).",
                }
            ],
        }
    )

    cases.append(
        {
            "case_id": "fhir_term_08_vn",
            "title": "[Mã hóa - VN] Viết tắt tiếng Việt lệch mã ICD-10 (Ghi THA nhưng gắn mã E11.9)",
            "primary_category": "TERMINOLOGY",
            "profile": "VN-FHIR",
            "bundle": _base_bundle(
                "fhir_term_08_vn", "pat-308", "Nguyen Van Vu", "male", "1958-12-09",
                "2025-10-14T08:30:00Z", "2025-10-14T10:00:00Z",
                [_cond("c-308", "pat-308", "E11.9", "THA nguyên phát độ 2 (Tăng huyết áp)")],
                [
                    _obs("o-308a", "pat-308", "8480-6", "Huyết áp tâm thu", 162.0, "mmHg"),
                    _obs("o-308b", "pat-308", "8462-4", "Huyết áp tâm trương", 98.0, "mmHg"),
                ],
                [_med("m-308", "pat-308", "197361", "Amlodipine 5 MG", "5 mg mỗi ngày")],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_t8",
                    "category": "TERMINOLOGY",
                    "error_code": "CODE_DISPLAY_MISMATCH",
                    "severity": "high",
                    "resources": ["Condition/c-308"],
                    "description": "Condition/c-308 mô tả 'THA nguyên phát độ 2 (Tăng huyết áp)' nhưng lại gắn sai mã ICD-10 E11.9 (Đái tháo đường type 2; đúng phải là I10).",
                }
            ],
        }
    )

    # =========================================================================
    # GROUP D: 8 CROSS-RESOURCE CLINICAL SEMANTIC ERRORS (Incl. Vietnam Dengue + NSAID)
    # =========================================================================
    cases.append(
        {
            "case_id": "fhir_sem_01",
            "title": "[Lâm sàng] Xung đột Thuốc–Dị ứng (Kê Amoxicillin dù dị ứng nặng Penicillin)",
            "primary_category": "CLINICAL_SEMANTIC",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_sem_01", "pat-401", "Nguyen Tien Dat", "male", "1982-06-18",
                "2025-10-10T08:30:00Z", "2025-10-12T10:00:00Z",
                [_cond("c-401", "pat-401", "J03.90", "Acute tonsillitis, unspecified")],
                [_obs("o-401", "pat-401", "8310-5", "Body temperature", 38.6, "Cel")],
                [_med("m-401", "pat-401", "308191", "Amoxicillin 500 MG Oral Capsule", "500 mg mỗi 8 giờ trong 7 ngày")],
                [_allergy("a-401", "pat-401", "Penicillin", "high", "Anaphylaxis and airway edema")],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_c1",
                    "category": "CLINICAL_SEMANTIC",
                    "error_code": "DRUG_ALLERGY_CONFLICT",
                    "severity": "critical",
                    "resources": ["MedicationRequest/m-401", "AllergyIntolerance/a-401"],
                    "description": "MedicationRequest/m-401 kê Amoxicillin (nhóm beta-lactam/Penicillin) dù AllergyIntolerance/a-401 ghi dị ứng phản vệ nặng với Penicillin.",
                }
            ],
        }
    )

    cases.append(
        {
            "case_id": "fhir_sem_02",
            "title": "[Lâm sàng] Chống chỉ định Thận–Thuốc (Kê Metformin liều cao khi eGFR = 18)",
            "primary_category": "CLINICAL_SEMANTIC",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_sem_02", "pat-402", "Tran Van Dung", "male", "1954-02-28",
                "2025-10-11T09:00:00Z", "2025-10-14T11:00:00Z",
                [
                    _cond("c-402a", "pat-402", "E11.22", "Type 2 diabetes mellitus with diabetic chronic kidney disease"),
                    _cond("c-402b", "pat-402", "N18.4", "Chronic kidney disease, stage 4"),
                ],
                [
                    _obs("o-402a", "pat-402", "33914-3", "Glomerular filtration rate (eGFR)", 18.0, "mL/min/1.73m2"),
                    _obs("o-402b", "pat-402", "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", 8.5, "%"),
                ],
                [_med("m-402", "pat-402", "861004", "Metformin Hydrochloride 1000 MG Oral Tablet", "1000 mg uống 2 lần/ngày")],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_c2",
                    "category": "CLINICAL_SEMANTIC",
                    "error_code": "RENAL_CONTRAINDICATION",
                    "severity": "critical",
                    "resources": ["MedicationRequest/m-402", "Observation/o-402a", "Condition/c-402b"],
                    "description": "Metformin 1000 mg x2/ngày chống chỉ định tuyệt đối khi eGFR < 30 mL/min/1.73m2 (Observation/o-402a = 18) do nguy cơ nhiễm toan lactic.",
                }
            ],
        }
    )

    cases.append(
        {
            "case_id": "fhir_sem_03",
            "title": "[Lâm sàng] Mâu thuẫn Giới tính–Chẩn đoán (Bệnh nhân nam chẩn đoán Tiền sản giật O14.9)",
            "primary_category": "CLINICAL_SEMANTIC",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_sem_03", "pat-403", "Pham Hoang Lam", "male", "1988-09-09",
                "2025-10-12T10:00:00Z", "2025-10-13T12:00:00Z",
                [_cond("c-403", "pat-403", "O14.9", "Unspecified pre-eclampsia in pregnancy")],
                [_obs("o-403", "pat-403", "8480-6", "Systolic blood pressure", 156.0, "mmHg")],
                [_med("m-403", "pat-403", "200033", "Labetalol 100 MG Oral Tablet", "100 mg uống 2 lần/ngày")],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_c3",
                    "category": "CLINICAL_SEMANTIC",
                    "error_code": "SEX_DIAGNOSIS_MISMATCH",
                    "severity": "high",
                    "resources": ["Patient/pat-403", "Condition/c-403"],
                    "description": "Patient/pat-403 giới tính 'male' mâu thuẫn lâm sàng với chẩn đoán sản khoa O14.9 (Tiền sản giật).",
                }
            ],
        }
    )

    cases.append(
        {
            "case_id": "fhir_sem_04",
            "title": "[Lâm sàng] Chống chỉ định Nhi khoa (Trẻ 4 tuổi cúm kê Aspirin — Nguy cơ Reye)",
            "primary_category": "CLINICAL_SEMANTIC",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_sem_04", "pat-404", "Tran Bao Ngoc", "female", "2021-07-10",
                "2025-10-13T15:00:00Z", "2025-10-13T17:00:00Z",
                [_cond("c-404", "pat-404", "J11.1", "Influenza due to unidentified influenza virus")],
                [_obs("o-404", "pat-404", "8310-5", "Body temperature", 39.1, "Cel")],
                [_med("m-404", "pat-404", "1191", "Aspirin 325 MG Oral Tablet", "325 mg mỗi 6 giờ hạ sốt")],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_c4",
                    "category": "CLINICAL_SEMANTIC",
                    "error_code": "PEDIATRIC_CONTRAINDICATION",
                    "severity": "critical",
                    "resources": ["Patient/pat-404", "Condition/c-404", "MedicationRequest/m-404"],
                    "description": "Aspirin chống chỉ định cho trẻ 4 tuổi mắc cúm virus (J11.1) vì nguy cơ gây hội chứng Reye tử vong.",
                }
            ],
        }
    )

    cases.append(
        {
            "case_id": "fhir_sem_05",
            "title": "[Lâm sàng] Mâu thuẫn Xét nghiệm–Thuốc (Đường huyết 41 mg/dL vẫn tiêm Insulin nhanh)",
            "primary_category": "CLINICAL_SEMANTIC",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_sem_05", "pat-405", "Doan Minh Tuan", "male", "1960-03-03",
                "2025-10-14T06:30:00Z", "2025-10-15T10:00:00Z",
                [_cond("c-405", "pat-405", "E11.649", "Type 2 diabetes mellitus with hypoglycemia")],
                [_obs("o-405", "pat-405", "2339-0", "Glucose [Mass/volume] in Blood", 41.0, "mg/dL")],
                [_med("m-405", "pat-405", "253182", "Insulin Regular 100 UNT/ML Injectable Solution", "15 đơn vị tiêm tĩnh mạch ngay lập tức")],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_c5",
                    "category": "CLINICAL_SEMANTIC",
                    "error_code": "LAB_MEDICATION_CONFLICT",
                    "severity": "critical",
                    "resources": ["Observation/o-405", "Condition/c-405", "MedicationRequest/m-405"],
                    "description": "Bệnh nhân đang hạ đường huyết nặng (41 mg/dL) nhưng vẫn y lệnh tiêm 15 đơn vị Insulin Regular tĩnh mạch, đe dọa tính mạng.",
                }
            ],
        }
    )

    cases.append(
        {
            "case_id": "fhir_sem_06",
            "title": "[Lâm sàng] Nguy cơ xuất huyết nặng (INR = 6.4 vẫn tăng Warfarin + kèm Ibuprofen)",
            "primary_category": "CLINICAL_SEMANTIC",
            "profile": "INTL",
            "bundle": _base_bundle(
                "fhir_sem_06", "pat-406", "Nguyen Thi Yen", "female", "1950-12-19",
                "2025-10-15T08:00:00Z", "2025-10-16T14:00:00Z",
                [_cond("c-406", "pat-406", "I48.91", "Unspecified atrial fibrillation")],
                [_obs("o-406", "pat-406", "6301-6", "INR in Platelet poor plasma by Coagulation assay", 6.4, "{INR}")],
                [
                    _med("m-406a", "pat-406", "855332", "Warfarin Sodium 10 MG Oral Tablet", "10 mg mỗi ngày (tăng liều)"),
                    _med("m-406b", "pat-406", "197806", "Ibuprofen 600 MG Oral Tablet", "600 mg 3 lần/ngày"),
                ],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_c6",
                    "category": "CLINICAL_SEMANTIC",
                    "error_code": "LAB_MEDICATION_CONFLICT",
                    "severity": "critical",
                    "resources": ["Observation/o-406", "MedicationRequest/m-406a", "MedicationRequest/m-406b"],
                    "description": "INR quá ngưỡng điều trị nguy hiểm (6.4) nhưng vẫn tăng liều Warfarin 10 mg và phối hợp NSAID (Ibuprofen), gây nguy cơ xuất huyết ồ ạt.",
                }
            ],
        }
    )

    # Vietnam-specific Clinical Semantic Errors: Dengue + NSAID (Ibuprofen) & Severe Asthma + Non-selective Beta-blocker
    cases.append(
        {
            "case_id": "fhir_sem_07_vn",
            "title": "[Lâm sàng - VN] Sốt xuất huyết Dengue giảm tiểu cầu nhưng kê Ibuprofen (NSAID)",
            "primary_category": "CLINICAL_SEMANTIC",
            "profile": "VN-FHIR",
            "bundle": _base_bundle(
                "fhir_sem_07_vn", "pat-407", "Tran Minh Khoa", "male", "1994-08-22",
                "2025-10-16T10:00:00Z", "2025-10-18T11:00:00Z",
                [_cond("c-407", "pat-407", "A97.1", "Sốt xuất huyết Dengue có dấu hiệu cảnh báo")],
                [
                    _obs("o-407a", "pat-407", "8310-5", "Thân nhiệt", 39.3, "Cel"),
                    _obs("o-407b", "pat-407", "777-3", "Tiểu cầu (Platelets)", 48.0, "10*9/L"),
                ],
                [_med("m-407", "pat-407", "197806", "Ibuprofen 400 MG Oral Tablet", "400 mg uống mỗi 8 giờ để hạ sốt")],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_c7",
                    "category": "CLINICAL_SEMANTIC",
                    "error_code": "CONDITION_MEDICATION_CONFLICT",
                    "severity": "critical",
                    "resources": ["Condition/c-407", "Observation/o-407b", "MedicationRequest/m-407"],
                    "description": "Chống chỉ định nghiêm trọng theo phác đồ Bộ Y tế VN: Bệnh nhân Sốt xuất huyết Dengue (A97.1) có tiểu cầu giảm 48 G/L tuyệt đối không được dùng NSAID (Ibuprofen) vì nguy cơ xuất huyết tiêu hóa nặng.",
                }
            ],
        }
    )

    cases.append(
        {
            "case_id": "fhir_sem_08_vn",
            "title": "[Lâm sàng - VN] Cơn hen phế quản cấp nhưng kê Propranolol (chẹn beta không chọn lọc)",
            "primary_category": "CLINICAL_SEMANTIC",
            "profile": "VN-FHIR",
            "bundle": _base_bundle(
                "fhir_sem_08_vn", "pat-408", "Bui Thi Lan", "female", "1979-03-14",
                "2025-10-17T07:00:00Z", "2025-10-18T12:00:00Z",
                [
                    _cond("c-408a", "pat-408", "J45.901", "Hen phế quản đợt cấp nặng (Asthma with acute exacerbation)"),
                    _cond("c-408b", "pat-408", "I10", "Tăng huyết áp"),
                ],
                [_obs("o-408", "pat-408", "2708-6", "Độ bão hòa oxy SpO2", 91.0, "%")],
                [_med("m-408", "pat-408", "8787", "Propranolol 40 MG Oral Tablet", "40 mg uống 2 lần/ngày")],
                [],
            ),
            "ground_truth_issues": [
                {
                    "issue_id": "gt_c8",
                    "category": "CLINICAL_SEMANTIC",
                    "error_code": "CONDITION_MEDICATION_CONFLICT",
                    "severity": "critical",
                    "resources": ["Condition/c-408a", "MedicationRequest/m-408"],
                    "description": "Propranolol (chẹn beta không chọn lọc) chống chỉ định trên bệnh nhân hen phế quản đợt cấp (J45.901) do gây co thắt phế quản nặng.",
                }
            ],
        }
    )

    return cases


def generate_and_save(output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    cases = build_benchmark_cases()
    out_path = os.path.join(output_dir, "fhir_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(cases)} FHIR R4 (International + VN-FHIR) benchmark cases at: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HL7 FHIR R4 audit benchmark dataset.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    generate_and_save(args.output_dir)
