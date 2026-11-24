"""
Baseline 1 (Conventional / Non-LLM): Deterministic Rule-Based HL7 FHIR R4 Validator.

Checks Bundle structure, mandatory R4 fields, reference integrity, temporal ordering,
Vietnam CCCD/BHYT identifier format (QĐ 130/QĐ-BYT), and basic UCUM unit compatibility.
Does NOT perform deep cross-resource clinical semantic reasoning.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Dict, List


REQUIRED_FIELDS = {
    "Patient": ["id", "gender", "birthDate"],
    "Encounter": ["id", "status", "class", "subject"],
    "Condition": ["id", "code", "subject"],
    "Observation": ["id", "status", "code", "subject"],
    "MedicationRequest": ["id", "status", "intent", "medicationCodeableConcept", "subject"],
    "AllergyIntolerance": ["id", "code", "patient"],
}

ALLOWED_LOINC_UNITS = {
    "4548-4": {"%", "mmol/mol"},
    "2339-0": {"mg/dL", "mmol/L"},
    "33914-3": {"mL/min/1.73m2", "mL/min"},
    "8480-6": {"mmHg"},
    "8462-4": {"mmHg"},
    "8310-5": {"Cel", "[degF]"},
    "2708-6": {"%"},
    "6301-6": {"{INR}", "1"},
    "718-7": {"g/dL"},
    "2093-3": {"mg/dL", "mmol/L"},
    "3016-3": {"mIU/L", "uIU/mL"},
    "777-3": {"10*9/L", "10*3/uL"},
}


def _parse_iso(date_str: str) -> datetime | None:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d")
        except Exception:
            return None


def validate_fhir_bundle_rule_based(bundle: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    issues: List[Dict[str, Any]] = []

    if bundle.get("resourceType") != "Bundle":
        issues.append(
            {
                "category": "STRUCTURAL",
                "error_code": "INVALID_BUNDLE",
                "severity": "critical",
                "resources": ["Bundle"],
                "description": "Root resourceType không phải là 'Bundle'.",
            }
        )

    entries = bundle.get("entry", [])
    resources = [e.get("resource", {}) for e in entries if isinstance(e, dict) and isinstance(e.get("resource"), dict)]
    present_refs = {f"{r.get('resourceType')}/{r.get('id')}" for r in resources if r.get("resourceType") and r.get("id")}

    patient_birth_dt = None
    patient_ref_id = None
    for r in resources:
        if r.get("resourceType") == "Patient":
            patient_birth_dt = _parse_iso(r.get("birthDate", ""))
            patient_ref_id = f"Patient/{r.get('id')}"

    for r in resources:
        rtype = r.get("resourceType", "Unknown")
        rid = r.get("id", "unknown")
        full_ref = f"{rtype}/{rid}"

        # 1. Required fields check
        missing = [field for field in REQUIRED_FIELDS.get(rtype, []) if field not in r or r.get(field) in (None, "", [])]
        if missing:
            issues.append(
                {
                    "category": "STRUCTURAL",
                    "error_code": "MISSING_REQUIRED_FIELD",
                    "severity": "high" if len(missing) > 1 else "medium",
                    "resources": [full_ref],
                    "description": f"{full_ref} thiếu trường bắt buộc FHIR R4: {', '.join(missing)}.",
                }
            )

        # 2. Vietnam QĐ 130/QĐ-BYT identifier check on Patient
        if rtype == "Patient":
            for ident in r.get("identifier", []):
                sys_uri = str(ident.get("system", "")).lower()
                val = str(ident.get("value", "")).strip()
                if "cccd" in sys_uri and not re.fullmatch(r"\d{12}", val):
                    issues.append(
                        {
                            "category": "STRUCTURAL",
                            "error_code": "INVALID_IDENTIFIER",
                            "severity": "medium",
                            "resources": [full_ref],
                            "description": f"{full_ref} có mã CCCD '{val}' không đúng chuẩn 12 chữ số.",
                        }
                    )
                if "bhyt" in sys_uri and not (re.fullmatch(r"\d{10}", val) or re.fullmatch(r"[A-Z0-9]{15}", val)):
                    issues.append(
                        {
                            "category": "STRUCTURAL",
                            "error_code": "INVALID_IDENTIFIER",
                            "severity": "medium",
                            "resources": [full_ref],
                            "description": f"{full_ref} có mã thẻ BHYT '{val}' không đúng chuẩn 10 hoặc 15 ký tự (QĐ 130/QĐ-BYT).",
                        }
                    )

        # 3. Reference integrity check
        for ref_key in ("subject", "patient"):
            ref_obj = r.get(ref_key)
            if isinstance(ref_obj, dict) and ref_obj.get("reference"):
                target = ref_obj["reference"]
                if target not in present_refs:
                    issues.append(
                        {
                            "category": "STRUCTURAL",
                            "error_code": "BROKEN_REFERENCE",
                            "severity": "high",
                            "resources": [full_ref],
                            "description": f"{full_ref}.{ref_key}.reference trỏ tới '{target}' không tồn tại trong Bundle.",
                        }
                    )

        # 4. Temporal checks
        if rtype == "Encounter":
            period = r.get("period", {})
            start_dt = _parse_iso(period.get("start", ""))
            end_dt = _parse_iso(period.get("end", ""))
            if start_dt and end_dt and end_dt < start_dt:
                issues.append(
                    {
                        "category": "STRUCTURAL",
                        "error_code": "INVALID_PERIOD",
                        "severity": "medium",
                        "resources": [full_ref],
                        "description": f"{full_ref}.period.end ({period.get('end')}) sớm hơn start ({period.get('start')}).",
                    }
                )
            if patient_birth_dt and start_dt:
                if patient_birth_dt.replace(tzinfo=None) > start_dt.replace(tzinfo=None):
                    issues.append(
                        {
                            "category": "STRUCTURAL",
                            "error_code": "INVALID_BIRTHDATE",
                            "severity": "high",
                            "resources": [patient_ref_id or "Patient", full_ref],
                            "description": f"Patient.birthDate lớn hơn ngày khám Encounter.period.start ({period.get('start')}).",
                        }
                    )

        # 5. Basic UCUM unit check
        if rtype == "Observation":
            codings = r.get("code", {}).get("coding", [])
            loinc = codings[0].get("code") if codings else None
            unit = r.get("valueQuantity", {}).get("unit")
            if loinc in ALLOWED_LOINC_UNITS and unit and unit not in ALLOWED_LOINC_UNITS[loinc]:
                issues.append(
                    {
                        "category": "TERMINOLOGY",
                        "error_code": "INVALID_UNIT",
                        "severity": "medium",
                        "resources": [full_ref],
                        "description": f"{full_ref} (LOINC {loinc}) dùng đơn vị '{unit}', chuẩn cho phép: {sorted(ALLOWED_LOINC_UNITS[loinc])}.",
                    }
                )

    latency_ms = max(1, int((time.perf_counter() - t0) * 1000))
    has_issues = len(issues) > 0
    return {
        "method": "rule_based",
        "has_issues": has_issues,
        "issues": issues,
        "summary": (
            f"Bộ kiểm tra xác định (Rule-based) phát hiện {len(issues)} lỗi cấu trúc/đơn vị."
            if has_issues
            else "Bộ kiểm tra xác định không phát hiện lỗi cú pháp/schema (không kiểm tra được ngữ nghĩa lâm sàng chéo)."
        ),
        "latency_ms": latency_ms,
        "total_tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "total_rounds": 1,
        "consensus_reached": True,
        "memory_hits": 0,
    }
