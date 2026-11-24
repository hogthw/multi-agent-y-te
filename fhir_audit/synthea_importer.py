"""
Synthea HL7 FHIR R4 Importer & Large-Scale Controlled Error Injector.

Allows importing 100-500 real Synthea-generated FHIR R4 patient bundles
(from https://github.com/synthetichealth/synthea) and automatically injecting
balanced, ground-truth-labeled errors across CLEAN, STRUCTURAL, TERMINOLOGY,
and CLINICAL_SEMANTIC categories for large-scale thesis benchmarking.

Usage:
  .venv/bin/python -m fhir_audit.synthea_importer \
      --synthea_dir /path/to/synthea/output/fhir \
      --num_cases 100 \
      --output my_datasets/processed/fhir_audit/synthea_benchmark_100.json
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple


KEEP_TYPES = {"Patient", "Encounter", "Condition", "Observation", "MedicationRequest", "AllergyIntolerance"}


def compact_synthea_bundle(raw_bundle: Dict[str, Any], max_per_type: int = 3) -> Dict[str, Any]:
    """Extract a concise, representative clinical slice from a verbose Synthea FHIR R4 Bundle."""
    counts: Dict[str, int] = {}
    selected_entries: List[Dict[str, Any]] = []
    patient_id = "pat-syn"

    for entry in raw_bundle.get("entry", []):
        res = entry.get("resource", {})
        rtype = res.get("resourceType")
        if rtype not in KEEP_TYPES:
            continue
        limit = 1 if rtype in ("Patient", "Encounter") else max_per_type
        if counts.get(rtype, 0) >= limit:
            continue
        counts[rtype] = counts.get(rtype, 0) + 1
        res_copy = copy.deepcopy(res)
        if rtype == "Patient":
            patient_id = res_copy.get("id", patient_id)
        selected_entries.append({"resource": res_copy})

    # Normalize subject/patient references to Patient/{patient_id}
    for e in selected_entries:
        r = e["resource"]
        if r.get("resourceType") in ("Encounter", "Condition", "Observation", "MedicationRequest"):
            r["subject"] = {"reference": f"Patient/{patient_id}"}
        elif r.get("resourceType") == "AllergyIntolerance":
            r["patient"] = {"reference": f"Patient/{patient_id}"}

    return {
        "resourceType": "Bundle",
        "id": raw_bundle.get("id", f"bundle-{patient_id}"),
        "type": "collection",
        "entry": selected_entries,
    }


def inject_error(bundle: Dict[str, Any], category: str, rng: random.Random) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Inject one controlled, ground-truth-labeled error into a compacted Synthea FHIR R4 Bundle."""
    b = copy.deepcopy(bundle)
    entries = b.get("entry", [])
    pat = next((e["resource"] for e in entries if e["resource"].get("resourceType") == "Patient"), None)
    pid = pat.get("id", "pat-1") if pat else "pat-1"

    if category == "CLEAN":
        return b, []

    if category == "STRUCTURAL":
        choice = rng.choice(["BROKEN_REFERENCE", "INVALID_PERIOD", "MISSING_REQUIRED_FIELD"])
        if choice == "BROKEN_REFERENCE":
            obs = next((e["resource"] for e in entries if e["resource"].get("resourceType") == "Observation"), None)
            if obs:
                obs["subject"] = {"reference": "Patient/nonexistent-999"}
                return b, [{
                    "category": "STRUCTURAL",
                    "error_code": "BROKEN_REFERENCE",
                    "severity": "high",
                    "resources": [f"Observation/{obs.get('id', 'obs')}"],
                    "description": "Observation.subject.reference trỏ tới Patient/nonexistent-999 không tồn tại trong Bundle.",
                }]
        if choice == "INVALID_PERIOD":
            enc = next((e["resource"] for e in entries if e["resource"].get("resourceType") == "Encounter"), None)
            if enc:
                enc["period"] = {"start": "2025-11-20T10:00:00Z", "end": "2025-11-15T08:00:00Z"}
                return b, [{
                    "category": "STRUCTURAL",
                    "error_code": "INVALID_PERIOD",
                    "severity": "medium",
                    "resources": [f"Encounter/{enc.get('id', 'enc')}"],
                    "description": "Encounter.period.end sớm hơn Encounter.period.start.",
                }]
        # Fallback: remove required Observation.status or Patient.gender
        if pat and "gender" in pat:
            del pat["gender"]
            return b, [{
                "category": "STRUCTURAL",
                "error_code": "MISSING_REQUIRED_FIELD",
                "severity": "medium",
                "resources": [f"Patient/{pid}"],
                "description": "Patient thiếu trường bắt buộc 'gender' theo chuẩn FHIR R4.",
            }]

    if category == "TERMINOLOGY":
        choice = rng.choice(["INVALID_UNIT", "IMPOSSIBLE_VALUE", "CODE_DISPLAY_MISMATCH"])
        if choice == "INVALID_UNIT":
            entries.append({
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-inj-unit",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "Hemoglobin A1c"}]},
                    "subject": {"reference": f"Patient/{pid}"},
                    "valueQuantity": {"value": 7.9, "unit": "mmHg", "system": "http://unitsofmeasure.org"},
                }
            })
            return b, [{
                "category": "TERMINOLOGY",
                "error_code": "INVALID_UNIT",
                "severity": "medium",
                "resources": ["Observation/obs-inj-unit"],
                "description": "HbA1c (LOINC 4548-4) ghi sai đơn vị 'mmHg' thay vì '%'.",
            }]
        if choice == "IMPOSSIBLE_VALUE":
            entries.append({
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-inj-val",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "2339-0", "display": "Glucose máu"}]},
                    "subject": {"reference": f"Patient/{pid}"},
                    "valueQuantity": {"value": 190.0, "unit": "mmol/L", "system": "http://unitsofmeasure.org"},
                }
            })
            return b, [{
                "category": "TERMINOLOGY",
                "error_code": "IMPOSSIBLE_VALUE",
                "severity": "high",
                "resources": ["Observation/obs-inj-val"],
                "description": "Glucose máu ghi 190 mmol/L bất khả thi sinh lý (nhầm đơn vị mg/dL sang mmol/L).",
            }]
        entries.append({
            "resource": {
                "resourceType": "Condition",
                "id": "cond-inj-icd",
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": "J45.9", "display": "Tăng huyết áp nguyên phát (THA)"}]},
                "subject": {"reference": f"Patient/{pid}"},
            }
        })
        return b, [{
            "category": "TERMINOLOGY",
            "error_code": "CODE_DISPLAY_MISMATCH",
            "severity": "high",
            "resources": ["Condition/cond-inj-icd"],
            "description": "Mã ICD-10 J45.9 (Hen phế quản) không khớp mô tả Tăng huyết áp (I10).",
        }]

    # CLINICAL_SEMANTIC
    choice = rng.choice(["DRUG_ALLERGY_CONFLICT", "RENAL_CONTRAINDICATION", "CONDITION_MEDICATION_CONFLICT"])
    if choice == "DRUG_ALLERGY_CONFLICT":
        entries.append({
            "resource": {
                "resourceType": "AllergyIntolerance",
                "id": "alg-inj-pen",
                "criticality": "high",
                "code": {"text": "Penicillin"},
                "patient": {"reference": f"Patient/{pid}"},
                "reaction": [{"manifestation": [{"text": "Anaphylaxis"}]}],
            }
        })
        entries.append({
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "med-inj-amox",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "308191", "display": "Amoxicillin 500 MG"}]},
                "subject": {"reference": f"Patient/{pid}"},
            }
        })
        return b, [{
            "category": "CLINICAL_SEMANTIC",
            "error_code": "DRUG_ALLERGY_CONFLICT",
            "severity": "critical",
            "resources": ["MedicationRequest/med-inj-amox", "AllergyIntolerance/alg-inj-pen"],
            "description": "Kê Amoxicillin dù bệnh nhân có tiền sử phản vệ nặng với Penicillin.",
        }]

    if choice == "RENAL_CONTRAINDICATION":
        entries.append({
            "resource": {
                "resourceType": "Observation",
                "id": "obs-inj-egfr",
                "status": "final",
                "code": {"coding": [{"system": "http://loinc.org", "code": "33914-3", "display": "eGFR"}]},
                "subject": {"reference": f"Patient/{pid}"},
                "valueQuantity": {"value": 19.0, "unit": "mL/min/1.73m2", "system": "http://unitsofmeasure.org"},
            }
        })
        entries.append({
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "med-inj-met",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "861004", "display": "Metformin 1000 MG"}]},
                "subject": {"reference": f"Patient/{pid}"},
            }
        })
        return b, [{
            "category": "CLINICAL_SEMANTIC",
            "error_code": "RENAL_CONTRAINDICATION",
            "severity": "critical",
            "resources": ["MedicationRequest/med-inj-met", "Observation/obs-inj-egfr"],
            "description": "Chống chỉ định kê Metformin khi eGFR = 19 mL/min/1.73m2 (< 30).",
        }]

    # Dengue + NSAID
    entries.append({
        "resource": {
            "resourceType": "Condition",
            "id": "cond-inj-dengue",
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": "A97.1", "display": "Sốt xuất huyết Dengue"}]},
            "subject": {"reference": f"Patient/{pid}"},
        }
    })
    entries.append({
        "resource": {
            "resourceType": "MedicationRequest",
            "id": "med-inj-ibu",
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "197806", "display": "Ibuprofen 400 MG"}]},
            "subject": {"reference": f"Patient/{pid}"},
        }
    })
    return b, [{
        "category": "CLINICAL_SEMANTIC",
        "error_code": "CONDITION_MEDICATION_CONFLICT",
        "severity": "critical",
        "resources": ["Condition/cond-inj-dengue", "MedicationRequest/med-inj-ibu"],
        "description": "Chống chỉ định kê Ibuprofen (NSAID) cho bệnh nhân Sốt xuất huyết Dengue (A97.1).",
    }]


def import_synthea_directory(synthea_dir: str, num_cases: int, output_path: str, seed: int = 42) -> None:
    files = sorted(glob.glob(os.path.join(synthea_dir, "*.json")))
    files = [f for f in files if "hospitalInformation" not in f and "practitionerInformation" not in f]
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file FHIR JSON của Synthea trong thư mục: {synthea_dir}")

    rng = random.Random(seed)
    rng.shuffle(files)
    selected = files[:num_cases]
    categories = ["CLEAN", "STRUCTURAL", "TERMINOLOGY", "CLINICAL_SEMANTIC"]

    dataset: List[Dict[str, Any]] = []
    for idx, fpath in enumerate(selected):
        with open(fpath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        compacted = compact_synthea_bundle(raw)
        cat = categories[idx % len(categories)]
        mutated, gt = inject_error(compacted, cat, rng)
        cid = f"synthea_{cat.lower()}_{idx + 1:03d}"
        mutated["id"] = cid
        dataset.append({
            "case_id": cid,
            "title": f"[Synthea - {cat}] Hồ sơ #{idx + 1:03d}",
            "primary_category": cat,
            "profile": "SYNTHEA-R4",
            "bundle": mutated,
            "ground_truth_issues": gt,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Imported {len(dataset)} Synthea FHIR R4 benchmark cases -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Synthea FHIR R4 bundles and inject controlled ground-truth errors.")
    parser.add_argument("--synthea_dir", type=str, required=True, help="Directory containing Synthea output/fhir/*.json")
    parser.add_argument("--num_cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="my_datasets/processed/fhir_audit/synthea_benchmark.json")
    args = parser.parse_args()
    import_synthea_directory(args.synthea_dir, args.num_cases, args.output, args.seed)
