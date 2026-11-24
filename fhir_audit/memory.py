"""
Two-Tier Memory-Augmented Module for HL7 FHIR R4 Multi-Agent Auditing (incl. Vietnam Profile).

Tier 1 — Semantic Memory (FHIRSemanticMemory):
  Stores HL7 FHIR R4 structural rules, Vietnam QĐ 130/QĐ-BYT identifier rules,
  clinical terminology mappings (LOINC/ICD-10/UCUM + Vietnam SI units & abbreviations),
  and clinical safety guidelines (including Vietnam Dengue + NSAID & Asthma + Beta-blocker).

Tier 2 — Episodic Memory (FHIREpisodicMemory):
  Stores past audited FHIR cases, subtle clinical error patterns, and False-Positive Traps
  (including Vietnam SI mmol/L clean cases and Dengue + Paracetamol valid cases).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np


TOKEN_RE = re.compile(r"[a-zA-Z0-9_\.\-/À-ỹ]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) >= 2]


class LightweightVectorIndex:
    """Pure-NumPy TF-IDF cosine similarity retriever for transparent, reproducible research."""

    def __init__(self, docs: List[Dict[str, Any]], text_key: str = "search_text"):
        self.docs = docs
        self.text_key = text_key
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._rebuild()

    def _rebuild(self) -> None:
        if not self.docs:
            self.vocab = {}
            self.idf = np.array([])
            self.matrix = np.zeros((0, 0), dtype=np.float32)
            return

        tokenized = [_tokenize(d.get(self.text_key, "")) for d in self.docs]
        df: Counter = Counter()
        for toks in tokenized:
            for tok in set(toks):
                df[tok] += 1

        self.vocab = {tok: idx for idx, tok in enumerate(sorted(df.keys()))}
        n_docs = len(self.docs)
        v_size = len(self.vocab)
        self.idf = np.zeros(v_size, dtype=np.float32)
        for tok, idx in self.vocab.items():
            self.idf[idx] = math.log((n_docs + 1) / (df[tok] + 1)) + 1.0

        mat = np.zeros((n_docs, v_size), dtype=np.float32)
        for row_i, toks in enumerate(tokenized):
            if not toks:
                continue
            tf = Counter(toks)
            for tok, cnt in tf.items():
                col = self.vocab[tok]
                mat[row_i, col] = (1.0 + math.log(cnt)) * self.idf[col]
            norm = np.linalg.norm(mat[row_i])
            if norm > 0:
                mat[row_i] /= norm
        self.matrix = mat

    def add(self, doc: Dict[str, Any]) -> None:
        self.docs.append(doc)
        self._rebuild()

    def search(
        self,
        query: str,
        top_k: int = 3,
        domain_filter: Optional[str] = None,
        min_score: float = 0.03,
    ) -> List[Dict[str, Any]]:
        if not self.docs or not self.vocab:
            return []

        toks = _tokenize(query)
        if not toks:
            return []

        q_vec = np.zeros(len(self.vocab), dtype=np.float32)
        tf = Counter(toks)
        for tok, cnt in tf.items():
            idx = self.vocab.get(tok)
            if idx is not None:
                q_vec[idx] = (1.0 + math.log(cnt)) * self.idf[idx]
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec /= q_norm

        scores = self.matrix @ q_vec
        ranked = np.argsort(-scores)
        results: List[Dict[str, Any]] = []
        for idx in ranked:
            sc = float(scores[idx])
            if sc < min_score:
                break
            doc = self.docs[int(idx)]
            if domain_filter and doc.get("domain") not in (domain_filter, "ALL"):
                continue
            results.append({**doc, "similarity": round(sc, 4)})
            if len(results) >= top_k:
                break
        return results


# =============================================================================
# TIER 1: HL7 FHIR R4 + VIETNAM PROFILE SEMANTIC KNOWLEDGE BASE
# =============================================================================
SEMANTIC_GUIDELINES: List[Dict[str, Any]] = [
    {
        "id": "SEM-STRUCT-01",
        "domain": "STRUCTURAL",
        "title": "HL7 FHIR R4 Mandatory Fields & Reference Integrity",
        "content": (
            "Every FHIR R4 Bundle must contain valid references: Observation.subject.reference, "
            "Condition.subject.reference, MedicationRequest.subject.reference, Encounter.subject.reference, "
            "and AllergyIntolerance.patient.reference MUST resolve to an existing Patient/{id} inside the Bundle. "
            "Required fields: Patient(id, gender, birthDate), Encounter(status, class, subject), "
            "Observation(status, code, subject), MedicationRequest(status, intent, medicationCodeableConcept, subject)."
        ),
        "search_text": "bundle patient encounter observation medicationrequest allergyintolerance reference broken missing required status intent subject gender",
    },
    {
        "id": "SEM-STRUCT-02",
        "domain": "STRUCTURAL",
        "title": "HL7 FHIR Temporal Consistency (Period & BirthDate)",
        "content": (
            "Encounter.period.start MUST be chronologically earlier than or equal to Encounter.period.end. "
            "Patient.birthDate MUST NOT be in the future relative to Encounter.period.start."
        ),
        "search_text": "encounter period start end birthdate temporal chronology date future earlier",
    },
    {
        "id": "SEM-STRUCT-VN03",
        "domain": "STRUCTURAL",
        "title": "Chuẩn định danh VN-FHIR / QĐ 130/QĐ-BYT (CCCD & BHYT)",
        "content": (
            "Theo chuẩn liên thông dữ liệu y tế Việt Nam (QĐ 130/QĐ-BYT): Nếu Patient.identifier chứa "
            "CCCD (system .../cccd) thì phải đúng 12 chữ số; nếu chứa mã thẻ BHYT (system .../bhyt) "
            "thì phải đúng định dạng 10 chữ số (BHXH) hoặc 15 ký tự (mã thẻ BHYT chuẩn). Sai độ dài là INVALID_IDENTIFIER."
        ),
        "search_text": "vietnam cccd bhyt identifier moh.gov.vn qd 130 patient invalid_identifier",
    },
    {
        "id": "SEM-TERM-01",
        "domain": "TERMINOLOGY",
        "title": "LOINC, UCUM & Đơn vị SI Việt Nam (mmol/L vs mg/dL)",
        "content": (
            "Standard LOINC + UCUM + Vietnam SI physiological ranges: "
            "4548-4 = HbA1c (% or mmol/mol, NEVER mmHg); "
            "2339-0 = Blood Glucose: mg/dL (plausible 20-600 mg/dL) OR Vietnam SI mmol/L (plausible 1.1-33.0 mmol/L; "
            "values like 185 mmol/L indicate mg/dL-to-mmol/L unit confusion = IMPOSSIBLE_VALUE); "
            "33914-3 = eGFR (mL/min/1.73m2); 777-3 = Platelets (10*9/L); "
            "8480-6 = Systolic BP (mmHg) > 8462-4 = Diastolic BP (mmHg); "
            "8310-5 = Body temperature (34-43 Cel); 2708-6 = SpO2 (0-100%, NEVER > 100%); 6301-6 = INR ({INR})."
        ),
        "search_text": "loinc ucum unit hba1c 4548-4 glucose 2339-0 mmol/l mg/dl egfr 33914-3 systolic 8480-6 diastolic 8462-4 temperature 8310-5 spo2 2708-6 inr 6301-6 777-3 platelets mmhg cel",
    },
    {
        "id": "SEM-TERM-02",
        "domain": "TERMINOLOGY",
        "title": "Đối chiếu mã ICD-10 & Viết tắt lâm sàng Việt Nam",
        "content": (
            "Condition.code.coding code MUST match its display text (including Vietnamese abbreviations): "
            "E11.* = Đái tháo đường type 2 (ĐTĐ type 2 / Type 2 diabetes); "
            "I10 = Tăng huyết áp nguyên phát (THA / Essential hypertension); "
            "J45.* = Hen phế quản (Asthma); J18.* = Viêm phổi (Pneumonia); "
            "N18.* = Suy thận mạn (STM / CKD); A97.* = Sốt xuất huyết Dengue (SXH Dengue); "
            "I48.* = Rung nhĩ (Atrial fibrillation); O14.* / Z34.* = Sản khoa/Thai kỳ."
        ),
        "search_text": "icd-10 condition code display mismatch e11 đtđ diabetes i10 tha hypertension j45 hen asthma j18 pneumonia n18 stm ckd a97 dengue o14 pregnancy",
    },
    {
        "id": "SEM-CLIN-01",
        "domain": "CLINICAL_SEMANTIC",
        "title": "Xung đột Dị ứng Penicillin / Beta-Lactam",
        "content": (
            "If AllergyIntolerance records high-criticality allergy or anaphylaxis to Penicillin, "
            "prescribing Penicillin-class beta-lactams (Amoxicillin, Ampicillin, Piperacillin, Penicillin) "
            "is a CRITICAL DRUG_ALLERGY_CONFLICT. Macrolides (Azithromycin) or Amlodipine are SAFE alternatives."
        ),
        "search_text": "allergyintolerance penicillin amoxicillin ampicillin beta-lactam anaphylaxis medicationrequest conflict azithromycin amlodipine",
    },
    {
        "id": "SEM-CLIN-02",
        "domain": "CLINICAL_SEMANTIC",
        "title": "Chống chỉ định Thận (eGFR < 30) — Metformin",
        "content": (
            "Metformin is ABSOLUTELY CONTRAINDICATED when eGFR < 30 mL/min/1.73m2 (CKD stage 4/5, N18.4/N18.5) "
            "due to lactic acidosis risk. Insulin Glargine is an acceptable safe alternative in CKD stage 4 with hyperglycemia."
        ),
        "search_text": "metformin egfr renal kidney ckd n18.4 33914-3 lactic acidosis insulin glargine contraindication",
    },
    {
        "id": "SEM-CLIN-03",
        "domain": "CLINICAL_SEMANTIC",
        "title": "Tương thích Giới tính & Chẩn đoán Sản khoa",
        "content": (
            "Obstetric and pregnancy conditions (ICD-10 O00-O99 such as O14.9 Pre-eclampsia, Z34.*) "
            "are clinically incompatible with Patient.gender = 'male' (SEX_DIAGNOSIS_MISMATCH)."
        ),
        "search_text": "patient gender male female pregnancy pre-eclampsia o14.9 z34 obstetric sex diagnosis mismatch",
    },
    {
        "id": "SEM-CLIN-04",
        "domain": "CLINICAL_SEMANTIC",
        "title": "An toàn Nhi khoa — Hội chứng Reye (Aspirin)",
        "content": (
            "Aspirin (salicylates) is CONTRAINDICATED in children (< 16 years old) with viral illness or influenza "
            "(J11.*, B34.*) due to Reye's syndrome risk. Paracetamol (Acetaminophen) is safe."
        ),
        "search_text": "pediatric child aspirin influenza viral j11 b34 reye syndrome acetaminophen paracetamol fever birthdate",
    },
    {
        "id": "SEM-CLIN-05",
        "domain": "CLINICAL_SEMANTIC",
        "title": "Xung đột Xét nghiệm nguy kịch – Thuốc (Hạ đường huyết/Insulin & INR cao/Warfarin)",
        "content": (
            "1) Blood Glucose < 70 mg/dL (or < 3.9 mmol/L) with IV/rapid Insulin order is a CRITICAL LAB_MEDICATION_CONFLICT. "
            "2) Supratherapeutic INR > 5.0 with increased Warfarin or concurrent NSAID (Ibuprofen) causes severe hemorrhage risk."
        ),
        "search_text": "hypoglycemia glucose insulin warfarin inr 6301-6 ibuprofen nsaid bleeding hemorrhage lab medication conflict",
    },
    {
        "id": "SEM-CLIN-VN06",
        "domain": "CLINICAL_SEMANTIC",
        "title": "Phác đồ Bộ Y tế VN: Sốt xuất huyết Dengue (A97) & Hen phế quản (J45)",
        "content": (
            "1) Sốt xuất huyết Dengue (ICD-10 A97.*, đặc biệt khi giảm tiểu cầu): TUYỆT ĐỐI CHỐNG CHỈ ĐỊNH NSAID "
            "(Ibuprofen, Aspirin, Diclofenac) vì gây xuất huyết tiêu hóa nặng (CONDITION_MEDICATION_CONFLICT). "
            "Paracetamol là thuốc hạ sốt an toàn được Bộ Y tế khuyến cáo. "
            "2) Hen phế quản đợt cấp (J45.*): Chống chỉ định thuốc chẹn beta không chọn lọc (Propranolol) do gây co thắt phế quản nặng."
        ),
        "search_text": "dengue a97 sốt xuất huyết tiểu cầu platelets ibuprofen aspirin nsaid paracetamol hen phế quản asthma j45 propranolol beta blocker",
    },
]


# =============================================================================
# TIER 2: EPISODIC MEMORY SEED (Past audit lessons & Vietnam False-Positive Guards)
# =============================================================================
DEFAULT_EPISODIC_LESSONS: List[Dict[str, Any]] = [
    {
        "id": "EPI-FP-01",
        "domain": "ALL",
        "outcome": "FALSE_POSITIVE_TRAP",
        "title": "Bài học: Dị ứng Penicillin dùng Azithromycin hoặc Amlodipine là HỢP LỆ",
        "lesson": (
            "Không báo lỗi nhầm khi bệnh nhân dị ứng Penicillin nhưng được kê thuốc khác nhóm (Azithromycin, Amlodipine). "
            "Chỉ cảnh báo khi thuốc thuộc nhóm Penicillin/Beta-lactam (Amoxicillin, Ampicillin)."
        ),
        "search_text": "penicillin allergy azithromycin amlodipine safe valid non-beta-lactam false positive trap",
    },
    {
        "id": "EPI-FP-02",
        "domain": "ALL",
        "outcome": "FALSE_POSITIVE_TRAP",
        "title": "Bài học: Suy thận độ 4 dùng Insulin (không dùng Metformin) là HỢP LỆ",
        "lesson": (
            "Khi eGFR < 30 kèm đái tháo đường nhưng bác sĩ đã chuyển sang Insulin Glargine (đường huyết cao > 180 mg/dL) "
            "và KHÔNG kê Metformin thì hồ sơ hoàn toàn hợp lệ."
        ),
        "search_text": "ckd stage 4 egfr insulin glargine diabetes hyperglycemia safe valid false positive",
    },
    {
        "id": "EPI-FP-VN03",
        "domain": "ALL",
        "outcome": "FALSE_POSITIVE_TRAP",
        "title": "Bài học VN: Glucose 5-15 mmol/L (đơn vị SI) & Dengue dùng Paracetamol là HỢP LỆ",
        "lesson": (
            "1) Bệnh viện Việt Nam dùng mmol/L cho Glucose (7.8 mmol/L ≈ 140 mg/dL là hợp lý, KHÔNG phải hạ đường huyết). "
            "2) Sốt xuất huyết Dengue (A97.0) dùng Paracetamol hạ sốt là đúng phác đồ Bộ Y tế — chỉ báo lỗi khi kê Ibuprofen/Aspirin (NSAID)."
        ),
        "search_text": "vietnam mmol/l glucose si unit dengue a97 paracetamol safe valid false positive",
    },
    {
        "id": "EPI-ERR-01",
        "domain": "CLINICAL_SEMANTIC",
        "outcome": "CONFIRMED_ERROR",
        "title": "Bài học: Đối chiếu chéo Giới tính vs ICD-10 sản khoa & Tuổi nhi vs Aspirin",
        "lesson": (
            "Lỗi ngữ nghĩa thường bị bỏ sót nếu chỉ đọc từng Resource rời: luôn đối chiếu Patient.gender với mã ICD-10 chương O (O14.9) "
            "và tính tuổi từ Patient.birthDate so với Encounter khi gặp Aspirin ở bệnh nhân cúm/virus."
        ),
        "search_text": "male gender condition o14 pregnancy pre-eclampsia obstetric mismatch pediatric aspirin child",
    },
    {
        "id": "EPI-ERR-02",
        "domain": "TERMINOLOGY",
        "outcome": "CONFIRMED_ERROR",
        "title": "Bài học: Kiểm tra lệch mã ICD-10/LOINC (gồm viết tắt VN) & nhầm đơn vị mmol/L vs mg/dL",
        "lesson": (
            "Các lỗi tinh vi đã xác nhận: (1) Viết tắt 'THA' nhưng gắn nhầm mã E11.9, (2) Glucose ghi 185 mmol/L (nhầm từ mg/dL), "
            "(3) LOINC 8480-6 (Huyết áp) gắn cho Glucose, (4) SBP < DBP, (5) SpO2 > 100% hoặc thân nhiệt > 45 Cel."
        ),
        "search_text": "loinc icd-10 code display mismatch tha đtđ mmol/l mg/dl systolic diastolic spo2 temperature unit ucum",
    },
    {
        "id": "EPI-ERR-03",
        "domain": "CLINICAL_SEMANTIC",
        "outcome": "CONFIRMED_ERROR",
        "title": "Bài học: Kiểm tra chéo Xét nghiệm nguy kịch & Chống chỉ định đặc thù VN (Dengue/Hen)",
        "lesson": (
            "Các lỗi mức CRITICAL đã xác nhận: Metformin khi eGFR < 30, Insulin khi Glucose < 55 mg/dL, "
            "Warfarin + Ibuprofen khi INR > 5.0, Ibuprofen (NSAID) ở Sốt xuất huyết Dengue (A97), và Propranolol ở cơn hen cấp (J45)."
        ),
        "search_text": "inr warfarin ibuprofen glucose hypoglycemia insulin egfr metformin dengue a97 asthma j45 propranolol",
    },
]


class FHIRMemorySystem:
    """Unified interface for Two-Tier Semantic Memory + Episodic Memory."""

    def __init__(self, episodic_store_path: Optional[str] = None, use_semantic: bool = True, use_episodic: bool = True):
        self.use_semantic = use_semantic
        self.use_episodic = use_episodic
        self.episodic_store_path = episodic_store_path

        self.semantic_index = LightweightVectorIndex(list(SEMANTIC_GUIDELINES), text_key="search_text")
        episodic_docs = list(DEFAULT_EPISODIC_LESSONS)
        if episodic_store_path and os.path.exists(episodic_store_path):
            try:
                with open(episodic_store_path, "r", encoding="utf-8") as f:
                    extra = json.load(f)
                if isinstance(extra, list):
                    existing_ids = {d["id"] for d in episodic_docs}
                    for item in extra:
                        if item.get("id") not in existing_ids:
                            episodic_docs.append(item)
            except Exception:
                pass
        self.episodic_index = LightweightVectorIndex(episodic_docs, text_key="search_text")

    def extract_bundle_query(self, bundle: Dict[str, Any]) -> str:
        parts: List[str] = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            rtype = res.get("resourceType", "")
            parts.append(rtype)
            if rtype == "Patient":
                parts.append(str(res.get("gender", "")))
                parts.append(str(res.get("birthDate", "")))
                for ident in res.get("identifier", []):
                    parts.extend([str(ident.get("system", "")), str(ident.get("value", ""))])
            elif rtype == "Condition":
                for c in res.get("code", {}).get("coding", []):
                    parts.extend([str(c.get("code", "")), str(c.get("display", ""))])
            elif rtype == "Observation":
                for c in res.get("code", {}).get("coding", []):
                    parts.extend([str(c.get("code", "")), str(c.get("display", ""))])
                vq = res.get("valueQuantity", {})
                parts.extend([str(vq.get("value", "")), str(vq.get("unit", ""))])
            elif rtype == "MedicationRequest":
                for c in res.get("medicationCodeableConcept", {}).get("coding", []):
                    parts.extend([str(c.get("code", "")), str(c.get("display", ""))])
            elif rtype == "AllergyIntolerance":
                parts.append(str(res.get("code", {}).get("text", "")))
                parts.append(str(res.get("criticality", "")))
        return " ".join(p for p in parts if p)

    def retrieve(self, bundle: Dict[str, Any], domain: Optional[str] = None, top_k_sem: int = 3, top_k_epi: int = 2) -> Dict[str, Any]:
        query = self.extract_bundle_query(bundle)
        semantic_hits = (
            self.semantic_index.search(query, top_k=top_k_sem, domain_filter=domain)
            if self.use_semantic
            else []
        )
        episodic_hits = (
            self.episodic_index.search(query, top_k=top_k_epi, domain_filter=domain)
            if self.use_episodic
            else []
        )
        return {
            "query_preview": query[:180],
            "semantic_hits": semantic_hits,
            "episodic_hits": episodic_hits,
            "total_hits": len(semantic_hits) + len(episodic_hits),
        }

    def format_memory_prompt(self, retrieved: Dict[str, Any]) -> str:
        sections: List[str] = []
        if retrieved.get("semantic_hits"):
            lines = ["[SEMANTIC MEMORY — Quy chuẩn HL7 FHIR R4, VN-FHIR & An toàn lâm sàng]"]
            for h in retrieved["semantic_hits"]:
                lines.append(f"- ({h['id']}, sim={h['similarity']}) {h['title']}: {h['content']}")
            sections.append("\n".join(lines))

        if retrieved.get("episodic_hits"):
            lines = ["[EPISODIC MEMORY — Kinh nghiệm rà soát quá khứ & Phòng tránh báo động giả]"]
            for h in retrieved["episodic_hits"]:
                lines.append(f"- ({h['id']}, {h['outcome']}, sim={h['similarity']}) {h['title']}: {h['lesson']}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def record_lesson(self, lesson_id: str, domain: str, outcome: str, title: str, lesson: str, search_text: str) -> None:
        doc = {
            "id": lesson_id,
            "domain": domain,
            "outcome": outcome,
            "title": title,
            "lesson": lesson,
            "search_text": search_text,
        }
        self.episodic_index.add(doc)
        if self.episodic_store_path:
            os.makedirs(os.path.dirname(self.episodic_store_path), exist_ok=True)
            with open(self.episodic_store_path, "w", encoding="utf-8") as f:
                json.dump(self.episodic_index.docs, f, ensure_ascii=False, indent=2)
