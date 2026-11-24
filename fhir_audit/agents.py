"""
Async Event-Driven LLM & Memory-Augmented Multi-Agent Engine for HL7 FHIR R4 Auditing.

Supports 6 experimental configurations for PPNCKH benchmarking & ablation study:
  1. rule_based         : Conventional deterministic FHIR R4 validator (0 tokens)
  2. single_llm         : Single-LLM zero-shot baseline (no memory)
  3. single_llm_memory  : Single-LLM + Semantic & Episodic Memory (RAG baseline)
  4. multi_agent_no_mem : ColaCare Multi-Agent (3 Auditors + Meta Adjudicator, no memory)
  5. multi_agent_sem    : Ablation — Multi-Agent + Semantic Memory only
  6. multi_agent_memory : Proposed — Multi-Agent + Two-Tier Memory (Semantic + Episodic)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from fhir_audit.memory import FHIRMemorySystem
from fhir_audit.rate_limiter import call_llm_with_rate_limit
from fhir_audit.rule_validator import validate_fhir_bundle_rule_based
from medagentboard.utils.json_utils import preprocess_response_string
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS


FHIR_LOGS_DIR = os.path.join("logs", "fhir_audit", "sessions")
EPISODIC_STORE_PATH = os.path.join("logs", "fhir_audit", "episodic_memory.json")

FHIR_AUDITORS = [
    {
        "id": "doctor_1",
        "name": "BS. Linh",
        "role_en": "FHIR Structural Auditor",
        "role_vi": "Kiểm toán Cấu trúc FHIR",
        "domain": "STRUCTURAL",
        "color": "#0284c7",
        "focus": (
            "Check HL7 FHIR R4 structural validity: mandatory fields (Patient.gender/birthDate, "
            "Observation.status/code/subject, MedicationRequest.status/intent/subject), reference integrity "
            "(all subject/patient references must exist in Bundle), and chronological validity "
            "(Encounter.period.start <= end; Patient.birthDate <= Encounter.period.start)."
        ),
    },
    {
        "id": "doctor_2",
        "name": "BS. Minh",
        "role_en": "Clinical Semantic Auditor",
        "role_vi": "Kiểm toán Lâm sàng Chéo",
        "domain": "CLINICAL_SEMANTIC",
        "color": "#e11d48",
        "focus": (
            "Check cross-resource clinical safety and semantic consistency across Patient, Condition, "
            "Observation, MedicationRequest, and AllergyIntolerance: drug-allergy conflicts, renal/eGFR "
            "contraindications (e.g., Metformin with eGFR < 30), sex-diagnosis mismatches (e.g., male pregnancy), "
            "pediatric contraindications (e.g., Aspirin in young child with viral illness), and critical lab-drug "
            "conflicts (hypoglycemia + Insulin, high INR + Warfarin/NSAID). Do NOT invent errors on safe regimens."
        ),
    },
    {
        "id": "doctor_3",
        "name": "BS. Hùng",
        "role_en": "Terminology & Lab Auditor",
        "role_vi": "Kiểm toán Mã hóa & Chỉ số",
        "domain": "TERMINOLOGY",
        "color": "#7c3aed",
        "focus": (
            "Check clinical coding and measurement validity: ICD-10 code vs display concordance, "
            "LOINC code vs display concordance, UCUM unit appropriateness (e.g., HbA1c in %, Glucose in mg/dL, "
            "BP in mmHg, Temp in Cel), and physiological plausibility (SpO2 <= 100%, Systolic BP > Diastolic BP, "
            "realistic body temperature)."
        ),
    },
]

FHIR_META = {
    "id": "meta",
    "name": "BS. Thảo",
    "role_en": "Chief FHIR Adjudicator",
    "role_vi": "Thẩm định trưởng (Meta)",
    "color": "#d97706",
}

VALID_CATEGORIES = {"STRUCTURAL", "TERMINOLOGY", "CLINICAL_SEMANTIC"}

VALID_ERROR_CODES = {
    "BROKEN_REFERENCE",
    "MISSING_REQUIRED_FIELD",
    "INVALID_PERIOD",
    "INVALID_BIRTHDATE",
    "INVALID_IDENTIFIER",
    "INVALID_BUNDLE",
    "INVALID_UNIT",
    "CODE_DISPLAY_MISMATCH",
    "IMPOSSIBLE_VALUE",
    "DRUG_ALLERGY_CONFLICT",
    "RENAL_CONTRAINDICATION",
    "SEX_DIAGNOSIS_MISMATCH",
    "PEDIATRIC_CONTRAINDICATION",
    "LAB_MEDICATION_CONFLICT",
    "CONDITION_MEDICATION_CONFLICT",
}


def _normalize_issues(raw_issues: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_issues, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for item in raw_issues:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("category", "CLINICAL_SEMANTIC")).upper().strip()
        if cat not in VALID_CATEGORIES:
            if "STRUCT" in cat or "SCHEMA" in cat or "REF" in cat:
                cat = "STRUCTURAL"
            elif "TERM" in cat or "CODE" in cat or "UNIT" in cat or "LOINC" in cat:
                cat = "TERMINOLOGY"
            else:
                cat = "CLINICAL_SEMANTIC"

        code = str(item.get("error_code", "OTHER")).upper().strip().replace(" ", "_")
        desc = str(item.get("description", "")).strip()
        if not desc and code == "OTHER":
            continue
        res_list = item.get("resources", [])
        if isinstance(res_list, str):
            res_list = [res_list]
        cleaned.append(
            {
                "category": cat,
                "error_code": code,
                "severity": str(item.get("severity", "medium")).lower().strip(),
                "resources": [str(r) for r in res_list] if isinstance(res_list, list) else [],
                "description": desc,
            }
        )
    return cleaned


def _parse_audit_json(raw_text: str, expect_review: bool = False) -> Dict[str, Any]:
    cleaned = preprocess_response_string(raw_text or "")
    for candidate in (cleaned, cleaned[cleaned.find("{") : cleaned.rfind("}") + 1] if "{" in cleaned and "}" in cleaned else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                if expect_review and isinstance(parsed.get("agree"), str):
                    parsed["agree"] = parsed["agree"].strip().lower() in ("true", "yes", "đồng ý", "1")
                if "issues" in parsed:
                    parsed["issues"] = _normalize_issues(parsed["issues"])
                return parsed
        except Exception:
            continue

    if expect_review:
        low = (raw_text or "").lower()
        ag = ("true" in low or "yes" in low or "đồng ý" in low) and ("false" not in low and "không đồng ý" not in low)
        return {"agree": ag, "reason": (raw_text or "")[:300], "short_bubble": "Đã xem xét bản tổng hợp.", "issues": []}
    return {"has_issues": False, "issues": [], "summary": (raw_text or "")[:400], "short_bubble": "Hoàn tất rà soát."}


async def _call_llm(
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 4,
) -> Tuple[str, int, Dict[str, int]]:
    return await call_llm_with_rate_limit(
        client=client,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.15,
        max_retries=max_retries,
    )


def _lang_note(language: str) -> str:
    if language == "en":
        return "Write 'summary', 'description', and 'short_bubble' concisely in English."
    return "Viết 'summary', 'description', 'reason' và 'short_bubble' súc tích bằng tiếng Việt (giữ nguyên tên Resource/code chuẩn FHIR)."


def _verdict_label(has_issues: bool, issues: List[Dict[str, Any]]) -> str:
    if not has_issues or not issues:
        return "VALID"
    cats = sorted({i["category"][:4] for i in issues})
    return f"ERR:{len(issues)}({','.join(cats)})"


async def stream_fhir_audit(
    bundle: Dict[str, Any],
    method: str = "multi_agent_memory",
    model_key: str = "gemini-3.1-flash-lite",
    max_rounds: int = 2,
    language: str = "vi",
    case_id: str = "custom_fhir",
    ground_truth_issues: Optional[List[Dict[str, Any]]] = None,
    save_session: bool = True,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Unified async event stream for auditing an HL7 FHIR R4 Bundle under any of the
    6 experimental configurations. Emits structured events for both CLI benchmark
    aggregation and the Light-Theme Pixel 2D UI.
    """
    started_at = time.perf_counter()
    session_id = f"fhir_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}"
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    recorded_events: List[Dict[str, Any]] = []
    bundle_str = json.dumps(bundle, ensure_ascii=False, indent=2)
    gt_issues = ground_truth_issues if ground_truth_issues is not None else []
    use_memory = method in ("single_llm_memory", "multi_agent_memory", "multi_agent_sem")
    use_semantic = method in ("single_llm_memory", "multi_agent_memory", "multi_agent_sem")
    use_episodic = method in ("single_llm_memory", "multi_agent_memory")

    def emit(ev: Dict[str, Any]) -> Dict[str, Any]:
        ev.setdefault("ts", round(time.perf_counter() - started_at, 2))
        recorded_events.append(ev)
        return ev

    yield emit(
        {
            "type": "session_start",
            "task_mode": "fhir",
            "session_id": session_id,
            "method": method,
            "use_memory": use_memory,
            "case_id": case_id,
            "model_key": "rule-engine" if method == "rule_based" else model_key,
            "max_rounds": 1 if method in ("rule_based", "single_llm", "single_llm_memory") else max_rounds,
            "ground_truth_issues": gt_issues,
            "auditors": FHIR_AUDITORS,
            "meta": FHIR_META,
        }
    )

    # =========================================================================
    # METHOD 1: RULE-BASED CONVENTIONAL VALIDATOR
    # =========================================================================
    if method == "rule_based":
        yield emit({"type": "round_start", "round": 1, "max_rounds": 1, "phase": "analyze", "label": "Baseline 1: Kiểm tra xác định (Rule-based FHIR R4 Validator)"})
        yield emit({"type": "agent_state", "agent_id": "doctor_1", "state": "thinking", "round": 1, "phase": "analyze", "status_text": "Chạy luật kiểm tra Schema FHIR..."})
        await asyncio.sleep(0.25)
        res = validate_fhir_bundle_rule_based(bundle)
        verdict = _verdict_label(res["has_issues"], res["issues"])
        yield emit(
            {
                "type": "agent_output",
                "agent_id": "doctor_1",
                "name": "Rule Validator",
                "specialty_vi": "FHIR R4 Schema Engine",
                "round": 1,
                "phase": "decision",
                "state": "final",
                "answer": verdict,
                "has_issues": res["has_issues"],
                "issues": res["issues"],
                "explanation": res["summary"] + ("\n" + "\n".join(f"- [{i['category']}/{i['error_code']}] {i['description']}" for i in res["issues"]) if res["issues"] else ""),
                "short_bubble": f"Phát hiện {len(res['issues'])} lỗi cú pháp/đơn vị." if res["issues"] else "Không thấy lỗi cú pháp (VALID).",
                "latency_ms": res["latency_ms"],
                "usage": res["total_tokens"],
                "total_tokens": res["total_tokens"],
                "system_prompt": "[Deterministic HL7 FHIR R4 Rule Validator — 0 LLM calls]",
                "raw_output": json.dumps(res, ensure_ascii=False, indent=2),
            }
        )
        yield emit(
            {
                "type": "session_complete",
                "task_mode": "fhir",
                "session_id": session_id,
                "method": method,
                "case_id": case_id,
                "has_issues": res["has_issues"],
                "issues": res["issues"],
                "final_answer": verdict,
                "final_explanation": res["summary"],
                "ground_truth_issues": gt_issues,
                "consensus_reached": True,
                "total_rounds": 1,
                "total_elapsed_ms": res["latency_ms"],
                "total_tokens": res["total_tokens"],
                "memory_hits": 0,
            }
        )
        return

    # Validate LLM credentials
    if model_key not in LLM_MODELS_SETTINGS or not LLM_MODELS_SETTINGS[model_key].get("api_key"):
        raise ValueError(f"Model '{model_key}' chưa được cấu hình API key trong .env")

    cfg = LLM_MODELS_SETTINGS[model_key]
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=60.0)
    model_name = cfg["model_name"]
    memory_sys = FHIRMemorySystem(episodic_store_path=EPISODIC_STORE_PATH, use_semantic=use_semantic, use_episodic=use_episodic) if use_memory else None
    lang_note = _lang_note(language)
    total_memory_hits = 0

    try:
        # =====================================================================
        # METHODS 2 & 3: SINGLE-LLM (WITH OR WITHOUT MEMORY)
        # =====================================================================
        if method in ("single_llm", "single_llm_memory"):
            yield emit(
                {
                    "type": "round_start",
                    "round": 1,
                    "max_rounds": 1,
                    "phase": "analyze",
                    "label": "Single-LLM + Memory (RAG)" if use_memory else "Single-LLM Zero-Shot Baseline",
                }
            )
            mem_block = ""
            retrieved = None
            if memory_sys:
                retrieved = memory_sys.retrieve(bundle, domain=None, top_k_sem=4, top_k_epi=3)
                total_memory_hits = retrieved["total_hits"]
                mem_block = memory_sys.format_memory_prompt(retrieved)
                yield emit(
                    {
                        "type": "memory_retrieve",
                        "agent_id": "meta",
                        "domain": "ALL",
                        "semantic_hits": retrieved["semantic_hits"],
                        "episodic_hits": retrieved["episodic_hits"],
                        "total_hits": retrieved["total_hits"],
                    }
                )
                yield emit({"type": "packet_fly", "from": "memory_vault", "to": "meta", "label": f"Memory ({retrieved['total_hits']})", "color": "#0d9488"})

            yield emit({"type": "agent_state", "agent_id": "meta", "state": "thinking", "round": 1, "phase": "analyze", "status_text": "Single-LLM đang rà soát toàn bộ FHIR Bundle..."})
            sys_p = (
                "You are an expert HL7 FHIR R4 clinical data auditor. Audit the entire FHIR R4 Bundle for "
                "STRUCTURAL errors (missing required fields, broken references, invalid dates), "
                "TERMINOLOGY errors (ICD-10/LOINC code-display mismatch, wrong UCUM unit, impossible values), and "
                "CLINICAL_SEMANTIC errors (drug-allergy conflict, renal contraindication, sex-diagnosis mismatch, "
                "pediatric contraindication, critical lab-medication conflict). "
                "Only flag genuine, evidence-backed errors; if the Bundle is valid, return has_issues=false and issues=[]. "
                f"{lang_note} "
                "Return ONLY valid JSON: "
                '{"has_issues": boolean, "issues": [{"category": "STRUCTURAL|TERMINOLOGY|CLINICAL_SEMANTIC", '
                '"error_code": "BROKEN_REFERENCE|MISSING_REQUIRED_FIELD|INVALID_PERIOD|INVALID_BIRTHDATE|INVALID_IDENTIFIER|INVALID_UNIT|'
                'CODE_DISPLAY_MISMATCH|IMPOSSIBLE_VALUE|DRUG_ALLERGY_CONFLICT|RENAL_CONTRAINDICATION|'
                'SEX_DIAGNOSIS_MISMATCH|PEDIATRIC_CONTRAINDICATION|LAB_MEDICATION_CONFLICT|CONDITION_MEDICATION_CONFLICT", '
                '"severity": "low|medium|high|critical", "resources": ["ResourceType/id"], "description": "..."}], '
                '"summary": "...", "short_bubble": "<=14 words"}'
            )
            usr_p = (f"{mem_block}\n\n" if mem_block else "") + f"FHIR R4 Bundle:\n```json\n{bundle_str}\n```\nReturn JSON only."
            raw, lat, usage = await _call_llm(client, model_name, sys_p, usr_p)
            for k in total_tokens:
                total_tokens[k] += usage.get(k, 0)
            parsed = _parse_audit_json(raw)
            issues = _normalize_issues(parsed.get("issues", []))
            has_issues = bool(parsed.get("has_issues", len(issues) > 0)) and len(issues) > 0
            verdict = _verdict_label(has_issues, issues)
            summary = str(parsed.get("summary", "")).strip()
            bubble = str(parsed.get("short_bubble") or (f"Phát hiện {len(issues)} lỗi FHIR!" if has_issues else "Hồ sơ FHIR hợp lệ (VALID)."))

            yield emit(
                {
                    "type": "agent_output",
                    "agent_id": "meta",
                    "name": "Single-LLM Auditor",
                    "specialty_vi": "Single-LLM + Memory" if use_memory else "Single-LLM Zero-Shot",
                    "round": 1,
                    "phase": "decision",
                    "state": "final",
                    "answer": verdict,
                    "has_issues": has_issues,
                    "issues": issues,
                    "explanation": summary + ("\n" + "\n".join(f"- [{i['category']}/{i['error_code']}] {i['description']}" for i in issues) if issues else ""),
                    "short_bubble": bubble,
                    "latency_ms": lat,
                    "usage": usage,
                    "total_tokens": dict(total_tokens),
                    "memory_retrieved": retrieved,
                    "system_prompt": sys_p,
                    "raw_output": raw,
                }
            )
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            yield emit(
                {
                    "type": "session_complete",
                    "task_mode": "fhir",
                    "session_id": session_id,
                    "method": method,
                    "case_id": case_id,
                    "has_issues": has_issues,
                    "issues": issues,
                    "final_answer": verdict,
                    "final_explanation": summary,
                    "ground_truth_issues": gt_issues,
                    "consensus_reached": True,
                    "total_rounds": 1,
                    "total_elapsed_ms": elapsed_ms,
                    "total_tokens": total_tokens,
                    "memory_hits": total_memory_hits,
                }
            )
            if save_session:
                _save_fhir_log(session_id, case_id, method, model_key, has_issues, issues, gt_issues, True, 1, elapsed_ms, total_tokens, total_memory_hits, recorded_events)
            return

        # =====================================================================
        # METHODS 4, 5, 6: COLACARE MULTI-AGENT (WITH / WITHOUT MEMORY)
        # =====================================================================
        current_round = 0
        consensus_reached = False
        final_issues: List[Dict[str, Any]] = []
        final_has_issues = False
        final_summary = ""
        last_reviews: List[Dict[str, Any]] = []
        prev_syntheses: List[Dict[str, Any]] = []

        # Pre-retrieve specialized memory per auditor once per Bundle
        auditor_memories: Dict[str, Optional[Dict[str, Any]]] = {}
        if memory_sys:
            for aud in FHIR_AUDITORS:
                ret = memory_sys.retrieve(bundle, domain=aud["domain"], top_k_sem=3, top_k_epi=2)
                auditor_memories[aud["id"]] = ret
                total_memory_hits += ret["total_hits"]
                yield emit(
                    {
                        "type": "memory_retrieve",
                        "agent_id": aud["id"],
                        "domain": aud["domain"],
                        "semantic_hits": ret["semantic_hits"],
                        "episodic_hits": ret["episodic_hits"],
                        "total_hits": ret["total_hits"],
                    }
                )
                yield emit(
                    {
                        "type": "packet_fly",
                        "from": "memory_vault",
                        "to": aud["id"],
                        "label": f"Memory ({ret['total_hits']})",
                        "color": "#0d9488",
                    }
                )
            meta_mem = memory_sys.retrieve(bundle, domain=None, top_k_sem=3, top_k_epi=3)
            auditor_memories["meta"] = meta_mem
            total_memory_hits += meta_mem["total_hits"]
            yield emit(
                {
                    "type": "memory_retrieve",
                    "agent_id": "meta",
                    "domain": "ALL",
                    "semantic_hits": meta_mem["semantic_hits"],
                    "episodic_hits": meta_mem["episodic_hits"],
                    "total_hits": meta_mem["total_hits"],
                }
            )
            yield emit({"type": "packet_fly", "from": "memory_vault", "to": "meta", "label": f"Memory ({meta_mem['total_hits']})", "color": "#0d9488"})

        while current_round < max_rounds and not consensus_reached:
            current_round += 1
            yield emit(
                {
                    "type": "round_start",
                    "round": current_round,
                    "max_rounds": max_rounds,
                    "phase": "analyze",
                    "label": f"Vòng {current_round}: 3 Kiểm toán viên FHIR rà soát song song"
                    + (" (kèm Memory-Augmented)" if use_memory else " (Không Memory)"),
                }
            )

            for aud in FHIR_AUDITORS:
                yield emit(
                    {
                        "type": "agent_state",
                        "agent_id": aud["id"],
                        "state": "thinking",
                        "round": current_round,
                        "phase": "analyze",
                        "status_text": f"Đang rà soát {aud['role_vi']}...",
                    }
                )

            async def run_auditor(aud: Dict[str, Any]) -> Dict[str, Any]:
                mem_ctx = ""
                ret = auditor_memories.get(aud["id"])
                if memory_sys and ret:
                    mem_ctx = memory_sys.format_memory_prompt(ret) + "\n\n"

                prev_ctx = ""
                if current_round > 1 and prev_syntheses:
                    prev_ctx = (
                        f"Previous Round {current_round - 1} Synthesis:\n{json.dumps(prev_syntheses[-1], ensure_ascii=False)}\n"
                        f"Peer Reviews:\n{json.dumps(last_reviews, ensure_ascii=False)}\n\n"
                        "Re-audit carefully and resolve any disagreements."
                    )

                sys_p = (
                    f"You are {aud['name']}, {aud['role_en']} ({aud['role_vi']}) in Round {current_round} of a "
                    f"ColaCare Multi-Agent HL7 FHIR R4 auditing panel. "
                    f"Primary responsibility: {aud['focus']} "
                    "Be strict and precise: only report genuine violations with concrete evidence from the Bundle. "
                    "If no issue exists in your domain, return has_issues=false and issues=[]. "
                    f"{lang_note} "
                    "Return ONLY valid JSON: "
                    '{"has_issues": boolean, "issues": [{"category": "STRUCTURAL|TERMINOLOGY|CLINICAL_SEMANTIC", '
                    '"error_code": "BROKEN_REFERENCE|MISSING_REQUIRED_FIELD|INVALID_PERIOD|INVALID_BIRTHDATE|INVALID_UNIT|'
                    'CODE_DISPLAY_MISMATCH|IMPOSSIBLE_VALUE|DRUG_ALLERGY_CONFLICT|RENAL_CONTRAINDICATION|'
                    'SEX_DIAGNOSIS_MISMATCH|PEDIATRIC_CONTRAINDICATION|LAB_MEDICATION_CONFLICT", '
                    '"severity": "low|medium|high|critical", "resources": ["ResourceType/id"], "description": "..."}], '
                    '"summary": "...", "short_bubble": "<=14 words"}'
                )
                usr_p = f"{mem_ctx}{prev_ctx}\nFHIR R4 Bundle:\n```json\n{bundle_str}\n```\nReturn JSON only."
                raw, lat, usage = await _call_llm(client, model_name, sys_p, usr_p)
                parsed = _parse_audit_json(raw)
                return {"aud": aud, "sys_p": sys_p, "raw": raw, "parsed": parsed, "lat": lat, "usage": usage, "ret": ret}

            tasks = [asyncio.create_task(run_auditor(a)) for a in FHIR_AUDITORS]
            round_findings: List[Dict[str, Any]] = []

            for fut in asyncio.as_completed(tasks):
                res = await fut
                aud = res["aud"]
                parsed = res["parsed"]
                for k in total_tokens:
                    total_tokens[k] += res["usage"].get(k, 0)

                issues = _normalize_issues(parsed.get("issues", []))
                has_issues = bool(parsed.get("has_issues", len(issues) > 0)) and len(issues) > 0
                verdict = _verdict_label(has_issues, issues)
                summary = str(parsed.get("summary", "")).strip()
                bubble = str(parsed.get("short_bubble") or (f"Phát hiện {len(issues)} lỗi {aud['domain']}!" if has_issues else f"{aud['role_vi']}: Hợp lệ.")).strip()

                finding = {
                    "agent_id": aud["id"],
                    "name": aud["name"],
                    "domain": aud["domain"],
                    "has_issues": has_issues,
                    "issues": issues,
                    "summary": summary,
                }
                round_findings.append(finding)

                yield emit(
                    {
                        "type": "agent_output",
                        "agent_id": aud["id"],
                        "name": aud["name"],
                        "specialty_vi": aud["role_vi"],
                        "round": current_round,
                        "phase": "analyze",
                        "state": "speaking",
                        "answer": verdict,
                        "has_issues": has_issues,
                        "issues": issues,
                        "explanation": summary + ("\n" + "\n".join(f"- [{i['category']}/{i['error_code']}] {i['description']}" for i in issues) if issues else ""),
                        "short_bubble": bubble,
                        "latency_ms": res["lat"],
                        "usage": res["usage"],
                        "total_tokens": dict(total_tokens),
                        "memory_retrieved": res["ret"],
                        "system_prompt": res["sys_p"],
                        "raw_output": res["raw"],
                    }
                )
                yield emit(
                    {
                        "type": "packet_fly",
                        "from": aud["id"],
                        "to": "meta",
                        "label": verdict,
                        "color": aud["color"],
                    }
                )

            order_map = {a["id"]: i for i, a in enumerate(FHIR_AUDITORS)}
            round_findings.sort(key=lambda x: order_map[x["agent_id"]])

            # STEP 2: Meta Adjudicator synthesizes & deduplicates findings
            yield emit(
                {
                    "type": "agent_state",
                    "agent_id": "meta",
                    "state": "thinking",
                    "round": current_round,
                    "phase": "synthesize",
                    "status_text": "Đang tổng hợp & lọc báo động giả...",
                }
            )
            meta_mem_ctx = ""
            if memory_sys and auditor_memories.get("meta"):
                meta_mem_ctx = memory_sys.format_memory_prompt(auditor_memories["meta"]) + "\n\n"

            meta_sys = (
                f"You are {FHIR_META['name']}, Chief FHIR Adjudicator facilitating Round {current_round}. "
                "Synthesize the 3 specialist auditors' reports into a unified, deduplicated FHIR audit report. "
                "Critically verify each claimed issue against the actual FHIR Bundle to eliminate hallucinations "
                "and False Positives, while preserving genuine structural, terminology, and clinical semantic errors. "
                f"{lang_note} "
                "Return ONLY valid JSON: "
                '{"has_issues": boolean, "issues": [{"category": "STRUCTURAL|TERMINOLOGY|CLINICAL_SEMANTIC", '
                '"error_code": "...", "severity": "low|medium|high|critical", "resources": ["..."], "description": "..."}], '
                '"summary": "...", "short_bubble": "<=14 words"}'
            )
            meta_usr = (
                f"{meta_mem_ctx}FHIR R4 Bundle:\n```json\n{bundle_str}\n```\n\n"
                f"Round {current_round} Specialist Auditor Reports:\n{json.dumps(round_findings, ensure_ascii=False, indent=2)}\n\n"
                "Synthesize and return JSON only."
            )
            meta_raw, meta_lat, meta_usage = await _call_llm(client, model_name, meta_sys, meta_usr)
            for k in total_tokens:
                total_tokens[k] += meta_usage.get(k, 0)
            syn = _parse_audit_json(meta_raw)
            syn_issues = _normalize_issues(syn.get("issues", []))
            syn_has = bool(syn.get("has_issues", len(syn_issues) > 0)) and len(syn_issues) > 0
            syn_verdict = _verdict_label(syn_has, syn_issues)
            syn_summary = str(syn.get("summary", "")).strip()
            syn_bubble = str(syn.get("short_bubble") or (f"Tổng hợp: phát hiện {len(syn_issues)} lỗi, mời phản biện!" if syn_has else "Tổng hợp: Hồ sơ sạch (VALID), mời xác nhận!")).strip()
            prev_syntheses.append({"round": current_round, "has_issues": syn_has, "issues": syn_issues, "summary": syn_summary})

            yield emit(
                {
                    "type": "agent_output",
                    "agent_id": "meta",
                    "name": FHIR_META["name"],
                    "specialty_vi": FHIR_META["role_vi"],
                    "round": current_round,
                    "phase": "synthesize",
                    "state": "speaking",
                    "answer": syn_verdict,
                    "has_issues": syn_has,
                    "issues": syn_issues,
                    "explanation": syn_summary + ("\n" + "\n".join(f"- [{i['category']}/{i['error_code']}] {i['description']}" for i in syn_issues) if syn_issues else ""),
                    "short_bubble": syn_bubble,
                    "latency_ms": meta_lat,
                    "usage": meta_usage,
                    "total_tokens": dict(total_tokens),
                    "memory_retrieved": auditor_memories.get("meta"),
                    "system_prompt": meta_sys,
                    "raw_output": meta_raw,
                }
            )
            for aud in FHIR_AUDITORS:
                yield emit({"type": "packet_fly", "from": "meta", "to": aud["id"], "label": syn_verdict, "color": FHIR_META["color"]})

            # STEP 3: Auditors review & verify synthesis in parallel
            yield emit(
                {
                    "type": "phase_change",
                    "round": current_round,
                    "phase": "review",
                    "label": f"Vòng {current_round}: 3 Kiểm toán viên phản biện & bỏ phiếu đồng thuận",
                }
            )
            for aud in FHIR_AUDITORS:
                yield emit(
                    {
                        "type": "agent_state",
                        "agent_id": aud["id"],
                        "state": "thinking",
                        "round": current_round,
                        "phase": "review",
                        "status_text": "Đang thẩm định bản tổng hợp...",
                    }
                )

            async def run_review(aud: Dict[str, Any]) -> Dict[str, Any]:
                own = next((f for f in round_findings if f["agent_id"] == aud["id"]), {})
                rev_sys = (
                    f"You are {aud['name']} ({aud['role_en']}), reviewing the Chief Adjudicator's Round {current_round} "
                    "FHIR audit synthesis. Check if any genuine error in your domain was missed OR if a false positive "
                    "was wrongly included. "
                    f"{lang_note} "
                    "Return ONLY valid JSON: "
                    '{"agree": boolean, "reason": "concise rationale", "short_bubble": "<=12 words", '
                    '"issues": [...same schema if disagreeing, else []]}'
                )
                rev_usr = (
                    f"Your Round {current_round} Report:\n{json.dumps(own, ensure_ascii=False)}\n\n"
                    f"Chief Adjudicator Synthesis:\n{json.dumps({'has_issues': syn_has, 'issues': syn_issues, 'summary': syn_summary}, ensure_ascii=False)}\n\n"
                    "Do you agree with this synthesized audit report? Return JSON only."
                )
                raw, lat, usage = await _call_llm(client, model_name, rev_sys, rev_usr)
                parsed = _parse_audit_json(raw, expect_review=True)
                return {"aud": aud, "sys_p": rev_sys, "raw": raw, "parsed": parsed, "lat": lat, "usage": usage}

            rev_tasks = [asyncio.create_task(run_review(a)) for a in FHIR_AUDITORS]
            round_reviews: List[Dict[str, Any]] = []
            agree_count = 0

            for fut in asyncio.as_completed(rev_tasks):
                res = await fut
                aud = res["aud"]
                parsed = res["parsed"]
                for k in total_tokens:
                    total_tokens[k] += res["usage"].get(k, 0)

                agrees = bool(parsed.get("agree", True))
                if agrees:
                    agree_count += 1
                reason = str(parsed.get("reason", "")).strip()
                bubble = str(parsed.get("short_bubble") or ("Đồng ý bản tổng hợp!" if agrees else "Chưa đồng ý, cần xem lại!")).strip()
                rev_issues = _normalize_issues(parsed.get("issues", []))

                rev_rec = {
                    "agent_id": aud["id"],
                    "name": aud["name"],
                    "agree": agrees,
                    "reason": reason,
                    "issues": rev_issues,
                }
                round_reviews.append(rev_rec)

                yield emit(
                    {
                        "type": "agent_output",
                        "agent_id": aud["id"],
                        "name": aud["name"],
                        "specialty_vi": aud["role_vi"],
                        "round": current_round,
                        "phase": "review",
                        "state": "agree" if agrees else "disagree",
                        "agree": agrees,
                        "answer": "ĐỒNG Ý" if agrees else "PHẢN ĐỐI",
                        "explanation": reason,
                        "short_bubble": bubble,
                        "latency_ms": res["lat"],
                        "usage": res["usage"],
                        "total_tokens": dict(total_tokens),
                        "system_prompt": res["sys_p"],
                        "raw_output": res["raw"],
                    }
                )
                yield emit(
                    {
                        "type": "packet_fly",
                        "from": aud["id"],
                        "to": "meta",
                        "label": "Đồng ý" if agrees else "Phản đối",
                        "color": "#16a34a" if agrees else "#dc2626",
                    }
                )

            last_reviews = round_reviews
            all_agree = agree_count == len(FHIR_AUDITORS)
            yield emit(
                {
                    "type": "consensus_update",
                    "round": current_round,
                    "agree_count": agree_count,
                    "total_doctors": len(FHIR_AUDITORS),
                    "all_agree": all_agree,
                }
            )

            # STEP 4: Final decision or proceed to next round
            is_final = all_agree or (current_round >= max_rounds)
            if all_agree:
                final_has_issues = syn_has
                final_issues = syn_issues
                final_summary = syn_summary
                consensus_reached = True
                yield emit(
                    {
                        "type": "agent_output",
                        "agent_id": "meta",
                        "name": FHIR_META["name"],
                        "specialty_vi": FHIR_META["role_vi"],
                        "round": current_round,
                        "phase": "decision",
                        "state": "final",
                        "answer": syn_verdict,
                        "has_issues": syn_has,
                        "issues": syn_issues,
                        "explanation": f"Đạt đồng thuận 3/3 sau Vòng {current_round}. {syn_summary}",
                        "short_bubble": f"CHỐT 3/3: {syn_verdict}!",
                        "latency_ms": 0,
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "total_tokens": dict(total_tokens),
                        "system_prompt": "[Consensus reached 3/3 — directly finalized verified synthesis]",
                        "raw_output": json.dumps({"has_issues": syn_has, "issues": syn_issues, "summary": syn_summary}, ensure_ascii=False, indent=2),
                    }
                )
                break

            yield emit(
                {
                    "type": "agent_state",
                    "agent_id": "meta",
                    "state": "thinking",
                    "round": current_round,
                    "phase": "decision",
                    "status_text": "Đang chốt kết luận cuối cùng..." if is_final else "Tổng kết bất đồng cho vòng sau...",
                }
            )
            dec_sys = (
                f"You are {FHIR_META['name']}, Chief FHIR Adjudicator making the "
                f"{'FINAL' if is_final else f'Round {current_round} interim'} audit decision after specialist reviews. "
                f"{lang_note} "
                'Return ONLY valid JSON: {"has_issues": boolean, "issues": [...], "summary": "...", "short_bubble": "<=14 words"}'
            )
            dec_usr = (
                f"Current Synthesis: {json.dumps({'has_issues': syn_has, 'issues': syn_issues, 'summary': syn_summary}, ensure_ascii=False)}\n"
                f"Reviews: {json.dumps(round_reviews, ensure_ascii=False)}\nReturn JSON only."
            )
            dec_raw, dec_lat, dec_usage = await _call_llm(client, model_name, dec_sys, dec_usr)
            for k in total_tokens:
                total_tokens[k] += dec_usage.get(k, 0)
            dec = _parse_audit_json(dec_raw)
            dec_issues = _normalize_issues(dec.get("issues", syn_issues))
            dec_has = bool(dec.get("has_issues", len(dec_issues) > 0)) and len(dec_issues) > 0
            dec_verdict = _verdict_label(dec_has, dec_issues)
            dec_sum = str(dec.get("summary", syn_summary)).strip()
            dec_bub = str(dec.get("short_bubble") or (f"CHỐT: {dec_verdict}" if is_final else "Chưa đồng thuận → Mở vòng tiếp!")).strip()

            yield emit(
                {
                    "type": "agent_output",
                    "agent_id": "meta",
                    "name": FHIR_META["name"],
                    "specialty_vi": FHIR_META["role_vi"],
                    "round": current_round,
                    "phase": "decision",
                    "state": "final" if is_final else "speaking",
                    "answer": dec_verdict,
                    "has_issues": dec_has,
                    "issues": dec_issues,
                    "explanation": dec_sum,
                    "short_bubble": dec_bub,
                    "latency_ms": dec_lat,
                    "usage": dec_usage,
                    "total_tokens": dict(total_tokens),
                    "system_prompt": dec_sys,
                    "raw_output": dec_raw,
                }
            )
            if is_final:
                final_has_issues = dec_has
                final_issues = dec_issues
                final_summary = dec_sum
                break

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        final_verdict = _verdict_label(final_has_issues, final_issues)
        yield emit(
            {
                "type": "session_complete",
                "task_mode": "fhir",
                "session_id": session_id,
                "method": method,
                "case_id": case_id,
                "has_issues": final_has_issues,
                "issues": final_issues,
                "final_answer": final_verdict,
                "final_explanation": final_summary,
                "ground_truth_issues": gt_issues,
                "consensus_reached": consensus_reached,
                "total_rounds": current_round,
                "total_elapsed_ms": elapsed_ms,
                "total_tokens": total_tokens,
                "memory_hits": total_memory_hits,
            }
        )
        if save_session:
            _save_fhir_log(
                session_id, case_id, method, model_key, final_has_issues, final_issues,
                gt_issues, consensus_reached, current_round, elapsed_ms, total_tokens, total_memory_hits, recorded_events,
            )
    finally:
        await client.close()


def _save_fhir_log(
    session_id: str,
    case_id: str,
    method: str,
    model_key: str,
    has_issues: bool,
    issues: List[Dict[str, Any]],
    gt_issues: List[Dict[str, Any]],
    consensus_reached: bool,
    total_rounds: int,
    elapsed_ms: int,
    total_tokens: Dict[str, int],
    memory_hits: int,
    events: List[Dict[str, Any]],
) -> None:
    os.makedirs(FHIR_LOGS_DIR, exist_ok=True)
    path = os.path.join(FHIR_LOGS_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "task_mode": "fhir",
                "case_id": case_id,
                "method": method,
                "model_key": model_key,
                "has_issues": has_issues,
                "issues": issues,
                "final_answer": _verdict_label(has_issues, issues),
                "ground_truth_issues": gt_issues,
                "consensus_reached": consensus_reached,
                "total_rounds": total_rounds,
                "total_elapsed_ms": elapsed_ms,
                "total_tokens": total_tokens,
                "memory_hits": memory_hits,
                "events": events,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


async def stream_fhir_demo(bundle: Dict[str, Any], case_id: str = "fhir_sem_02", ground_truth_issues: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[Dict[str, Any]]:
    """
    Zero-cost 2-round scripted FHIR + Memory-Augmented Demo for classroom / thesis defense presentations.
    Demonstrates: Memory Vault retrieval -> Parallel 3 Auditors -> Round 1 conflict -> Round 2 3/3 Consensus.
    """
    started_at = time.perf_counter()
    session_id = f"fhir_demo_{datetime.now().strftime('%H%M%S')}"
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    mem_sys = FHIRMemorySystem(use_semantic=True, use_episodic=True)

    def tick(ev: Dict[str, Any], add_tok: int = 0) -> Dict[str, Any]:
        if add_tok:
            p = int(add_tok * 0.68)
            c = add_tok - p
            total_tokens["prompt_tokens"] += p
            total_tokens["completion_tokens"] += c
            total_tokens["total_tokens"] += add_tok
            ev["usage"] = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": add_tok}
            ev["total_tokens"] = dict(total_tokens)
        ev["ts"] = round(time.perf_counter() - started_at, 2)
        return ev

    gt = ground_truth_issues if ground_truth_issues is not None else [
        {
            "category": "CLINICAL_SEMANTIC",
            "error_code": "RENAL_CONTRAINDICATION",
            "severity": "critical",
            "resources": ["MedicationRequest/m-402", "Observation/o-402a"],
            "description": "Metformin 1000 mg BID chống chỉ định khi eGFR = 18 mL/min/1.73m2 (< 30).",
        }
    ]

    yield tick(
        {
            "type": "session_start",
            "task_mode": "fhir",
            "session_id": session_id,
            "method": "multi_agent_memory",
            "use_memory": True,
            "case_id": case_id,
            "model_key": "fhir-memory-demo",
            "max_rounds": 2,
            "ground_truth_issues": gt,
            "auditors": FHIR_AUDITORS,
            "meta": FHIR_META,
        }
    )

    # Step 0: Memory Vault retrieval animation
    total_hits = 0
    mem_cache: Dict[str, Dict[str, Any]] = {}
    for aud in FHIR_AUDITORS:
        ret = mem_sys.retrieve(bundle, domain=aud["domain"], top_k_sem=2, top_k_epi=2)
        mem_cache[aud["id"]] = ret
        total_hits += ret["total_hits"]
        yield tick({"type": "memory_retrieve", "agent_id": aud["id"], "domain": aud["domain"], "semantic_hits": ret["semantic_hits"], "episodic_hits": ret["episodic_hits"], "total_hits": ret["total_hits"]})
        yield tick({"type": "packet_fly", "from": "memory_vault", "to": aud["id"], "label": f"Memory ({ret['total_hits']})", "color": "#0d9488"})
        await asyncio.sleep(0.3)

    # Round 1
    yield tick({"type": "round_start", "round": 1, "max_rounds": 2, "phase": "analyze", "label": "Vòng 1: 3 Kiểm toán viên FHIR rà soát song song (kèm Memory)"})
    for a in FHIR_AUDITORS:
        yield tick({"type": "agent_state", "agent_id": a["id"], "state": "thinking", "round": 1, "phase": "analyze", "status_text": f"Đang rà soát {a['role_vi']}..."})

    r1_items = [
        ("doctor_1", "BS. Linh", "Kiểm toán Cấu trúc FHIR", "VALID", False, [], "Cấu trúc Bundle FHIR R4 hợp lệ: đủ trường bắt buộc, không gãy reference Patient/pat-402, thời gian Encounter đúng.", "Cấu trúc JSON & Reference đều hợp lệ!", 680, 460),
        ("doctor_3", "BS. Hùng", "Kiểm toán Mã hóa & Chỉ số", "VALID", False, [], "Mã ICD-10 (E11.22, N18.4), LOINC (33914-3 eGFR, 4548-4 HbA1c) và đơn vị UCUM (mL/min/1.73m2, %) đều chính xác.", "Mã ICD-10, LOINC & đơn vị chuẩn xác!", 840, 490),
        (
            "doctor_2", "BS. Minh", "Kiểm toán Lâm sàng Chéo", "ERR:1(CLIN)", True,
            [{"category": "CLINICAL_SEMANTIC", "error_code": "RENAL_CONTRAINDICATION", "severity": "critical", "resources": ["MedicationRequest/m-402", "Observation/o-402a", "Condition/c-402b"], "description": "Chống chỉ định nghiêm trọng: kê Metformin 1000 mg x2/ngày trong khi eGFR = 18 mL/min (< 30, CKD giai đoạn 4) gây nguy cơ nhiễm toan lactic."}],
            "Tra cứu Semantic Memory [SEM-CLIN-02] & Episodic Memory [EPI-ERR-03]: phát hiện mâu thuẫn chéo giữa Observation (eGFR = 18) và MedicationRequest (Metformin 1000 mg BID) → Chống chỉ định tuyệt đối!",
            "BÁO ĐỘNG ĐỎ: Metformin chống chỉ định khi eGFR=18!",
            1120, 620,
        ),
    ]
    for aid, name, spec, ans, has_i, iss, expl, bub, lat, tok in r1_items:
        await asyncio.sleep(0.85)
        col = next(a["color"] for a in FHIR_AUDITORS if a["id"] == aid)
        yield tick(
            {
                "type": "agent_output",
                "agent_id": aid,
                "name": name,
                "specialty_vi": spec,
                "round": 1,
                "phase": "analyze",
                "state": "speaking",
                "answer": ans,
                "has_issues": has_i,
                "issues": iss,
                "explanation": expl,
                "short_bubble": bub,
                "latency_ms": lat,
                "memory_retrieved": mem_cache.get(aid),
                "system_prompt": f"[DEMO] {name} ({spec}) with FHIR Memory-Augmented prompt.",
                "raw_output": json.dumps({"has_issues": has_i, "issues": iss, "summary": expl}, ensure_ascii=False, indent=2),
            },
            add_tok=tok,
        )
        yield tick({"type": "packet_fly", "from": aid, "to": "meta", "label": ans, "color": col})

    await asyncio.sleep(0.6)
    yield tick({"type": "agent_state", "agent_id": "meta", "state": "thinking", "round": 1, "phase": "synthesize", "status_text": "Tổng hợp & kiểm chứng chéo..."})
    await asyncio.sleep(1.0)
    crit_issue = r1_items[2][5]
    syn_expl = "Hồ sơ đúng cú pháp FHIR R4 & mã LOINC/ICD-10 (nên Rule-based Validator sẽ bỏ lọt), nhưng BS. Minh kết hợp Semantic Memory [SEM-CLIN-02] phát hiện lỗi ngữ nghĩa lâm sàng mức CRITICAL: kê Metformin khi eGFR = 18."
    yield tick(
        {
            "type": "agent_output",
            "agent_id": "meta",
            "name": FHIR_META["name"],
            "specialty_vi": FHIR_META["role_vi"],
            "round": 1,
            "phase": "synthesize",
            "state": "speaking",
            "answer": "ERR:1(CLIN)",
            "has_issues": True,
            "issues": crit_issue,
            "explanation": syn_expl,
            "short_bubble": "Ghi nhận 1 lỗi CRITICAL (Metformin/eGFR=18). Mời phản biện!",
            "latency_ms": 960,
            "system_prompt": "[DEMO] Chief FHIR Adjudicator synthesizing Round 1.",
            "raw_output": json.dumps({"has_issues": True, "issues": crit_issue, "summary": syn_expl}, ensure_ascii=False, indent=2),
        },
        add_tok=580,
    )
    for a in FHIR_AUDITORS:
        yield tick({"type": "packet_fly", "from": "meta", "to": a["id"], "label": "ERR:1(CLIN)", "color": FHIR_META["color"]})

    await asyncio.sleep(0.6)
    yield tick({"type": "phase_change", "round": 1, "phase": "review", "label": "Vòng 1: Phản biện & xác nhận đồng thuận"})
    r1_revs = [
        ("doctor_1", "BS. Linh", "Kiểm toán Cấu trúc FHIR", True, "Đồng ý. Cấu trúc JSON hợp lệ nhưng cảnh báo lâm sàng chéo của BS. Minh là hoàn toàn chính xác.", "Đồng ý! Lỗi này vượt ngoài schema.", 610, 290),
        ("doctor_2", "BS. Minh", "Kiểm toán Lâm sàng Chéo", True, "Đồng ý hoàn toàn với phân loại CRITICAL RENAL_CONTRAINDICATION.", "Đồng ý 100%, cần chặn y lệnh ngay!", 580, 280),
        ("doctor_3", "BS. Hùng", "Kiểm toán Mã hóa & Chỉ số", False, "Chưa đồng ý chốt vội: cần xác nhận lại đơn vị eGFR đúng là mL/min/1.73m2 chứ không phải lỗi nhập đơn vị trước khi kết luận chống chỉ định thuốc.", "Cần kiểm lại đơn vị eGFR có bị nhập nhầm không!", 790, 340),
    ]
    for aid, name, spec, ag, reason, bub, lat, tok in r1_revs:
        await asyncio.sleep(0.75)
        yield tick(
            {
                "type": "agent_output",
                "agent_id": aid,
                "name": name,
                "specialty_vi": spec,
                "round": 1,
                "phase": "review",
                "state": "agree" if ag else "disagree",
                "agree": ag,
                "answer": "ĐỒNG Ý" if ag else "PHẢN ĐỐI",
                "explanation": reason,
                "short_bubble": bub,
                "latency_ms": lat,
                "system_prompt": f"[DEMO] {name} reviewing Round 1.",
                "raw_output": json.dumps({"agree": ag, "reason": reason}, ensure_ascii=False, indent=2),
            },
            add_tok=tok,
        )
        yield tick({"type": "packet_fly", "from": aid, "to": "meta", "label": "Đồng ý" if ag else "Phản đối", "color": "#16a34a" if ag else "#dc2626"})

    yield tick({"type": "consensus_update", "round": 1, "agree_count": 2, "total_doctors": 3, "all_agree": False})
    await asyncio.sleep(0.7)
    yield tick(
        {
            "type": "agent_output",
            "agent_id": "meta",
            "name": FHIR_META["name"],
            "specialty_vi": FHIR_META["role_vi"],
            "round": 1,
            "phase": "decision",
            "state": "speaking",
            "answer": "ERR:1(CLIN)",
            "explanation": "BS. Hùng yêu cầu xác minh đơn vị eGFR khớp với chẩn đoán CKD giai đoạn 4 (N18.4). Mở Vòng 2 để đối chiếu chéo.",
            "short_bubble": "Mới đạt 2/3 → Mở Vòng 2 xác minh eGFR & N18.4!",
            "latency_ms": 620,
            "system_prompt": "[DEMO] Interim Round 1 adjudication.",
            "raw_output": json.dumps({"round": 1, "consensus": False}, indent=2),
        },
        add_tok=310,
    )

    # Round 2
    await asyncio.sleep(0.8)
    yield tick({"type": "round_start", "round": 2, "max_rounds": 2, "phase": "analyze", "label": "Vòng 2: Xác minh chéo & đạt đồng thuận 3/3"})
    r2_revs = [
        ("doctor_3", "BS. Hùng", "Kiểm toán Mã hóa & Chỉ số", True, "Đã kiểm tra chéo Condition/c-402b có mã N18.4 (CKD giai đoạn 4, ngưỡng eGFR 15–29). Vậy giá trị eGFR = 18 mL/min/1.73m2 là thật, không phải lỗi đơn vị! Tôi đồng ý chốt lỗi lâm sàng.", "Khớp N18.4 (CKD 4) → eGFR=18 là thật! Đồng ý!", 720, 350),
        ("doctor_1", "BS. Linh", "Kiểm toán Cấu trúc FHIR", True, "Xác nhận đồng thuận 100% báo cáo kiểm toán FHIR.", "Đồng ý chốt báo cáo kiểm toán!", 510, 240),
        ("doctor_2", "BS. Minh", "Kiểm toán Lâm sàng Chéo", True, "Đồng ý chốt lỗi CRITICAL RENAL_CONTRAINDICATION (Metformin + eGFR=18).", "Chốt CRITICAL: Ngưng Metformin!", 540, 260),
    ]
    for aid, name, spec, ag, reason, bub, lat, tok in r2_revs:
        await asyncio.sleep(0.7)
        yield tick(
            {
                "type": "agent_output",
                "agent_id": aid,
                "name": name,
                "specialty_vi": spec,
                "round": 2,
                "phase": "review",
                "state": "agree",
                "agree": True,
                "answer": "ĐỒNG Ý",
                "explanation": reason,
                "short_bubble": bub,
                "latency_ms": lat,
                "system_prompt": f"[DEMO] {name} Round 2 verification.",
                "raw_output": json.dumps({"agree": True, "reason": reason}, ensure_ascii=False, indent=2),
            },
            add_tok=tok,
        )
        yield tick({"type": "packet_fly", "from": aid, "to": "meta", "label": "Đồng ý", "color": "#16a34a"})

    yield tick({"type": "consensus_update", "round": 2, "agree_count": 3, "total_doctors": 3, "all_agree": True})
    await asyncio.sleep(0.6)
    final_sum = "Hội đồng kiểm toán FHIR + Memory đạt đồng thuận 3/3 sau 2 vòng: Phát hiện 1 lỗi CRITICAL [CLINICAL_SEMANTIC / RENAL_CONTRAINDICATION] — Chống chỉ định kê Metformin 1000 mg BID khi bệnh nhân CKD giai đoạn 4 (N18.4) có eGFR = 18 mL/min/1.73m2."
    yield tick(
        {
            "type": "agent_output",
            "agent_id": "meta",
            "name": FHIR_META["name"],
            "specialty_vi": FHIR_META["role_vi"],
            "round": 2,
            "phase": "decision",
            "state": "final",
            "answer": "ERR:1(CLIN)",
            "has_issues": True,
            "issues": crit_issue,
            "explanation": final_sum,
            "short_bubble": "CHỐT ĐỒNG THUẬN 3/3: PHÁT HIỆN LỖI CRITICAL!",
            "latency_ms": 680,
            "system_prompt": "[DEMO] Final FHIR Audit Decision.",
            "raw_output": json.dumps({"has_issues": True, "issues": crit_issue, "summary": final_sum}, ensure_ascii=False, indent=2),
        },
        add_tok=340,
    )
    yield tick(
        {
            "type": "session_complete",
            "task_mode": "fhir",
            "session_id": session_id,
            "method": "multi_agent_memory",
            "case_id": case_id,
            "has_issues": True,
            "issues": crit_issue,
            "final_answer": "ERR:1(CLIN)",
            "final_explanation": final_sum,
            "ground_truth_issues": gt,
            "consensus_reached": True,
            "total_rounds": 2,
            "total_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            "total_tokens": total_tokens,
            "memory_hits": total_hits,
        }
    )
