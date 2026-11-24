"""
Async event-driven ColaCare Multi-Disciplinary Team (MDT) engine.
Adapts the ColaCare algorithm from medagentboard/medqa/multi_agent_colacare.py
with parallel agent execution (asyncio), telemetry tracking, and real-time event streaming.
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

from fhir_audit.rate_limiter import call_llm_with_rate_limit
from medagentboard.utils.json_utils import preprocess_response_string
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS


DOCTORS_META = [
    {
        "id": "doctor_1",
        "name": "BS. Linh",
        "specialty": "Internal Medicine",
        "specialty_vi": "Nội khoa",
        "color": "#38bdf8",
        "accent": "#0284c7",
    },
    {
        "id": "doctor_2",
        "name": "BS. Minh",
        "specialty": "Surgery",
        "specialty_vi": "Ngoại khoa",
        "color": "#fb7185",
        "accent": "#e11d48",
    },
    {
        "id": "doctor_3",
        "name": "BS. Hùng",
        "specialty": "Radiology",
        "specialty_vi": "Chẩn đoán hình ảnh",
        "color": "#a78bfa",
        "accent": "#7c3aed",
    },
]

META_INFO = {
    "id": "meta",
    "name": "BS. Thảo",
    "specialty": "Chief Coordinator",
    "specialty_vi": "Bác sĩ trưởng (Meta)",
    "color": "#fbbf24",
    "accent": "#d97706",
}

LOGS_DIR = os.path.join("logs", "pixel_mdt")


def get_available_models() -> List[Dict[str, Any]]:
    models = []
    for key, cfg in LLM_MODELS_SETTINGS.items():
        if cfg.get("api_key"):
            models.append(
                {
                    "key": key,
                    "label": cfg.get("comment", key),
                    "model_name": cfg.get("model_name", key),
                    "reasoning": bool(cfg.get("reasoning", False)),
                }
            )
    return models


def parse_json_fallback(response_text: str, expect_review: bool = False) -> Dict[str, Any]:
    cleaned = preprocess_response_string(response_text or "")
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            if expect_review and isinstance(parsed.get("agree"), str):
                parsed["agree"] = parsed["agree"].strip().lower() in ("true", "yes", "đồng ý", "1")
            return parsed
    except Exception:
        pass

    # Try extracting first JSON object substring
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                if expect_review and isinstance(parsed.get("agree"), str):
                    parsed["agree"] = parsed["agree"].strip().lower() in ("true", "yes", "đồng ý", "1")
                return parsed
        except Exception:
            pass

    if expect_review:
        lower = (response_text or "").lower()
        agrees = ("yes" in lower or "đồng ý" in lower or "true" in lower) and (
            "không đồng ý" not in lower and '"agree": false' not in lower
        )
        return {
            "agree": agrees,
            "reason": (response_text or "No reason provided")[:400],
            "answer": "",
        }
    return {
        "explanation": (response_text or "No explanation provided")[:800],
        "answer": "?",
    }


def _format_question(question: str, options: Optional[Dict[str, str]]) -> str:
    if options:
        opts = "\n".join([f"{k}: {v}" for k, v in options.items()])
        return f"{question}\n\nOptions:\n{opts}"
    return question


def _lang_instruction(language: str) -> str:
    if language == "en":
        return "Respond concisely in English (2-4 sentences for explanation/reason)."
    return "Trả lời ngắn gọn, súc tích bằng tiếng Việt (2-4 câu cho phần giải thích, có thể giữ thuật ngữ y khoa chuẩn)."


async def _call_llm_json(
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 4,
) -> Tuple[str, int, Dict[str, int]]:
    """Call OpenAI-compatible chat completions API with RPM limiting and smart 429 retry."""
    return await call_llm_with_rate_limit(
        client=client,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.35,
        max_retries=max_retries,
    )


async def stream_colacare_live(
    question: str,
    options: Optional[Dict[str, str]],
    model_key: str,
    max_rounds: int = 2,
    language: str = "vi",
    qid: str = "custom_case",
    ground_truth: Optional[str] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Execute ColaCare MDT asynchronously and yield structured UI events.
    Doctors run in parallel during analysis and review phases.
    """
    if model_key not in LLM_MODELS_SETTINGS or not LLM_MODELS_SETTINGS[model_key].get("api_key"):
        raise ValueError(f"Model '{model_key}' không khả dụng hoặc chưa cấu hình API key trong .env")

    cfg = LLM_MODELS_SETTINGS[model_key]
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=45.0)
    model_name = cfg["model_name"]

    session_id = f"mdt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    started_at = time.perf_counter()
    lang_note = _lang_instruction(language)
    q_text = _format_question(question, options)

    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    recorded_events: List[Dict[str, Any]] = []
    doctor_memories: Dict[str, List[Dict[str, Any]]] = {d["id"]: [] for d in DOCTORS_META}
    previous_syntheses: List[Dict[str, Any]] = []

    def emit(event: Dict[str, Any]) -> Dict[str, Any]:
        event.setdefault("ts", round(time.perf_counter() - started_at, 2))
        recorded_events.append(event)
        return event

    yield emit(
        {
            "type": "session_start",
            "session_id": session_id,
            "mode": "live",
            "qid": qid,
            "question": question,
            "options": options or {},
            "ground_truth": ground_truth,
            "model_key": model_key,
            "model_name": model_name,
            "max_rounds": max_rounds,
            "doctors": DOCTORS_META,
            "meta": META_INFO,
        }
    )

    current_round = 0
    consensus_reached = False
    final_decision: Optional[Dict[str, Any]] = None
    last_reviews: List[Dict[str, Any]] = []

    try:
        while current_round < max_rounds and not consensus_reached:
            current_round += 1
            yield emit(
                {
                    "type": "round_start",
                    "round": current_round,
                    "max_rounds": max_rounds,
                    "phase": "analyze",
                    "label": f"Vòng {current_round}: Phân tích độc lập từ 3 chuyên khoa",
                }
            )

            # ------------------------------------------------------
            # STEP 1: Doctors analyze in parallel
            # ------------------------------------------------------
            for doc in DOCTORS_META:
                yield emit(
                    {
                        "type": "agent_state",
                        "agent_id": doc["id"],
                        "state": "thinking",
                        "round": current_round,
                        "phase": "analyze",
                        "status_text": "Đang phân tích ca bệnh...",
                    }
                )

            async def run_doctor_analysis(doc: Dict[str, Any]) -> Dict[str, Any]:
                prev_ctx = ""
                if current_round > 1 and previous_syntheses:
                    last_syn = previous_syntheses[-1]
                    prev_reviews_txt = "\n".join(
                        f"- {r['name']} ({r['specialty_vi']}): agree={r['agree']}, answer={r['answer']}, reason={r['reason']}"
                        for r in last_reviews
                    )
                    prev_ctx = (
                        f"\n\nPrevious Round {current_round - 1} Synthesis:\n"
                        f"Answer: {last_syn.get('answer')}\nExplanation: {last_syn.get('explanation')}\n"
                        f"Peer Reviews:\n{prev_reviews_txt}\n"
                        f"Re-evaluate carefully from your {doc['specialty']} perspective."
                    )

                sys_prompt = (
                    f"You are {doc['name']}, a specialist doctor in {doc['specialty']} ({doc['specialty_vi']}) "
                    f"participating in Round {current_round} of a ColaCare multidisciplinary team (MDT) consultation. "
                    f"Analyze the medical case from your specialty perspective. {lang_note} "
                    f"Output ONLY a valid JSON object with keys: "
                    f"'explanation' (concise clinical reasoning), "
                    f"'short_bubble' (1 short sentence <= 14 words summarizing your stance for a speech bubble), "
                    f"'answer' (option letter like 'A', 'B', 'C' if options are provided, else concise answer)."
                )
                usr_prompt = f"Case:\n{q_text}{prev_ctx}\n\nReturn JSON only."
                raw, latency_ms, usage = await _call_llm_json(client, model_name, sys_prompt, usr_prompt)
                parsed = parse_json_fallback(raw, expect_review=False)
                return {
                    "doc": doc,
                    "system_prompt": sys_prompt,
                    "user_prompt": usr_prompt,
                    "raw": raw,
                    "parsed": parsed,
                    "latency_ms": latency_ms,
                    "usage": usage,
                }

            tasks = [asyncio.create_task(run_doctor_analysis(d)) for d in DOCTORS_META]
            round_opinions: List[Dict[str, Any]] = []

            for fut in asyncio.as_completed(tasks):
                res = await fut
                doc = res["doc"]
                parsed = res["parsed"]
                usage = res["usage"]
                for k in total_tokens:
                    total_tokens[k] += usage.get(k, 0)

                answer = str(parsed.get("answer", "?")).strip()
                explanation = str(parsed.get("explanation", "")).strip()
                bubble = str(parsed.get("short_bubble") or explanation[:90]).strip()

                opinion_record = {
                    "doctor_id": doc["id"],
                    "name": doc["name"],
                    "specialty": doc["specialty"],
                    "specialty_vi": doc["specialty_vi"],
                    "answer": answer,
                    "explanation": explanation,
                    "short_bubble": bubble,
                }
                doctor_memories[doc["id"]].append({"round": current_round, "type": "analysis", **opinion_record})
                round_opinions.append(opinion_record)

                yield emit(
                    {
                        "type": "agent_output",
                        "agent_id": doc["id"],
                        "name": doc["name"],
                        "specialty_vi": doc["specialty_vi"],
                        "round": current_round,
                        "phase": "analyze",
                        "state": "speaking",
                        "answer": answer,
                        "explanation": explanation,
                        "short_bubble": bubble,
                        "latency_ms": res["latency_ms"],
                        "usage": usage,
                        "total_tokens": dict(total_tokens),
                        "system_prompt": res["system_prompt"],
                        "raw_output": res["raw"],
                    }
                )
                yield emit(
                    {
                        "type": "packet_fly",
                        "from": doc["id"],
                        "to": "meta",
                        "label": f"Đề xuất: {answer}",
                        "color": doc["color"],
                    }
                )

            # Sort opinions consistently by doctor order for Meta prompt
            order_map = {d["id"]: i for i, d in enumerate(DOCTORS_META)}
            round_opinions.sort(key=lambda x: order_map[x["doctor_id"]])

            # ------------------------------------------------------
            # STEP 2: Meta agent synthesizes opinions
            # ------------------------------------------------------
            yield emit(
                {
                    "type": "agent_state",
                    "agent_id": "meta",
                    "state": "thinking",
                    "round": current_round,
                    "phase": "synthesize",
                    "status_text": "Đang tổng hợp ý kiến 3 chuyên khoa...",
                }
            )

            opinions_block = "\n\n".join(
                f"{op['name']} ({op['specialty']}):\nAnswer: {op['answer']}\nExplanation: {op['explanation']}"
                for op in round_opinions
            )
            meta_sys = (
                f"You are {META_INFO['name']}, Chief Coordinator (Meta Agent) facilitating Round {current_round} "
                f"of a ColaCare MDT consultation. Synthesize the 3 specialist doctors' opinions, highlight agreements "
                f"or conflicts, and propose a consensus conclusion. {lang_note} "
                f"Output ONLY a valid JSON object with keys: "
                f"'explanation' (synthesized reasoning), "
                f"'short_bubble' (1 short sentence <= 14 words for speech bubble), "
                f"'answer' (option letter like 'A', 'B', 'C' if options exist)."
            )
            meta_usr = f"Case:\n{q_text}\n\nRound {current_round} Specialist Opinions:\n{opinions_block}\n\nReturn JSON only."
            meta_raw, meta_latency, meta_usage = await _call_llm_json(client, model_name, meta_sys, meta_usr)
            synthesis = parse_json_fallback(meta_raw, expect_review=False)
            for k in total_tokens:
                total_tokens[k] += meta_usage.get(k, 0)

            syn_answer = str(synthesis.get("answer", "?")).strip()
            syn_expl = str(synthesis.get("explanation", "")).strip()
            syn_bubble = str(synthesis.get("short_bubble") or syn_expl[:90]).strip()
            previous_syntheses.append({"round": current_round, "answer": syn_answer, "explanation": syn_expl})

            yield emit(
                {
                    "type": "agent_output",
                    "agent_id": "meta",
                    "name": META_INFO["name"],
                    "specialty_vi": META_INFO["specialty_vi"],
                    "round": current_round,
                    "phase": "synthesize",
                    "state": "speaking",
                    "answer": syn_answer,
                    "explanation": syn_expl,
                    "short_bubble": syn_bubble,
                    "latency_ms": meta_latency,
                    "usage": meta_usage,
                    "total_tokens": dict(total_tokens),
                    "system_prompt": meta_sys,
                    "raw_output": meta_raw,
                }
            )
            for doc in DOCTORS_META:
                yield emit(
                    {
                        "type": "packet_fly",
                        "from": "meta",
                        "to": doc["id"],
                        "label": f"Tổng hợp: {syn_answer}",
                        "color": META_INFO["color"],
                    }
                )

            # ------------------------------------------------------
            # STEP 3: Doctors review synthesis in parallel
            # ------------------------------------------------------
            yield emit(
                {
                    "type": "phase_change",
                    "round": current_round,
                    "phase": "review",
                    "label": f"Vòng {current_round}: Các bác sĩ phản biện & bỏ phiếu đồng thuận",
                }
            )
            for doc in DOCTORS_META:
                yield emit(
                    {
                        "type": "agent_state",
                        "agent_id": doc["id"],
                        "state": "thinking",
                        "round": current_round,
                        "phase": "review",
                        "status_text": "Đang phản biện bản tổng hợp...",
                    }
                )

            async def run_doctor_review(doc: Dict[str, Any]) -> Dict[str, Any]:
                own_op = next((o for o in round_opinions if o["doctor_id"] == doc["id"]), {})
                rev_sys = (
                    f"You are {doc['name']}, specialist in {doc['specialty']} ({doc['specialty_vi']}), "
                    f"reviewing the Chief Coordinator's Round {current_round} synthesis. "
                    f"Critically check whether you AGREE or DISAGREE based on clinical evidence. {lang_note} "
                    f"Output ONLY a valid JSON object with keys: "
                    f"'agree' (boolean true or false), "
                    f"'reason' (concise rationale), "
                    f"'short_bubble' (1 short sentence <= 12 words), "
                    f"'answer' (your supported option letter or conclusion)."
                )
                rev_usr = (
                    f"Case:\n{q_text}\n\n"
                    f"Your Round {current_round} Analysis:\nAnswer: {own_op.get('answer')}\nExplanation: {own_op.get('explanation')}\n\n"
                    f"Meta Agent Synthesis:\nAnswer: {syn_answer}\nExplanation: {syn_expl}\n\n"
                    f"Do you agree? Return JSON only."
                )
                raw, latency_ms, usage = await _call_llm_json(client, model_name, rev_sys, rev_usr)
                parsed = parse_json_fallback(raw, expect_review=True)
                return {
                    "doc": doc,
                    "system_prompt": rev_sys,
                    "raw": raw,
                    "parsed": parsed,
                    "latency_ms": latency_ms,
                    "usage": usage,
                }

            rev_tasks = [asyncio.create_task(run_doctor_review(d)) for d in DOCTORS_META]
            round_reviews: List[Dict[str, Any]] = []
            agree_count = 0

            for fut in asyncio.as_completed(rev_tasks):
                res = await fut
                doc = res["doc"]
                parsed = res["parsed"]
                usage = res["usage"]
                for k in total_tokens:
                    total_tokens[k] += usage.get(k, 0)

                agrees = bool(parsed.get("agree", False))
                if agrees:
                    agree_count += 1
                reason = str(parsed.get("reason", "")).strip()
                ans = str(parsed.get("answer") or syn_answer).strip()
                bubble = str(parsed.get("short_bubble") or reason[:80]).strip()

                rev_record = {
                    "doctor_id": doc["id"],
                    "name": doc["name"],
                    "specialty_vi": doc["specialty_vi"],
                    "agree": agrees,
                    "reason": reason,
                    "answer": ans,
                    "short_bubble": bubble,
                }
                doctor_memories[doc["id"]].append({"round": current_round, "type": "review", **rev_record})
                round_reviews.append(rev_record)

                yield emit(
                    {
                        "type": "agent_output",
                        "agent_id": doc["id"],
                        "name": doc["name"],
                        "specialty_vi": doc["specialty_vi"],
                        "round": current_round,
                        "phase": "review",
                        "state": "agree" if agrees else "disagree",
                        "agree": agrees,
                        "answer": ans,
                        "explanation": reason,
                        "short_bubble": bubble,
                        "latency_ms": res["latency_ms"],
                        "usage": usage,
                        "total_tokens": dict(total_tokens),
                        "system_prompt": res["system_prompt"],
                        "raw_output": res["raw"],
                    }
                )
                yield emit(
                    {
                        "type": "packet_fly",
                        "from": doc["id"],
                        "to": "meta",
                        "label": "Đồng ý" if agrees else f"Phản đối ({ans})",
                        "color": "#22c55e" if agrees else "#ef4444",
                    }
                )

            last_reviews = round_reviews
            all_agree = agree_count == len(DOCTORS_META)

            yield emit(
                {
                    "type": "consensus_update",
                    "round": current_round,
                    "agree_count": agree_count,
                    "total_doctors": len(DOCTORS_META),
                    "all_agree": all_agree,
                }
            )

            # ------------------------------------------------------
            # STEP 4: Meta agent makes round/final decision
            # ------------------------------------------------------
            is_final = all_agree or (current_round >= max_rounds)
            yield emit(
                {
                    "type": "agent_state",
                    "agent_id": "meta",
                    "state": "thinking",
                    "round": current_round,
                    "phase": "decision",
                    "status_text": "Đang chốt kết luận cuối cùng..." if is_final else "Chưa đồng thuận, tổng kết vòng...",
                }
            )

            reviews_block = "\n".join(
                f"- {r['name']} ({r['specialty_vi']}): Agree={'Yes' if r['agree'] else 'No'}, Answer={r['answer']}, Reason={r['reason']}"
                for r in round_reviews
            )
            dec_sys = (
                f"You are {META_INFO['name']}, Chief Coordinator making the "
                f"{'FINAL' if is_final else f'Round {current_round} interim'} decision in a ColaCare MDT consultation. "
                f"{'All doctors reached consensus.' if all_agree else ('Max rounds reached; use majority/strongest evidence.' if is_final else 'Consensus not yet reached; summarize remaining disagreements for the next round.')} "
                f"{lang_note} "
                f"Output ONLY a valid JSON object with keys: "
                f"'explanation' (final medical conclusion), "
                f"'short_bubble' (1 short sentence <= 14 words), "
                f"'answer' (final option letter if options provided)."
            )
            dec_usr = (
                f"Case:\n{q_text}\n\n"
                f"Current Synthesis: Answer={syn_answer}, Explanation={syn_expl}\n\n"
                f"Specialist Reviews:\n{reviews_block}\n\nReturn JSON only."
            )
            dec_raw, dec_latency, dec_usage = await _call_llm_json(client, model_name, dec_sys, dec_usr)
            decision = parse_json_fallback(dec_raw, expect_review=False)
            for k in total_tokens:
                total_tokens[k] += dec_usage.get(k, 0)

            dec_ans = str(decision.get("answer") or syn_answer).strip()
            dec_expl = str(decision.get("explanation", "")).strip()
            dec_bubble = str(decision.get("short_bubble") or dec_expl[:90]).strip()

            yield emit(
                {
                    "type": "agent_output",
                    "agent_id": "meta",
                    "name": META_INFO["name"],
                    "specialty_vi": META_INFO["specialty_vi"],
                    "round": current_round,
                    "phase": "decision",
                    "state": "final" if is_final else "speaking",
                    "answer": dec_ans,
                    "explanation": dec_expl,
                    "short_bubble": dec_bubble,
                    "latency_ms": dec_latency,
                    "usage": dec_usage,
                    "total_tokens": dict(total_tokens),
                    "system_prompt": dec_sys,
                    "raw_output": dec_raw,
                }
            )

            if is_final:
                consensus_reached = all_agree
                final_decision = {"answer": dec_ans, "explanation": dec_expl}
                break

        total_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        final_event = emit(
            {
                "type": "session_complete",
                "session_id": session_id,
                "final_answer": (final_decision or {}).get("answer", "?"),
                "final_explanation": (final_decision or {}).get("explanation", ""),
                "ground_truth": ground_truth,
                "consensus_reached": consensus_reached,
                "total_rounds": current_round,
                "total_elapsed_ms": total_elapsed_ms,
                "total_tokens": total_tokens,
            }
        )
        yield final_event

        # Save replayable log to logs/pixel_mdt/
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_path = os.path.join(LOGS_DIR, f"{session_id}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "qid": qid,
                    "model_key": model_key,
                    "final_answer": (final_decision or {}).get("answer", "?"),
                    "ground_truth": ground_truth,
                    "consensus_reached": consensus_reached,
                    "total_rounds": current_round,
                    "total_elapsed_ms": total_elapsed_ms,
                    "total_tokens": total_tokens,
                    "events": recorded_events,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    finally:
        await client.close()


async def stream_colacare_demo(
    question: str,
    options: Optional[Dict[str, str]],
    qid: str = "demo_case",
    ground_truth: Optional[str] = "A",
) -> AsyncIterator[Dict[str, Any]]:
    """
    Zero-cost 2-round scripted ColaCare simulation showing real disagreement in Round 1
    and consensus convergence in Round 2. Ideal for testing animations & offline demos.
    """
    session_id = f"demo_{datetime.now().strftime('%H%M%S')}"
    started_at = time.perf_counter()
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def tick(ev: Dict[str, Any], add_tokens: int = 0) -> Dict[str, Any]:
        if add_tokens:
            total_tokens["prompt_tokens"] += int(add_tokens * 0.65)
            total_tokens["completion_tokens"] += int(add_tokens * 0.35)
            total_tokens["total_tokens"] = total_tokens["prompt_tokens"] + total_tokens["completion_tokens"]
            ev["usage"] = {
                "prompt_tokens": int(add_tokens * 0.65),
                "completion_tokens": int(add_tokens * 0.35),
                "total_tokens": add_tokens,
            }
            ev["total_tokens"] = dict(total_tokens)
        ev["ts"] = round(time.perf_counter() - started_at, 2)
        return ev

    yield tick(
        {
            "type": "session_start",
            "session_id": session_id,
            "mode": "demo",
            "qid": qid,
            "question": question,
            "options": options or {"A": "Tái tưới máu mạch vành khẩn cấp (PCI)", "B": "Cho về theo dõi ngoại trú", "C": "Điều trị bảo tồn bằng thuốc & siêu âm tim"},
            "ground_truth": ground_truth or "A",
            "model_key": "simulation-colacare-2round",
            "model_name": "ColaCare Demo Simulator",
            "max_rounds": 2,
            "doctors": DOCTORS_META,
            "meta": META_INFO,
        }
    )

    # ---- ROUND 1 ----
    yield tick({"type": "round_start", "round": 1, "max_rounds": 2, "phase": "analyze", "label": "Vòng 1: Phân tích độc lập từ 3 chuyên khoa"})
    for d in DOCTORS_META:
        yield tick({"type": "agent_state", "agent_id": d["id"], "state": "thinking", "round": 1, "phase": "analyze", "status_text": "Đang đọc hồ sơ & ECG..."})

    r1_ops = [
        ("doctor_1", "BS. Linh", "Nội khoa", "A", "Đau ngực trái điển hình kèm ST chênh lên V1–V4 là nhồi máu cơ tim cấp STEMI thành trước. Cần tái tưới máu khẩn cấp trong cửa sổ vàng.", "STEMI thành trước rất rõ, ưu tiên tái tưới máu ngay!", 920, 410),
        ("doctor_2", "BS. Minh", "Ngoại khoa", "A", "Mức độ tổn thương cơ tim tiến triển theo từng phút. Chuyển thẳng phòng thông tim can thiệp mạch vành (PCI) là ưu tiên số 1.", "Đề xuất can thiệp mạch vành khẩn (A)!", 1180, 390),
        ("doctor_3", "BS. Hùng", "Chẩn đoán hình ảnh", "C", "Chưa có men tim troponin và hình ảnh siêu âm tim loại trừ phình tách động mạch chủ. Nên chụp/siêu âm kiểm tra nhanh trước khi can thiệp xâm lấn.", "Cần siêu âm loại trừ tách thành ĐMC trước (C).", 1450, 430),
    ]
    for aid, name, spec, ans, expl, bub, lat, tok in r1_ops:
        await asyncio.sleep(0.85)
        doc_color = next(d["color"] for d in DOCTORS_META if d["id"] == aid)
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
                "explanation": expl,
                "short_bubble": bub,
                "latency_ms": lat,
                "system_prompt": f"[DEMO] You are {name} ({spec}) in ColaCare Round 1.",
                "raw_output": json.dumps({"answer": ans, "explanation": expl, "short_bubble": bub}, ensure_ascii=False, indent=2),
            },
            add_tokens=tok,
        )
        yield tick({"type": "packet_fly", "from": aid, "to": "meta", "label": f"Đề xuất: {ans}", "color": doc_color})

    await asyncio.sleep(0.6)
    yield tick({"type": "agent_state", "agent_id": "meta", "state": "thinking", "round": 1, "phase": "synthesize", "status_text": "Đang tổng hợp 3 ý kiến..."})
    await asyncio.sleep(1.1)
    r1_syn_expl = "Nội khoa và Ngoại khoa thống nhất chọn A (STEMI cần tái tưới máu khẩn). Chẩn đoán hình ảnh thận trọng đề nghị C để loại trừ tách thành ĐMC. Theo hướng dẫn ESC/AHA, không được trì hoãn tái tưới máu khi ECG đã có ST chênh lên điển hình → đề xuất sơ bộ: A."
    yield tick(
        {
            "type": "agent_output",
            "agent_id": "meta",
            "name": META_INFO["name"],
            "specialty_vi": META_INFO["specialty_vi"],
            "round": 1,
            "phase": "synthesize",
            "state": "speaking",
            "answer": "A",
            "explanation": r1_syn_expl,
            "short_bubble": "Đa số chọn A (PCI khẩn), mời 3 khoa phản biện!",
            "latency_ms": 1120,
            "system_prompt": "[DEMO] Chief Coordinator synthesizing Round 1 opinions.",
            "raw_output": json.dumps({"answer": "A", "explanation": r1_syn_expl}, ensure_ascii=False, indent=2),
        },
        add_tokens=560,
    )
    for d in DOCTORS_META:
        yield tick({"type": "packet_fly", "from": "meta", "to": d["id"], "label": "Tổng hợp: A", "color": META_INFO["color"]})

    await asyncio.sleep(0.7)
    yield tick({"type": "phase_change", "round": 1, "phase": "review", "label": "Vòng 1: Phản biện & kiểm tra đồng thuận"})
    for d in DOCTORS_META:
        yield tick({"type": "agent_state", "agent_id": d["id"], "state": "thinking", "round": 1, "phase": "review", "status_text": "Đang xem bản tổng hợp..."})

    r1_revs = [
        ("doctor_1", "BS. Linh", "Nội khoa", True, "A", "Hoàn toàn đồng ý. ST chênh lên ở V1-V4 kèm đau ngực điển hình cần kích hoạt CathLab ngay.", "Đồng ý 100% chọn A!", 740, 310),
        ("doctor_2", "BS. Minh", "Ngoại khoa", True, "A", "Đồng ý với Bác sĩ trưởng, trì hoãn thêm sẽ làm hoại tử rộng cơ tim thành trước.", "Đồng ý chọn A, thời gian là cơ tim.", 880, 300),
        ("doctor_3", "BS. Hùng", "Chẩn đoán hình ảnh", False, "C", "Tôi vẫn phản đối nếu bỏ qua đánh giá hình ảnh ban đầu — ít nhất cần siêu âm tại giường (POCUS) song song để chắc chắn an toàn.", "Chưa đồng ý! Cần làm rõ nguy cơ ĐMC trước.", 1150, 340),
    ]
    for aid, name, spec, ag, ans, reason, bub, lat, tok in r1_revs:
        await asyncio.sleep(0.8)
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
                "answer": ans,
                "explanation": reason,
                "short_bubble": bub,
                "latency_ms": lat,
                "system_prompt": f"[DEMO] {name} reviewing Round 1 synthesis.",
                "raw_output": json.dumps({"agree": ag, "answer": ans, "reason": reason}, ensure_ascii=False, indent=2),
            },
            add_tokens=tok,
        )
        yield tick({"type": "packet_fly", "from": aid, "to": "meta", "label": "Đồng ý" if ag else "Phản đối (C)", "color": "#22c55e" if ag else "#ef4444"})

    yield tick({"type": "consensus_update", "round": 1, "agree_count": 2, "total_doctors": 3, "all_agree": False})
    await asyncio.sleep(0.8)
    yield tick(
        {
            "type": "agent_output",
            "agent_id": "meta",
            "name": META_INFO["name"],
            "specialty_vi": META_INFO["specialty_vi"],
            "round": 1,
            "phase": "decision",
            "state": "speaking",
            "answer": "A",
            "explanation": "Chưa đạt đồng thuận 3/3 do BS. Hùng lo ngại bỏ sót tách thành ĐMC. Chuyển sang Vòng 2 để các chuyên khoa thảo luận phương án phối hợp POCUS tại giường không làm chậm PCI.",
            "short_bubble": "Mới đạt 2/3 đồng thuận → Mở Vòng 2 phản biện!",
            "latency_ms": 840,
            "system_prompt": "[DEMO] Interim decision after Round 1.",
            "raw_output": json.dumps({"answer": "A", "note": "Proceed to Round 2"}, ensure_ascii=False, indent=2),
        },
        add_tokens=420,
    )

    # ---- ROUND 2 ----
    await asyncio.sleep(0.9)
    yield tick({"type": "round_start", "round": 2, "max_rounds": 2, "phase": "analyze", "label": "Vòng 2: Đánh giá lại sau phản biện"})
    for d in DOCTORS_META:
        yield tick({"type": "agent_state", "agent_id": d["id"], "state": "thinking", "round": 2, "phase": "analyze", "status_text": "Đang cân nhắc phản biện vòng 1..."})

    r2_ops = [
        ("doctor_1", "BS. Linh", "Nội khoa", "A", "Ghi nhận lưu ý của BS. Hùng: có thể siêu âm tim tại giường trong 2 phút trong lúc đẩy bệnh nhân sang phòng thông tim, nhưng hướng xử trí ưu tiên hàng đầu vẫn là A.", "Giữ nguyên A + làm POCUS song song!", 820, 380),
        ("doctor_2", "BS. Minh", "Ngoại khoa", "A", "Tán thành kết hợp đánh giá mạch 2 tay & POCUS nhanh, nhưng không trì hoãn tái tưới máu mạch vành (A).", "Chốt A, không làm chậm cửa sổ PCI.", 950, 360),
        ("doctor_3", "BS. Hùng", "Chẩn đoán hình ảnh", "A", "Nếu thực hiện siêu âm tại giường song song mà không trì hoãn CathLab thì rủi ro được kiểm soát tốt. Tôi điều chỉnh sang chọn A.", "Hợp lý! Tôi đổi sang phương án A.", 1080, 410),
    ]
    for aid, name, spec, ans, expl, bub, lat, tok in r2_ops:
        await asyncio.sleep(0.85)
        doc_color = next(d["color"] for d in DOCTORS_META if d["id"] == aid)
        yield tick(
            {
                "type": "agent_output",
                "agent_id": aid,
                "name": name,
                "specialty_vi": spec,
                "round": 2,
                "phase": "analyze",
                "state": "speaking",
                "answer": ans,
                "explanation": expl,
                "short_bubble": bub,
                "latency_ms": lat,
                "system_prompt": f"[DEMO] {name} re-evaluating in Round 2.",
                "raw_output": json.dumps({"answer": ans, "explanation": expl}, ensure_ascii=False, indent=2),
            },
            add_tokens=tok,
        )
        yield tick({"type": "packet_fly", "from": aid, "to": "meta", "label": f"Đề xuất: {ans}", "color": doc_color})

    await asyncio.sleep(0.6)
    yield tick({"type": "agent_state", "agent_id": "meta", "state": "thinking", "round": 2, "phase": "synthesize", "status_text": "Tổng hợp đồng thuận Vòng 2..."})
    await asyncio.sleep(0.9)
    yield tick(
        {
            "type": "agent_output",
            "agent_id": "meta",
            "name": META_INFO["name"],
            "specialty_vi": META_INFO["specialty_vi"],
            "round": 2,
            "phase": "synthesize",
            "state": "speaking",
            "answer": "A",
            "explanation": "Cả 3 chuyên khoa hiện đều thống nhất phương án A (Tái tưới máu mạch vành khẩn cấp) kết hợp đánh giá POCUS nhanh tại giường mà không gây trì hoãn.",
            "short_bubble": "Cả 3 khoa đều chọn A! Mời xác nhận cuối.",
            "latency_ms": 910,
            "system_prompt": "[DEMO] Round 2 synthesis.",
            "raw_output": json.dumps({"answer": "A", "consensus": True}, ensure_ascii=False, indent=2),
        },
        add_tokens=490,
    )
    for d in DOCTORS_META:
        yield tick({"type": "packet_fly", "from": "meta", "to": d["id"], "label": "Tổng hợp: A", "color": META_INFO["color"]})

    await asyncio.sleep(0.6)
    yield tick({"type": "phase_change", "round": 2, "phase": "review", "label": "Vòng 2: Xác nhận đồng thuận cuối cùng"})
    for d in DOCTORS_META:
        await asyncio.sleep(0.55)
        yield tick(
            {
                "type": "agent_output",
                "agent_id": d["id"],
                "name": d["name"],
                "specialty_vi": d["specialty_vi"],
                "round": 2,
                "phase": "review",
                "state": "agree",
                "agree": True,
                "answer": "A",
                "explanation": f"{d['name']} ({d['specialty_vi']}) xác nhận đồng thuận hoàn toàn với kết luận A.",
                "short_bubble": "Đồng ý chốt phương án A!",
                "latency_ms": 620,
                "system_prompt": f"[DEMO] {d['name']} Round 2 review.",
                "raw_output": json.dumps({"agree": True, "answer": "A"}, ensure_ascii=False, indent=2),
            },
            add_tokens=240,
        )
        yield tick({"type": "packet_fly", "from": d["id"], "to": "meta", "label": "Đồng ý", "color": "#22c55e"})

    yield tick({"type": "consensus_update", "round": 2, "agree_count": 3, "total_doctors": 3, "all_agree": True})
    await asyncio.sleep(0.6)
    final_expl = "Hội đồng đa chuyên khoa (Nội khoa, Ngoại khoa, CĐHA) đạt đồng thuận 3/3 sau 2 vòng thảo luận: Chốt đáp án A — Tái tưới máu mạch vành khẩn cấp (PCI), kết hợp POCUS nhanh tại giường không gây trì hoãn."
    yield tick(
        {
            "type": "agent_output",
            "agent_id": "meta",
            "name": META_INFO["name"],
            "specialty_vi": META_INFO["specialty_vi"],
            "round": 2,
            "phase": "decision",
            "state": "final",
            "answer": "A",
            "explanation": final_expl,
            "short_bubble": "CHỐT ĐỒNG THUẬN 3/3: ĐÁP ÁN A!",
            "latency_ms": 780,
            "system_prompt": "[DEMO] Final MDT Decision.",
            "raw_output": json.dumps({"answer": "A", "explanation": final_expl}, ensure_ascii=False, indent=2),
        },
        add_tokens=380,
    )

    yield tick(
        {
            "type": "session_complete",
            "session_id": session_id,
            "final_answer": "A",
            "final_explanation": final_expl,
            "ground_truth": ground_truth or "A",
            "consensus_reached": True,
            "total_rounds": 2,
            "total_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            "total_tokens": total_tokens,
        }
    )
