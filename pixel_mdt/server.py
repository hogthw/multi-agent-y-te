"""
Starlette ASGI server for MedAgentBoard:
  - Light-Theme Pixel 2D Hospital Room for HL7 FHIR R4 Memory-Augmented Multi-Agent Auditing
  - Live Benchmark Results viewer (Bootstrap F1, FPR, Latency, Tokens)
  - MedQA ColaCare mode preserved for direct baseline comparison
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from fhir_audit.agents import (
    FHIR_AUDITORS,
    FHIR_LOGS_DIR,
    FHIR_META,
    stream_fhir_audit,
    stream_fhir_demo,
)
from fhir_audit.dataset_generator import DEFAULT_OUTPUT_DIR, generate_and_save
from fhir_audit.evaluate import RESULTS_DIR
from pixel_mdt.engine import (
    DOCTORS_META,
    LOGS_DIR as MEDQA_LOGS_DIR,
    META_INFO,
    get_available_models,
    stream_colacare_demo,
    stream_colacare_live,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

FHIR_METHODS = [
    {"id": "fhir_demo", "label": "Mô phỏng 2 vòng FHIR + Bộ nhớ (0đ — Trình chiếu)"},
    {"id": "multi_agent_memory", "label": "Multi-Agent + Bộ nhớ 2 tầng (Đề xuất)"},
    {"id": "multi_agent_sem", "label": "Ablation: Multi-Agent + Chỉ Bộ nhớ Ngữ nghĩa"},
    {"id": "multi_agent_no_mem", "label": "Đối chứng 4: Multi-Agent ColaCare (Không Bộ nhớ)"},
    {"id": "single_llm_memory", "label": "Đối chứng 3: Single-LLM + Bộ nhớ (RAG)"},
    {"id": "single_llm", "label": "Đối chứng 2: Single-LLM Zero-Shot"},
    {"id": "rule_based", "label": "Đối chứng 1: Bộ kiểm tra Luật FHIR (0 token)"},
]


def load_fhir_cases() -> List[Dict[str, Any]]:
    path = os.path.join(DEFAULT_OUTPUT_DIR, "fhir_benchmark.json")
    if not os.path.exists(path):
        generate_and_save(DEFAULT_OUTPUT_DIR)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_medqa_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = [
        {
            "qid": "vn_stemi_case_01",
            "title": "🏥 [VN] Nhồi máu cơ tim cấp ST chênh lên (STEMI)",
            "question": (
                "Bệnh nhân nam 58 tuổi đau thắt ngực trái lan lên vai kèm vã mồ hôi 30 phút. "
                "Điện tâm đồ có ST chênh lên ở V1–V4. Hướng xử trí ưu tiên hàng đầu là gì?"
            ),
            "options": {
                "A": "Tái tưới máu mạch vành khẩn cấp (PCI)",
                "B": "Cho về theo dõi ngoại trú",
                "C": "Trì hoãn can thiệp, nhập viện theo dõi bảo tồn",
            },
            "answer": "A",
        }
    ]
    pubmed_path = os.path.join("my_datasets", "processed", "medqa", "PubMedQA", "medqa_mc_test.json")
    if os.path.exists(pubmed_path):
        try:
            with open(pubmed_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                first_line = item.get("question", "").split("\n")[0][:72]
                cases.append(
                    {
                        "qid": item.get("qid", "pubmedqa"),
                        "title": f"📄 [PubMedQA] {first_line}...",
                        "question": item.get("question", ""),
                        "options": item.get("options", {}),
                        "answer": item.get("answer", ""),
                    }
                )
        except Exception:
            pass
    return cases


def list_all_logs() -> List[Dict[str, Any]]:
    os.makedirs(FHIR_LOGS_DIR, exist_ok=True)
    os.makedirs(MEDQA_LOGS_DIR, exist_ok=True)
    files = glob.glob(os.path.join(FHIR_LOGS_DIR, "*.json")) + glob.glob(os.path.join(MEDQA_LOGS_DIR, "*.json"))
    files.sort(key=os.path.getmtime, reverse=True)
    result = []
    for path in files[:35]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            result.append(
                {
                    "filename": os.path.basename(path),
                    "folder": "fhir" if FHIR_LOGS_DIR in path else "medqa",
                    "session_id": data.get("session_id", os.path.basename(path)),
                    "task_mode": data.get("task_mode", "medqa"),
                    "case_id": data.get("case_id") or data.get("qid", ""),
                    "method": data.get("method", "colacare"),
                    "model_key": data.get("model_key", ""),
                    "final_answer": data.get("final_answer", "?"),
                    "consensus_reached": data.get("consensus_reached", False),
                    "total_rounds": data.get("total_rounds", 1),
                    "created_at": data.get("created_at", ""),
                }
            )
        except Exception:
            continue
    return result


def load_benchmark_summary() -> Dict[str, Any]:
    path = os.path.join(RESULTS_DIR, "benchmark_summary.json")
    if not os.path.exists(path):
        return {"model_key": None, "summaries": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"model_key": None, "summaries": []}


async def homepage(request: Request):
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def api_config(request: Request):
    return JSONResponse(
        {
            "models": get_available_models(),
            "fhir_cases": load_fhir_cases(),
            "fhir_methods": FHIR_METHODS,
            "medqa_cases": load_medqa_cases(),
            "logs": list_all_logs(),
            "benchmark": load_benchmark_summary(),
            "fhir_auditors": FHIR_AUDITORS,
            "fhir_meta": FHIR_META,
            "medqa_doctors": DOCTORS_META,
            "medqa_meta": META_INFO,
        }
    )


async def api_benchmark(request: Request):
    return JSONResponse(load_benchmark_summary())


async def api_get_log(request: Request):
    filename = os.path.basename(request.path_params["filename"])
    for folder in (FHIR_LOGS_DIR, MEDQA_LOGS_DIR):
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return JSONResponse(json.load(f))
    return JSONResponse({"error": "Log not found"}, status_code=404)


async def api_fhir_audit(request: Request):
    payload = await request.json()
    method = payload.get("method", "fhir_demo")
    bundle = payload.get("bundle")
    if isinstance(bundle, str):
        try:
            bundle = json.loads(bundle)
        except Exception as exc:
            return JSONResponse({"error": f"JSON FHIR không hợp lệ: {exc}"}, status_code=400)
    if not isinstance(bundle, dict):
        return JSONResponse({"error": "Vui lòng chọn hoặc dán một HL7 FHIR R4 Bundle JSON hợp lệ."}, status_code=400)

    model_key = payload.get("model_key") or "gemini-3.1-flash-lite"
    max_rounds = max(1, min(int(payload.get("max_rounds", 2)), 3))
    language = payload.get("language", "vi")
    case_id = payload.get("case_id", "custom_fhir")
    gt_issues = payload.get("ground_truth_issues")

    async def event_generator():
        try:
            if method == "fhir_demo":
                stream = stream_fhir_demo(bundle=bundle, case_id=case_id, ground_truth_issues=gt_issues)
            else:
                stream = stream_fhir_audit(
                    bundle=bundle,
                    method=method,
                    model_key=model_key,
                    max_rounds=max_rounds,
                    language=language,
                    case_id=case_id,
                    ground_truth_issues=gt_issues,
                    save_session=True,
                )
            async for ev in stream:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def api_consult(request: Request):
    """MedQA ColaCare consultation SSE endpoint."""
    payload = await request.json()
    mode = payload.get("mode", "demo")
    question = (payload.get("question") or "").strip()
    options = payload.get("options") or None
    model_key = payload.get("model_key") or "gemini-3.1-flash-lite"
    max_rounds = max(1, min(int(payload.get("max_rounds", 2)), 3))
    language = payload.get("language", "vi")
    qid = payload.get("qid", "custom_case")
    ground_truth = payload.get("ground_truth") or None

    if not question:
        return JSONResponse({"error": "Vui lòng nhập nội dung câu hỏi y khoa."}, status_code=400)

    async def event_generator():
        try:
            if mode == "demo":
                stream = stream_colacare_demo(question=question, options=options, qid=qid, ground_truth=ground_truth)
            else:
                stream = stream_colacare_live(
                    question=question, options=options, model_key=model_key,
                    max_rounds=max_rounds, language=language, qid=qid, ground_truth=ground_truth,
                )
            async for ev in stream:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


app = Starlette(
    debug=False,
    routes=[
        Route("/", homepage),
        Route("/api/config", api_config, methods=["GET"]),
        Route("/api/benchmark", api_benchmark, methods=["GET"]),
        Route("/api/logs/{filename:str}", api_get_log, methods=["GET"]),
        Route("/api/fhir/audit", api_fhir_audit, methods=["POST"]),
        Route("/api/consult", api_consult, methods=["POST"]),
        Mount("/static", app=StaticFiles(directory=STATIC_DIR), name="static"),
    ],
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    print(f"\n🏥 MedAgentBoard FHIR Memory-Augmented Pixel 2D Room (Light Slide Theme): http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
