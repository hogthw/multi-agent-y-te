"""
Batch Benchmark Runner for HL7 FHIR R4 Multi-Agent & Memory-Augmented Evaluation.

Runs all baselines + ablation + proposed method on my_datasets/processed/fhir_audit/fhir_benchmark.json,
caches per-case results to allow resuming, and generates CSV/JSON/Markdown reports.

Usage examples:
  # Quick rule-based only verification (0 API calls)
  .venv/bin/python -m fhir_audit.run_benchmark --methods rule_based

  # Full 5-method comparison on 12 balanced cases (3 per category) with Gemini Flash Lite
  .venv/bin/python -m fhir_audit.run_benchmark --model gemini-3.1-flash-lite --per_category 3

  # Full 24-case publication benchmark (including semantic-only ablation)
  .venv/bin/python -m fhir_audit.run_benchmark --model gemini-3.1-flash-lite --methods rule_based single_llm single_llm_memory multi_agent_no_mem multi_agent_sem multi_agent_memory
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List

from fhir_audit.agents import stream_fhir_audit
from fhir_audit.dataset_generator import DEFAULT_OUTPUT_DIR, generate_and_save
from fhir_audit.evaluate import RESULTS_DIR, build_and_save_reports


ALL_METHODS = [
    "rule_based",
    "single_llm",
    "single_llm_memory",
    "multi_agent_no_mem",
    "multi_agent_sem",
    "multi_agent_memory",
]


def load_benchmark_dataset(per_category: int = 0) -> List[Dict[str, Any]]:
    path = os.path.join(DEFAULT_OUTPUT_DIR, "fhir_benchmark.json")
    if not os.path.exists(path):
        generate_and_save(DEFAULT_OUTPUT_DIR)
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if per_category > 0:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for c in cases:
            cat = c.get("primary_category", "OTHER")
            buckets.setdefault(cat, [])
            if len(buckets[cat]) < per_category:
                buckets[cat].append(c)
        selected = []
        for cat in ("CLEAN", "STRUCTURAL", "TERMINOLOGY", "CLINICAL_SEMANTIC"):
            selected.extend(buckets.get(cat, []))
        return selected
    return cases


async def run_one_case(
    case: Dict[str, Any],
    method: str,
    model_key: str,
    max_rounds: int,
    language: str,
) -> Dict[str, Any]:
    final_ev: Dict[str, Any] = {}
    async for ev in stream_fhir_audit(
        bundle=case["bundle"],
        method=method,
        model_key=model_key,
        max_rounds=max_rounds,
        language=language,
        case_id=case["case_id"],
        ground_truth_issues=case.get("ground_truth_issues", []),
        save_session=False,
    ):
        if ev.get("type") == "session_complete":
            final_ev = ev
    final_ev["primary_category"] = case.get("primary_category", "CLEAN")
    return final_ev


async def run_benchmark(
    methods: List[str],
    model_key: str,
    max_rounds: int,
    per_category: int,
    language: str,
    force_rerun: bool,
) -> None:
    cases = load_benchmark_dataset(per_category=per_category)
    print(f"Loaded {len(cases)} FHIR R4 benchmark cases. Methods: {methods}. Model: {model_key}")

    cache_dir = os.path.join(RESULTS_DIR, "raw_runs", model_key)
    os.makedirs(cache_dir, exist_ok=True)
    method_results: Dict[str, List[Dict[str, Any]]] = {}

    for method in methods:
        method_cache_path = os.path.join(cache_dir, f"{method}.json")
        cached_map: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(method_cache_path) and not force_rerun:
            try:
                with open(method_cache_path, "r", encoding="utf-8") as f:
                    for rec in json.load(f):
                        if rec.get("case_id"):
                            cached_map[rec["case_id"]] = rec
            except Exception:
                cached_map = {}

        records: List[Dict[str, Any]] = []
        print(f"\n=== Running method: {method} ===")
        for idx, case in enumerate(cases, 1):
            cid = case["case_id"]
            if cid in cached_map:
                rec = cached_map[cid]
                print(f"  [{idx}/{len(cases)}] {cid} (cached) -> {rec.get('final_answer')}")
            else:
                try:
                    rec = await run_one_case(case, method, model_key, max_rounds, language)
                    cached_map[cid] = rec
                    # Persist incrementally so partial progress is never lost
                    with open(method_cache_path, "w", encoding="utf-8") as f:
                        json.dump(list(cached_map.values()), f, ensure_ascii=False, indent=2)
                    print(
                        f"  [{idx}/{len(cases)}] {cid} ({case['primary_category']}) -> "
                        f"{rec.get('final_answer')} | {rec.get('total_elapsed_ms', 0)}ms | "
                        f"tok={rec.get('total_tokens', {}).get('total_tokens', 0)}"
                    )
                except Exception as exc:
                    print(f"  [{idx}/{len(cases)}] {cid} FAILED: {exc}")
                    continue
            records.append(rec)
        method_results[method] = records

    build_and_save_reports(method_results, model_key=model_key, output_dir=RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HL7 FHIR R4 Multi-Agent & Memory-Augmented Benchmark.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["rule_based", "single_llm", "single_llm_memory", "multi_agent_no_mem", "multi_agent_memory"],
        choices=ALL_METHODS,
    )
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite")
    parser.add_argument("--max_rounds", type=int, default=2)
    parser.add_argument("--per_category", type=int, default=0, help="If >0, limit to N cases per category (e.g. 2 or 3 for fast pilot).")
    parser.add_argument("--language", type=str, default="vi", choices=["vi", "en"])
    parser.add_argument("--force", action="store_true", help="Ignore cached results and rerun all cases.")
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            methods=args.methods,
            model_key=args.model,
            max_rounds=args.max_rounds,
            per_category=args.per_category,
            language=args.language,
            force_rerun=args.force,
        )
    )
