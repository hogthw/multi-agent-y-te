"""
Statistical Evaluation & Bootstrap Reporting for HL7 FHIR R4 Multi-Agent Auditing.

Computes:
  - Case-level Detection: Accuracy, Precision, Recall, F1, False Positive Rate (FPR)
  - Category-level Recall & F1: STRUCTURAL, TERMINOLOGY, CLINICAL_SEMANTIC
  - Code-level Exact Match Rate (error_code match against Ground Truth)
  - System Efficiency: Mean Latency (s), Mean Tokens / Bundle, Est. Cost, Consensus Rate, Memory Hits
  - Bootstrap 95% Confidence Intervals (1000 resamples, seed=42) following MedAgentBoard methodology

Exports JSON, CSV, and publication-ready Markdown tables to logs/fhir_audit/results/.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


RESULTS_DIR = os.path.join("logs", "fhir_audit", "results")
CATEGORIES = ["STRUCTURAL", "TERMINOLOGY", "CLINICAL_SEMANTIC"]


def _match_case(record: Dict[str, Any]) -> Dict[str, Any]:
    gt_issues = record.get("ground_truth_issues", [])
    pred_issues = record.get("issues", [])
    gt_has = len(gt_issues) > 0
    pred_has = bool(record.get("has_issues", len(pred_issues) > 0)) and len(pred_issues) > 0

    # Category-level match
    gt_cats = {i["category"] for i in gt_issues}
    pred_cats = {i["category"] for i in pred_issues}
    cat_hit = (not gt_has and not pred_has) or bool(gt_cats & pred_cats)

    # Error-code level match
    gt_codes = {i["error_code"] for i in gt_issues}
    pred_codes = {i["error_code"] for i in pred_issues}
    code_hit = (not gt_has and not pred_has) or bool(gt_codes & pred_codes)

    return {
        "case_id": record.get("case_id"),
        "primary_category": record.get("primary_category", "CLEAN" if not gt_has else next(iter(gt_cats))),
        "gt_has": gt_has,
        "pred_has": pred_has,
        "tp": int(gt_has and pred_has),
        "fp": int((not gt_has) and pred_has),
        "fn": int(gt_has and (not pred_has)),
        "tn": int((not gt_has) and (not pred_has)),
        "cat_hit": int(cat_hit),
        "code_hit": int(code_hit),
        "gt_cats": list(gt_cats),
        "pred_cats": list(pred_cats),
        "latency_s": (record.get("total_elapsed_ms", 0) or 0) / 1000.0,
        "tokens": (record.get("total_tokens", {}) or {}).get("total_tokens", 0) or 0,
        "rounds": record.get("total_rounds", 1) or 1,
        "consensus": int(bool(record.get("consensus_reached", True))),
        "memory_hits": record.get("memory_hits", 0) or 0,
    }


def _compute_prf(rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, float]:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    tn = sum(r["tn"] for r in rows)
    total = max(1, tp + fp + fn + tn)
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return acc, prec, rec, f1, fpr


def _bootstrap_f1(rows: List[Dict[str, Any]], n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float, float]:
    if not rows:
        return 0.0, 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(rows)
    f1s = []
    for _ in range(n_boot):
        idxs = rng.integers(0, n, size=n)
        sample = [rows[i] for i in idxs]
        _, _, _, f1, _ = _compute_prf(sample)
        f1s.append(f1)
    arr = np.array(f1s)
    return float(arr.mean()), float(arr.std()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def evaluate_method_records(method: str, records: List[Dict[str, Any]], n_boot: int = 1000) -> Dict[str, Any]:
    rows = [_match_case(r) for r in records]
    acc, prec, rec, f1, fpr = _compute_prf(rows)
    b_mean, b_std, b_lo, b_hi = _bootstrap_f1(rows, n_boot=n_boot)

    # Per-category recall (detection rate of true error cases in each category)
    per_cat: Dict[str, Dict[str, float]] = {}
    for cat in CATEGORIES:
        cat_rows = [r for r in rows if cat in r["gt_cats"]]
        if cat_rows:
            detect_rec = sum(r["tp"] for r in cat_rows) / len(cat_rows)
            cat_exact_rec = sum(1 for r in cat_rows if cat in r["pred_cats"]) / len(cat_rows)
            code_exact_rec = sum(r["code_hit"] for r in cat_rows) / len(cat_rows)
        else:
            detect_rec = cat_exact_rec = code_exact_rec = 0.0
        per_cat[cat] = {
            "count": len(cat_rows),
            "detection_recall": round(detect_rec * 100, 1),
            "category_recall": round(cat_exact_rec * 100, 1),
            "code_match_recall": round(code_exact_rec * 100, 1),
        }

    mean_lat = float(np.mean([r["latency_s"] for r in rows])) if rows else 0.0
    mean_tok = float(np.mean([r["tokens"] for r in rows])) if rows else 0.0
    mean_rounds = float(np.mean([r["rounds"] for r in rows])) if rows else 1.0
    consensus_rate = float(np.mean([r["consensus"] for r in rows]) * 100) if rows else 100.0
    mean_mem = float(np.mean([r["memory_hits"] for r in rows])) if rows else 0.0
    code_acc = float(np.mean([r["code_hit"] for r in rows]) * 100) if rows else 0.0

    # Estimated cost (rough reference $0.10 / 1M blended tokens for Flash Lite class)
    est_cost_per_100 = round((mean_tok * 100 / 1_000_000) * 0.15, 4)

    return {
        "method": method,
        "n_cases": len(rows),
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "f1_bootstrap_mean": round(b_mean * 100, 2),
        "f1_bootstrap_std": round(b_std * 100, 2),
        "f1_ci95": [round(b_lo * 100, 2), round(b_hi * 100, 2)],
        "fpr": round(fpr * 100, 2),
        "code_accuracy": round(code_acc, 2),
        "per_category": per_cat,
        "mean_latency_s": round(mean_lat, 2),
        "mean_tokens": int(round(mean_tok)),
        "mean_rounds": round(mean_rounds, 2),
        "consensus_rate": round(consensus_rate, 1),
        "mean_memory_hits": round(mean_mem, 1),
        "est_cost_per_100_usd": est_cost_per_100,
    }


def build_and_save_reports(
    method_results: Dict[str, List[Dict[str, Any]]],
    model_key: str,
    output_dir: str = RESULTS_DIR,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    summaries = [evaluate_method_records(m, recs) for m, recs in method_results.items() if recs]

    # Save JSON summary (read by Web UI Benchmark Tab)
    payload = {
        "model_key": model_key,
        "summaries": summaries,
    }
    json_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Save CSV for Excel / plotting
    flat_rows = []
    for s in summaries:
        flat_rows.append(
            {
                "Method": s["method"],
                "N": s["n_cases"],
                "Accuracy (%)": s["accuracy"],
                "Precision (%)": s["precision"],
                "Recall (%)": s["recall"],
                "F1 (%)": s["f1"],
                "F1 Bootstrap (Mean±Std)": f"{s['f1_bootstrap_mean']} ± {s['f1_bootstrap_std']}",
                "FPR (%)": s["fpr"],
                "Code Match (%)": s["code_accuracy"],
                "Struct Recall (%)": s["per_category"]["STRUCTURAL"]["category_recall"],
                "Term Recall (%)": s["per_category"]["TERMINOLOGY"]["category_recall"],
                "ClinSem Recall (%)": s["per_category"]["CLINICAL_SEMANTIC"]["category_recall"],
                "Mean Latency (s)": s["mean_latency_s"],
                "Mean Tokens": s["mean_tokens"],
                "Mean Rounds": s["mean_rounds"],
                "Consensus (%)": s["consensus_rate"],
                "Memory Hits": s["mean_memory_hits"],
            }
        )
    df = pd.DataFrame(flat_rows)
    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Save publication-ready Markdown report
    md_lines = [
        "# Báo cáo Thực nghiệm: Rà soát Dữ liệu HL7 FHIR R4 bằng Multi-Agent & Memory-Augmented",
        "",
        f"- **Mô hình LLM nền:** `{model_key}`",
        f"- **Phương pháp thống kê:** Bootstrap 1,000 lần (95% CI), kế thừa chuẩn đánh giá MedAgentBoard (NeurIPS 2025).",
        "",
        "## Bảng 1. Hiệu quả phát hiện lỗi tổng thể (Effectiveness)",
        "",
        "| Phương pháp | N | Acc (%) | Prec (%) | Rec (%) | F1 (%) | Bootstrap F1 (Mean±Std) | FPR (%) ↓ | Code Match (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in summaries:
        md_lines.append(
            f"| `{s['method']}` | {s['n_cases']} | {s['accuracy']} | {s['precision']} | {s['recall']} | "
            f"**{s['f1']}** | {s['f1_bootstrap_mean']} ± {s['f1_bootstrap_std']} | {s['fpr']} | {s['code_accuracy']} |"
        )

    md_lines.extend(
        [
            "",
            "## Bảng 2. Phân tích Recall theo từng nhóm lỗi FHIR & Hiệu năng hệ thống (Efficiency)",
            "",
            "| Phương pháp | Structural Rec (%) | Terminology Rec (%) | Clinical Semantic Rec (%) | Latency (s/bundle) | Tokens/bundle | Rounds | Memory Hits |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for s in summaries:
        pc = s["per_category"]
        md_lines.append(
            f"| `{s['method']}` | {pc['STRUCTURAL']['category_recall']} | {pc['TERMINOLOGY']['category_recall']} | "
            f"**{pc['CLINICAL_SEMANTIC']['category_recall']}** | {s['mean_latency_s']}s | {s['mean_tokens']} | "
            f"{s['mean_rounds']} | {s['mean_memory_hits']} |"
        )

    md_path = os.path.join(output_dir, "benchmark_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nSaved benchmark reports:\n  - {json_path}\n  - {csv_path}\n  - {md_path}")

    # Automatically render 300-DPI PNG & SVG charts for reports and slides
    try:
        from fhir_audit.plot_charts import generate_all_charts

        generate_all_charts(summary_path=json_path, output_dir=os.path.join(output_dir, "charts"))
    except Exception as exc:
        print(f"Warning: could not auto-generate charts: {exc}")

    return payload
