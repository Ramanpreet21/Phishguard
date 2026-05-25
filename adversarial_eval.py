#!/usr/bin/env python3
"""
adversarial_eval.py
===================
Adversarial evaluation & false-positive stress testing.

Runs trained models against the benign hard-negatives dataset
(data/benign_hard_negatives_v2.csv) and reports:

  1. Overall false-positive rate
  2. Per-category FP breakdown (google_search, youtube, aws_signed, …)
  3. Per-difficulty-flag FP analysis (high_entropy, oauth_redirect, …)
  4. Worst offenders — benign URLs scored highest (most likely to be FPs)
  5. Confidence distribution statistics
  6. Conflict / arbitration diagnostics

All results are saved to adversarial_eval_results.json.

Usage:
  python adversarial_eval.py [--csv data/benign_hard_negatives_v2.csv]
                             [--top 15]
                             [--threshold 0.35]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import PhishingPredictor


# ── Difficulty flag columns in the CSV ────────────────────────────
DIFFICULTY_FLAGS = [
    "high_entropy", "oauth_redirect", "nested_parameters",
    "url_encoding", "excessive_length", "base64_payload",
    "signature_params", "temporary_token", "subdomain_randomness",
    "lookalike_structure",
]


def load_adversarial_dataset(csv_path: str) -> pd.DataFrame:
    """Load and validate the adversarial CSV."""
    df = pd.read_csv(csv_path)
    required = {"id", "category", "url", "label", "label_int"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    log.info(f"Loaded {len(df)} adversarial URLs from {csv_path}")
    log.info(f"  Categories: {df['category'].nunique()} → {sorted(df['category'].unique())}")
    log.info(f"  Labels:     {dict(df['label'].value_counts())}")
    return df


def evaluate(
    predictor: PhishingPredictor,
    df: pd.DataFrame,
    safe_threshold: float,
) -> List[Dict[str, Any]]:
    """Run every URL through the predictor and collect results."""
    results = []
    total = len(df)

    for idx, row in df.iterrows():
        url = row["url"]
        expected_label = row["label"]      # "benign" (all rows)
        expected_int   = int(row["label_int"])  # 0 (all rows)
        category       = row["category"]
        row_id         = row["id"]

        try:
            t0 = time.perf_counter()
            pred = predictor.predict(url, include_shap=False)
            latency = round((time.perf_counter() - t0) * 1000, 2)

            predicted_label = pred["label"]
            confidence      = pred["confidence"]
            conflict        = pred.get("conflict_detected", False)
            arb_reason      = pred.get("arbitration_reason", None)
            model_votes     = pred.get("model_votes", {})

            # False positive = benign URL flagged as phishing or suspicious
            is_fp = predicted_label in ("phishing", "suspicious")

            results.append({
                "id":                 row_id,
                "category":           category,
                "url":                url,
                "expected":           expected_label,
                "predicted":          predicted_label,
                "confidence":         confidence,
                "is_fp":              is_fp,
                "conflict_detected":  conflict,
                "arbitration_reason": arb_reason,
                "model_votes":        model_votes,
                "latency_ms":         latency,
                "notes":              row.get("notes", ""),
                "difficulty_reason":  row.get("difficulty_reason", ""),
                # Individual difficulty flags
                **{flag: int(row.get(flag, 0)) for flag in DIFFICULTY_FLAGS},
            })

        except Exception as e:
            log.warning(f"  ERROR on {row_id}: {e}")
            results.append({
                "id":         row_id,
                "category":   category,
                "url":        url,
                "expected":   expected_label,
                "predicted":  "error",
                "confidence": 0.0,
                "is_fp":      False,
                "error":      str(e),
            })

        done = idx + 1
        if done % 25 == 0 or done == total:
            fp_so_far = sum(1 for r in results if r.get("is_fp"))
            log.info(f"  [{done:3d}/{total}]  FPs so far: {fp_so_far}")

    return results


def analyse_results(
    results: List[Dict[str, Any]],
    safe_threshold: float,
    top_n: int,
) -> Dict[str, Any]:
    """Compute all analytics from raw results."""

    valid = [r for r in results if r["predicted"] != "error"]
    errors = [r for r in results if r["predicted"] == "error"]
    fps = [r for r in valid if r["is_fp"]]

    total = len(valid)
    fp_count = len(fps)
    fp_rate = fp_count / total if total > 0 else 0.0

    confidences = [r["confidence"] for r in valid]

    # ── Per-category breakdown ────────────────────────────────────
    categories = sorted(set(r["category"] for r in valid))
    per_category = {}
    for cat in categories:
        cat_rows = [r for r in valid if r["category"] == cat]
        cat_fps  = [r for r in cat_rows if r["is_fp"]]
        cat_confs = [r["confidence"] for r in cat_rows]
        per_category[cat] = {
            "total":    len(cat_rows),
            "fp_count": len(cat_fps),
            "fp_rate":  round(len(cat_fps) / len(cat_rows), 4) if cat_rows else 0,
            "conf_mean": round(statistics.mean(cat_confs), 4) if cat_confs else 0,
            "conf_max":  round(max(cat_confs), 4) if cat_confs else 0,
            "conf_min":  round(min(cat_confs), 4) if cat_confs else 0,
        }

    # ── Per-difficulty-flag breakdown ─────────────────────────────
    per_flag = {}
    for flag in DIFFICULTY_FLAGS:
        flagged = [r for r in valid if r.get(flag, 0) == 1]
        flag_fps = [r for r in flagged if r["is_fp"]]
        flag_confs = [r["confidence"] for r in flagged]
        per_flag[flag] = {
            "total":    len(flagged),
            "fp_count": len(flag_fps),
            "fp_rate":  round(len(flag_fps) / len(flagged), 4) if flagged else 0,
            "conf_mean": round(statistics.mean(flag_confs), 4) if flag_confs else 0,
            "conf_max":  round(max(flag_confs), 4) if flag_confs else 0,
        }

    # ── Worst offenders (highest confidence on benign URLs) ───────
    sorted_by_conf = sorted(valid, key=lambda r: r["confidence"], reverse=True)
    worst_offenders = [
        {
            "id":         r["id"],
            "category":   r["category"],
            "url":        r["url"][:100],
            "confidence": r["confidence"],
            "predicted":  r["predicted"],
            "conflict":   r.get("conflict_detected", False),
            "arbitration": r.get("arbitration_reason", None),
            "difficulty":  r.get("difficulty_reason", ""),
        }
        for r in sorted_by_conf[:top_n]
    ]

    # ── Conflict & arbitration stats ──────────────────────────────
    conflict_count = sum(1 for r in valid if r.get("conflict_detected"))
    arb_reasons = {}
    for r in valid:
        reason = r.get("arbitration_reason", "none") or "none"
        arb_reasons[reason] = arb_reasons.get(reason, 0) + 1

    # ── Confidence distribution ───────────────────────────────────
    conf_stats = {
        "mean":   round(statistics.mean(confidences), 4) if confidences else 0,
        "median": round(statistics.median(confidences), 4) if confidences else 0,
        "stdev":  round(statistics.stdev(confidences), 4) if len(confidences) > 1 else 0,
        "min":    round(min(confidences), 4) if confidences else 0,
        "max":    round(max(confidences), 4) if confidences else 0,
        "p90":    round(sorted(confidences)[int(0.9 * len(confidences))] if confidences else 0, 4),
        "p95":    round(sorted(confidences)[int(0.95 * len(confidences))] if confidences else 0, 4),
        "p99":    round(sorted(confidences)[min(int(0.99 * len(confidences)), len(confidences) - 1)] if confidences else 0, 4),
    }

    # ── Latency ───────────────────────────────────────────────────
    latencies = [r.get("latency_ms", 0) for r in valid if r.get("latency_ms")]
    latency_stats = {
        "mean_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "p95_ms":  round(sorted(latencies)[int(0.95 * len(latencies))], 2) if latencies else 0,
    }

    return {
        "total_urls":         total,
        "errors":             len(errors),
        "false_positives":    fp_count,
        "false_positive_rate": round(fp_rate, 4),
        "safe_threshold":     safe_threshold,
        "confidence_stats":   conf_stats,
        "per_category":       per_category,
        "per_difficulty_flag": per_flag,
        "worst_offenders":    worst_offenders,
        "conflict_count":     conflict_count,
        "arbitration_reasons": arb_reasons,
        "latency":            latency_stats,
    }


def print_report(analysis: Dict[str, Any]) -> None:
    """Pretty-print the adversarial evaluation report."""

    print("\n" + "=" * 90)
    print("  ADVERSARIAL EVALUATION — FALSE-POSITIVE STRESS TEST")
    print("=" * 90)

    total  = analysis["total_urls"]
    fp     = analysis["false_positives"]
    fp_pct = analysis["false_positive_rate"]

    print(f"\n  Total benign URLs tested:    {total}")
    print(f"  False positives:             {fp}  ({fp_pct:.1%})")
    print(f"  True negatives (correct):    {total - fp}  ({1 - fp_pct:.1%})")

    # ── Confidence distribution ───────────────────────────────────
    cs = analysis["confidence_stats"]
    print(f"\n  {'CONFIDENCE DISTRIBUTION':^86}")
    print("  " + "-" * 86)
    print(f"  Mean={cs['mean']:.4f}  Median={cs['median']:.4f}  "
          f"Stdev={cs['stdev']:.4f}  Min={cs['min']:.4f}  Max={cs['max']:.4f}")
    print(f"  P90={cs['p90']:.4f}  P95={cs['p95']:.4f}  P99={cs['p99']:.4f}")

    # ── Per-category breakdown ────────────────────────────────────
    print(f"\n  {'PER-CATEGORY BREAKDOWN':^86}")
    print("  " + "-" * 86)
    print(f"  {'Category':<28} {'Total':>5} {'FPs':>5} {'FP Rate':>8} "
          f"{'Mean Conf':>10} {'Max Conf':>10}")
    print("  " + "-" * 86)
    for cat, stats in sorted(analysis["per_category"].items()):
        marker = " ❌" if stats["fp_count"] > 0 else " ✅"
        print(f"  {cat:<28} {stats['total']:>5} {stats['fp_count']:>5} "
              f"{stats['fp_rate']:>7.1%} {stats['conf_mean']:>10.4f} "
              f"{stats['conf_max']:>10.4f}{marker}")

    # ── Per-difficulty-flag breakdown ─────────────────────────────
    print(f"\n  {'PER-DIFFICULTY-FLAG ANALYSIS':^86}")
    print("  " + "-" * 86)
    print(f"  {'Flag':<28} {'Total':>5} {'FPs':>5} {'FP Rate':>8} "
          f"{'Mean Conf':>10} {'Max Conf':>10}")
    print("  " + "-" * 86)
    for flag, stats in sorted(analysis["per_difficulty_flag"].items(),
                               key=lambda x: x[1]["fp_rate"], reverse=True):
        marker = " ⚠️" if stats["fp_rate"] > 0 else ""
        print(f"  {flag:<28} {stats['total']:>5} {stats['fp_count']:>5} "
              f"{stats['fp_rate']:>7.1%} {stats['conf_mean']:>10.4f} "
              f"{stats['conf_max']:>10.4f}{marker}")

    # ── Worst offenders ───────────────────────────────────────────
    print(f"\n  {'WORST OFFENDERS (Highest-Confidence Benign URLs)':^86}")
    print("  " + "-" * 86)
    for i, w in enumerate(analysis["worst_offenders"], 1):
        status = "❌ FP" if w["predicted"] in ("phishing", "suspicious") else "✅ OK"
        conflict = "⚡" if w.get("conflict") else " "
        print(f"  {i:>2}. [{w['confidence']:.4f}] {status} {conflict} "
              f"{w['id']:<12} {w['url'][:65]}")
        if w.get("arbitration"):
            print(f"      └─ arbitration: {w['arbitration']}")

    # ── Conflict & arbitration stats ──────────────────────────────
    print(f"\n  {'CONFLICT & ARBITRATION':^86}")
    print("  " + "-" * 86)
    print(f"  Conflicts detected:  {analysis['conflict_count']}/{analysis['total_urls']}")
    print(f"  Arbitration reasons:")
    for reason, count in sorted(analysis["arbitration_reasons"].items(),
                                 key=lambda x: x[1], reverse=True):
        print(f"    {reason:<40} {count:>4}")

    # ── Final verdict ─────────────────────────────────────────────
    print(f"\n  {'VERDICT':^86}")
    print("  " + "=" * 86)

    checks = {
        "Zero false positives":              fp == 0,
        "Mean confidence < 0.20":            cs["mean"] < 0.20,
        "Max confidence < safe threshold":   cs["max"] < analysis["safe_threshold"],
        "P95 confidence < 0.25":             cs["p95"] < 0.25,
        "No category > 10% FP rate":         all(
            s["fp_rate"] <= 0.10 for s in analysis["per_category"].values()
        ),
    }
    for check, passed in checks.items():
        print(f"  {'✅' if passed else '❌'} {check}")

    if all(checks.values()):
        print(f"\n  🎉 ALL ADVERSARIAL CHECKS PASSED — robust against hard negatives!")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"\n  ⚠️  {len(failed)} check(s) failed — review results above.")

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial evaluation & false-positive stress test"
    )
    parser.add_argument(
        "--csv", default="data/benign_hard_negatives_v2.csv",
        help="Path to adversarial CSV dataset",
    )
    parser.add_argument(
        "--top", type=int, default=15,
        help="Number of worst offenders to show",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.35,
        help="Safe-ceiling threshold (should match predictor)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  ADVERSARIAL EVALUATION — FALSE-POSITIVE STRESS TEST")
    log.info("=" * 60)

    # Load dataset
    df = load_adversarial_dataset(args.csv)

    # Load predictor
    log.info("Loading predictor…")
    predictor = PhishingPredictor()

    # Run evaluation
    log.info(f"Evaluating {len(df)} adversarial URLs…")
    t0 = time.time()
    results = evaluate(predictor, df, safe_threshold=args.threshold)
    elapsed = time.time() - t0
    log.info(f"Evaluation complete in {elapsed:.1f}s")

    # Analyse
    analysis = analyse_results(results, safe_threshold=args.threshold, top_n=args.top)

    # Print report
    print_report(analysis)

    # Save results
    output_file = Path("adversarial_eval_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "timestamp":      datetime.now().isoformat(),
            "dataset":        args.csv,
            "elapsed_s":      round(elapsed, 2),
            "analysis":       analysis,
            "raw_results":    results,
        }, f, indent=2, default=str)

    log.info(f"✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
