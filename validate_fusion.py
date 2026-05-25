#!/usr/bin/env python3
"""
validate_fusion.py
==================
Diagnostic validation for the Conflict-Aware Adaptive Fusion engine.

Runs test URLs through the predictor and reports:
  - Per-URL verdict with conflict detection and arbitration info
  - Before/after comparison with old validation results
  - Distribution spread analysis
  - False positive / false negative counts

Usage:
  python validate_fusion.py [--include-adversarial]
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import PhishingPredictor

ADVERSARIAL_CSV = Path(__file__).resolve().parent / "data" / "benign_hard_negatives_v2.csv"

# ── Test cases ────────────────────────────────────────────────────
test_cases = [
    # Safe sites (should all be labeled "safe", confidence < 35%)
    ("https://www.google.com",         "safe",     "Google Search"),
    ("https://www.github.com",         "safe",     "GitHub"),
    ("https://www.amazon.com",         "safe",     "Amazon"),
    ("https://www.linkedin.com",       "safe",     "LinkedIn"),
    ("https://www.stackoverflow.com",  "safe",     "Stack Overflow"),
    ("https://www.hdfc.co.in",         "safe",     "HDFC Bank India"),
    ("https://www.wikipedia.org",      "safe",     "Wikipedia"),
    ("https://www.reddit.com",         "safe",     "Reddit"),

    # Phishing sites (should all be labeled "phishing", confidence > 65%)
    ("https://paypal-verify-account.com",  "phishing", "PayPal typosquat"),
    ("https://amazon-login-verify.com",    "phishing", "Amazon typosquat"),
    ("https://google-account-verify.com",  "phishing", "Google typosquat"),
]

# ── Old results for comparison ────────────────────────────────────
OLD_RESULTS_FILE = Path(__file__).resolve().parent / "validation_results.json"


def load_old_results():
    """Load previous validation results for comparison."""
    if OLD_RESULTS_FILE.exists():
        with open(OLD_RESULTS_FILE) as f:
            data = json.load(f)
        return {r["url"]: r for r in data.get("results", [])}
    return {}


def run_adversarial_phase(predictor: PhishingPredictor) -> dict:
    """Run adversarial hard-negatives evaluation as a supplementary phase."""
    if not ADVERSARIAL_CSV.exists():
        log.info(f"\n  ⚠️  Adversarial dataset not found at {ADVERSARIAL_CSV}")
        return {}

    log.info(f"\n{'ADVERSARIAL HARD-NEGATIVES STRESS TEST':^110}")
    log.info("=" * 110)

    df = pd.read_csv(ADVERSARIAL_CSV)
    log.info(f"  Loaded {len(df)} benign hard-negative URLs across {df['category'].nunique()} categories")

    categories = sorted(df["category"].unique())
    adv_results = []
    total_fp = 0

    for _, row in df.iterrows():
        try:
            result = predictor.predict(row["url"], include_shap=False)
            predicted  = result["label"]
            confidence = result["confidence"]
            is_fp = predicted in ("phishing", "suspicious")
            if is_fp:
                total_fp += 1
            adv_results.append({
                "id":         row["id"],
                "category":   row["category"],
                "url":        row["url"],
                "predicted":  predicted,
                "confidence": confidence,
                "is_fp":      is_fp,
            })
        except Exception as e:
            adv_results.append({
                "id": row["id"], "category": row["category"],
                "url": row["url"], "predicted": "error",
                "confidence": 0, "is_fp": False,
            })

    # Per-category summary
    log.info(f"\n  {'Category':<28} {'Total':>5} {'FPs':>5} {'FP Rate':>8} {'Max Conf':>10}")
    log.info("  " + "-" * 70)
    for cat in categories:
        cat_rows = [r for r in adv_results if r["category"] == cat]
        cat_fps  = [r for r in cat_rows if r["is_fp"]]
        cat_confs = [r["confidence"] for r in cat_rows if r["predicted"] != "error"]
        max_conf = max(cat_confs) if cat_confs else 0
        fp_rate = len(cat_fps) / len(cat_rows) if cat_rows else 0
        marker = " ❌" if cat_fps else " ✅"
        log.info(f"  {cat:<28} {len(cat_rows):>5} {len(cat_fps):>5} "
                 f"{fp_rate:>7.1%} {max_conf:>10.4f}{marker}")

    fp_rate = total_fp / len(adv_results) if adv_results else 0
    log.info("  " + "-" * 70)
    log.info(f"  {'TOTAL':<28} {len(adv_results):>5} {total_fp:>5} {fp_rate:>7.1%}")

    if total_fp == 0:
        log.info(f"\n  🎉 Zero false positives on adversarial dataset!")
    else:
        log.info(f"\n  ⚠️  {total_fp} false positive(s) on adversarial dataset — review above.")

    return {
        "total": len(adv_results),
        "false_positives": total_fp,
        "fp_rate": round(fp_rate, 4),
        "results": adv_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Fusion validation")
    parser.add_argument("--include-adversarial", action="store_true",
                        help="Also run adversarial hard-negatives stress test")
    args = parser.parse_args()

    log.info("Loading predictor...")
    predictor = PhishingPredictor()

    log.info("\n" + "=" * 110)
    log.info("  CONFLICT-AWARE ADAPTIVE FUSION — DIAGNOSTIC VALIDATION")
    log.info("=" * 110)

    old_results = load_old_results()
    results = []
    false_positives = 0
    false_negatives = 0

    # Header
    log.info(
        f"\n{'URL':<45} {'Exp':<10} {'Got':<12} {'Conf':>6}  "
        f"{'Old':>6}  {'Conflict':<9} {'Arbitration':<30} {'Status':<10}"
    )
    log.info("-" * 110)

    for url, expected, description in test_cases:
        try:
            result = predictor.predict(url, include_shap=False)
            predicted  = result["label"]
            confidence = result["confidence"]
            conflict   = result.get("conflict_detected", False)
            arb_reason = result.get("arbitration_reason", "—")

            # Old result for comparison
            old = old_results.get(url, {})
            old_conf = old.get("confidence", None)
            old_str  = f"{old_conf:.1%}" if old_conf is not None else "  —  "

            # Determine correctness
            is_correct = predicted == expected or (
                expected == "phishing" and predicted == "suspicious"
            )
            if not is_correct:
                if expected == "safe" and predicted in ("phishing", "suspicious"):
                    false_positives += 1
                    status = "❌ FP"
                elif expected == "phishing" and predicted == "safe":
                    false_negatives += 1
                    status = "❌ FN"
                else:
                    status = "⚠️ WRONG"
            else:
                status = "✅ OK"

            # Format output
            conf_str = f"{confidence:.1%}"
            conflict_str = "YES ⚡" if conflict else "no"
            arb_str = str(arb_reason or "—")[:28]

            log.info(
                f"{url:<45} {expected:<10} {predicted:<12} {conf_str:>6}  "
                f"{old_str:>6}  {conflict_str:<9} {arb_str:<30} {status:<10}"
            )

            results.append({
                "url":                url,
                "description":       description,
                "expected":          expected,
                "predicted":         predicted,
                "confidence":        confidence,
                "old_confidence":    old_conf,
                "conflict_detected": conflict,
                "arbitration_reason": arb_reason,
                "correct":           is_correct,
                "type": (
                    "FP" if (expected == "safe" and predicted in ("phishing", "suspicious")) else
                    "FN" if (expected == "phishing" and predicted == "safe") else
                    "TP/TN"
                ),
            })

        except Exception as e:
            log.info(f"{url:<45} {expected:<10} ERROR: {str(e)}")
            results.append({
                "url": url, "description": description,
                "expected": expected, "predicted": "error",
                "confidence": 0, "correct": False, "type": "ERROR",
            })

    log.info("-" * 110)

    # ── Summary ───────────────────────────────────────────────────
    total    = len([r for r in results if r["predicted"] != "error"])
    correct  = len([r for r in results if r["correct"]])
    accuracy = correct / total * 100 if total > 0 else 0

    safe_confs    = [r["confidence"] for r in results if r["expected"] == "safe" and r["predicted"] != "error"]
    phish_confs   = [r["confidence"] for r in results if r["expected"] == "phishing" and r["predicted"] != "error"]

    log.info(f"\n{'SUMMARY':^110}")
    log.info("=" * 110)
    log.info(f"  Total tests:        {total}")
    log.info(f"  Correct:            {correct}/{total} ({accuracy:.1f}%)")
    log.info(f"  False Positives:    {false_positives}  (safe sites flagged)")
    log.info(f"  False Negatives:    {false_negatives}  (phishing missed)")

    # ── Distribution Analysis ─────────────────────────────────────
    log.info(f"\n{'DISTRIBUTION ANALYSIS':^110}")
    log.info("=" * 110)

    if safe_confs:
        log.info(f"  Safe sites:     min={min(safe_confs):.1%}  max={max(safe_confs):.1%}  "
                 f"mean={sum(safe_confs)/len(safe_confs):.1%}")
    if phish_confs:
        log.info(f"  Phishing sites: min={min(phish_confs):.1%}  max={max(phish_confs):.1%}  "
                 f"mean={sum(phish_confs)/len(phish_confs):.1%}")

    if safe_confs and phish_confs:
        gap = min(phish_confs) - max(safe_confs)
        old_gap_str = ""
        old_safe = [r.get("old_confidence", 0) for r in results if r["expected"] == "safe" and r.get("old_confidence") is not None]
        old_phish = [r.get("old_confidence", 0) for r in results if r["expected"] == "phishing" and r.get("old_confidence") is not None]
        if old_safe and old_phish:
            old_gap = min(old_phish) - max(old_safe)
            old_gap_str = f"  (was {old_gap:+.1%})"
        log.info(f"\n  Separation gap:    {gap:+.1%}{old_gap_str}")
        if gap > 0:
            log.info(f"  ✅ Clean separation — no overlap between safe and phishing distributions")
        else:
            log.info(f"  ❌ Overlap detected — safe and phishing distributions still cross")

    spread = max(r["confidence"] for r in results if r["predicted"] != "error") - \
             min(r["confidence"] for r in results if r["predicted"] != "error")
    log.info(f"  Dynamic range:     {spread:.1%}")

    # ── Before/After comparison ───────────────────────────────────
    if old_results:
        log.info(f"\n{'BEFORE / AFTER COMPARISON':^110}")
        log.info("=" * 110)
        log.info(f"  {'URL':<45} {'Old Conf':>10} {'New Conf':>10} {'Delta':>10} {'Old Label':>12} {'New Label':>12}")
        log.info("  " + "-" * 100)
        for r in results:
            if r.get("old_confidence") is not None:
                delta = r["confidence"] - r["old_confidence"]
                old_label = old_results.get(r["url"], {}).get("predicted", "?")
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
                log.info(
                    f"  {r['url']:<45} {r['old_confidence']:>9.1%} {r['confidence']:>9.1%} "
                    f"{arrow}{abs(delta):>8.1%} {old_label:>12} {r['predicted']:>12}"
                )

    # ── Verdict ───────────────────────────────────────────────────
    log.info(f"\n{'FINAL VERDICT':^110}")
    log.info("=" * 110)

    checks = {
        "Zero false positives":    false_positives == 0,
        "Zero false negatives":    false_negatives == 0,
        "Dynamic range > 50pp":    spread > 0.50,
        "Clean separation":        (safe_confs and phish_confs and min(phish_confs) > max(safe_confs)),
    }
    all_pass = all(checks.values())

    for check, passed in checks.items():
        log.info(f"  {'✅' if passed else '❌'} {check}")

    if all_pass:
        log.info(f"\n  🎉 ALL CHECKS PASSED — Adaptive fusion is working correctly!")
    else:
        log.info(f"\n  ⚠️  Some checks failed — review the results above.")

    log.info("\n" + "=" * 110)

    # ── Adversarial phase (optional) ───────────────────────────────
    adv_summary = {}
    if args.include_adversarial:
        adv_summary = run_adversarial_phase(predictor)
        if adv_summary:
            # Update overall checks
            adv_pass = adv_summary["false_positives"] == 0
            checks["Zero adversarial FPs"] = adv_pass
            all_pass = all_pass and adv_pass
            log.info(f"  {'✅' if adv_pass else '❌'} Zero adversarial FPs")
            if all_pass:
                log.info(f"\n  🎉 ALL CHECKS PASSED (including adversarial)!")

    log.info("\n" + "=" * 110)

    # ── Save results ──────────────────────────────────────────────
    output_file = Path("fusion_validation_results.json")
    save_data = {
        "timestamp":       datetime.now().isoformat(),
        "engine":          "conflict_aware_adaptive_fusion",
        "total_tests":     total,
        "correct":         correct,
        "accuracy":        accuracy,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "dynamic_range":   round(spread, 4),
        "checks":          {k: v for k, v in checks.items()},
        "all_pass":        all_pass,
        "results":         results,
    }
    if adv_summary:
        save_data["adversarial"] = adv_summary

    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2)

    log.info(f"✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
