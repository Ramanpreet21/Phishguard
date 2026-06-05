#!/usr/bin/env python3
"""Feature importance analysis for trained phishing-detection models.

Loads trained RF and XGBoost models, computes feature importances,
identifies low-contribution features, and generates a visual report.

Usage:
    python analyze_features.py [--threshold 0.01] [--output logs/feature_importance.png]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_models() -> tuple[Any, Any, list[str]]:
    """Load trained RF and XGBoost models plus feature column names."""
    artifacts = Config.ARTIFACTS_DIR
    rf = joblib.load(artifacts / "rf.pkl")
    xgb = joblib.load(artifacts / "xgb.pkl")
    feature_cols: list[str] = joblib.load(artifacts / "feature_cols.pkl")
    return rf, xgb, feature_cols


def compute_importances(
    rf: Any,
    xgb: Any,
    feature_cols: list[str],
) -> dict[str, dict[str, float]]:
    """Compute feature importances from both models."""
    rf_imp = rf.feature_importances_
    xgb_imp = xgb.feature_importances_

    # Normalize to sum=1
    rf_imp = rf_imp / rf_imp.sum()
    xgb_imp = xgb_imp / xgb_imp.sum()

    # Average importance across both models
    avg_imp = (rf_imp + xgb_imp) / 2

    results = {}
    for i, name in enumerate(feature_cols):
        results[name] = {
            "rf": round(float(rf_imp[i]), 5),
            "xgb": round(float(xgb_imp[i]), 5),
            "average": round(float(avg_imp[i]), 5),
        }
    return results


def print_report(
    importances: dict[str, dict[str, float]],
    threshold: float,
) -> tuple[list[str], list[str]]:
    """Print a formatted feature importance report.
    
    Returns:
        (keep_features, drop_features) lists.
    """
    sorted_features = sorted(
        importances.items(),
        key=lambda x: x[1]["average"],
        reverse=True,
    )

    print("\n" + "=" * 80)
    print("  FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    print(f"\n  {'Rank':<5} {'Feature':<28} {'RF':>8} {'XGBoost':>8} {'Average':>8} {'Cumul':>8}  Status")
    print("  " + "-" * 78)

    cumulative = 0.0
    keep = []
    drop = []

    for rank, (name, scores) in enumerate(sorted_features, 1):
        cumulative += scores["average"]
        is_low = scores["average"] < threshold
        status = "❌ DROP" if is_low else "✅ KEEP"
        if is_low:
            drop.append(name)
        else:
            keep.append(name)

        print(
            f"  {rank:<5} {name:<28} {scores['rf']:>7.4f} "
            f"{scores['xgb']:>8.4f} {scores['average']:>8.4f} "
            f"{cumulative:>7.1%}  {status}"
        )

    print("  " + "-" * 78)
    print(f"\n  Features to KEEP: {len(keep)} (≥{threshold:.1%} average importance)")
    print(f"  Features to DROP: {len(drop)} (<{threshold:.1%} average importance)")

    if drop:
        print(f"\n  Candidates for removal: {', '.join(drop)}")
        cum_drop = sum(importances[f]["average"] for f in drop)
        print(f"  Combined importance of dropped features: {cum_drop:.2%}")

    print("\n" + "=" * 80)
    return keep, drop


def save_plot(
    importances: dict[str, dict[str, float]],
    output_path: str,
    threshold: float,
) -> None:
    """Generate and save a horizontal bar chart of feature importances."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plot generation.")
        return

    sorted_features = sorted(
        importances.items(),
        key=lambda x: x[1]["average"],
    )
    names = [f[0] for f in sorted_features]
    rf_vals = [f[1]["rf"] for f in sorted_features]
    xgb_vals = [f[1]["xgb"] for f in sorted_features]

    y_pos = np.arange(len(names))
    height = 0.35

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.4)))
    ax.barh(y_pos - height / 2, rf_vals, height, label="Random Forest", color="#3b82f6", alpha=0.85)
    ax.barh(y_pos + height / 2, xgb_vals, height, label="XGBoost", color="#f59e0b", alpha=0.85)
    ax.axvline(x=threshold, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.1%})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Normalized Importance", fontsize=11)
    ax.set_title("PhishGuard — Feature Importance (RF vs XGBoost)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    log.info(f"  Plot saved → {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature importance analysis")
    parser.add_argument(
        "--threshold", type=float, default=0.01,
        help="Importance threshold below which features are flagged for removal (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--output", default="logs/feature_importance.png",
        help="Output path for the bar chart",
    )
    args = parser.parse_args()

    log.info("Loading trained models...")
    rf, xgb, feature_cols = load_models()
    log.info(f"  Loaded {len(feature_cols)} features: {feature_cols}")

    log.info("Computing feature importances...")
    importances = compute_importances(rf, xgb, feature_cols)

    keep, drop = print_report(importances, threshold=args.threshold)

    save_plot(importances, output_path=args.output, threshold=args.threshold)

    log.info("\nDone. Review the report above and decide which features to prune.")
    log.info("To prune, update URL_FEATURE_NAMES in src/features.py and retrain.")


if __name__ == "__main__":
    main()
