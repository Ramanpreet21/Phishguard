#!/usr/bin/env python3
"""
extract_features.py
===================
Batch feature extraction pipeline.

Reads the raw URL dataset (data/phishing_site_urls.csv) and computes
22 URL-derived features for every row, then saves the result as both
Parquet (fast I/O) and CSV (inspectable).

Output schema (per row):
  url           : str   — original URL
  label         : int   — 0 = benign, 1 = phishing
  + 22 float columns matching URL_FEATURE_NAMES

Usage:
  python extract_features.py [--csv data/phishing_site_urls.csv]
                             [--out data/extracted_features]
                             [--workers 4]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config
from src.features import URL_FEATURE_NAMES, extract_url_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _extract_batch(urls: list[str]) -> list[dict]:
    """Extract features for a batch of URLs (runs in worker process)."""
    results = []
    for url in urls:
        try:
            feats = extract_url_features(str(url))
        except Exception:
            feats = {k: 0.0 for k in URL_FEATURE_NAMES}
        results.append(feats)
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract URL features to parquet/CSV")
    parser.add_argument("--csv", default="data/phishing_site_urls.csv",
                        help="Path to raw URL dataset")
    parser.add_argument("--out", default="data/extracted_features",
                        help="Output path prefix (without extension)")
    parser.add_argument("--workers", type=int, default=Config.EXTRACTION_WORKERS,
                        help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=Config.EXTRACTION_BATCH_SIZE,
                        help="URLs per worker batch")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  FEATURE EXTRACTION PIPELINE")
    log.info("=" * 60)

    # ── Load raw CSV ──────────────────────────────────────────────
    log.info(f"Loading {args.csv}…")
    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    log.info(f"  Rows: {len(df):,}")
    log.info(f"  Columns: {list(df.columns)}")

    # Standardise label column
    df["label"] = (df["Label"].str.strip().str.lower() == "bad").astype(int)
    urls = df["URL"].tolist()

    log.info(f"  Label distribution: {dict(df['label'].value_counts())}")
    log.info(f"  Features to extract: {len(URL_FEATURE_NAMES)}")

    # ── Batch extraction (parallel) ───────────────────────────────
    log.info(f"\nExtracting features with {args.workers} workers, "
             f"batch_size={args.batch_size}…")
    t0 = time.time()

    # Split into batches
    batches = [urls[i:i + args.batch_size]
               for i in range(0, len(urls), args.batch_size)]

    all_features: list[dict] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_extract_batch, batch): i
                   for i, batch in enumerate(batches)}

        # Collect results in submission order
        results_map = {}
        for future in as_completed(futures):
            batch_idx = futures[future]
            results_map[batch_idx] = future.result()
            completed += len(results_map[batch_idx])
            if completed % 50000 < args.batch_size or completed == len(urls):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                log.info(f"  {completed:>7,}/{len(urls):,}  "
                         f"({completed / len(urls):.1%})  "
                         f"{rate:,.0f} URLs/s")

        # Reassemble in order
        for i in range(len(batches)):
            all_features.extend(results_map[i])

    elapsed = time.time() - t0
    log.info(f"\n  Extraction complete: {len(all_features):,} URLs in {elapsed:.1f}s "
             f"({len(all_features) / elapsed:,.0f} URLs/s)")

    # ── Build output DataFrame ────────────────────────────────────
    feat_df = pd.DataFrame(all_features)

    # Ensure column order matches URL_FEATURE_NAMES
    feat_df = feat_df[URL_FEATURE_NAMES]

    # Combine with url + label
    out_df = pd.DataFrame({
        "url": urls,
        "label": df["label"].values,
    })
    out_df = pd.concat([out_df, feat_df], axis=1)

    # ── Sanity checks ─────────────────────────────────────────────
    log.info(f"\n  Output shape: {out_df.shape}")
    log.info(f"  Columns: {list(out_df.columns)}")
    log.info(f"  Null count: {out_df.isnull().sum().sum()}")
    log.info(f"  Label dist: {dict(out_df['label'].value_counts())}")

    # Quick stats for numeric features
    log.info(f"\n  Feature statistics:")
    for col in URL_FEATURE_NAMES[:5]:
        log.info(f"    {col:<28} mean={out_df[col].mean():.2f}  "
                 f"std={out_df[col].std():.2f}  "
                 f"min={out_df[col].min():.0f}  max={out_df[col].max():.0f}")
    log.info(f"    … ({len(URL_FEATURE_NAMES) - 5} more features)")

    # ── Save ──────────────────────────────────────────────────────
    out_parquet = f"{args.out}.parquet"
    out_csv = f"{args.out}.csv"

    log.info(f"\n  Saving to {out_parquet}…")
    out_df.to_parquet(out_parquet, index=False, engine="pyarrow")

    log.info(f"  Saving to {out_csv}…")
    out_df.to_csv(out_csv, index=False)

    # File sizes
    parquet_size = Path(out_parquet).stat().st_size / (1024 * 1024)
    csv_size = Path(out_csv).stat().st_size / (1024 * 1024)
    log.info(f"\n  Parquet: {parquet_size:.1f} MB")
    log.info(f"  CSV:     {csv_size:.1f} MB")

    log.info(f"\n{'=' * 60}")
    log.info(f"  ✓ Feature extraction complete")
    log.info(f"  {len(out_df):,} URLs × {len(URL_FEATURE_NAMES)} features")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
