"""Centralized configuration for PhishGuard.

All tunable parameters in a single location. Values are loaded from
environment variables with sensible defaults so the system works
out-of-the-box without a `.env` file.

Usage::

    from config import Config

    path = Config.ARTIFACTS_DIR / "rf.pkl"
    if confidence > Config.PHISHING_FLOOR:
        ...
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (no-op if file missing)
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    """Read an env-var, falling back to *default*."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


class Config:
    """Global, read-only configuration namespace."""

    # ── Paths ──────────────────────────────────────────────────────
    PROJECT_ROOT: Path = _PROJECT_ROOT
    ARTIFACTS_DIR: Path = _PROJECT_ROOT / _env("ARTIFACTS_DIR", "src/models/artifacts")
    LOGS_DIR: Path = _PROJECT_ROOT / _env("LOGS_DIR", "logs")
    DATA_DIR: Path = _PROJECT_ROOT / _env("DATA_DIR", "data")

    # ── API Server ─────────────────────────────────────────────────
    API_HOST: str = _env("API_HOST", "0.0.0.0")
    API_PORT: int = _env_int("API_PORT", 8000)
    PREDICTOR_DEVICE: str = _env("PREDICTOR_DEVICE", "cpu")
    API_WORKERS: int = _env_int("WORKERS", 1)

    # ── Prediction Thresholds ──────────────────────────────────────
    SAFE_CEILING: float = _env_float("SAFE_CEILING", 0.35)
    PHISHING_FLOOR: float = _env_float("PHISHING_FLOOR", 0.65)
    CONFLICT_THRESHOLD: float = _env_float("CONFLICT_THRESHOLD", 0.40)

    # ── Training Hyper-parameters ──────────────────────────────────
    RF_N_ESTIMATORS: int = _env_int("RF_N_ESTIMATORS", 200)
    RF_MAX_DEPTH: int = _env_int("RF_MAX_DEPTH", 25)
    XGB_N_ESTIMATORS: int = _env_int("XGB_N_ESTIMATORS", 200)
    SVM_C: float = _env_float("SVM_C", 1.0)
    RANDOM_SEED: int = _env_int("RANDOM_SEED", 42)
    DL_EPOCHS: int = _env_int("DL_EPOCHS", 10)
    DL_BATCH_SIZE: int = _env_int("DL_BATCH_SIZE", 256)
    DL_LEARNING_RATE: float = _env_float("DL_LEARNING_RATE", 3e-4)

    # ── Feature Extraction ─────────────────────────────────────────
    EXTRACTION_WORKERS: int = _env_int("EXTRACTION_WORKERS", 4)
    EXTRACTION_BATCH_SIZE: int = _env_int("EXTRACTION_BATCH_SIZE", 5000)
