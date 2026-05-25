"""
api.py
======
FastAPI phishing-detection service.

Endpoints:
  POST /predict        → full prediction with explainability
  GET  /health         → liveness check
  GET  /metrics        → aggregate request stats (last 1000)

Logs (JSONL) written to:
  logs/requests.jsonl      every inbound request
  logs/predictions.jsonl   every successful prediction
  logs/errors.jsonl        every error

Run:
  uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import PhishingPredictor

# ── Structured logging setup ─────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
slog = structlog.get_logger()

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

_REQUEST_LOG    = LOGS_DIR / "requests.jsonl"
_PREDICTION_LOG = LOGS_DIR / "predictions.jsonl"
_ERROR_LOG      = LOGS_DIR / "errors.jsonl"


def _append_log(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── In-memory metrics ring-buffer ───────────────────────────────
_LATENCIES: Deque[float] = deque(maxlen=1000)
_ERROR_COUNT: int = 0
_REQUEST_COUNT: int = 0


# ── Startup / shutdown ───────────────────────────────────────────
predictor: Optional[PhishingPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logging.basicConfig(level=logging.INFO)
    slog.info("startup", msg="Loading PhishingPredictor…")
    predictor = PhishingPredictor(device=os.getenv("PREDICTOR_DEVICE", "cpu"))
    slog.info("startup", msg="Ready.")
    yield
    slog.info("shutdown", msg="Bye.")


# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Phishing Detection API",
    description=(
        "Six-model ensemble (RF · XGBoost · SVM · LSTM · CNN · Transformer) "
        "with SHAP explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten for production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Latency middleware ────────────────────────────────────────────
@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    global _REQUEST_COUNT
    _REQUEST_COUNT += 1
    rid    = str(uuid.uuid4())[:8]
    t0     = time.perf_counter()
    record = {
        "request_id": rid,
        "ts":         datetime.now(timezone.utc).isoformat(),
        "method":     request.method,
        "path":       request.url.path,
        "client_ip":  request.client.host if request.client else "unknown",
    }
    _append_log(_REQUEST_LOG, record)

    response: Response = await call_next(request)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    _LATENCIES.append(latency_ms)
    response.headers["X-Request-Id"]  = rid
    response.headers["X-Latency-Ms"]  = str(latency_ms)
    slog.info("request", rid=rid, path=record["path"], latency_ms=latency_ms, status=response.status_code)
    return response


# ── Schemas ──────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    url:          str  = Field(..., description="URL to classify")
    include_shap: bool = Field(True, description="Include SHAP values in response")
    fetch_html:   bool = Field(False, description="Fetch page HTML for extra metadata (slower)")

    @field_validator("url")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("url must not be empty")
        return v


class ModelVote(BaseModel):
    label:      str
    confidence: float


class TopFeature(BaseModel):
    feature:    str
    value:      float
    importance: float


class PredictResponse(BaseModel):
    url:                str
    label:              str                   # "safe" | "suspicious" | "phishing"
    risk_level:         str                   # "low"  | "medium"     | "high"
    is_phishing:        bool
    confidence:         float
    conflict_detected:  bool
    arbitration_reason: Optional[str]
    model_votes:        Dict[str, ModelVote]
    top_features:       List[TopFeature]
    shap_values:        Dict[str, float]
    metadata:           Dict[str, Any]
    latency_ms:         float
    request_id:         str


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status":   "ok",
        "models":   "loaded" if predictor else "not_loaded",
        "requests": _REQUEST_COUNT,
        "ts":       datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics", tags=["System"])
def metrics():
    lats = list(_LATENCIES)
    if not lats:
        return {"requests": _REQUEST_COUNT, "errors": _ERROR_COUNT, "latency": {}}
    return {
        "requests":        _REQUEST_COUNT,
        "errors":          _ERROR_COUNT,
        "latency": {
            "mean_ms":    round(sum(lats) / len(lats), 2),
            "p95_ms":     round(sorted(lats)[int(0.95 * len(lats))], 2),
            "p99_ms":     round(sorted(lats)[int(0.99 * len(lats))], 2),
            "worst_ms":   round(max(lats), 2),
            "best_ms":    round(min(lats), 2),
            "samples":    len(lats),
        },
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(req: PredictRequest, request: Request):
    global _ERROR_COUNT
    rid = request.headers.get("X-Request-Id", str(uuid.uuid4())[:8])

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        t0     = time.perf_counter()
        result = predictor.predict(req.url, include_shap=req.include_shap)

        # Optional metadata (WHOIS / DNS / SSL)
        meta: Dict[str, Any] = {}
        if req.fetch_html:
            from src.features import get_metadata
            try:
                meta = get_metadata(req.url, fetch_html=True)
            except Exception as e:
                meta = {"error": str(e)}
        else:
            from src.features import get_metadata
            try:
                meta = get_metadata(req.url, fetch_html=False)
            except Exception:
                meta = {}

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        pred_log = {
            "request_id":  rid,
            "ts":          datetime.now(timezone.utc).isoformat(),
            "url":         req.url,
            "label":       result["label"],
            "confidence":  result["confidence"],
            "latency_ms":  latency_ms,
            "model_votes": {k: v["confidence"] for k, v in result["model_votes"].items()},
        }
        _append_log(_PREDICTION_LOG, pred_log)
        slog.info("prediction", **{k: v for k, v in pred_log.items() if k != "model_votes"})

        return PredictResponse(
            url=req.url,
            label=result["label"],
            risk_level=result["risk_level"],
            is_phishing=result["is_phishing"],
            confidence=result["confidence"],
            conflict_detected=result["conflict_detected"],
            arbitration_reason=result.get("arbitration_reason"),
            model_votes={k: ModelVote(**v) for k, v in result["model_votes"].items()},
            top_features=[TopFeature(**f) for f in result["top_features"]],
            shap_values=result["shap_values"],
            metadata=meta,
            latency_ms=latency_ms,
            request_id=rid,
        )

    except HTTPException:
        raise
    except Exception as exc:
        _ERROR_COUNT += 1
        tb = traceback.format_exc()
        err_log = {
            "request_id": rid,
            "ts":         datetime.now(timezone.utc).isoformat(),
            "url":        req.url,
            "error":      str(exc),
            "traceback":  tb,
        }
        _append_log(_ERROR_LOG, err_log)
        slog.error("prediction_error", rid=rid, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


# ── Dev run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
