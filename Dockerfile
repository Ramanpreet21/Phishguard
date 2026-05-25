# ── Stage 1: Builder ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a prefix we can copy
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Group 32 — Dept. of AI & Emerging Technologies"
LABEL description="Phishing Detection API — 6-model ensemble"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/      ./src/
COPY api.py    .
COPY predict.py .

# Logs and model artifacts directories
RUN mkdir -p logs src/models/artifacts

# Non-root user for security
RUN useradd -r -u 1001 -g root appuser \
 && chown -R appuser:root /app
USER appuser

# ── Env defaults (override via docker run -e) ────────────────────
ENV PREDICTOR_DEVICE=cpu
ENV PORT=8000
ENV WORKERS=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", \
  "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS} --no-access-log"]
