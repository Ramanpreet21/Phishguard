# 🛡️ Phishing Detection System

**Dept. of AI & Emerging Technologies**

Six-model ensemble (3 classical ML + 3 deep learning) served via FastAPI,
packaged in Docker, with a Chrome extension for real-time tab analysis.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Chrome Extension                      │
│  popup.html / popup.js  →  background.js (SW)           │
└────────────────────┬────────────────────────────────────┘
                     │  POST /predict
┌────────────────────▼────────────────────────────────────┐
│                   FastAPI  (api.py)                     │
│  Latency middleware · Request/Prediction/Error logs     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              PhishingPredictor  (predict.py)            │
│                                                         │
│  ┌─────────────────────┐   ┌──────────────────────────┐ │
│  │  Structured ML      │   │  Deep Learning           │ │
│  │  (ARFF features)    │   │  (URL char sequences)    │ │
│  │  ── Random Forest   │   │  ── LSTM (BiDir)         │ │
│  │  ── XGBoost         │   │  ── Character CNN        │ │
│  │  ── SVM (RBF)       │   │  ── Transformer encoder  │ │
│  └──────────┬──────────┘   └────────────┬─────────────┘ │
│             └──────────────┬────────────┘               │
│                     Weighted Fusion                     │
│                     (F1-proportional)                   │
│                            │                            │
│                     SHAP Explainability                 │
│                     Top-N Feature Report                │
└─────────────────────────────────────────────────────────┘
```

---

## Repo layout

```
phishing-detector/
├── train.py               ← train all 6 models
├── predict.py             ← inference engine (importable)
├── api.py                 ← FastAPI app
├── benchmark.py           ← latency benchmark suite
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore / .dockerignore
│
├── src/
│   ├── features.py        ← URL / WHOIS / DNS / SSL / HTML features
│   └── models/
│       ├── dl_models.py   ← LSTM · CNN · Transformer (PyTorch)
│       └── artifacts/     ← saved .pkl / .pt  (git-ignored)
│
├── data/                  ← put your CSV + ARFF here (git-ignored)
│   ├── phishing_site_urls.csv
│   └── Training_Dataset.arff
│
├── logs/                  ← JSONL request / prediction / error logs
│   ├── requests.jsonl
│   ├── predictions.jsonl
│   └── errors.jsonl
│
└── extension/             ← Chrome / Edge extension (MV3)
    ├── manifest.json
    ├── background.js
    ├── popup.html
    ├── popup.js
    └── icons/             
```

---

## Quick start

### 1 · Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 · Place datasets

```
data/phishing_site_urls.csv   (549k URLs, columns: URL, Label)
data/Training_Dataset.arff    (11k samples, 30 features + Result)
```

### 3 · Train all models

```bash
python train.py \
  --csv   data/phishing_site_urls.csv \
  --arff  data/Training_Dataset.arff  \
  --sample 50000 \
  --epochs 10    \
  --device cpu
```

Artifacts saved to `src/models/artifacts/`.

### 4 · Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 5 · Test a prediction

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json"         \
  -d '{"url":"http://login-paypal-verify.com/update?account=true"}' \
  | python -m json.tool
```

### 6 · Run benchmarks

```bash
python benchmark.py --requests 200 --concurrency 8
```

---

## Docker

### Build & run

```bash
# Build
docker build -t phishing-detector:latest .

# Run
docker run -d -p 8000:8000 \
  -v $(pwd)/src/models/artifacts:/app/src/models/artifacts:ro \
  -v $(pwd)/logs:/app/logs \
  --name phishing-api \
  phishing-detector:latest
```

### With Compose

```bash
docker compose up -d
docker compose logs -f api
```

---

## API reference

| Method | Endpoint   | Description                          |
|--------|------------|--------------------------------------|
| POST   | `/predict` | Classify a URL (full ensemble)       |
| GET    | `/health`  | Liveness check                       |
| GET    | `/metrics` | Aggregate latency stats (last 1000)  |

### `POST /predict` — request body

```json
{
  "url":          "https://example.com",
  "include_shap": true,
  "fetch_html":   false
}
```

### Response schema

```json
{
  "url":         "...",
  "label":       "phishing | safe",
  "is_phishing": true,
  "confidence":  0.87,
  "model_votes": {
    "rf":          {"label":"phishing","confidence":0.91},
    "xgb":         {"label":"phishing","confidence":0.85},
    "svm":         {"label":"phishing","confidence":0.79},
    "lstm":        {"label":"phishing","confidence":0.88},
    "cnn":         {"label":"phishing","confidence":0.86},
    "transformer": {"label":"phishing","confidence":0.92}
  },
  "top_features": [
    {"feature":"has_suspicious_words","value":1.0,"importance":0.23}
  ],
  "shap_values":  {"has_suspicious_words": 0.18, "...": "..."},
  "metadata": {
    "domain": "login-paypal-verify.com",
    "domain_age_days": 12,
    "ssl_valid":       false,
    "has_mx":          false
  },
  "latency_ms":   14.3,
  "request_id":   "a1b2c3d4"
}
```
<img width="1274" height="949" alt="response" src="https://github.com/user-attachments/assets/35467929-9977-4600-8626-0612f79dbed9" />


---

## Chrome extension

1. Open Chrome → `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** → select the `extension/` folder
4. Add icons to `extension/icons/` (icon16/48/128.png)
5. Change `API_BASE` in `background.js` to match your server

**Chrome Extention Interface**
<img width="289" height="464" alt="phishguard_extention" src="https://github.com/user-attachments/assets/95134fd2-1db8-4f9f-b71c-22bec9deed25" />


Live feed alternatives: [OpenPhish](https://openphish.com) · [PhishTank](https://phishtank.org)
