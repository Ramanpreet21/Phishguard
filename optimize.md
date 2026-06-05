# PhishGuard — Codebase & ML Backend Optimization

Comprehensive restructuring of the phishing-detector project across three axes: modular architecture, code quality, and ML performance.

## User Review Required

> [!IMPORTANT]
> **RF Model Pruning — Retraining Required**: Adding `max_depth` and dropping low-importance features means the RF model must be retrained. The current `rf.pkl` is **86 MB** (unlimited tree depth, 200 estimators, 22 features). After pruning, it should drop to ~15–25 MB. This requires running `python train.py` after the changes. **Do you want me to also retrain the models, or just prepare the code changes?**

> [!WARNING]
> **Config Migration — Breaking Change for Docker**: Moving hardcoded values to a centralized `config.py` + `.env` means Docker deployments will need the `.env` file mounted or env vars set. The Dockerfile will be updated to set sensible defaults, but existing `docker-compose.yml` setups should be tested.

## Open Questions

> [!IMPORTANT]
> 1. **Feature pruning aggressiveness**: Should I drop features contributing <1% importance (conservative) or <5% (aggressive)? The aggressive option could reduce from 22 to ~15 features but requires retraining all 6 models.
> 2. **Ruff strictness**: Should I enforce `ALL` rules or a curated subset (`E`, `F`, `W`, `I`, `UP`, `ANN`, `B`, `SIM`)? The `ANN` (type annotations) rule set would flag every untyped function as an error.
> 3. **Extension content script separation**: The popup.js currently handles both rendering and API communication. Should I split this into a separate `api-client.js` module, or is the current structure acceptable given it routes through `background.js`?

---

## Proposed Changes

### 1. Centralized Configuration Module

Create a single source of truth for all configurable parameters. Currently scattered across 7+ files.

#### [NEW] [config.py](file:///home/rs/Projects/phishing-detector/config.py)

Centralized configuration loaded from environment variables with sensible defaults:

```python
# All constants currently hardcoded across api.py, predict.py, train.py, etc.
class Config:
    # Paths
    ARTIFACTS_DIR: Path
    LOGS_DIR: Path
    DATA_DIR: Path
    
    # Model thresholds
    SAFE_CEILING: float = 0.35
    PHISHING_FLOOR: float = 0.65
    CONFLICT_THRESHOLD: float = 0.40
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    PREDICTOR_DEVICE: str = "cpu"
    
    # Training
    RF_MAX_DEPTH: int = 25          # NEW: was None (unlimited)
    RF_N_ESTIMATORS: int = 200
    XGB_N_ESTIMATORS: int = 200
    RANDOM_SEED: int = 42
```

#### [NEW] [.env.example](file:///home/rs/Projects/phishing-detector/.env.example)

Template `.env` with all configurable variables documented.

#### [MODIFY] [predict.py](file:///home/rs/Projects/phishing-detector/predict.py)

- Replace hardcoded `ARTIFACTS`, threshold constants, and fusion parameters with imports from `config.py`
- Remove `SAFE_CEILING`, `PHISHING_FLOOR`, `CONFLICT_THRESHOLD` class-level constants

#### [MODIFY] [api.py](file:///home/rs/Projects/phishing-detector/api.py)

- Replace hardcoded host/port, log paths with `Config` imports
- Remove inline `LOGS_DIR` construction

#### [MODIFY] [train.py](file:///home/rs/Projects/phishing-detector/train.py)

- Replace hardcoded `ARTIFACTS` path, RF/XGB hyperparameters with `Config` imports
- Add `max_depth=Config.RF_MAX_DEPTH` to `RandomForestClassifier`

#### [MODIFY] [extract_features.py](file:///home/rs/Projects/phishing-detector/extract_features.py)

- Replace hardcoded data/output paths with `Config` imports

---

### 2. Strict Typing & Ruff Linting

#### [NEW] [pyproject.toml](file:///home/rs/Projects/phishing-detector/pyproject.toml)

Ruff configuration with curated rule sets:
- `E`/`F`/`W` — pycodestyle errors/warnings, pyflakes
- `I` — isort import ordering
- `UP` — pyupgrade (modernize syntax)
- `ANN` — flake8-annotations (enforce type hints)
- `B` — flake8-bugbear
- `SIM` — flake8-simplify
- `RUF` — Ruff-specific rules
- Target: Python 3.11

#### [MODIFY] [src/features.py](file:///home/rs/Projects/phishing-detector/src/features.py)

Add missing type hints to helper functions:
- `_entropy(s: str) -> float` ✅ already typed
- `_is_ip(domain: str) -> bool` ✅ already typed
- `extract_url_features(url: str) -> Dict[str, Any]` ✅ already typed
- `check_brand_impersonation(url: str) -> float` ✅ already typed
- All functions already have return types — minor gaps in local variables only

#### [MODIFY] [predict.py](file:///home/rs/Projects/phishing-detector/predict.py)

Add comprehensive type annotations:
- `PhishingPredictor.__init__(self, device: str = "cpu") -> None`
- `_load_models(self) -> None`
- `_ml_predict_proba(self, url: str) -> Dict[str, float]`
- `_dl_predict_proba(self, url: str) -> Dict[str, float]`
- `_weighted_average(self, probs: Dict[str, float]) -> float`
- `_detect_conflict(self, all_probs: Dict[str, float]) -> Dict[str, Any]`
- `_contextual_arbitrate(...)` — add full parameter and return types
- `_adaptive_fuse(...)` — add full parameter and return types
- `predict(self, url: str, include_shap: bool = True) -> Dict[str, Any]`

#### [MODIFY] [train.py](file:///home/rs/Projects/phishing-detector/train.py)

Add type annotations to:
- `load_features()` return type — already partially typed, complete it
- `train_dl_model()` — already typed ✅
- `dl_predict_proba()` — already typed ✅
- `main()` — add `-> None`

#### [MODIFY] [src/models/dl_models.py](file:///home/rs/Projects/phishing-detector/src/models/dl_models.py)

- All `forward()` methods already typed ✅
- Add `-> None` to `__init__` methods
- Type `URLDataset.__init__` parameters more precisely

---

### 3. ML Backend Optimization

#### 3a. Random Forest Pruning

#### [MODIFY] [train.py](file:///home/rs/Projects/phishing-detector/train.py)

Add `max_depth=25` (or configurable via `Config.RF_MAX_DEPTH`) to the RF constructor. Current state:
```python
# BEFORE (unlimited depth — trees memorize noise, 86MB model)
RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)

# AFTER (bounded depth — generalize better, ~15-25MB model)
RandomForestClassifier(
    n_estimators=Config.RF_N_ESTIMATORS,
    max_depth=Config.RF_MAX_DEPTH,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=Config.RANDOM_SEED,
)
```

#### 3b. Feature Importance Analysis

#### [NEW] [analyze_features.py](file:///home/rs/Projects/phishing-detector/analyze_features.py)

New script that:
1. Loads the trained RF and XGBoost models
2. Computes `feature_importances_` from both
3. Generates a ranked importance table
4. Identifies features below the configurable threshold (default: 1% cumulative importance)
5. Outputs a recommended pruned feature list
6. Generates a bar chart visualization saved to `logs/feature_importance.png`

This is a **diagnostic tool** — it reports which features to consider dropping but doesn't auto-prune. The user decides the cutoff.

#### 3c. Vectorized Feature Extraction

#### [MODIFY] [src/features.py](file:///home/rs/Projects/phishing-detector/src/features.py)

**Line 147** — `num_digits` via generator expression (char-by-char):
```python
# BEFORE
"num_digits": sum(c.isdigit() for c in url),

# AFTER — regex is ~3x faster for digit counting
"num_digits": len(_RE_DIGITS.findall(url)),
# where _RE_DIGITS = re.compile(r"\d") is pre-compiled at module level
```

**Lines 148** — `num_params` is already using `.count()` (optimal) ✅

**Lines 141–146** — `.count()` calls are already optimal (C-level string ops) ✅

**Line 158** — `has_suspicious_words` via `any()` loop:
```python
# BEFORE
"has_suspicious_words": int(any(w in url.lower() for w in SUSPICIOUS_WORDS)),

# AFTER — pre-compiled regex alternation (~2x faster for 24 keywords)
"has_suspicious_words": int(bool(_RE_SUSPICIOUS.search(url))),
# where _RE_SUSPICIOUS = re.compile(r"login|signin|verify|...", re.IGNORECASE)
```

**Line 154** — `has_shortener` via `any()` loop:
```python
# BEFORE
"has_shortener": int(any(s in domain for s in URL_SHORTENERS)),

# AFTER — pre-compiled regex for exact domain matching
"has_shortener": int(bool(_RE_SHORTENERS.search(domain))),
# where _RE_SHORTENERS = re.compile(r"bit\.ly|tinyurl\.com|...", re.IGNORECASE)
```

**Line 103** — `_entropy()` uses `Counter` + generator — this is already efficient for Shannon entropy. No change needed.

**Lines 296–308** — `check_brand_impersonation()` duplicates logic from `extract_url_features()` (lines 122–135). This function will be refactored to call the existing feature extractor instead of reimplementing the same loop.

#### 3d. Model Serialization

> [!NOTE]
> **Good news**: The codebase **already uses `joblib`** for serialization in both [train.py](file:///home/rs/Projects/phishing-detector/train.py) (line 261, 262, 286) and [predict.py](file:///home/rs/Projects/phishing-detector/predict.py) (lines 79–86). The `requirements.txt` also already lists `joblib==1.4.2`. No migration is needed here — this optimization is already in place.

The real serialization win comes from **pruning the RF model** (Section 3a): reducing it from 86 MB to ~15–25 MB will cut load time proportionally.

---

### 4. Extension Script Separation

#### [MODIFY] [extension/background.js](file:///home/rs/Projects/phishing-detector/extension/background.js)

- Extract `API_BASE` to a shared constants file (see below)

#### [NEW] [extension/config.js](file:///home/rs/Projects/phishing-detector/extension/config.js)

Shared constants for the extension:
```javascript
export const API_BASE = "http://localhost:8000";
export const CACHE_TTL_MS = 5 * 60 * 1000;
```

> [!NOTE]
> Since Manifest V3 service workers don't support ES modules in all contexts, this will be implemented as a simple shared object pattern importable via `importScripts()` in the background worker and `<script>` in the popup.

---

### 5. Dockerfile Update

#### [MODIFY] [Dockerfile](file:///home/rs/Projects/phishing-detector/Dockerfile)

- Add `python-dotenv` usage comment
- Ensure `.env` defaults are embedded as `ENV` directives for container deployment

#### [MODIFY] [docker-compose.yml](file:///home/rs/Projects/phishing-detector/docker-compose.yml)

- Add `env_file: .env` directive

---

## Summary of Impact

| Area | Before | After |
|------|--------|-------|
| **Config locations** | 7+ files with duplicated constants | Single `config.py` + `.env` |
| **Type coverage** | ~40% (partial in features/predict) | ~95% (all public APIs typed) |
| **Linting** | None | Ruff with 8 rule categories |
| **RF model size** | 86 MB (unlimited depth) | ~15–25 MB (max_depth=25) |
| **RF load time** | ~2–3s | ~0.5–1s (estimated) |
| **Feature extraction** | Char-by-char loops | Pre-compiled regex (3–5x faster) |
| **Model serialization** | Already joblib ✅ | No change needed |
| **Extension config** | Hardcoded in background.js | Shared `config.js` |

---

## Verification Plan

### Automated Tests
```bash
# 1. Ruff lint pass
ruff check . --config pyproject.toml

# 2. Type check (optional, informational)
# mypy --config-file pyproject.toml src/ predict.py api.py

# 3. Feature extraction regression test
python -c "
from src.features import extract_url_features
result = extract_url_features('http://login-verify-paypal.com/update?account=true')
assert len(result) == 22, f'Expected 22 features, got {len(result)}'
assert result['has_suspicious_words'] == 1
assert result['has_brand_impersonation'] == 1
print('Feature extraction: PASS')
"

# 4. Config import test
python -c "from config import Config; print(Config.SAFE_CEILING, Config.RF_MAX_DEPTH)"

# 5. API startup smoke test
timeout 15 python -c "
from predict import PhishingPredictor
p = PhishingPredictor()
r = p.predict('https://www.google.com', include_shap=False)
assert r['label'] == 'safe', f'Expected safe, got {r[\"label\"]}'
print('Predictor smoke test: PASS')
"
```

### Manual Verification
- Run `python benchmark.py` before and after to compare latency
- Run `python validate_fusion.py` to ensure no regression in predictions
- Run `python adversarial_eval.py` to ensure no new false positives
- Review `logs/feature_importance.png` from the new analysis script
