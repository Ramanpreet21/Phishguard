"""
predict.py
==========
Inference engine.  Loads all 6 trained models once at startup, then
provides a single predict(url) call that returns:

  {
    "label":        "phishing" | "safe",
    "is_phishing":  bool,
    "confidence":   float,            # fused probability
    "model_votes":  { model: {label, confidence} },
    "top_features": [ {feature, value, importance} ],
    "shap_values":  { feature: shap_value },
    "latency_ms":   float,
  }

Usage:
  from predict import PhishingPredictor
  p = PhishingPredictor()
  result = p.predict("http://login-verify-paypal.com/update")
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import torch
import shap

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.features import (
    URL_FEATURE_NAMES, url_feature_vector, url_to_ids,
    VOCAB_SIZE, MAX_URL_LEN,
)
from src.models.dl_models import (
    URLLSTMClassifier, URLCNNClassifier, URLTransformerClassifier,
)

log = logging.getLogger(__name__)

ARTIFACTS = Path(__file__).resolve().parent / "src" / "models" / "artifacts"


class PhishingPredictor:
    """Thread-safe predictor.  Instantiate once, call predict() many times."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._load_models()
        self._build_shap_explainers()
        log.info("PhishingPredictor ready.")

    # ── Loading ─────────────────────────────────────────────────

    def _load_models(self):
        log.info("Loading model artifacts…")

        # Scalers
        self.scaler_arff = joblib.load(ARTIFACTS / "scaler_arff.pkl")
        self.scaler_csv  = joblib.load(ARTIFACTS / "scaler_csv.pkl")
        self.feature_cols: List[str] = joblib.load(ARTIFACTS / "feature_cols.pkl")
        self.fusion_weights: Dict[str, float] = joblib.load(ARTIFACTS / "fusion_weights.pkl")

        # Sklearn models  (trained on ARFF 30-feature space)
        self.rf  = joblib.load(ARTIFACTS / "rf.pkl")
        self.xgb = joblib.load(ARTIFACTS / "xgb.pkl")
        self.svm = joblib.load(ARTIFACTS / "svm.pkl")

        # PyTorch DL models
        self.lstm        = self._load_pt(URLLSTMClassifier(),        "lstm.pt")
        self.cnn         = self._load_pt(URLCNNClassifier(),         "cnn.pt")
        self.transformer = self._load_pt(URLTransformerClassifier(), "transformer.pt")

        # Map model names → callables
        self._ml_models = {"rf": self.rf, "xgb": self.xgb, "svm": self.svm}
        self._dl_models = {
            "lstm":        self.lstm,
            "cnn":         self.cnn,
            "transformer": self.transformer,
        }

    def _load_pt(self, arch: torch.nn.Module, filename: str) -> torch.nn.Module:
        arch.load_state_dict(torch.load(ARTIFACTS / filename, map_location=self.device))
        arch.to(self.device)
        arch.eval()
        return arch

    def _build_shap_explainers(self):
        """Build SHAP explainers for tree-based models."""
        # Use a small background sample (all zeros is a valid baseline for URL features)
        bg = np.zeros((1, len(URL_FEATURE_NAMES)), dtype=np.float32)
        try:
            self._shap_rf  = shap.TreeExplainer(self.rf)
            self._shap_xgb = shap.TreeExplainer(self.xgb)
        except Exception as e:
            log.warning(f"SHAP explainer init failed: {e}")
            self._shap_rf  = None
            self._shap_xgb = None

    # ── Inference helpers ────────────────────────────────────────

    def _url_to_arff_space(self, url: str) -> np.ndarray:
        """
        Map URL-derived features into the ARFF 30-feature space.
        Features that can be derived from URL are filled; others default to 0.
        """
        url_feat = url_feature_vector(url)  # 21-dim
        arff_vec = np.zeros(len(self.feature_cols), dtype=np.float32)
        name_set = set(self.feature_cols)

        # Direct name matches
        for i, name in enumerate(URL_FEATURE_NAMES):
            if name in name_set:
                idx = self.feature_cols.index(name)
                arff_vec[idx] = url_feat[i]

        # ARFF-specific feature heuristics from URL signals
        mapping = {
            "having_IP_Address":     url_feat[URL_FEATURE_NAMES.index("has_ip")],
            "URL_Length":            url_feat[URL_FEATURE_NAMES.index("url_length")],
            "Shortining_Service":    url_feat[URL_FEATURE_NAMES.index("has_shortener")],
            "having_At_Symbol":      url_feat[URL_FEATURE_NAMES.index("has_at_symbol")],
            "double_slash_redirecting": url_feat[URL_FEATURE_NAMES.index("double_slash")],
            "Prefix_Suffix":         url_feat[URL_FEATURE_NAMES.index("prefix_suffix")],
            "having_Sub_Domain":     url_feat[URL_FEATURE_NAMES.index("subdomain_level")],
            "HTTPS_token":           url_feat[URL_FEATURE_NAMES.index("has_https")],
        }
        for feat_name, val in mapping.items():
            if feat_name in name_set:
                arff_vec[self.feature_cols.index(feat_name)] = val

        return arff_vec

    def _ml_predict_proba(self, url: str) -> Dict[str, float]:
        vec_arff = self._url_to_arff_space(url).reshape(1, -1)
        vec_sc   = self.scaler_arff.transform(vec_arff)
        probs = {}
        for name, clf in self._ml_models.items():
            probs[name] = float(clf.predict_proba(vec_sc)[0, 1])
        return probs

    def _dl_predict_proba(self, url: str) -> Dict[str, float]:
        ids    = torch.tensor(url_to_ids(url), dtype=torch.long).unsqueeze(0).to(self.device)
        probs  = {}
        with torch.no_grad():
            for name, model in self._dl_models.items():
                logit = model(ids)
                probs[name] = float(torch.sigmoid(logit).item())
        return probs

    def _fuse(self, all_probs: Dict[str, float]) -> float:
        """Weighted average fusion using pre-computed F1 weights."""
        total_w = sum(self.fusion_weights.get(k, 1.0) for k in all_probs)
        fused   = sum(
            self.fusion_weights.get(k, 1.0) * v for k, v in all_probs.items()
        ) / max(total_w, 1e-9)
        return float(fused)

    def _shap_explanation(self, url: str) -> Dict[str, float]:
        """
        Return per-feature SHAP values from the RF explainer
        (best proxy for the overall structured-feature importance).
        """
        if self._shap_rf is None:
            return {}
        vec_arff = self._url_to_arff_space(url).reshape(1, -1)
        vec_sc   = self.scaler_arff.transform(vec_arff)
        try:
            sv = self._shap_rf.shap_values(vec_sc)
            # sv shape: [n_classes, 1, n_features] for RF, or [1, n_features]
            if isinstance(sv, list):
                sv = sv[1]   # phishing class
            sv = np.array(sv).flatten()
            return {self.feature_cols[i]: round(float(sv[i]), 5) for i in range(len(sv))}
        except Exception as e:
            log.debug(f"SHAP computation failed: {e}")
            return {}

    def _top_url_features(self, url: str, n: int = 8) -> List[Dict[str, Any]]:
        """Return top-N URL features by RF feature importance."""
        vec      = url_feature_vector(url)
        vec_arff = self._url_to_arff_space(url)

        try:
            importances = self.rf.feature_importances_
        except Exception:
            importances = np.zeros(len(self.feature_cols))

        # Build list from URL features (always available)
        items = []
        for i, name in enumerate(URL_FEATURE_NAMES):
            imp = 0.0
            if name in self.feature_cols:
                imp = float(importances[self.feature_cols.index(name)])
            items.append({
                "feature":    name,
                "value":      float(vec[i]),
                "importance": round(imp, 5),
            })
        items.sort(key=lambda x: x["importance"], reverse=True)
        return items[:n]

    # ── Public API ───────────────────────────────────────────────

    def predict(self, url: str, include_shap: bool = True) -> Dict[str, Any]:
        t0 = time.perf_counter()

        ml_probs  = self._ml_predict_proba(url)
        dl_probs  = self._dl_predict_proba(url)
        all_probs = {**ml_probs, **dl_probs}

        fused_prob = self._fuse(all_probs)
        is_phishing = fused_prob >= 0.5

        model_votes = {
            name: {
                "label":      "phishing" if p >= 0.5 else "safe",
                "confidence": round(p, 4),
            }
            for name, p in all_probs.items()
        }

        shap_vals  = self._shap_explanation(url) if include_shap else {}
        top_feats  = self._top_url_features(url)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "label":        "phishing" if is_phishing else "safe",
            "is_phishing":  is_phishing,
            "confidence":   round(fused_prob, 4),
            "model_votes":  model_votes,
            "top_features": top_feats,
            "shap_values":  shap_vals,
            "latency_ms":   latency_ms,
        }


# ── CLI quick-test ───────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    url = sys.argv[1] if len(sys.argv) > 1 else "http://paypal-login-verify.com/update?account=true"
    predictor = PhishingPredictor()
    result    = predictor.predict(url)
    print(json.dumps(result, indent=2))
