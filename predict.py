"""
predict.py
==========
Inference engine with **Conflict-Aware Adaptive Fusion**.

Loads all 6 trained models once at startup, then provides a single
predict(url) call that returns:

  {
    "label":              "phishing" | "suspicious" | "safe",
    "risk_level":         "high" | "medium" | "low",
    "is_phishing":        bool,
    "confidence":         float,            # fused probability
    "conflict_detected":  bool,
    "arbitration_reason":  str | None,
    "model_votes":        { model: {label, confidence} },
    "top_features":       [ {feature, value, importance} ],
    "shap_values":        { feature: shap_value },
    "latency_ms":         float,
  }

Fusion Strategy:
  1. Conflict Detection — measures ML-vs-DL disagreement
  2. Contextual Arbitration — uses brand impersonation, domain trust,
     and shortener/IP signals to break ties when models disagree
  3. Tri-Zone Classification — safe (<0.35) / suspicious (0.35–0.65) /
     phishing (>0.65)

Usage:
  from predict import PhishingPredictor
  p = PhishingPredictor()
  result = p.predict("http://login-verify-paypal.com/update")
"""

from __future__ import annotations

import time
import math
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
from config import Config
from src.features import (
    URL_FEATURE_NAMES, url_feature_vector, url_to_ids,
    VOCAB_SIZE, MAX_URL_LEN, get_trust_signals,
)
from src.models.dl_models import (
    URLLSTMClassifier, URLCNNClassifier, URLTransformerClassifier, PhishingVisualCNN
)
import base64
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms

log = logging.getLogger(__name__)

ARTIFACTS = Config.ARTIFACTS_DIR


class PhishingPredictor:
    """Thread-safe predictor.  Instantiate once, call predict() many times."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._load_models()
        self._build_shap_explainers()
        log.info("PhishingPredictor ready.")

    # ── Loading ─────────────────────────────────────────────────

    def _load_models(self) -> None:
        log.info("Loading model artifacts…")

        # Unified scaler (trained on 22 URL-derived features)
        self.scaler = joblib.load(ARTIFACTS / "scaler.pkl")
        self.feature_cols: List[str] = joblib.load(ARTIFACTS / "feature_cols.pkl")
        self.fusion_weights: Dict[str, float] = joblib.load(ARTIFACTS / "fusion_weights.pkl")

        # Sklearn models (trained on 22 URL-derived features)
        self.rf  = joblib.load(ARTIFACTS / "rf.pkl")
        self.xgb = joblib.load(ARTIFACTS / "xgb.pkl")
        self.svm = joblib.load(ARTIFACTS / "svm.pkl")

        # PyTorch DL models (multimodal: text + tabular features)
        n_tab = len(self.feature_cols)
        self.lstm        = self._load_pt(URLLSTMClassifier(n_tab_features=n_tab),        "lstm.pt")
        self.cnn         = self._load_pt(URLCNNClassifier(n_tab_features=n_tab),         "cnn.pt")
        self.transformer = self._load_pt(URLTransformerClassifier(n_tab_features=n_tab), "transformer.pt")

        # Map model names → callables
        self._ml_models = {"rf": self.rf, "xgb": self.xgb, "svm": self.svm}
        self._dl_models = {
            "lstm":        self.lstm,
            "cnn":         self.cnn,
            "transformer": self.transformer,
        }

        # Visual CNN Model
        self.visual_cnn = PhishingVisualCNN(pretrained=True).to(self.device)
        try:
            self.visual_cnn.load_state_dict(torch.load(ARTIFACTS / "visual_cnn.pt", map_location=self.device))
        except FileNotFoundError:
            log.warning("visual_cnn.pt not found. Using untrained/ImageNet weights for visual fallback.")
        self.visual_cnn.eval()

    def _load_pt(self, arch: torch.nn.Module, filename: str) -> torch.nn.Module:
        try:
            arch.load_state_dict(torch.load(ARTIFACTS / filename, map_location=self.device))
        except FileNotFoundError:
            log.warning(f"{filename} not found.")
        arch.to(self.device)
        arch.eval()
        return arch

    def _build_shap_explainers(self) -> None:
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

    def _get_features(self, url: str) -> np.ndarray:
        """Extract and scale URL features. Returns shape (1, 22) float32."""
        vec = url_feature_vector(url).reshape(1, -1)
        return self.scaler.transform(vec).astype(np.float32)

    def _ml_predict_proba(self, url: str) -> Dict[str, float]:
        vec_sc = self._get_features(url)
        probs = {}
        for name, clf in self._ml_models.items():
            probs[name] = float(clf.predict_proba(vec_sc)[0, 1])
        return probs

    def _dl_predict_proba(self, url: str) -> Dict[str, float]:
        ids    = torch.tensor(url_to_ids(url), dtype=torch.long).unsqueeze(0).to(self.device)
        tab    = torch.tensor(self._get_features(url), dtype=torch.float32).to(self.device)
        probs  = {}
        with torch.no_grad():
            for name, model in self._dl_models.items():
                logit = model(ids, tab)
                probs[name] = float(torch.sigmoid(logit).item())
        return probs

    def _visual_predict_proba(self, screenshot_b64: str | None) -> Dict[str, float]:
        if not screenshot_b64:
            return {}
        try:
            # Strip data URI prefix if present
            if "," in screenshot_b64:
                screenshot_b64 = screenshot_b64.split(",")[1]
            img_bytes = base64.b64decode(screenshot_b64)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logit = self.visual_cnn(tensor)
                prob = float(torch.sigmoid(logit).item())
            return {"visual_cnn": prob}
        except Exception as e:
            log.error(f"Visual prediction failed: {e}")
            return {}

    # ── Conflict-Aware Adaptive Fusion ───────────────────────────

    # Tri-zone thresholds (from centralised config)
    SAFE_CEILING      = Config.SAFE_CEILING
    PHISHING_FLOOR    = Config.PHISHING_FLOOR
    # Conflict sensitivity
    CONFLICT_THRESHOLD = Config.CONFLICT_THRESHOLD

    def _weighted_average(self, probs: Dict[str, float]) -> float:
        """Standard weighted average fusion using pre-computed F1 weights."""
        total_w = sum(self.fusion_weights.get(k, 1.0) for k in probs)
        fused   = sum(
            self.fusion_weights.get(k, 1.0) * v for k, v in probs.items()
        ) / max(total_w, 1e-9)
        return float(fused)

    def _detect_conflict(self, all_probs: Dict[str, float]) -> Dict[str, Any]:
        """
        Measure disagreement between ML (structural) and DL (text) model families.

        Returns:
          {
            "conflict":    bool,
            "score":       float,    # |ML_avg - DL_avg|
            "ml_avg":      float,
            "dl_avg":      float,
            "ml_side":     str,      # "safe" or "phishing"
            "dl_side":     str,
          }
        """
        ml_keys = [k for k in all_probs if k in self._ml_models]
        dl_keys = [k for k in all_probs if k in self._dl_models]

        ml_avg = np.mean([all_probs[k] for k in ml_keys]) if ml_keys else 0.5
        dl_avg = np.mean([all_probs[k] for k in dl_keys]) if dl_keys else 0.5
        score  = abs(float(ml_avg) - float(dl_avg))

        return {
            "conflict":  score > self.CONFLICT_THRESHOLD,
            "score":     round(score, 4),
            "ml_avg":    round(float(ml_avg), 4),
            "dl_avg":    round(float(dl_avg), 4),
            "ml_side":   "phishing" if ml_avg >= 0.5 else "safe",
            "dl_side":   "phishing" if dl_avg >= 0.5 else "safe",
        }

    def _contextual_arbitrate(
        self,
        all_probs: Dict[str, float],
        conflict_info: Dict[str, Any],
        trust_signals: Dict[str, Any],
    ) -> tuple[float, str | None]:
        """
        Dynamic Escalation Layer — smooth mathematical fusion.

        Instead of clamping scores to hardcoded values, each rule computes:
          signal_strength  ∈ [0, 1]  — how strong the ground-truth signal is
          target           ∈ [0, 1]  — where the signal points
          base             = weighted average of all models

        Final blend:
          fused = base + signal_strength × (target − base)

        This preserves natural variation from model outputs while shifting
        the distribution toward the correct direction.

        Returns:
          (fused_probability, arbitration_reason)
        """
        ml_avg = conflict_info["ml_avg"]
        dl_avg = conflict_info["dl_avg"]
        conflict_score = conflict_info["score"]
        base = self._weighted_average(all_probs)

        # ── Escalation 1: Brand Impersonation ────────────────────
        # The URL contains a known brand name on a non-official domain.
        # Escalate toward phishing, proportional to DL confidence.
        if trust_signals.get("brand_impersonation", 0) == 1:
            # Signal strength: scale by DL confidence and suspicious-word density
            sus_words = trust_signals.get("has_suspicious_words", 0)
            strength = 0.75 + 0.15 * dl_avg + 0.10 * sus_words
            strength = min(strength, 0.98)

            # Target: DL models' reading + a phishing bias from the signal
            target = 0.5 + 0.5 * dl_avg   # maps dl_avg ∈ [0,1] → target ∈ [0.5, 1.0]

            fused = base + strength * (target - base)
            return fused, "brand_impersonation_detected"

        # ── Escalation 2: URL Shortener / Raw IP ─────────────────
        # Obfuscation signal — shorteners hide the real destination.
        # Note: uses exact domain match, not substring.
        if trust_signals.get("is_shortener_domain", 0) == 1 or trust_signals.get("has_ip", 0) == 1:
            strength = 0.60 + 0.25 * dl_avg   # stronger if DL is also suspicious
            target   = 0.50 + 0.40 * dl_avg   # target ∈ [0.5, 0.9]

            fused = base + strength * (target - base)
            reason = "url_shortener_detected" if trust_signals.get("is_shortener_domain") else "raw_ip_detected"
            return fused, reason

        # ── De-escalation 3: Domain Trust Bundle ─────────────────
        # Established domain (old, valid SSL, DNS records).
        # De-escalate toward safe, proportional to trust evidence.
        age    = trust_signals.get("domain_age_days", -1)
        ssl_ok = trust_signals.get("ssl_valid", False)
        dns_ok = trust_signals.get("has_dns", False)
        mx_ok  = trust_signals.get("has_mx", False)

        # Continuous trust score from 4 boolean signals
        trust_signals_count = sum([age > 365, ssl_ok, dns_ok, mx_ok])
        trust_strength = trust_signals_count / 4.0   # ∈ [0, 1]

        # Age bonus: older domains get additional trust weight (log-scaled)
        if age > 0:
            age_factor = min(math.log1p(age / 365.0) / 3.0, 0.25)
            trust_strength = min(trust_strength + age_factor, 1.0)

        if trust_strength >= 0.60:
            # Target: shift toward ML's lower reading
            # ML models anchor at their own value, scaled down for safety
            target = ml_avg * 0.40   # e.g. ML=0.10 → target=0.04; ML=0.40 → target=0.16

            # Strength proportional to trust evidence
            strength = 0.50 + 0.45 * trust_strength   # ∈ [0.77, 0.95] for trust≥0.60

            fused = base + strength * (target - base)
            reason = "domain_trust_established" if trust_strength >= 0.75 else "domain_moderately_trusted"
            return fused, reason

        # ── Escalation 4: Young Domain + Suspicious Text ─────────
        # New domain combined with text-model alarm.
        if (0 <= age < 90) and dl_avg > 0.65:
            # Strength grows with DL confidence and domain youth
            youth_factor = 1.0 - (age / 90.0)   # ∈ (0, 1] — younger = stronger
            strength = 0.50 + 0.30 * dl_avg + 0.15 * youth_factor
            target   = 0.40 + 0.50 * dl_avg   # target ∈ [0.73, 0.90]

            fused = base + strength * (target - base)
            return fused, "young_domain_suspicious_text"

        # ── Escalation 5: Visual/CNN Phishing Override ───────────
        # Zero-text evasion: URL looks benign, but visual model strongly 
        # detects a phishing/login screenshot.
        visual_prob = all_probs.get("visual_cnn", 0.0)
        if visual_prob > 0.90 and ml_avg < 0.40:
            # Tabular models think it's safe, but CNN is screaming phishing.
            # This triggers the Fusion layer block.
            strength = 0.85
            target   = 0.95
            fused = base + strength * (target - base)
            return fused, "visual_cnn_phishing_override"

        # ── No signal: pass through ──────────────────────────────
        return base, "no_clear_arbitration_signal"

    def _adaptive_fuse(
        self,
        all_probs: Dict[str, float],
        url: str,
    ) -> Dict[str, Any]:
        """
        Three-stage adaptive fusion pipeline:
          1. Always fetch trust signals (brand impersonation, domain trust)
          2. Detect conflict between ML and DL model families
          3. Apply contextual arbitration unconditionally — critical
             ground-truth signals (brand impersonation, domain trust)
             override regardless of conflict level

        Returns a dict with fused probability and diagnostic metadata.
        """
        conflict_info = self._detect_conflict(all_probs)

        # Always fetch trust signals — they are the ground-truth tiebreakers
        # that resolve the midline overlap even when models nominally agree
        trust_signals = get_trust_signals(url)

        if conflict_info["conflict"]:
            log.info(
                f"Conflict detected (score={conflict_info['score']:.3f}): "
                f"ML={conflict_info['ml_side']}({conflict_info['ml_avg']:.3f}) "
                f"vs DL={conflict_info['dl_side']}({conflict_info['dl_avg']:.3f})"
            )

        # Always run arbitration — critical signals like brand impersonation
        # must be checked even when both model families agree in the gray zone
        fused, reason = self._contextual_arbitrate(
            all_probs, conflict_info, trust_signals
        )

        if reason and reason != "no_clear_arbitration_signal":
            log.info(f"Arbitration applied: {reason} → fused={fused:.4f}")
        else:
            # No ground-truth override triggered — use standard fusion
            fused  = self._weighted_average(all_probs)
            reason = None

        # Clamp to [0, 1]
        fused = max(0.0, min(1.0, fused))

        return {
            "fused_prob":         round(fused, 4),
            "conflict_detected":  conflict_info["conflict"],
            "conflict_score":     conflict_info["score"],
            "arbitration_reason": reason,
            "ml_avg":             conflict_info["ml_avg"],
            "dl_avg":             conflict_info["dl_avg"],
        }

    def _shap_explanation(self, url: str) -> Dict[str, float]:
        """
        Return per-feature SHAP values from the RF explainer
        (best proxy for the overall structured-feature importance).
        """
        if self._shap_rf is None:
            return {}
        vec_sc = self._get_features(url)
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
        vec = url_feature_vector(url)

        try:
            importances = self.rf.feature_importances_
        except Exception:
            importances = np.zeros(len(self.feature_cols))

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

    # ── Tri-zone classification ───────────────────────────────────

    @staticmethod
    def _classify_tri_zone(prob: float) -> tuple:
        """
        Map a fused probability to a three-tier verdict.

        Returns:
          (label, risk_level)
          label:      "safe" | "suspicious" | "phishing"
          risk_level: "low"  | "medium"     | "high"
        """
        if prob < PhishingPredictor.SAFE_CEILING:
            return "safe", "low"
        elif prob > PhishingPredictor.PHISHING_FLOOR:
            return "phishing", "high"
        else:
            return "suspicious", "medium"

    # ── Public API ───────────────────────────────────────────────

    def predict(self, url: str, include_shap: bool = True, screenshot_b64: str | None = None) -> Dict[str, Any]:
        t0 = time.perf_counter()

        ml_probs  = self._ml_predict_proba(url)
        dl_probs  = self._dl_predict_proba(url)
        vis_probs = self._visual_predict_proba(screenshot_b64)
        all_probs = {**ml_probs, **dl_probs, **vis_probs}

        # Adaptive fusion (conflict detection + arbitration)
        fusion = self._adaptive_fuse(all_probs, url)
        fused_prob = fusion["fused_prob"]

        # Tri-zone classification
        label, risk_level = self._classify_tri_zone(fused_prob)
        is_phishing = label == "phishing"

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
            "label":              label,
            "risk_level":         risk_level,
            "is_phishing":        is_phishing,
            "confidence":         fused_prob,
            "conflict_detected":  fusion["conflict_detected"],
            "arbitration_reason": fusion["arbitration_reason"],
            "model_votes":        model_votes,
            "top_features":       top_feats,
            "shap_values":        shap_vals,
            "latency_ms":         latency_ms,
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
