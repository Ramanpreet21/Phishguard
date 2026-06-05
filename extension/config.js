// config.js — Shared constants for PhishGuard extension
const PHISHGUARD_CONFIG = Object.freeze({
  API_BASE: "http://localhost:8000",
  CACHE_TTL_MS: 5 * 60 * 1000,   // 5 minutes
  BADGE_MAP: {
    safe:       { text: "✓", color: "#22c55e" },
    suspicious: { text: "?", color: "#f59e0b" },
    phishing:   { text: "⚠", color: "#ef4444" },
  },
  MODEL_LABELS: {
    rf: "RF", xgb: "XGB", svm: "SVM",
    lstm: "LSTM", cnn: "CNN", transformer: "Tfmr",
  },
});
