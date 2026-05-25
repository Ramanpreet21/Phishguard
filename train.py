"""
train.py
========
Trains 6 separate phishing-detection models and saves them to
src/models/artifacts/.

Dataset layout expected in data/:
  data/phishing_site_urls.csv   → columns: URL, Label  (549k rows)
  data/Training_Dataset.arff    → 30 structured features + Result

Model split:
  Structured ML (trained on ARFF 30-feature dataset):
    → rf.pkl   Random Forest
    → xgb.pkl  XGBoost
    → svm.pkl  Support Vector Machine

  Deep Learning (trained on CSV URL strings, character-level):
    → lstm.pt        Bidirectional LSTM
    → cnn.pt         Character-level CNN
    → transformer.pt Transformer encoder

Also saves:
  scaler_arff.pkl  StandardScaler for ARFF features
  scaler_csv.pkl   StandardScaler for URL features
  feature_cols.pkl List of ARFF feature column names
  fusion_weights.pkl Dict of model_name → float weight (F1-based)

Usage:
  python train.py [--csv data/phishing_site_urls.csv]
                  [--arff data/Training_Dataset.arff]
                  [--sample 50000]
                  [--epochs 10]
                  [--device cpu]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from collections import Counter
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import (
    URL_FEATURE_NAMES, extract_url_features, url_to_ids,
    VOCAB_SIZE, MAX_URL_LEN,
)
from src.models.dl_models import (
    URLLSTMClassifier, URLCNNClassifier, URLTransformerClassifier,
    URLDataset,
)

# ── Paths ────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "src" / "models" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────

def load_arff(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw, _ = arff.loadarff(path)
    df = pd.DataFrame(raw)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: int(x.decode()) if isinstance(x, bytes) else int(x))
    y = (df["Result"] == 1).astype(int).values
    feature_cols = [c for c in df.columns if c != "Result"]
    X = df[feature_cols].values.astype(np.float32)
    return X, y, feature_cols


def load_csv(path: str, sample: int = 50_000) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["label"] = (df["Label"].str.strip().str.lower() == "bad").astype(int)
    df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
    return df["URL"].tolist(), df["label"].values


def csv_to_url_features(urls: list[str]) -> np.ndarray:
    rows = [extract_url_features(u) for u in urls]
    return np.array([[r[k] for k in URL_FEATURE_NAMES] for r in rows], dtype=np.float32)


# ────────────────────────────────────────────────────────────────
# DL training loop
# ────────────────────────────────────────────────────────────────

def train_dl_model(
    model: nn.Module,
    train_ids: list[list[int]],
    train_labels: list[int],
    val_ids: list[list[int]],
    val_labels: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    name: str,
) -> nn.Module:
    train_ds  = URLDataset(train_ids, train_labels)
    val_ds    = URLDataset(val_ids,   val_labels)
    train_dl  = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl    = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_dl), epochs=epochs
    )
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item()
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, zero_division=0)
        log.info(
            f"  [{name}] epoch {epoch:2d}/{epochs}  "
            f"train_loss={total_loss/len(train_dl):.4f}  "
            f"val_loss={val_loss/len(val_dl):.4f}  "
            f"val_f1={f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def dl_predict_proba(
    model: nn.Module,
    ids: list[list[int]],
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    ds     = URLDataset(ids, [0] * len(ids))
    dl     = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs  = []
    with torch.no_grad():
        for xb, _ in dl:
            xb    = xb.to(device)
            logits = model(xb)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
    return np.array(probs)


# ────────────────────────────────────────────────────────────────
# Main training routine
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train all phishing-detection models")
    parser.add_argument("--csv",    default="data/phishing_site_urls.csv")
    parser.add_argument("--arff",   default="data/Training_Dataset.arff")
    parser.add_argument("--sample", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch",  type=int, default=256)
    parser.add_argument("--lr",     type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    log.info("=" * 60)
    log.info("  PHISHING DETECTOR — TRAINING")
    log.info("=" * 60)

    # ── 1. Load ARFF (structured features) ──────────────────────
    log.info("[1/4] Loading ARFF dataset…")
    X_arff, y_arff, feature_cols = load_arff(args.arff)
    log.info(f"  ARFF: {X_arff.shape}  phishing={y_arff.mean():.1%}")

    # ── 2. Load CSV (URL strings) ────────────────────────────────
    log.info("[2/4] Loading CSV dataset…")
    urls, y_csv = load_csv(args.csv, sample=args.sample)
    X_csv       = csv_to_url_features(urls)
    log.info(f"  CSV: {X_csv.shape}  phishing={y_csv.mean():.1%}")

    # ── 3. Train structured ML models on ARFF ───────────────────
    log.info("[3/4] Training structured ML models (RF / XGB / SVM)…")

    X_a_tr, X_a_te, y_a_tr, y_a_te = train_test_split(
        X_arff, y_arff, test_size=0.2, random_state=42, stratify=y_arff
    )
    scaler_arff = StandardScaler()
    X_a_tr_sc   = scaler_arff.fit_transform(X_a_tr)
    X_a_te_sc   = scaler_arff.transform(X_a_te)

    smote       = SMOTE(random_state=42)
    X_a_bal, y_a_bal = smote.fit_resample(X_a_tr_sc, y_a_tr)
    log.info(f"  After SMOTE: {Counter(y_a_bal)}")

    ml_results: dict[str, dict] = {}

    for name, clf in [
        ("rf",  RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=200, eval_metric="logloss", verbosity=0, random_state=42)),
        ("svm", SVC(kernel="rbf", probability=True, C=1.0, random_state=42)),
    ]:
        t0 = time.time()
        clf.fit(X_a_bal, y_a_bal)
        y_pred = clf.predict(X_a_te_sc)
        y_prob = clf.predict_proba(X_a_te_sc)[:, 1]
        f1  = f1_score(y_a_te, y_pred)
        auc = roc_auc_score(y_a_te, y_prob)
        log.info(f"  {name:4s}  F1={f1:.4f}  AUC={auc:.4f}  ({time.time()-t0:.1f}s)")
        ml_results[name] = {"f1": f1, "auc": auc}
        joblib.dump(clf, ARTIFACTS / f"{name}.pkl")

    joblib.dump(scaler_arff, ARTIFACTS / "scaler_arff.pkl")
    joblib.dump(feature_cols, ARTIFACTS / "feature_cols.pkl")

    # Also train + save a scaler for CSV URL features
    X_c_tr, X_c_te, y_c_tr, y_c_te = train_test_split(
        X_csv, y_csv, test_size=0.2, random_state=42, stratify=y_csv
    )
    scaler_csv = StandardScaler()
    scaler_csv.fit(X_c_tr)
    joblib.dump(scaler_csv, ARTIFACTS / "scaler_csv.pkl")

    # ── 4. Train DL models on URL strings ───────────────────────
    log.info("[4/4] Training DL models (LSTM / CNN / Transformer)…")

    ids_all    = [url_to_ids(u) for u in urls]
    tr_idx, te_idx = train_test_split(
        range(len(urls)), test_size=0.2, random_state=42, stratify=y_csv
    )
    tr_ids  = [ids_all[i] for i in tr_idx]
    te_ids  = [ids_all[i] for i in te_idx]
    tr_lbl  = [int(y_csv[i]) for i in tr_idx]
    te_lbl  = [int(y_csv[i]) for i in te_idx]

    dl_models = {
        "lstm":        URLLSTMClassifier(),
        "cnn":         URLCNNClassifier(),
        "transformer": URLTransformerClassifier(),
    }

    dl_results: dict[str, dict] = {}
    for name, model in dl_models.items():
        log.info(f"  Training {name}…")
        trained = train_dl_model(
            model, tr_ids, tr_lbl, te_ids, te_lbl,
            epochs=args.epochs, batch_size=args.batch,
            lr=args.lr, device=device, name=name,
        )
        probs  = dl_predict_proba(trained, te_ids, device)
        preds  = (probs > 0.5).astype(int)
        f1  = f1_score(te_lbl, preds)
        auc = roc_auc_score(te_lbl, probs)
        log.info(f"  {name:12s}  F1={f1:.4f}  AUC={auc:.4f}")
        dl_results[name] = {"f1": f1, "auc": auc}
        torch.save(trained.state_dict(), ARTIFACTS / f"{name}.pt")

    # ── Fusion weights (F1-proportional) ────────────────────────
    all_results = {**ml_results, **dl_results}
    total_f1    = sum(v["f1"] for v in all_results.values())
    weights     = {k: v["f1"] / total_f1 for k, v in all_results.items()}
    joblib.dump(weights, ARTIFACTS / "fusion_weights.pkl")

    log.info("\n" + "=" * 60)
    log.info("  TRAINING COMPLETE")
    log.info("=" * 60)
    for k, v in all_results.items():
        w = weights[k]
        log.info(f"  {k:14s}  F1={v['f1']:.4f}  AUC={v['auc']:.4f}  weight={w:.3f}")
    log.info(f"\n  Artifacts saved → {ARTIFACTS}")


if __name__ == "__main__":
    main()
