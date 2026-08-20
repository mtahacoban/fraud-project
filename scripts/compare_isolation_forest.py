"""
Isolation Forest baseline vs. the existing LogReg baseline and the
production XGBoost model, evaluated on the exact same train/val/test
split train_models.py/evaluate_model.py already used (data/*.parquet,
RANDOM_STATE=42) — no re-splitting, so the comparison is apples-to-apples.

Comparison artifact only. Does not touch the production model, scoring
path, or any file under backend/ — Isolation Forest is fit and scored
here and nowhere else.

Isolation Forest is unsupervised: fit(X_train) never sees y_train. Labels
are only used afterward, the same way they're used for every model here:
to pick an F1-max decision threshold on val and to evaluate on test.
`contamination` is set to the training set's true fraud rate (~0.30%) —
a defensible, undisputed choice for a baseline (not a tuned value).

Usage (from the project root):
    venv/Scripts/python.exe scripts/compare_isolation_forest.py
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR, MODEL_DIR = "data", "models"
ML_FEATURES = ["amount", "step_hour", "errorBalanceOrig", "errorBalanceDest", "is_transfer", "is_cashout"]
TARGET, RANDOM_STATE = "isFraud", 42


def f1max_threshold(y_true, y_score):
    """Same F1-maximizing threshold search train_models.py uses for LogReg/XGBoost."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    denom = prec[:-1] + rec[:-1]
    f1s = np.where(denom > 0, 2 * prec[:-1] * rec[:-1] / denom, 0.0)
    idx = int(np.argmax(f1s))
    return float(thr[idx]), float(f1s[idx])


def evaluate(name, y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Model": name, "threshold": round(thr, 6),
        "PR-AUC": round(average_precision_score(y_true, y_score), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_score), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "FPR": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 6),
        "FNR": round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def main() -> None:
    train_raw = pd.read_parquet(f"{DATA_DIR}/train.parquet")
    val_raw = pd.read_parquet(f"{DATA_DIR}/val.parquet")
    test_raw = pd.read_parquet(f"{DATA_DIR}/test.parquet")
    val_sc = pd.read_parquet(f"{DATA_DIR}/val_scaled.parquet")
    test_sc = pd.read_parquet(f"{DATA_DIR}/test_scaled.parquet")

    y_train, y_val, y_test = train_raw[TARGET].values, val_raw[TARGET].values, test_raw[TARGET].values
    X_train = train_raw[ML_FEATURES].values
    X_val, X_test = val_raw[ML_FEATURES].values, test_raw[ML_FEATURES].values

    # --- Isolation Forest ---
    contamination = round(float(y_train.mean()), 4)
    print(f"Isolation Forest: fitting on X_train ({X_train.shape}), contamination={contamination} (train fraud rate)")
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train)

    # score_samples(): higher = more normal. Flip sign so higher = more
    # anomalous, matching the "higher score = more likely fraud"
    # convention predict_proba(...)[:, 1] gives for the other two models.
    iso_val_score = -iso.score_samples(X_val)
    iso_test_score = -iso.score_samples(X_test)
    ISO_THR, iso_val_f1 = f1max_threshold(y_val, iso_val_score)
    print(f"Isolation Forest F1-max threshold (val): {ISO_THR:.6f}  (val F1={iso_val_f1:.4f})")
    iso_m = evaluate("Isolation Forest", y_test, iso_test_score, ISO_THR)

    # --- LogReg (existing baseline) — reproduced on the identical split ---
    logreg = joblib.load(f"{MODEL_DIR}/logreg_v1.pkl")
    X_val_sc, X_test_sc = val_sc[ML_FEATURES].values, test_sc[ML_FEATURES].values
    lr_val_proba = logreg.predict_proba(X_val_sc)[:, 1]
    lr_test_proba = logreg.predict_proba(X_test_sc)[:, 1]
    LR_THR, _ = f1max_threshold(y_val, lr_val_proba)
    lr_m = evaluate("LogReg (F1-max)", y_test, lr_test_proba, LR_THR)

    # --- XGBoost (production model) — reproduced on the identical split ---
    xgb_model = joblib.load(f"{MODEL_DIR}/xgb_v1.pkl")
    xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
    XGB_THR, _ = f1max_threshold(y_val, xgb_val_proba)
    xgb_m = evaluate("XGBoost (production)", y_test, xgb_test_proba, XGB_THR)

    print("\n" + "=" * 74)
    print("THREE-WAY BASELINE COMPARISON — TEST SET (same split, RANDOM_STATE=42)")
    print("=" * 74)
    print(f"{'Model':<22}{'PR-AUC':>9}{'ROC-AUC':>9}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    for m in [lr_m, iso_m, xgb_m]:
        print(f"{m['Model']:<22}{m['PR-AUC']:>9}{m['ROC-AUC']:>9}{m['Precision']:>11}{m['Recall']:>9}{m['F1']:>8}")

    print(f"\nIsolation Forest confusion (test, thr={iso_m['threshold']}): "
          f"TP={iso_m['TP']} FP={iso_m['FP']} TN={iso_m['TN']} FN={iso_m['FN']}")
    print(f"Isolation Forest FPR={iso_m['FPR']}  FNR={iso_m['FNR']}")


if __name__ == "__main__":
    main()
