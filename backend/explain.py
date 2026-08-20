from __future__ import annotations

import numpy as np
import shap

ML_FEATURES = [
    "amount", "step_hour", "errorBalanceOrig",
    "errorBalanceDest", "is_transfer", "is_cashout",
]


def build_explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def _extract_shap_row(raw, row_idx: int = 0) -> np.ndarray:
    # With the currently pinned shap/xgboost versions, TreeExplainer.
    # shap_values() on this binary XGBClassifier returns a plain
    # (n_samples, n_features) ndarray for the positive-class margin —
    # there's no class axis to select, so the plain-ndarray branch below is
    # already correct and is the only one exercised today (verified by
    # comparing expected_value + sum(shap_values) against predict_proba on
    # real cases). The two branches below are defensive for SHAP/model
    # combinations *not* currently in use:
    if isinstance(raw, list):
        # Older list-of-per-class-arrays format: index 1 is the positive
        # (fraud) class, matching predict_proba's column order.
        return np.array(raw[1][row_idx])
    arr = np.array(raw)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) format some newer SHAP/model
        # combinations use for binary classifiers — select the positive
        # (fraud) class, the last axis.
        return arr[row_idx, :, -1]
    return arr[row_idx]


def explain_transaction(
    explainer: shap.TreeExplainer,
    feature_values: dict | list | np.ndarray,
    top_n: int = 6,
) -> list[dict]:
    if isinstance(feature_values, dict):
        x = np.array([[feature_values[f] for f in ML_FEATURES]], dtype=float)
    else:
        x = np.array(feature_values, dtype=float).reshape(1, -1)

    sv = _extract_shap_row(explainer.shap_values(x), row_idx=0)

    factors = [
        {
            "feature":    f,
            "value":      float(x[0, i]),
            "shap_value": float(sv[i]),
            "direction":  "increasing" if sv[i] > 0 else "decreasing",
        }
        for i, f in enumerate(ML_FEATURES)
    ]
    factors.sort(key=lambda d: abs(d["shap_value"]), reverse=True)
    return factors[:top_n]


def compute_shap_matrix(
    explainer: shap.TreeExplainer,
    X: np.ndarray,
) -> np.ndarray:
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        return np.array(raw[1])
    return np.array(raw)
