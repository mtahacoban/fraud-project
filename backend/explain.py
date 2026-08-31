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
    if isinstance(raw, list):
        return np.array(raw[1][row_idx])
    arr = np.array(raw)
    if arr.ndim == 3:
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
