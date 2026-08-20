from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


class XGBCalibrator:
    def __init__(self, method: str = "isotonic") -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError("method: 'isotonic' or 'sigmoid'")
        self.method = method
        self._cal = None

    def fit(self, raw_proba: np.ndarray, y_true: np.ndarray) -> "XGBCalibrator":
        if self.method == "isotonic":
            self._cal = IsotonicRegression(out_of_bounds="clip")
            self._cal.fit(raw_proba, y_true)
        else:
            self._cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            self._cal.fit(raw_proba.reshape(-1, 1), y_true)
        return self

    def predict_proba(self, raw_proba: np.ndarray) -> np.ndarray:
        if self._cal is None:
            raise RuntimeError(".fit() must be called first")
        if self.method == "isotonic":
            return np.array(self._cal.predict(raw_proba), dtype=float)
        return self._cal.predict_proba(raw_proba.reshape(-1, 1))[:, 1]


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_proba))


def brier_skill_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    ref_proba: np.ndarray | None = None,
) -> float:
    if ref_proba is None:
        ref_proba = np.full(len(y_true), float(y_true.mean()))
    bs     = brier_score_loss(y_true, y_proba)
    bs_ref = brier_score_loss(y_true, ref_proba)
    return float(1 - bs / bs_ref) if bs_ref > 0 else 0.0
