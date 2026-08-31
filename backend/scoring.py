from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np

from backend.calibrate import XGBCalibrator
from backend.config import settings
from backend.explain import ML_FEATURES, build_explainer, explain_transaction
from backend.rule_engine import RULE_FIELDS, _THRESHOLD_RED, score_transaction

HIGH_CONFIDENCE_THRESHOLD = 0.95
OVERRIDE_SPREAD = 5.0
RISKY_TYPES = frozenset({"TRANSFER", "CASH_OUT"})


def is_model_relevant(txn_type: str) -> bool:
    return txn_type in RISKY_TYPES


@dataclass
class ScoringEngine:
    xgb_model: object
    calibrator: XGBCalibrator
    explainer: object

    def score(self, txn: dict, include_shap: bool = True) -> dict:
        error_orig = txn["newbalanceOrig"] + txn["amount"] - txn["oldbalanceOrg"]
        error_dest = txn["oldbalanceDest"] + txn["amount"] - txn["newbalanceDest"]
        step_hour = int(txn["step"]) % 24
        is_transfer = int(txn["type"] == "TRANSFER")
        is_cashout = int(txn["type"] == "CASH_OUT")

        derived = {
            "step_hour": step_hour,
            "errorBalanceOrig": error_orig,
            "errorBalanceDest": error_dest,
            "is_transfer": is_transfer,
            "is_cashout": is_cashout,
        }

        if not is_model_relevant(txn["type"]):
            return {
                **derived,
                "hard_rule_hits": [],
                "hard_rule_flag": False,
                "clean_rule_hits": [],
                "soft_rule_hits": [],
                "soft_score": 0,
                "ml_score": 0.0,
                "hybrid_score": 0.0,
                "risk_band": "GREEN",
                "calibrated_proba": 0.0,
                "shap_factors": [],
                "model_version": "fast_path_v1",
                "band_reason": "fast_path",
            }

        ml_row = {
            "amount": txn["amount"],
            "step_hour": step_hour,
            "errorBalanceOrig": error_orig,
            "errorBalanceDest": error_dest,
            "is_transfer": is_transfer,
            "is_cashout": is_cashout,
        }
        x = np.array([[ml_row[f] for f in ML_FEATURES]], dtype=float)

        raw_proba = float(self.xgb_model.predict_proba(x)[:, 1][0])
        calibrated_proba = float(self.calibrator.predict_proba(np.array([raw_proba]))[0])

        rule_input = {k: txn[k] for k in RULE_FIELDS}
        result = score_transaction(rule_input, raw_proba)

        band_reason = None
        if result["risk_band"] != "RED" and calibrated_proba >= HIGH_CONFIDENCE_THRESHOLD:
            result["risk_band"] = "RED"
            result["hybrid_score"] = round(
                _THRESHOLD_RED + OVERRIDE_SPREAD * (result["hybrid_score"] / _THRESHOLD_RED), 2
            )
            band_reason = "high_confidence_override"

        shap_factors = explain_transaction(self.explainer, ml_row) if include_shap else []

        return {
            **result,
            **derived,
            "calibrated_proba": calibrated_proba,
            "shap_factors": shap_factors,
            "model_version": settings.model_version,
            "band_reason": band_reason,
        }


def load_scoring_engine() -> ScoringEngine:
    xgb_model = joblib.load(settings.xgb_model_path)
    calibrator = joblib.load(settings.calibrator_path)
    explainer = build_explainer(xgb_model)
    return ScoringEngine(xgb_model=xgb_model, calibrator=calibrator, explainer=explainer)
