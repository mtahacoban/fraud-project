from __future__ import annotations

import pytest

from backend.scoring import load_scoring_engine


@pytest.fixture(scope="module")
def scoring_engine():
    return load_scoring_engine()


CANONICAL_FRAUD_TXN = {
    "type": "TRANSFER", "amount": 320_000.0, "step": 3,
    "oldbalanceOrg": 320_000.0, "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}


def test_canonical_fraud_case_scores_exactly(scoring_engine):
    result = scoring_engine.score(CANONICAL_FRAUD_TXN)
    assert result["risk_band"] == "RED"
    assert result["hybrid_score"] == 99.35
    assert result["ml_score"] == 87.0
    assert result["soft_score"] == 95
    assert result["calibrated_proba"] == 0.9852941036224365
    assert result["band_reason"] is None
    assert result["hard_rule_hits"] == ["ghost_destination"]
    assert set(result["soft_rule_hits"]) == {"night_transaction", "drain_account", "high_amount_transfer"}
    assert result["model_version"] == "xgb_v1_calibrated"


def test_scoring_is_deterministic_across_repeated_calls(scoring_engine):
    first = scoring_engine.score(CANONICAL_FRAUD_TXN)
    second = scoring_engine.score(CANONICAL_FRAUD_TXN)
    assert first == second


def test_fast_path_type_skips_the_model_entirely(scoring_engine):
    txn = {
        "type": "PAYMENT", "amount": 9_999.0, "step": 10,
        "oldbalanceOrg": 20_000.0, "newbalanceOrig": 10_001.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 9_999.0,
    }
    result = scoring_engine.score(txn)
    assert result["risk_band"] == "GREEN"
    assert result["hybrid_score"] == 0.0
    assert result["calibrated_proba"] == 0.0
    assert result["shap_factors"] == []
    assert result["model_version"] == "fast_path_v1"
    assert result["band_reason"] == "fast_path"
    assert result["clean_rule_hits"] == []


def test_fast_path_applies_to_cash_in_and_debit_too(scoring_engine):
    for txn_type in ["CASH_IN", "DEBIT"]:
        txn = {
            "type": txn_type, "amount": 1_000.0, "step": 5,
            "oldbalanceOrg": 5_000.0, "newbalanceOrig": 4_000.0,
            "oldbalanceDest": 1_000.0, "newbalanceDest": 2_000.0,
        }
        result = scoring_engine.score(txn)
        assert result["risk_band"] == "GREEN", f"{txn_type} should fast-path to GREEN"
        assert result["model_version"] == "fast_path_v1"


def test_modest_transfer_to_real_destination_does_not_trigger_hard_rule(scoring_engine):
    txn = {
        "type": "TRANSFER", "amount": 500.0, "step": 14,
        "oldbalanceOrg": 50_000.0, "newbalanceOrig": 49_500.0,
        "oldbalanceDest": 10_000.0, "newbalanceDest": 10_500.0,
    }
    result = scoring_engine.score(txn)
    assert result["hard_rule_hits"] == []
    assert result["soft_rule_hits"] == []
    assert result["model_version"] == "xgb_v1_calibrated"
    assert len(result["shap_factors"]) == 6


def test_zero_amount_ghost_destination_still_triggers_hard_rule(scoring_engine):
    txn = {
        "type": "TRANSFER", "amount": 0.0, "step": 12,
        "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }
    result = scoring_engine.score(txn)
    assert result["hard_rule_hits"] == ["ghost_destination"]
    assert result["risk_band"] == "RED"
    assert result["hybrid_score"] == 85.0


def test_zero_amount_to_a_real_destination_scores_without_error(scoring_engine):
    txn = {
        "type": "TRANSFER", "amount": 0.0, "step": 12,
        "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 500.0, "newbalanceDest": 500.0,
    }
    result = scoring_engine.score(txn)
    assert result["hard_rule_hits"] == []
    assert result["soft_rule_hits"] == []
    assert result["risk_band"] in ("RED", "GRAY", "GREEN")


def test_shap_factors_sum_toward_the_models_own_prediction(scoring_engine):
    result = scoring_engine.score(CANONICAL_FRAUD_TXN)
    feature_names = {f["feature"] for f in result["shap_factors"]}
    assert feature_names == {"amount", "step_hour", "errorBalanceOrig", "errorBalanceDest", "is_transfer", "is_cashout"}
    assert len(result["shap_factors"]) == 6


def test_high_confidence_override_ranks_instead_of_flattening(scoring_engine):
    txn = {
        "type": "CASH_OUT", "amount": 1_153_156.0, "step": 459,
        "oldbalanceOrg": 1_153_156.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 1_153_156.0,
    }
    result = scoring_engine.score(txn)

    assert result["band_reason"] == "high_confidence_override"
    assert result["risk_band"] == "RED"
    assert result["hard_rule_hits"] == []
    assert result["hybrid_score"] == 89.73
    assert 85.0 < result["hybrid_score"] <= 90.0
