from __future__ import annotations

from backend.rule_engine import (
    _THRESHOLD_GREEN,
    _THRESHOLD_RED,
    check_clean_confirming_rules,
    check_hard_rules,
    check_soft_rules,
    compute_hybrid_score,
    score_transaction,
)


def test_ghost_destination_fires_for_transfer_with_zero_balance_dest_both_sides():
    txn = {"type": "TRANSFER", "oldbalanceDest": 0, "newbalanceDest": 0}
    result = check_hard_rules(txn)
    assert result["hard_rule_hits"] == ["ghost_destination"]
    assert result["hard_rule_flag"] is True


def test_ghost_destination_fires_for_cash_out_too():
    txn = {"type": "CASH_OUT", "oldbalanceDest": 0, "newbalanceDest": 0}
    result = check_hard_rules(txn)
    assert result["hard_rule_hits"] == ["ghost_destination"]


def test_ghost_destination_does_not_fire_when_destination_has_balance():
    txn = {"type": "TRANSFER", "oldbalanceDest": 500.0, "newbalanceDest": 500.0}
    result = check_hard_rules(txn)
    assert result["hard_rule_hits"] == []
    assert result["hard_rule_flag"] is False


def test_ghost_destination_does_not_fire_for_non_risky_type():
    txn = {"type": "PAYMENT", "oldbalanceDest": 0, "newbalanceDest": 0}
    result = check_hard_rules(txn)
    assert result["hard_rule_hits"] == []


def test_ghost_destination_requires_both_before_and_after_zero():
    txn = {"type": "TRANSFER", "oldbalanceDest": 0, "newbalanceDest": 1000.0}
    result = check_hard_rules(txn)
    assert result["hard_rule_hits"] == []


def test_clean_confirmed_fires_for_modest_amount_to_real_destination():
    txn = {"oldbalanceOrg": 10_000.0, "amount": 1_000.0, "oldbalanceDest": 500.0, "newbalanceDest": 1500.0}
    result = check_clean_confirming_rules(txn)
    assert result["clean_rule_hits"] == ["clean_confirmed"]
    assert result["clean_rule_flag"] is True


def test_clean_confirmed_does_not_fire_when_amount_drains_half_or_more_of_balance():
    txn = {"oldbalanceOrg": 10_000.0, "amount": 6_000.0, "oldbalanceDest": 500.0, "newbalanceDest": 6500.0}
    result = check_clean_confirming_rules(txn)
    assert result["clean_rule_hits"] == []


def test_clean_confirmed_does_not_fire_for_ghost_destination_even_with_modest_amount():
    txn = {"oldbalanceOrg": 10_000.0, "amount": 1_000.0, "oldbalanceDest": 0, "newbalanceDest": 0}
    result = check_clean_confirming_rules(txn)
    assert result["clean_rule_hits"] == []


def test_clean_confirmed_does_not_fire_with_zero_source_balance():
    txn = {"oldbalanceOrg": 0, "amount": 0, "oldbalanceDest": 500.0, "newbalanceDest": 500.0}
    result = check_clean_confirming_rules(txn)
    assert result["clean_rule_hits"] == []


def test_night_transaction_fires_at_step_hour_boundary():
    for step in [0, 5, 24, 29]:
        result = check_soft_rules({"step": step, "oldbalanceOrg": 0, "amount": 0, "type": "TRANSFER"})
        assert "night_transaction" in result["soft_rule_hits"], f"step={step} should be night"


def test_night_transaction_does_not_fire_just_past_the_boundary():
    result = check_soft_rules({"step": 6, "oldbalanceOrg": 0, "amount": 0, "type": "TRANSFER"})
    assert "night_transaction" not in result["soft_rule_hits"]


def test_drain_account_fires_at_99_percent_threshold():
    result = check_soft_rules({"step": 12, "oldbalanceOrg": 1000.0, "amount": 990.0, "type": "TRANSFER"})
    assert "drain_account" in result["soft_rule_hits"]


def test_drain_account_does_not_fire_below_threshold():
    result = check_soft_rules({"step": 12, "oldbalanceOrg": 1000.0, "amount": 500.0, "type": "TRANSFER"})
    assert "drain_account" not in result["soft_rule_hits"]


def test_high_amount_transfer_fires_above_200k_for_transfer_only():
    result = check_soft_rules({"step": 12, "oldbalanceOrg": 0, "amount": 200_001.0, "type": "TRANSFER"})
    assert "high_amount_transfer" in result["soft_rule_hits"]


def test_high_amount_transfer_does_not_fire_for_cash_out():
    result = check_soft_rules({"step": 12, "oldbalanceOrg": 0, "amount": 500_000.0, "type": "CASH_OUT"})
    assert "high_amount_transfer" not in result["soft_rule_hits"]


def test_soft_score_sums_all_triggered_weights():
    result = check_soft_rules({"step": 3, "oldbalanceOrg": 320_000.0, "amount": 320_000.0, "type": "TRANSFER"})
    assert set(result["soft_rule_hits"]) == {"night_transaction", "drain_account", "high_amount_transfer"}
    assert result["soft_score"] == 95


def test_soft_score_caps_at_100():
    result = check_soft_rules({"step": 0, "oldbalanceOrg": 100.0, "amount": 500_000.0, "type": "TRANSFER"})
    assert result["soft_score"] <= 100


def test_hybrid_formula_is_070_ml_plus_030_soft_when_no_hard_rule():
    result = compute_hybrid_score(ml_proba=0.60, hard_rule_flag=False, soft_score=50)
    assert result["ml_score"] == 60.0
    assert result["hybrid_score"] == 57.0
    assert result["risk_band"] == "GRAY"


def test_hard_rule_floors_at_red_threshold():
    result = compute_hybrid_score(ml_proba=0.10, hard_rule_flag=True, soft_score=0)
    assert result["hybrid_score"] == 85.0
    assert result["risk_band"] == "RED"

    result_higher_ml = compute_hybrid_score(ml_proba=0.90, hard_rule_flag=True, soft_score=0)
    assert result_higher_ml["hybrid_score"] == 90.0
    assert result_higher_ml["risk_band"] == "RED"

    with_soft = compute_hybrid_score(ml_proba=0.10, hard_rule_flag=True, soft_score=95)
    assert with_soft["hybrid_score"] == 97.35


def test_soft_evidence_ranks_within_the_red_band():
    only_hard = compute_hybrid_score(ml_proba=0.872, hard_rule_flag=True, soft_score=0)
    all_soft = compute_hybrid_score(ml_proba=0.872, hard_rule_flag=True, soft_score=95)
    assert only_hard["risk_band"] == all_soft["risk_band"] == "RED"
    assert all_soft["hybrid_score"] > only_hard["hybrid_score"]
    assert all_soft["hybrid_score"] == 99.55
    assert only_hard["hybrid_score"] == 87.20


def test_hard_rule_path_never_exceeds_100():
    assert compute_hybrid_score(ml_proba=1.0, hard_rule_flag=True, soft_score=95)["hybrid_score"] == 100.0


def test_red_band_boundary_is_inclusive_at_85():
    just_below = compute_hybrid_score(ml_proba=1.0, hard_rule_flag=False, soft_score=49.97)
    at_boundary = compute_hybrid_score(ml_proba=1.0, hard_rule_flag=False, soft_score=50.0)
    assert just_below["hybrid_score"] == 84.99
    assert just_below["risk_band"] == "GRAY"
    assert at_boundary["hybrid_score"] == _THRESHOLD_RED
    assert at_boundary["risk_band"] == "RED"


def test_green_band_boundary_is_inclusive_at_15():
    just_below = compute_hybrid_score(ml_proba=0.0, hard_rule_flag=False, soft_score=49.9)
    at_boundary = compute_hybrid_score(ml_proba=0.0, hard_rule_flag=False, soft_score=50.0)
    assert just_below["hybrid_score"] == 14.97
    assert just_below["risk_band"] == "GREEN"
    assert at_boundary["hybrid_score"] == _THRESHOLD_GREEN
    assert at_boundary["risk_band"] == "GRAY"


def test_hybrid_score_caps_at_100():
    result = compute_hybrid_score(ml_proba=1.0, hard_rule_flag=False, soft_score=100)
    assert result["hybrid_score"] == 100.0


def test_score_transaction_composes_all_four_checks_for_the_canonical_fraud_case():
    txn = {
        "amount": 320_000.0, "oldbalanceOrg": 320_000.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "type": "TRANSFER", "step": 3,
    }
    result = score_transaction(txn, ml_proba=0.87)
    assert result["hard_rule_hits"] == ["ghost_destination"]
    assert set(result["soft_rule_hits"]) == {"night_transaction", "drain_account", "high_amount_transfer"}
    assert result["clean_rule_hits"] == []
    assert result["risk_band"] == "RED"
    assert result["hybrid_score"] == 99.35
