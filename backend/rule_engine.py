from __future__ import annotations

RULE_FIELDS = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "type", "step",
]

# lift = multiplier of the fraud rate over baseline when the rule fires, on train+val
_SOFT_WEIGHTS: dict[str, int] = {
    "night_transaction":    45,  # lift=24x
    "drain_account":        20,  # lift=2.3x
    "high_amount_transfer": 30,  # lift=2.3x
}

_ML_WEIGHT   = 0.70
_SOFT_WEIGHT = 0.30
_THRESHOLD_RED   = 85.0
_THRESHOLD_GREEN = 15.0


def check_hard_rules(txn: dict) -> dict:
    hits: list[str] = []

    if (
        txn.get("type") in ("TRANSFER", "CASH_OUT")
        and txn.get("oldbalanceDest", -1) == 0
        and txn.get("newbalanceDest", -1) == 0
    ):
        hits.append("ghost_destination")

    return {
        "hard_rule_hits": hits,
        "hard_rule_flag": len(hits) > 0,
    }


def check_clean_confirming_rules(txn: dict) -> dict:
    """Clean-direction hard rules: signals strong enough to contradict an
    AI #2 "fraud" suggestion. Deliberately NEVER read by
    compute_hybrid_score() or anything else in the scoring path — the only
    consumer is automation.py's evaluate_auto_decision() (Gate B). Kept in
    a separate function (not folded into check_hard_rules()) specifically
    so hard_rule_flag stays fraud-direction-only; a single flag can't mean
    two opposite things.

    clean_confirmed: amount is a modest fraction of the source balance
    (opposite of the drain_account soft rule) AND the destination is not a
    ghost account (logical negation of check_hard_rules()'s
    ghost_destination). Empirically verified on train+val
    (TRANSFER/CASH_OUT only): coverage 5.63%, fraud_rate 0.0008% vs. a
    0.30% baseline — a 370x reduction, symmetric to ghost_destination's
    237.8x lift in the fraud direction. errorBalanceOrig≈0 ("balance
    consistency") was tested and deliberately excluded: on this dataset it
    has lift=9.78x (i.e. it's a mild fraud indicator, not a clean one) —
    PaySim's full-drain fraud pattern is itself exactly balance-consistent,
    so including it would have mislabeled genuine fraud as clean-confirmed.
    See the README's Human-Confirmed Automation section for the full
    analysis."""
    hits: list[str] = []

    orig_bal = txn.get("oldbalanceOrg", 0)
    amount = txn.get("amount", 0)
    dest_is_ghost = txn.get("oldbalanceDest", -1) == 0 and txn.get("newbalanceDest", -1) == 0

    if orig_bal > 0 and amount <= 0.5 * orig_bal and not dest_is_ghost:
        hits.append("clean_confirmed")

    return {
        "clean_rule_hits": hits,
        "clean_rule_flag": len(hits) > 0,
    }


def check_soft_rules(txn: dict) -> dict:
    hits:  list[str] = []
    score: int       = 0

    step_hour = int(txn.get("step", 0)) % 24
    if step_hour <= 5:
        hits.append("night_transaction")
        score += _SOFT_WEIGHTS["night_transaction"]

    orig_bal = txn.get("oldbalanceOrg", 0)
    amount   = txn.get("amount", 0)
    if orig_bal > 0 and amount >= orig_bal * 0.99:
        hits.append("drain_account")
        score += _SOFT_WEIGHTS["drain_account"]

    if txn.get("type") == "TRANSFER" and amount > 200_000:
        hits.append("high_amount_transfer")
        score += _SOFT_WEIGHTS["high_amount_transfer"]

    return {
        "soft_rule_hits": hits,
        "soft_score":     min(score, 100),
    }


def compute_hybrid_score(
    ml_proba: float,
    hard_rule_flag: bool,
    soft_score: int | float,
) -> dict:
    ml_score = round(float(ml_proba) * 100, 2)

    if hard_rule_flag:
        hybrid_score = round(max(_THRESHOLD_RED, ml_score), 2)
        risk_band    = "RED"
    else:
        hybrid_score = round(
            _ML_WEIGHT * ml_score + _SOFT_WEIGHT * float(soft_score), 2
        )
        hybrid_score = min(hybrid_score, 100.0)
        if hybrid_score >= _THRESHOLD_RED:
            risk_band = "RED"
        elif hybrid_score >= _THRESHOLD_GREEN:
            risk_band = "GRAY"
        else:
            risk_band = "GREEN"

    return {
        "ml_score":     ml_score,
        "soft_score":   int(soft_score),
        "hybrid_score": hybrid_score,
        "risk_band":    risk_band,
    }


def score_transaction(txn: dict, ml_proba: float) -> dict:
    hard   = check_hard_rules(txn)
    clean  = check_clean_confirming_rules(txn)
    soft   = check_soft_rules(txn)
    # compute_hybrid_score() only ever reads hard["hard_rule_flag"] — clean
    # rides along in the merged result for downstream persistence (Gate B
    # in automation.py) but never reaches scoring. See
    # check_clean_confirming_rules().
    hybrid = compute_hybrid_score(ml_proba, hard["hard_rule_flag"], soft["soft_score"])
    return {**hard, **clean, **soft, **hybrid}
