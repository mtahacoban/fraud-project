from __future__ import annotations

from sqlalchemy.orm import Session

from backend import db_models as m

FEATURE_LABELS: dict[str, str] = {
    "amount": "the transaction amount",
    "step_hour": "the hour of day the transaction occurred",
    "errorBalanceOrig": "an inconsistency in the sender's account balance",
    "errorBalanceDest": "an inconsistency in the recipient's account balance",
    "is_transfer": "the transaction being a transfer",
    "is_cashout": "the transaction being a cash-out",
}

RULE_LABELS: dict[str, str] = {
    "ghost_destination": "the destination account had zero balance both before and after the transaction",
    "night_transaction": "the transaction occurred during nighttime hours",
    "drain_account": "the transaction amount was nearly the entire sender balance",
    "high_amount_transfer": "the transfer amount exceeded the high-amount threshold (200,000)",
}

BAND_REASON_LABELS: dict[str, str] = {
    "fast_path": "the transaction type is outside the model's scope (PAYMENT/CASH_IN/DEBIT), so it was routed by a fast-path type gate without ever reaching the model",
    "high_confidence_override": "the calibrated probability was very high (>=0.95) even without rule confirmation, which promoted the case to RED",
}

TOP_N_SHAP = 5

_DIRECTION_VERBS: dict[str, str] = {
    "increasing": "increased",
    "decreasing": "decreased",
}


def _shap_sentence(feature_name: str, feature_value: float, direction: str, rank: int) -> str:
    label = FEATURE_LABELS.get(feature_name, feature_name)
    tag = " - most influential factor" if rank == 0 else ""
    verb = _DIRECTION_VERBS.get(direction)
    if verb is None:
        return f"Key factor: {label} (value: {feature_value:.2f}){tag}."
    return f"Key factor: {label} (value: {feature_value:.2f}) {verb} the risk{tag}."


def _rule_sentence(rule_name: str, rule_type: str) -> str:
    label = RULE_LABELS.get(rule_name, rule_name.replace("_", " "))
    kind = "Hard rule" if rule_type == "hard" else "Soft rule"
    return f"{kind} triggered: {label} ({rule_name})."


def build_findings(case: m.Case, db: Session) -> list[str]:
    txn_id = case.transaction_id

    score = (
        db.query(m.Score)
        .filter(m.Score.transaction_id == txn_id)
        .order_by(m.Score.id.desc())
        .first()
    )
    rule_hits = (
        db.query(m.RuleHit)
        .filter(m.RuleHit.transaction_id == txn_id, m.RuleHit.rule_type.in_(["hard", "soft"]))
        .all()
    )
    shap_rows = db.query(m.ShapExplanation).filter(m.ShapExplanation.transaction_id == txn_id).all()

    findings: list[str] = []

    top_shap = sorted(shap_rows, key=lambda r: abs(r.shap_value), reverse=True)[:TOP_N_SHAP]
    for rank, row in enumerate(top_shap):
        findings.append(_shap_sentence(row.feature_name, row.feature_value, row.direction, rank))

    for hit in rule_hits:
        findings.append(_rule_sentence(hit.rule_name, hit.rule_type))

    if score is not None:
        findings.append(
            f"Overall risk band is {score.risk_band} with a hybrid score of "
            f"{score.hybrid_score:.0f}/100 and a calibrated fraud probability of "
            f"{score.calibrated_proba * 100:.2f}%."
        )
        if score.band_reason:
            reason = BAND_REASON_LABELS.get(score.band_reason, score.band_reason)
            findings.append(f"Band reason: {reason}.")

    return findings


def txn_summary_text(txn: m.Transaction) -> str:
    return (
        f"{txn.type} of {txn.amount:.2f}, sender balance {txn.oldbalance_org:.2f} -> "
        f"{txn.newbalance_orig:.2f}, recipient balance {txn.oldbalance_dest:.2f} -> "
        f"{txn.newbalance_dest:.2f}, hour of day {txn.step_hour}"
    )
