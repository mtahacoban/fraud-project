from __future__ import annotations

from sqlalchemy.orm import Session

from backend import db_models as m
from backend import precedent
from backend.database import SessionLocal
from backend.precedent import latest_decision_label


def get_active_policy(db: Session) -> m.AutomationPolicyVersion | None:
    return (
        db.query(m.AutomationPolicyVersion)
        .filter(m.AutomationPolicyVersion.active.is_(True))
        .order_by(m.AutomationPolicyVersion.id.desc())
        .first()
    )


def get_pending_proposal(case_id: int, db: Session) -> m.AutoBlockLog | None:
    return (
        db.query(m.AutoBlockLog)
        .filter(m.AutoBlockLog.case_id == case_id, m.AutoBlockLog.review_status == "proposed")
        .order_by(m.AutoBlockLog.id.desc())
        .first()
    )


_DIRECTION_BY_SUGGESTION = {
    "confirm_fraud": "fraud",
    "approve_clean": "clean",
    "escalate": "escalate",
}


def evaluate_auto_decision(
    precedent_summary: dict,
    calibrated_proba: float,
    hard_rule_hits: list[str],
    policy: m.AutomationPolicyVersion,
    clean_rule_hits: list[str] | None = None,
) -> dict:
    gates: list[dict] = []
    clean_rule_hits = clean_rule_hits or []

    suggested = precedent_summary.get("suggested_decision")
    direction = _DIRECTION_BY_SUGGESTION.get(suggested)

    if suggested is None:
        gates.append({"gate": "direction_automatable", "passed": False, "actual": None, "threshold": None,
                       "detail": "no Precedent Analysis suggestion (insufficient precedent) - nothing to automate"})
    elif suggested == "escalate":
        gates.append({"gate": "direction_automatable", "passed": False, "actual": None, "threshold": None,
                       "detail": "direction=escalate - escalation is never automated, always human-routed"})
    elif suggested == "approve_clean" and not policy.auto_clean_enabled:
        gates.append({"gate": "direction_automatable", "passed": False, "actual": None, "threshold": None,
                       "detail": "direction=clean - auto-clean is disabled (policy.auto_clean_enabled=false)"})
    else:
        gates.append({"gate": "direction_automatable", "passed": True, "actual": None, "threshold": None,
                       "detail": f"direction={direction} - automatable under current policy"})

    avg_sim = precedent_summary.get("avg_similarity")
    min_sim = policy.fraud_similarity_threshold or 1.0
    sim_ok = avg_sim is not None and avg_sim >= min_sim
    gates.append({
        "gate": "similarity", "passed": sim_ok, "actual": avg_sim, "threshold": min_sim,
        "detail": (
            f"avg_similarity={avg_sim:.4f} >= {min_sim} required" if sim_ok
            else f"avg_similarity={avg_sim if avg_sim is not None else 'None'} < {min_sim} required"
        ),
    })

    count = precedent_summary.get("precedent_count") or 0
    min_count = policy.min_precedent_count or 0
    count_ok = count >= min_count
    gates.append({
        "gate": "precedent_count", "passed": count_ok, "actual": count, "threshold": min_count,
        "detail": f"precedent_count={count} {'>=' if count_ok else '<'} {min_count} required",
    })

    consensus = precedent_summary.get("consensus_ratio")
    min_consensus = policy.min_consensus_ratio or 1.0
    consensus_ok = consensus is not None and consensus >= min_consensus
    gates.append({
        "gate": "consensus", "passed": consensus_ok, "actual": consensus, "threshold": min_consensus,
        "detail": (
            f"consensus_ratio={consensus:.4f} >= {min_consensus} required" if consensus_ok
            else f"consensus_ratio={consensus if consensus is not None else 'None'} < {min_consensus} required"
        ),
    })

    min_proba = policy.min_calibrated_proba or 1.0
    proba_ok = calibrated_proba >= min_proba
    gates.append({
        "gate": "calibrated_proba", "passed": proba_ok, "actual": calibrated_proba, "threshold": min_proba,
        "detail": f"calibrated_proba={calibrated_proba:.4f} {'>=' if proba_ok else '<'} {min_proba} required (model confidence)",
    })

    has_hard_rule = len(hard_rule_hits) > 0
    has_clean_confirming_rule = len(clean_rule_hits) > 0

    if direction == "fraud" and has_clean_confirming_rule:
        hard_rule_ok = False
        hard_rule_detail = f"clean-confirming hard rule triggered ({clean_rule_hits}) - contradicts the Precedent Analysis fraud suggestion"
    elif policy.hard_rule_required:
        hard_rule_ok = has_hard_rule
        hard_rule_detail = (
            f"hard_rule_required=true, hard_rule_hits={hard_rule_hits}" if hard_rule_ok
            else "hard_rule_required=true but no hard rule fired - required confirmation missing"
        )
    else:
        hard_rule_ok = True
        hard_rule_detail = "hard_rule_required=false, no clean-confirming rule conflict - passes by default"
    gates.append({"gate": "hard_rule_conflict", "passed": hard_rule_ok, "actual": None, "threshold": None,
                   "detail": hard_rule_detail})

    failed_gates = [f"{g['gate']}: {g['detail']}" for g in gates if not g["passed"]]

    return {
        "eligible": len(failed_gates) == 0,
        "direction": direction,
        "reason": [g["detail"] for g in gates],
        "failed_gates": failed_gates,
        "gates": gates,
    }


def log_shadow_evaluation(
    case: m.Case,
    db: Session,
    precedent_summary: dict,
    calibrated_proba: float,
    hard_rule_hits: list[str],
    policy: m.AutomationPolicyVersion,
    clean_rule_hits: list[str] | None = None,
) -> m.AutoBlockLog | None:
    if policy.mode != "shadow":
        return None

    result = evaluate_auto_decision(precedent_summary, calibrated_proba, hard_rule_hits, policy, clean_rule_hits)

    row = m.AutoBlockLog(
        transaction_id=case.transaction_id,
        case_id=case.id,
        policy_version_id=policy.id,
        review_status="shadow",
        triggered_conditions=result,
    )
    db.add(row)
    return row


def shadow_agreement_stats(db: Session) -> dict:
    rows = db.query(m.AutoBlockLog).filter(m.AutoBlockLog.review_status == "shadow").all()

    would_confirm = 0
    would_be_wrong = 0
    not_eligible = 0
    shadow_evaluation_case_ids: list[int] = []
    not_eligible_case_ids: list[int] = []
    would_have_confirmed_correctly_case_ids: list[int] = []
    would_have_been_wrong_case_ids: list[int] = []

    for row in rows:
        if row.case_id is not None:
            shadow_evaluation_case_ids.append(row.case_id)

        result = row.triggered_conditions
        if not result.get("eligible"):
            not_eligible += 1
            if row.case_id is not None:
                not_eligible_case_ids.append(row.case_id)
            continue

        case = db.get(m.Case, row.case_id) if row.case_id is not None else None
        actual = latest_decision_label(case, db) if case is not None else None
        if actual is None:
            not_eligible += 1
            if row.case_id is not None:
                not_eligible_case_ids.append(row.case_id)
            continue

        if actual == "confirm_fraud":
            would_confirm += 1
            would_have_confirmed_correctly_case_ids.append(row.case_id)
        else:
            would_be_wrong += 1
            would_have_been_wrong_case_ids.append(row.case_id)

    comparable = would_confirm + would_be_wrong

    return {
        "n_shadow_evaluations": len(rows),
        "n_eligible": comparable,
        "n_not_eligible": not_eligible,
        "would_have_confirmed_correctly": would_confirm,
        "would_have_been_wrong": would_be_wrong,
        "shadow_accuracy": round(would_confirm / comparable, 4) if comparable else None,
        "shadow_evaluation_case_ids": shadow_evaluation_case_ids,
        "not_eligible_case_ids": not_eligible_case_ids,
        "eligible_case_ids": would_have_confirmed_correctly_case_ids + would_have_been_wrong_case_ids,
        "would_have_confirmed_correctly_case_ids": would_have_confirmed_correctly_case_ids,
        "would_have_been_wrong_case_ids": would_have_been_wrong_case_ids,
    }


_POLICY_FIELDS = (
    "mode", "fraud_similarity_threshold", "clean_similarity_threshold",
    "min_precedent_count", "min_consensus_ratio", "min_calibrated_proba",
    "hard_rule_required", "auto_clean_enabled",
    "circuit_breaker_max_reversal_rate", "circuit_breaker_min_confirmations",
)


def activate_new_policy_version(
    db: Session, *, notes: str, auto_triggered: bool = False, **overrides,
) -> m.AutomationPolicyVersion:
    current = get_active_policy(db)
    base = {field: getattr(current, field) for field in _POLICY_FIELDS} if current is not None else {
        "mode": "off",
        "fraud_similarity_threshold": 0.95,
        "clean_similarity_threshold": None,
        "min_precedent_count": 10,
        "min_consensus_ratio": 0.90,
        "min_calibrated_proba": 0.95,
        "hard_rule_required": False,
        "auto_clean_enabled": False,
        "circuit_breaker_max_reversal_rate": 0.20,
        "circuit_breaker_min_confirmations": 5,
    }
    base.update(overrides)

    if current is not None:
        current.active = False

    next_num = db.query(m.AutomationPolicyVersion).count() + 1
    new_policy = m.AutomationPolicyVersion(
        version=f"v{next_num}", active=True, notes=notes, auto_triggered=auto_triggered, **base,
    )
    db.add(new_policy)
    db.commit()
    db.refresh(new_policy)
    return new_policy


def propose_auto_decision(
    case: m.Case,
    db: Session,
    precedent_summary: dict,
    calibrated_proba: float,
    hard_rule_hits: list[str],
    policy: m.AutomationPolicyVersion,
    clean_rule_hits: list[str] | None = None,
) -> m.AutoBlockLog | None:
    if policy.mode != "propose":
        return None

    existing = (
        db.query(m.AutoBlockLog)
        .filter(m.AutoBlockLog.case_id == case.id, m.AutoBlockLog.review_status == "proposed")
        .first()
    )
    if existing is not None:
        return existing

    result = evaluate_auto_decision(precedent_summary, calibrated_proba, hard_rule_hits, policy, clean_rule_hits)
    if not result["eligible"] or result["direction"] != "fraud":
        return None

    row = m.AutoBlockLog(
        transaction_id=case.transaction_id,
        case_id=case.id,
        policy_version_id=policy.id,
        review_status="proposed",
        triggered_conditions=result,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def evaluate_and_propose_for_case(case_id: int) -> None:
    db = SessionLocal()
    try:
        policy = get_active_policy(db)
        if policy is None or policy.mode != "propose":
            return

        case = db.get(m.Case, case_id)
        if case is None:
            return

        scaler = precedent.load_precedent_scaler()
        if scaler is None:
            return

        neighbors = precedent.find_precedents(case, db, scaler, k=precedent.DEFAULT_K)
        precedent_summary = precedent.summarize_precedents(neighbors)

        score = (
            db.query(m.Score)
            .filter(m.Score.transaction_id == case.transaction_id)
            .order_by(m.Score.id.desc())
            .first()
        )
        hard_rule_hits = [
            h.rule_name for h in
            db.query(m.RuleHit)
            .filter(m.RuleHit.transaction_id == case.transaction_id, m.RuleHit.rule_type == "hard")
            .all()
        ]
        clean_rule_hits = [
            h.rule_name for h in
            db.query(m.RuleHit)
            .filter(m.RuleHit.transaction_id == case.transaction_id, m.RuleHit.rule_type == "clean")
            .all()
        ]

        propose_auto_decision(case, db, precedent_summary, score.calibrated_proba, hard_rule_hits, policy, clean_rule_hits)
    finally:
        db.close()


DEFAULT_CIRCUIT_BREAKER_MIN_CONFIRMATIONS = 5


def reject_rate_stats(db: Session, policy_version_id: int | None = None) -> dict:
    query = db.query(m.AutoBlockLog).filter(m.AutoBlockLog.review_status.in_(["confirmed", "rejected"]))
    if policy_version_id is not None:
        query = query.filter(m.AutoBlockLog.policy_version_id == policy_version_id)
    rows = query.all()

    confirmed = sum(1 for r in rows if r.review_status == "confirmed")
    rejected = sum(1 for r in rows if r.review_status == "rejected")
    n = confirmed + rejected
    confirmed_case_ids = [r.case_id for r in rows if r.review_status == "confirmed" and r.case_id is not None]
    rejected_case_ids = [r.case_id for r in rows if r.review_status == "rejected" and r.case_id is not None]

    pending_query = db.query(m.AutoBlockLog).filter(m.AutoBlockLog.review_status == "proposed")
    if policy_version_id is not None:
        pending_query = pending_query.filter(m.AutoBlockLog.policy_version_id == policy_version_id)
    pending_rows = pending_query.all()
    pending = len(pending_rows)
    pending_case_ids = [r.case_id for r in pending_rows if r.case_id is not None]

    if policy_version_id is not None:
        policy = db.get(m.AutomationPolicyVersion, policy_version_id)
    else:
        policy = get_active_policy(db)
    min_n = (
        (policy.circuit_breaker_min_confirmations if policy is not None else None)
        or DEFAULT_CIRCUIT_BREAKER_MIN_CONFIRMATIONS
    )

    if n < min_n:
        return {
            "n": n, "confirmed": confirmed, "rejected": rejected, "pending": pending,
            "confirmed_case_ids": confirmed_case_ids, "rejected_case_ids": rejected_case_ids,
            "pending_case_ids": pending_case_ids,
            "reject_rate": None,
            "note": f"insufficient data (n={n} < {min_n} required)",
        }

    return {
        "n": n, "confirmed": confirmed, "rejected": rejected, "pending": pending,
        "confirmed_case_ids": confirmed_case_ids, "rejected_case_ids": rejected_case_ids,
        "pending_case_ids": pending_case_ids,
        "reject_rate": round(rejected / n, 4),
        "note": None,
    }


def check_circuit_breaker(db: Session) -> dict:
    policy = get_active_policy(db)
    if policy is None:
        return {"tripped": False, "reason": "no active policy", "stats": None}
    if policy.mode != "propose":
        return {"tripped": False, "reason": f"mode={policy.mode} - only 'propose' can trip", "stats": None}

    stats = reject_rate_stats(db, policy_version_id=policy.id)
    if stats["reject_rate"] is None:
        return {"tripped": False, "reason": stats["note"], "stats": stats}

    max_rate = policy.circuit_breaker_max_reversal_rate
    if max_rate is None or stats["reject_rate"] < max_rate:
        return {
            "tripped": False,
            "reason": f"reject_rate={stats['reject_rate']} < threshold={max_rate}",
            "stats": stats,
        }

    new_policy = activate_new_policy_version(
        db,
        notes=(
            f"Circuit breaker auto-tripped: policy {policy.version}'s reject rate "
            f"({stats['reject_rate']:.2%}, n={stats['n']}) reached or exceeded "
            f"circuit_breaker_max_reversal_rate ({max_rate:.2%}). Mode automatically "
            f"downgraded {policy.mode!r} -> 'shadow'."
        ),
        auto_triggered=True,
        mode="shadow",
    )
    return {
        "tripped": True,
        "reason": f"reject_rate={stats['reject_rate']} >= threshold={max_rate}",
        "stats": stats,
        "new_policy_version": new_policy.version,
    }


def bias_monitoring_stats(db: Session) -> dict:
    rows = db.query(m.AutoBlockLog).filter(m.AutoBlockLog.review_status.in_(["confirmed", "rejected"])).all()
    breakdown: dict[str, dict[str, int]] = {}
    for row in rows:
        txn = db.get(m.Transaction, row.transaction_id)
        key = txn.type if txn is not None else "unknown"
        breakdown.setdefault(key, {"confirmed": 0, "rejected": 0})
        breakdown[key][row.review_status] += 1
    return {"by_transaction_type": breakdown}


_GATE_ORDER = (
    "direction_automatable", "similarity", "precedent_count",
    "consensus", "calibrated_proba", "hard_rule_conflict",
)

_DIRECTION_BUCKET_ORDER = ("passed", "clean_blocked", "no_suggestion", "escalate")


def _classify_direction_detail(detail: str | None) -> str:
    if not detail:
        return "unrecognized"
    if detail.startswith("direction=fraud"):
        return "passed"
    if detail.startswith("direction=clean"):
        return "clean_blocked"
    if detail.startswith("direction=escalate"):
        return "escalate"
    if detail.startswith("no ") and "suggestion" in detail:
        return "no_suggestion"
    return "unrecognized"


def gate_bottleneck_stats(db: Session) -> dict:
    rows = db.query(m.AutoBlockLog).all()
    counts = {gate: {"passed": 0, "failed": 0} for gate in _GATE_ORDER}
    direction_buckets = {bucket: 0 for bucket in _DIRECTION_BUCKET_ORDER}
    direction_unrecognized = 0

    for row in rows:
        tc = row.triggered_conditions
        gates = tc.get("gates")
        if gates:
            normalized = [(g.get("gate"), g.get("passed"), g.get("detail")) for g in gates]
        else:
            reason = tc.get("reason") or []
            failed_gates = tc.get("failed_gates") or []
            if len(reason) != len(_GATE_ORDER):
                continue
            normalized = [
                (gate, f"{gate}: {detail}" not in failed_gates, detail)
                for gate, detail in zip(_GATE_ORDER, reason)
            ]

        for name, passed, detail in normalized:
            if name in counts:
                counts[name]["passed" if passed else "failed"] += 1
            if name == "direction_automatable":
                bucket = _classify_direction_detail(detail)
                if bucket == "unrecognized":
                    direction_unrecognized += 1
                else:
                    direction_buckets[bucket] += 1

    return {
        "n_evaluations": len(rows),
        "gates": [
            {
                "gate": gate,
                "passed_count": counts[gate]["passed"],
                "failed_count": counts[gate]["failed"],
                "total": counts[gate]["passed"] + counts[gate]["failed"],
            }
            for gate in _GATE_ORDER
        ],
        "direction_breakdown": {
            **{bucket: direction_buckets[bucket] for bucket in _DIRECTION_BUCKET_ORDER},
            "unrecognized": direction_unrecognized,
        },
    }
