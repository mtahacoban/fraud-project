from __future__ import annotations

from sqlalchemy.orm import Session

from backend import db_models as m
from backend import precedent
from backend.database import SessionLocal
from backend.precedent import latest_decision_label


def get_active_policy(db: Session) -> m.AutomationPolicyVersion | None:
    """Reads the single active automation policy version. Code never
    hardcodes a threshold — every gate value comes from whatever this
    returns. None means no automation is configured at all (empty table,
    e.g. a fresh deploy before ensure_default_automation_policy() has ever
    run) — callers must treat that exactly like mode="off": evaluate
    nothing, propose nothing, stay silent. Should be effectively
    unreachable once the startup migration seeds v1, but the guard costs
    nothing and keeps this function honest about what "no policy" means."""
    return (
        db.query(m.AutomationPolicyVersion)
        .filter(m.AutomationPolicyVersion.active.is_(True))
        .order_by(m.AutomationPolicyVersion.id.desc())
        .first()
    )


def get_pending_proposal(case_id: int, db: Session) -> m.AutoBlockLog | None:
    """The live "proposed" AutoBlockLog row for a case, if any — what the
    confirm/reject endpoints act on, and what Case Detail reads to show
    the "AI #2 Proposed Decision" card. None if the case was never
    proposed, or its proposal was already confirmed/rejected/withdrawn."""
    return (
        db.query(m.AutoBlockLog)
        .filter(m.AutoBlockLog.case_id == case_id, m.AutoBlockLog.review_status == "proposed")
        .order_by(m.AutoBlockLog.id.desc())
        .first()
    )


# suggested_decision (AI #2's label) -> automation direction. Only "fraud"
# can ever be eligible in v1 — see the asymmetry gate below.
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
    """Pure deterministic evaluation — no DB writes, no side effects, no
    LLM. Takes AI #2's summarize_precedents() output as-is (never
    recomputes k-NN/consensus itself) plus the two independent-source
    signals the multi-gate design requires, and returns whether this case
    clears every gate for automation.

    Returns {eligible, direction, reason, failed_gates}:
      - direction: what AI #2 suggested ("fraud"/"clean"/"escalate"),
        reported even when not eligible — useful for shadow-mode analysis
        of what automation *would* have proposed by direction.
      - reason: every gate's outcome, passed or failed (the full rationale).
      - failed_gates: just the ones that failed, human-readable — this is
        what Case Detail and shadow logging show for "why not".

    Five gates, ALL must pass for eligible=True:
      1. direction is automatable (fraud only — clean is disabled in v1
         via policy.auto_clean_enabled, escalate is never automatable)
      2. avg_similarity >= policy.fraud_similarity_threshold
      3. precedent_count >= policy.min_precedent_count
      4. consensus_ratio >= policy.min_consensus_ratio
      5. calibrated_proba >= policy.min_calibrated_proba (Gate A — AI #1's
         independent signal; this is the actual "AI #1 + AI #2 agree" check)
      + hard-rule gate (Gate B): two independent checks —
        (1) unconditional, regardless of policy: if direction is "fraud"
        AND a clean-confirming hard rule fired (rule_engine.py's
        check_clean_confirming_rules(), e.g. "clean_confirmed"), that's a
        direct contradiction between AI #2's suggestion and a rule strong
        enough to say "this looks routine" — fails the gate outright.
        (2) policy.hard_rule_required=False (default) means "no conflict"
        and passes by default otherwise; policy.hard_rule_required=True
        means a hard rule must have actually fired in the fraud direction.
    """
    gates: list[tuple[str, bool, str]] = []
    clean_rule_hits = clean_rule_hits or []

    suggested = precedent_summary.get("suggested_decision")
    direction = _DIRECTION_BY_SUGGESTION.get(suggested)

    if suggested is None:
        gates.append(("direction_automatable", False, "no AI #2 suggestion (insufficient precedent) — nothing to automate"))
    elif suggested == "escalate":
        gates.append(("direction_automatable", False, "direction=escalate — escalation is never automated, always human-routed"))
    elif suggested == "approve_clean" and not policy.auto_clean_enabled:
        gates.append(("direction_automatable", False, "direction=clean — auto-clean is disabled (policy.auto_clean_enabled=false)"))
    else:
        gates.append(("direction_automatable", True, f"direction={direction} — automatable under current policy"))

    avg_sim = precedent_summary.get("avg_similarity")
    min_sim = policy.fraud_similarity_threshold or 1.0
    sim_ok = avg_sim is not None and avg_sim >= min_sim
    gates.append((
        "similarity",
        sim_ok,
        f"avg_similarity={avg_sim:.4f} >= {min_sim} required" if sim_ok
        else f"avg_similarity={avg_sim if avg_sim is not None else 'None'} < {min_sim} required",
    ))

    count = precedent_summary.get("precedent_count") or 0
    min_count = policy.min_precedent_count or 0
    count_ok = count >= min_count
    gates.append((
        "precedent_count",
        count_ok,
        f"precedent_count={count} {'>=' if count_ok else '<'} {min_count} required",
    ))

    consensus = precedent_summary.get("consensus_ratio")
    min_consensus = policy.min_consensus_ratio or 1.0
    consensus_ok = consensus is not None and consensus >= min_consensus
    gates.append((
        "consensus",
        consensus_ok,
        f"consensus_ratio={consensus:.4f} >= {min_consensus} required" if consensus_ok
        else f"consensus_ratio={consensus if consensus is not None else 'None'} < {min_consensus} required",
    ))

    min_proba = policy.min_calibrated_proba or 1.0
    proba_ok = calibrated_proba >= min_proba
    gates.append((
        "calibrated_proba",
        proba_ok,
        f"calibrated_proba={calibrated_proba:.4f} {'>=' if proba_ok else '<'} {min_proba} required (AI #1 agreement)",
    ))

    has_hard_rule = len(hard_rule_hits) > 0
    has_clean_confirming_rule = len(clean_rule_hits) > 0

    if direction == "fraud" and has_clean_confirming_rule:
        hard_rule_ok = False
        hard_rule_detail = f"clean-confirming hard rule triggered ({clean_rule_hits}) — contradicts AI #2's fraud suggestion"
    elif policy.hard_rule_required:
        hard_rule_ok = has_hard_rule
        hard_rule_detail = (
            f"hard_rule_required=true, hard_rule_hits={hard_rule_hits}" if hard_rule_ok
            else "hard_rule_required=true but no hard rule fired — required confirmation missing"
        )
    else:
        hard_rule_ok = True
        hard_rule_detail = "hard_rule_required=false, no clean-confirming rule conflict — passes by default"
    gates.append(("hard_rule_conflict", hard_rule_ok, hard_rule_detail))

    failed_gates = [f"{name}: {detail}" for name, passed, detail in gates if not passed]

    return {
        "eligible": len(failed_gates) == 0,
        "direction": direction,
        "reason": [detail for _, _, detail in gates],
        "failed_gates": failed_gates,
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
    """Runs evaluate_auto_decision() and logs the result — review_status
    ALWAYS "shadow" here, whatever eligible/direction came out as. This is
    the whole point of shadow mode: log every evaluation, surface none of
    them. Never returned in any API response, never rendered in Case
    Detail — the only way to see this data is a direct DB/metrics query
    (shadow_agreement_stats() below), which is exactly what keeps the
    measurement blind.

    No-op (returns None, nothing evaluated or logged) unless
    policy.mode == "shadow" — "off" means automation isn't observing at
    all yet; "propose" mode logs eligible cases as "proposed" instead, via
    propose_auto_decision() below. Does not commit; caller controls the
    transaction, same as precedent.add_to_precedent_index()."""
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
    """Blind agreement: for every shadow-logged evaluation, compares what
    automation WOULD have decided (only when eligible AND direction=="fraud"
    — the only case that could ever become an automatic action) against
    what the analyst actually, independently decided — without the analyst
    ever having seen the shadow verdict.

    This is conceptually distinct from precedent.compute_agreement_stats():
    that measures suggestion-agreement, where the analyst COULD see the
    "Similar Past Cases" panel while deciding (a visible, potentially
    influencing signal). This measures automation's stricter gate against a
    decision the analyst made with zero visibility into what automation
    would have done — genuinely blind, not an exposure-inflated upper
    bound. Rows where eligible=False are counted (n_not_eligible) but
    excluded from shadow_accuracy's denominator — nothing to compare
    against a verdict that was never reached."""
    rows = db.query(m.AutoBlockLog).filter(m.AutoBlockLog.review_status == "shadow").all()

    would_confirm = 0
    would_be_wrong = 0
    not_eligible = 0

    for row in rows:
        result = row.triggered_conditions
        if not result.get("eligible"):
            not_eligible += 1
            continue

        case = db.get(m.Case, row.case_id) if row.case_id is not None else None
        actual = latest_decision_label(case, db) if case is not None else None
        if actual is None:
            not_eligible += 1  # decided case with no reconstructable label — can't compare, don't silently drop it
            continue

        if actual == "confirm_fraud":
            would_confirm += 1
        else:
            would_be_wrong += 1

    comparable = would_confirm + would_be_wrong

    return {
        "n_shadow_evaluations": len(rows),
        "n_eligible": comparable,
        "n_not_eligible": not_eligible,
        "would_have_confirmed_correctly": would_confirm,
        "would_have_been_wrong": would_be_wrong,
        "shadow_accuracy": round(would_confirm / comparable, 4) if comparable else None,
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
    """Deactivates the current active policy (if any) and creates+activates
    a new version — a threshold is never edited in place, so every change
    (deliberate tuning or, later, a circuit-breaker trip) leaves a
    permanent, auditable row in automation_policy_versions. Copies every
    gate field from the current active policy except what's passed in
    `overrides`; falls back to v1's own defaults if there's somehow no
    active policy yet (shouldn't happen post-bootstrap, but this function
    has to be safe to call regardless). Self-committing: unlike the
    per-request evaluation helpers above, a policy switch is a standalone
    administrative action, not a step inside some caller's existing
    transaction."""
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
    """Only acts in mode="propose". If the case clears every gate AND the
    direction is "fraud" (the only automatable direction — see
    evaluate_auto_decision's asymmetry gate), creates a "proposed"
    AutoBlockLog row: a pending AI #2 decision awaiting analyst approval.

    Still no action: this does NOT touch case.status (stays OPEN), does
    NOT create an AnalystDecision, does NOT call add_to_precedent_index —
    the case is completely unchanged; only a new, inert record exists for
    the approve/reject endpoints to act on later.

    Idempotent: if a "proposed" row already exists for this case, returns
    it unchanged rather than creating a second one — one case, at most one
    live proposal, same discipline as AI #2's precedent upsert.

    Not eligible, or eligible but direction != "fraud" (clean/escalate) ->
    returns None, nothing written — the asymmetry holds in propose mode
    exactly as it does in evaluation (auto_clean_enabled=false and
    escalate-never-automated aren't shadow-mode-only rules)."""
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
    """Runs in a FastAPI BackgroundTask right after a new RED/GRAY case is
    created (wired into main.py's _write_score(), the same proactive
    generation-at-case-creation pattern used by report_worker and
    precedent_worker) — proactively checks whether the brand-new case is
    automation-eligible, so a pending proposal (if any) is usually ready
    before the analyst ever opens Case Detail. Opens its own DB session;
    the request-scoped one is already closed by the time a background task
    runs.

    Cheap early exit outside "propose" mode (the common case — "shadow" and
    "off" never produce a proposal) skips the k-NN pass entirely, so this
    costs nothing when automation isn't actively proposing. Delegates all
    real logic to propose_auto_decision(), which is itself idempotent and
    already a no-op in every mode except "propose" — this wrapper only
    supplies the case-creation trigger point that was otherwise missing."""
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
    """confirmed vs. rejected ratio among AI #2 proposals the analyst
    actually acted on (still-"proposed"/pending rows are excluded — they
    haven't been judged yet). A low reject rate is a plausible
    rubber-stamping signal, but only above a minimum sample size — the
    same statistical discipline as AI #2's own confidence gates (n=2 at
    50% is noise, not a finding, exactly like n=3 precedents isn't enough
    for a suggestion there).

    policy_version_id=None (default): all-time, across every policy
    version that has ever proposed anything — a global "how has AI #2's
    automation performed overall" view. Pass a specific id to scope to one
    policy's own track record, which is what check_circuit_breaker() below
    actually evaluates (a policy should be judged on its own proposals,
    not diluted by a since-superseded version's history)."""
    query = db.query(m.AutoBlockLog).filter(m.AutoBlockLog.review_status.in_(["confirmed", "rejected"]))
    if policy_version_id is not None:
        query = query.filter(m.AutoBlockLog.policy_version_id == policy_version_id)
    rows = query.all()

    confirmed = sum(1 for r in rows if r.review_status == "confirmed")
    rejected = sum(1 for r in rows if r.review_status == "rejected")
    n = confirmed + rejected

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
            "n": n, "confirmed": confirmed, "rejected": rejected,
            "reject_rate": None,
            "note": f"insufficient data (n={n} < {min_n} required)",
        }

    return {
        "n": n, "confirmed": confirmed, "rejected": rejected,
        "reject_rate": round(rejected / n, 4),
        "note": None,
    }


def check_circuit_breaker(db: Session) -> dict:
    """Evaluates the ACTIVE policy's own reject rate against its
    circuit_breaker_max_reversal_rate, guarded by
    circuit_breaker_min_confirmations. If tripped, downgrades mode
    "propose" -> "shadow" via activate_new_policy_version() — reusing the
    exact same versioning mechanism a manual policy change uses, not a new
    path: a trip IS a policy change, permanently logged with a `notes`
    explanation, never a silent in-place flag flip.

    Trigger signal is reject_rate (confirmed vs. rejected) ONLY — not
    post-confirmation reversal (a confirmed AI decision whose case later
    gets reopened). Reversal tracking is deliberately NOT implemented:
    reopen_case()'s reason is free text, not a structured "this reopening
    specifically contradicts an AI proposal" flag, and reopens are a slow,
    infrequent, behaviorally-confounded proxy (an analyst may reopen for
    reasons that have nothing to do with regretting an automation
    confirmation — new evidence, a data-entry correction, anything).
    reject_rate is immediate, unambiguous (an explicit reject click with a
    mandatory reason), and already cleanly measured. This is a documented
    scope boundary, not a gap papered over — see the README's Human-
    Confirmed Automation section for the backlog note.

    Only "propose" mode can trip: "shadow" and "off" never produce
    confirmed/rejected rows (nothing is ever surfaced to confirm or reject
    in those modes), so there's nothing for this function to act on."""
    policy = get_active_policy(db)
    if policy is None:
        return {"tripped": False, "reason": "no active policy", "stats": None}
    if policy.mode != "propose":
        return {"tripped": False, "reason": f"mode={policy.mode} — only 'propose' can trip", "stats": None}

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
    """Minimal bias smoke-check: confirmed vs. rejected counts grouped by
    transaction type. A systematic gap (e.g. AI #2 proposals on CASH_OUT
    get rejected far more often than on TRANSFER) would be a concrete,
    checkable trace of AI #2's known limitation: it imitates the analyst,
    it doesn't audit for fairness, so it can inherit and amplify a
    systematic blind spot. Deliberately one dimension and nothing more —
    a real fairness audit needs protected attributes this dataset doesn't
    have, and going deeper here would overstate what this data can
    actually support. This is a smoke alarm, not a certification."""
    rows = db.query(m.AutoBlockLog).filter(m.AutoBlockLog.review_status.in_(["confirmed", "rejected"])).all()
    breakdown: dict[str, dict[str, int]] = {}
    for row in rows:
        txn = db.get(m.Transaction, row.transaction_id)
        key = txn.type if txn is not None else "unknown"
        breakdown.setdefault(key, {"confirmed": 0, "rejected": 0})
        breakdown[key][row.review_status] += 1
    return {"by_transaction_type": breakdown}
