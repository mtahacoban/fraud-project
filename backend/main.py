from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend import automation
from backend import db_models as m
from backend import precedent
from backend import precedent_worker
from backend import report_worker
from backend import schemas as s
from backend.config import settings
from backend.database import (
    Base, engine, get_db,
    # Add/drop-column migrations (delegate to database.py's
    # _add_columns_if_missing / _drop_columns_if_present)
    ensure_llm_reports_source_column,
    ensure_precedent_explanation_pool_size_column,
    ensure_analyst_decisions_ai2_suggestion_column,
    ensure_automation_policy_columns,
    ensure_auto_block_log_columns,
    ensure_auto_block_log_review_columns,
    ensure_analyst_decisions_ai_proposal_columns,
    ensure_automation_policy_auto_triggered_column,
    ensure_automation_policy_legacy_columns_dropped,
    ensure_case_assigned_to_dropped,
    ensure_rule_hit_score_impact_dropped,
    # Bespoke migrations (their own logic, not the add/drop-column shape)
    ensure_precedent_index_label_width,
    ensure_default_automation_policy,
    ensure_shap_explanation_direction_backfill,
    ensure_reason_code_rename,
)
from backend.scoring import ScoringEngine, load_scoring_engine
from backend.simulation_templates import SIMULATION_TEMPLATES


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_precedent_index_label_width()  # must run before create_all — may drop+recreate on SQLite
    Base.metadata.create_all(bind=engine)
    ensure_llm_reports_source_column()
    ensure_precedent_explanation_pool_size_column()
    ensure_analyst_decisions_ai2_suggestion_column()
    ensure_automation_policy_columns()
    ensure_automation_policy_auto_triggered_column()
    ensure_automation_policy_legacy_columns_dropped()
    ensure_case_assigned_to_dropped()
    ensure_rule_hit_score_impact_dropped()
    ensure_default_automation_policy()
    ensure_auto_block_log_columns()
    ensure_auto_block_log_review_columns()
    ensure_analyst_decisions_ai_proposal_columns()
    ensure_shap_explanation_direction_backfill()
    ensure_reason_code_rename()
    app.state.scoring_engine = load_scoring_engine()
    yield


app = FastAPI(title="Fraud Detection & Investigation System — API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine() -> ScoringEngine:
    return app.state.scoring_engine


if os.path.isdir("reports"):
    app.mount("/report-assets", StaticFiles(directory="reports"), name="report-assets")


def _latest_score(db: Session, transaction_id: int) -> m.Score | None:
    return (
        db.query(m.Score)
        .filter(m.Score.transaction_id == transaction_id)
        .order_by(m.Score.id.desc())
        .first()
    )


def _rule_hits(db: Session, transaction_id: int, rule_type: str) -> list[str]:
    return [
        h.rule_name for h in
        db.query(m.RuleHit)
        .filter(m.RuleHit.transaction_id == transaction_id, m.RuleHit.rule_type == rule_type)
        .all()
    ]


def _get_case_or_404(db: Session, case_id: int) -> m.Case:
    case = db.get(m.Case, case_id)
    if case is None:
        raise HTTPException(404, "Case not found")
    return case


def _write_score(
    db: Session,
    txn_in: s.TransactionIn,
    result: dict,
    *,
    source: str = "live",
    is_demo: bool = False,
    background_tasks: BackgroundTasks | None = None,
) -> tuple[m.Transaction, m.Case | None]:
    """Persists a scored transaction (+ rule hits, SHAP, case). Shared by
    /score and /simulation/run so both write through the identical path;
    source/is_demo are the only things that differ between them."""
    txn_row = m.Transaction(
        step=txn_in.step,
        type=txn_in.type,
        amount=txn_in.amount,
        name_orig=txn_in.nameOrig,
        oldbalance_org=txn_in.oldbalanceOrg,
        newbalance_orig=txn_in.newbalanceOrig,
        name_dest=txn_in.nameDest,
        oldbalance_dest=txn_in.oldbalanceDest,
        newbalance_dest=txn_in.newbalanceDest,
        error_balance_orig=result["errorBalanceOrig"],
        error_balance_dest=result["errorBalanceDest"],
        step_hour=result["step_hour"],
        is_transfer=result["is_transfer"],
        is_cashout=result["is_cashout"],
        device_id=txn_in.device_id,
        is_known_device=txn_in.is_known_device,
        login_country=txn_in.login_country,
        geo_velocity_flag=txn_in.geo_velocity_flag,
        channel=txn_in.channel,
        is_fraud=txn_in.isFraud,
        source=source,
        is_demo=is_demo,
    )
    db.add(txn_row)
    db.flush()

    score_row = m.Score(
        transaction_id=txn_row.id,
        ml_score=result["ml_score"],
        rule_score=result["soft_score"],
        hybrid_score=result["hybrid_score"],
        risk_band=result["risk_band"],
        calibrated_proba=result["calibrated_proba"],
        model_version=result["model_version"],
        band_reason=result["band_reason"],
    )
    db.add(score_row)

    for name in result["hard_rule_hits"]:
        db.add(m.RuleHit(transaction_id=txn_row.id, rule_name=name, rule_type="hard", severity="critical"))
    for name in result["soft_rule_hits"]:
        db.add(m.RuleHit(transaction_id=txn_row.id, rule_name=name, rule_type="soft", severity="warning"))
    # Persisted so Gate B (automation.py) can query it later from a
    # background task — never read by scoring. Deliberately excluded from
    # every rule_hits/hard_rule_hits/soft_rule_hits API response (see the
    # rule_type filters below and in findings.py) — this is a clean signal,
    # not a risk indicator, and the frontend has no rendering for it.
    for name in result["clean_rule_hits"]:
        db.add(m.RuleHit(transaction_id=txn_row.id, rule_name=name, rule_type="clean", severity="info"))

    for f in result["shap_factors"]:
        db.add(m.ShapExplanation(
            transaction_id=txn_row.id,
            feature_name=f["feature"],
            shap_value=f["shap_value"],
            feature_value=f["value"],
            direction=f["direction"],
        ))

    case_row = None
    if result["risk_band"] in ("RED", "GRAY"):
        case_row = m.Case(
            transaction_id=txn_row.id,
            status="OPEN",
            priority="HIGH" if result["risk_band"] == "RED" else "NORMAL",
        )
        db.add(case_row)
        db.flush()
        # Proactive: try to have the report ready before an analyst opens
        # Case Detail. Runs after the response is sent — does not add to
        # this request's latency. Lazy fallback (if this never ran, or
        # failed silently) lives in GET /cases/{id}/report.
        if background_tasks is not None:
            report_worker.ensure_report_generation(case_row.id, background_tasks, db)
            # Same proactive-after-response pattern — checks automation
            # eligibility for the brand-new case. A no-op in every mode
            # except "propose" (see evaluate_and_propose_for_case), so this
            # doesn't add work in the system's normal shadow/off state.
            background_tasks.add_task(automation.evaluate_and_propose_for_case, case_row.id)

    return txn_row, case_row


@app.post("/score", response_model=s.ScoreOut)
def score_transaction_endpoint(
    txn_in: s.TransactionIn,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    engine_: ScoringEngine = Depends(get_engine),
):
    result = engine_.score(txn_in.model_dump())
    txn_row, case_row = _write_score(db, txn_in, result, background_tasks=background_tasks)
    db.commit()

    return s.ScoreOut(
        txn_id=txn_row.id,
        case_id=case_row.id if case_row else None,
        ml_score=result["ml_score"],
        soft_score=result["soft_score"],
        hybrid_score=result["hybrid_score"],
        risk_band=result["risk_band"],
        calibrated_proba=result["calibrated_proba"],
        hard_rule_hits=result["hard_rule_hits"],
        soft_rule_hits=result["soft_rule_hits"],
        shap_factors=result["shap_factors"],
        model_version=result["model_version"],
        band_reason=result["band_reason"],
    )


MAX_CONCURRENT_SIMULATION_SCORING = 15


def _jitter_transaction(base: dict) -> dict:
    """Small, balance-consistent variation around a template, used when
    count>1.

    Origin side: amount and oldbalanceOrg are jittered together, keeping
    the template's amount/oldbalanceOrg ratio (so a full drain stays a full
    drain and a partial transfer stays partial — amount never exceeds the
    jittered origin balance). newbalanceOrig is re-derived to preserve the
    template's own errorBalanceOrig offset, clamped at zero, instead of
    being left stale relative to the new amount.

    Destination side: left completely untouched so hard-rule patterns that
    depend on exact values (e.g. ghost_destination requires
    oldbalanceDest == newbalanceDest == 0) survive the jitter exactly;
    templates where the destination is a real, reconciled account will see
    errorBalanceDest drift slightly as amount moves, which is a minor,
    intentional trade-off to keep the hard-rule case robust.
    """
    jittered = dict(base)
    old_orig = base["oldbalanceOrg"]

    if old_orig > 0:
        jittered["oldbalanceOrg"] = round(old_orig * (1 + random.uniform(-0.10, 0.10)), 2)
        ratio = base["amount"] / old_orig
        amount_factor = 1 + random.uniform(-0.15, 0.15)
        jittered["amount"] = round(
            min(jittered["oldbalanceOrg"], jittered["oldbalanceOrg"] * ratio * amount_factor), 2
        )
        error_orig = base["newbalanceOrig"] + base["amount"] - old_orig
        jittered["newbalanceOrig"] = round(
            max(jittered["oldbalanceOrg"] - jittered["amount"] + error_orig, 0.0), 2
        )
    else:
        jittered["amount"] = round(max(base["amount"] * (1 + random.uniform(-0.15, 0.15)), 0.01), 2)

    return jittered


@app.get("/simulation/templates", response_model=list[s.SimulationTemplateOut])
def list_simulation_templates():
    return SIMULATION_TEMPLATES


@app.post("/simulation/run", response_model=s.SimulationRunOut)
def run_simulation(
    payload: s.SimulationRunIn,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    engine_: ScoringEngine = Depends(get_engine),
):
    base = payload.transaction.model_dump()
    count = payload.count
    include_shap = count == 1

    inputs = [base] if count == 1 else [_jitter_transaction(base) for _ in range(count)]

    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_SIMULATION_SCORING, count)) as pool:
        scored = list(pool.map(lambda txn: engine_.score(txn, include_shap=include_shap), inputs))

    band_counts = {"RED": 0, "GRAY": 0, "GREEN": 0}
    results: list[s.SimulationResultOut] = []
    for txn_dict, result in zip(inputs, scored):
        txn_in = s.TransactionIn(**txn_dict)
        txn_row, case_row = _write_score(
            db, txn_in, result, source="simulator", is_demo=True, background_tasks=background_tasks,
        )
        band_counts[result["risk_band"]] = band_counts.get(result["risk_band"], 0) + 1
        results.append(s.SimulationResultOut(
            txn_id=txn_row.id,
            case_id=case_row.id if case_row else None,
            risk_band=result["risk_band"],
            hybrid_score=result["hybrid_score"],
            calibrated_proba=result["calibrated_proba"],
            shap_factors=[s.ShapFactorOut(**f) for f in result["shap_factors"]],
        ))

    db.commit()

    return s.SimulationRunOut(
        requested=count,
        scored=len(results),
        band_counts=band_counts,
        results=results,
    )


# GREEN transactions never open a Case; added to the list as read-only virtual rows.
AUTO_CLEAN_LIMIT = 200


_CASE_SORT_COLUMNS = {
    "hybrid_score": lambda: m.Score.hybrid_score,
    "created_at": lambda: m.Case.created_at,
}


def _auto_clean_virtual_rows(db: Session, q: str | None, limit: int, offset: int) -> s.CaseListOut:
    """status=AUTO_CLEAN's distinct virtual view: GREEN transactions that
    never opened a Case, capped at AUTO_CLEAN_LIMIT. See list_cases()'s
    docstring for how this fits into the overall /cases contract."""
    query = (
        db.query(m.Score, m.Transaction)
        .join(m.Transaction, m.Transaction.id == m.Score.transaction_id)
        .outerjoin(m.Case, m.Case.transaction_id == m.Score.transaction_id)
        .filter(m.Score.risk_band == "GREEN", m.Case.id.is_(None))
    )
    if q and q.strip().isdigit():
        query = query.filter(m.Transaction.id == int(q.strip()))
    total = min(query.count(), AUTO_CLEAN_LIMIT)
    rows = (
        query.order_by(m.Score.id.desc())
        .limit(min(limit, AUTO_CLEAN_LIMIT))
        .offset(offset)
        .all()
    )
    items = [
        s.CaseSummaryOut(
            case_id=None, transaction_id=txn.id, status="AUTO_CLEAN", priority="LOW",
            hybrid_score=score.hybrid_score, risk_band=score.risk_band, created_at=txn.created_at,
        )
        for score, txn in rows
    ]
    return s.CaseListOut(items=items, total=total)


@app.get("/cases", response_model=s.CaseListOut)
def list_cases(
    status: str | None = None,
    risk_band: str | None = None,
    q: str | None = None,
    sort: str = "hybrid_score",
    order: str = "desc",
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Paginated, sortable, searchable case list — the analyst's queue.
    status=AUTO_CLEAN is a distinct virtual view (GREEN transactions that
    never opened a Case, capped at AUTO_CLEAN_LIMIT); every other status
    (including no filter, "every real case") is a normal paginated query
    against Case+Score. q matches an exact case or transaction ID.
    pending_ai_proposal flags cases with a live "proposed" automation
    row, so the queue can surface it without a per-case lookup."""
    status_upper = status.upper() if status else None
    risk_band_upper = risk_band.upper() if risk_band else None
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    if status_upper == "AUTO_CLEAN":
        return _auto_clean_virtual_rows(db, q, limit, offset)

    query = db.query(m.Case, m.Score).join(m.Score, m.Score.transaction_id == m.Case.transaction_id)
    if status_upper:
        query = query.filter(m.Case.status == status_upper)
    if risk_band_upper:
        query = query.filter(m.Score.risk_band == risk_band_upper)
    if q and q.strip().isdigit():
        qid = int(q.strip())
        query = query.filter((m.Case.id == qid) | (m.Case.transaction_id == qid))

    total = query.count()

    sort_col = _CASE_SORT_COLUMNS.get(sort, _CASE_SORT_COLUMNS["hybrid_score"])()
    sort_col = sort_col.asc() if order == "asc" else sort_col.desc()
    rows = query.order_by(sort_col).limit(limit).offset(offset).all()

    case_ids = [case.id for case, _ in rows]
    pending_ids: set[int] = set()
    if case_ids:
        pending_ids = {
            row.case_id for row in
            db.query(m.AutoBlockLog.case_id)
            .filter(m.AutoBlockLog.review_status == "proposed", m.AutoBlockLog.case_id.in_(case_ids))
            .all()
        }

    items = [
        s.CaseSummaryOut(
            case_id=case.id, transaction_id=case.transaction_id, status=case.status,
            priority=case.priority, hybrid_score=score.hybrid_score, risk_band=score.risk_band,
            created_at=case.created_at, pending_ai_proposal=case.id in pending_ids,
        )
        for case, score in rows
    ]
    return s.CaseListOut(items=items, total=total)


@app.get("/cases/{case_id}", response_model=s.CaseDetailOut)
def get_case(case_id: int, db: Session = Depends(get_db)):
    case = _get_case_or_404(db, case_id)

    txn = db.get(m.Transaction, case.transaction_id)
    score = _latest_score(db, case.transaction_id)
    # rule_type="clean" is deliberately excluded — API-facing, no rendering
    # for it, gate-only signal (see automation.py Gate B).
    rule_hits = (
        db.query(m.RuleHit)
        .filter(m.RuleHit.transaction_id == case.transaction_id, m.RuleHit.rule_type.in_(["hard", "soft"]))
        .all()
    )
    shap_rows = db.query(m.ShapExplanation).filter(m.ShapExplanation.transaction_id == case.transaction_id).all()

    score_out = None
    if score is not None:
        score_out = s.ScoreOut(
            txn_id=txn.id,
            case_id=case.id,
            ml_score=score.ml_score,
            soft_score=score.rule_score,
            hybrid_score=score.hybrid_score,
            risk_band=score.risk_band,
            calibrated_proba=score.calibrated_proba,
            hard_rule_hits=[h.rule_name for h in rule_hits if h.rule_type == "hard"],
            soft_rule_hits=[h.rule_name for h in rule_hits if h.rule_type == "soft"],
            shap_factors=[
                s.ShapFactorOut(
                    feature=r.feature_name, value=r.feature_value,
                    shap_value=r.shap_value, direction=r.direction,
                )
                for r in shap_rows
            ],
            model_version=score.model_version,
            band_reason=score.band_reason,
        )

    return s.CaseDetailOut(
        case_id=case.id,
        status=case.status,
        priority=case.priority,
        created_at=case.created_at,
        closed_at=case.closed_at,
        transaction=s.TransactionOut.model_validate(txn),
        score=score_out,
        rule_hits=[s.RuleHitOut.model_validate(h) for h in rule_hits],
        shap_explanations=[
            s.ShapFactorOut(feature=r.feature_name, value=r.feature_value,
                             shap_value=r.shap_value, direction=r.direction)
            for r in shap_rows
        ],
        llm_reports=[s.LlmReportOut.model_validate(r) for r in case.llm_reports],
        decisions=[s.AnalystDecisionOut.model_validate(d) for d in case.decisions],
    )


@app.post("/cases/{case_id}/decision", response_model=s.DecisionOut)
def decide_case(case_id: int, decision: s.DecisionIn, db: Session = Depends(get_db)):
    case = _get_case_or_404(db, case_id)
    if case.status == "CLOSED":
        raise HTTPException(409, "Case is already closed")

    # Capture what AI #2 suggests for this case BEFORE recording the
    # analyst's decision or adding this case to precedent_index — this is
    # what the analyst could actually see in the "Similar Past Cases"
    # panel right now, for the agreement-rate metric. Self-exclusion is
    # automatic (the case isn't indexed yet). None if there's no scaler
    # yet (true cold start) or no suggestion cleared the confidence gates
    # — both mean "AI #2 had nothing to say here".
    scaler = precedent.load_precedent_scaler()
    ai2_suggested_decision = None
    precedent_summary = None
    if scaler is not None:
        neighbors = precedent.find_precedents(case, db, scaler, k=precedent.DEFAULT_K)
        precedent_summary = precedent.summarize_precedents(neighbors)
        ai2_suggested_decision = precedent_summary["suggested_decision"]

    # Shadow-mode automation evaluation — reuses the exact precedent_summary
    # just computed above (no recomputation), logs unconditionally
    # (whatever eligible/direction came out as), and is
    # NEVER shown to the analyst or returned from this endpoint. Cheap
    # (in-memory arithmetic, no LLM, no extra query beyond what's below) —
    # doesn't change this endpoint's latency in any way that matters.
    # log_shadow_evaluation() itself no-ops unless policy.mode=="shadow".
    policy = automation.get_active_policy(db)
    if policy is not None and precedent_summary is not None:
        score_for_shadow = _latest_score(db, case.transaction_id)
        hard_rule_hits = _rule_hits(db, case.transaction_id, "hard")
        clean_rule_hits = _rule_hits(db, case.transaction_id, "clean")
        automation.log_shadow_evaluation(
            case, db, precedent_summary, score_for_shadow.calibrated_proba, hard_rule_hits, policy, clean_rule_hits,
        )

    decision_row = m.AnalystDecision(
        case_id=case.id,
        action_taken=decision.action_taken,
        analyst_reason_code=decision.analyst_reason_code,
        analyst_note=decision.analyst_note,
        auto_processed=False,
        ai2_suggested_decision=ai2_suggested_decision,
    )
    db.add(decision_row)

    case.status = "CLOSED"
    case.closed_at = datetime.now(timezone.utc)

    # This decision just became precedent. flush() first —
    # latest_decision_label() (inside add_to_precedent_index) queries
    # AnalystDecision, and this session has autoflush=False, so it wouldn't
    # see decision_row yet otherwise. Reuses the existing fitted scaler —
    # no refit, no leakage. If no scaler exists yet (true cold start, never
    # backfilled), skip silently; the decision is still recorded normally.
    db.flush()
    if scaler is not None:
        precedent.add_to_precedent_index(case, db, scaler)

    db.commit()
    db.refresh(decision_row)

    return s.DecisionOut(
        case_id=case.id,
        status=case.status,
        action_taken=decision_row.action_taken,
        decided_at=decision_row.decided_at,
    )


@app.post("/cases/{case_id}/confirm-ai-decision", response_model=s.ConfirmAiDecisionOut)
def confirm_ai_decision(case_id: int, payload: s.ConfirmAiDecisionIn, db: Session = Depends(get_db)):
    """Human confirms AI #2's proposed decision — the ONLY way a proposal
    ever finalizes into a real decision. Mirrors decide_case()'s
    CLOSED transition and precedent_index write exactly, but the resulting
    AnalystDecision is marked ai_proposed=True + ai_proposal_id (provenance:
    AI proposed, human confirmed — never auto_processed=True, that field
    stays False here same as everywhere else in this system)."""
    case = _get_case_or_404(db, case_id)
    if case.status == "CLOSED":
        raise HTTPException(409, "Case is already closed")

    proposal = automation.get_pending_proposal(case_id, db)
    if proposal is None:
        raise HTTPException(404, "No pending AI #2 proposal for this case")

    direction = proposal.triggered_conditions.get("direction")
    if direction != "fraud":
        # Not reachable by construction (propose_auto_decision only ever
        # creates fraud-direction proposals) — defensive, not a real path.
        raise HTTPException(409, f"Pending proposal has non-automatable direction={direction!r}")

    decision_row = m.AnalystDecision(
        case_id=case.id,
        action_taken="confirm_fraud",
        analyst_reason_code="ai2_proposal_confirmed",
        analyst_note=payload.analyst_note,
        auto_processed=False,
        ai2_suggested_decision="confirm_fraud",
        ai_proposed=True,
        ai_proposal_id=proposal.id,
    )
    db.add(decision_row)

    case.status = "CLOSED"
    case.closed_at = datetime.now(timezone.utc)

    proposal.review_status = "confirmed"
    proposal.reviewed_at = datetime.now(timezone.utc)

    # Same reasoning as decide_case(): flush so this decision is visible to
    # add_to_precedent_index()'s latest_decision_label() query before it runs.
    db.flush()
    scaler = precedent.load_precedent_scaler()
    if scaler is not None:
        precedent.add_to_precedent_index(case, db, scaler)

    db.commit()
    db.refresh(decision_row)

    return s.ConfirmAiDecisionOut(
        case_id=case.id,
        status=case.status,
        action_taken=decision_row.action_taken,
        ai_proposal_id=proposal.id,
        decided_at=decision_row.decided_at,
    )


@app.post("/cases/{case_id}/reject-ai-decision", response_model=s.RejectAiDecisionOut)
def reject_ai_decision(case_id: int, payload: s.RejectAiDecisionIn, db: Session = Depends(get_db)):
    """Human rejects AI #2's proposed decision. rejection_reason is
    required at the schema level (min_length=1) — rubber-stamping
    friction, and the raw material for the reject-rate metric. Does
    NOT decide the case: the proposal is marked "rejected" and the case
    stays OPEN — the analyst must still call POST /cases/{id}/decision
    separately with their own real decision, exactly as if no proposal had
    ever existed. The rejected proposal is never deleted — its
    triggered_conditions + rejection_reason stay in auto_block_log,
    permanently auditable."""
    case = _get_case_or_404(db, case_id)
    if case.status == "CLOSED":
        raise HTTPException(409, "Case is already closed")

    proposal = automation.get_pending_proposal(case_id, db)
    if proposal is None:
        raise HTTPException(404, "No pending AI #2 proposal for this case")

    proposal.review_status = "rejected"
    proposal.rejection_reason = payload.rejection_reason
    proposal.reviewed_at = datetime.now(timezone.utc)
    db.commit()

    # This is where the circuit breaker gets wired into the live flow — a
    # reject is exactly the event that can raise the active policy's
    # reject rate past its threshold, so check right after one lands.
    # No-ops instantly (mode check first) unless the active policy is in
    # "propose" mode; see automation.check_circuit_breaker.
    automation.check_circuit_breaker(db)

    return s.RejectAiDecisionOut(
        case_id=case.id,
        status=case.status,
        auto_block_log_id=proposal.id,
        rejection_reason=proposal.rejection_reason,
        rejected_at=proposal.reviewed_at,
    )


@app.get("/cases/{case_id}/pending-ai-decision", response_model=s.PendingAiDecisionOut | None)
def get_pending_ai_decision(case_id: int, db: Session = Depends(get_db)):
    """What Case Detail polls to decide whether to show the
    "AI #2 Proposed Decision" card. Returns null (200, not 404) when
    there's no live proposal — the common case for almost every case —
    so the frontend can check without special-casing an error response."""
    case = _get_case_or_404(db, case_id)

    proposal = automation.get_pending_proposal(case_id, db)
    if proposal is None:
        return None

    policy = db.get(m.AutomationPolicyVersion, proposal.policy_version_id)
    return s.PendingAiDecisionOut(
        auto_block_log_id=proposal.id,
        case_id=case.id,
        direction=proposal.triggered_conditions.get("direction"),
        triggered_conditions=proposal.triggered_conditions,
        policy_version=policy.version if policy is not None else "unknown",
        proposed_at=proposal.created_at,
    )


@app.post("/cases/{case_id}/reopen", response_model=s.ReopenOut)
def reopen_case(case_id: int, payload: s.ReopenIn, db: Session = Depends(get_db)):
    case = _get_case_or_404(db, case_id)
    if case.status == "OPEN":
        raise HTTPException(409, "Case is already open")

    reopened_at = datetime.now(timezone.utc)
    db.add(m.AnalystDecision(
        case_id=case.id,
        action_taken="reopened",
        analyst_reason_code=payload.analyst_reason_code,
        analyst_note=payload.analyst_note,
        auto_processed=False,
        decided_at=reopened_at,
    ))
    case.status = "OPEN"
    case.closed_at = None
    db.commit()

    return s.ReopenOut(case_id=case.id, status=case.status, reopened_at=reopened_at)


@app.get("/cases/{case_id}/report", response_model=s.ReportStatusOut)
def get_case_report(case_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    case = _get_case_or_404(db, case_id)

    status = report_worker.ensure_report_generation(case.id, background_tasks, db)
    if status == "ready":
        report = (
            db.query(m.LlmReport)
            .filter(m.LlmReport.case_id == case.id)
            .order_by(m.LlmReport.id.desc())
            .first()
        )
        return s.ReportStatusOut(status="ready", report=s.LlmReportOut.model_validate(report))

    return s.ReportStatusOut(status="generating", report=None)


@app.get("/cases/{case_id}/precedents", response_model=s.PrecedentsOut)
def get_case_precedents(case_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """AI #2 — precedent decision support. The neighbor list and summary
    are always computed fresh (cheap, deterministic, and precedent_index
    grows over time as analysts decide cases — caching them would go
    stale). The LLM explanation is only generated (and only cached) when
    there's an actual suggested_decision; with no suggestion there's
    nothing for an LLM to explain, so that branch is instant and never
    touches precedent_worker at all."""
    case = _get_case_or_404(db, case_id)

    txn = db.get(m.Transaction, case.transaction_id)
    scaler = precedent.load_precedent_scaler()

    if scaler is None:
        # Cold start: precedent_index has never been backfilled/fit yet.
        neighbors: list[dict] = []
    else:
        neighbors = precedent.find_precedents(case, db, scaler, k=precedent.DEFAULT_K)

    summary = precedent.summarize_precedents(neighbors)

    if summary["suggested_decision"] is None:
        result = precedent.explain_precedents(summary, txn)
        explanation = s.PrecedentExplanationOut(status="ready", text=result["text"], source=result["source"])
    else:
        exp = precedent_worker.ensure_precedent_explanation(case.id, summary, background_tasks, db)
        explanation = s.PrecedentExplanationOut(**exp)

    return s.PrecedentsOut(
        precedents=[s.PrecedentNeighborOut(**n) for n in neighbors],
        summary=s.PrecedentSummaryOut(**summary),
        explanation=explanation,
    )


@app.get("/model-info")
def get_model_info():
    xgb_meta_path = os.path.join(settings.model_dir, "xgb_v1_meta.json")
    cal_meta_path = os.path.join(settings.model_dir, "xgb_v1_calibrated_meta.json")

    xgb_meta = {}
    cal_meta = {}
    if os.path.exists(xgb_meta_path):
        with open(xgb_meta_path, "r", encoding="utf-8") as f:
            xgb_meta = json.load(f)
    if os.path.exists(cal_meta_path):
        with open(cal_meta_path, "r", encoding="utf-8") as f:
            cal_meta = json.load(f)

    return {"xgb": xgb_meta, "calibration": cal_meta}


@app.get("/metrics", response_model=s.MetricsOut)
def get_metrics(db: Session = Depends(get_db)):
    total_scored = db.query(func.count(m.Score.id)).scalar() or 0
    demo_scored = (
        db.query(func.count(m.Score.id))
        .join(m.Transaction, m.Transaction.id == m.Score.transaction_id)
        .filter(m.Transaction.is_demo.is_(True))
        .scalar() or 0
    )
    band_rows = db.query(m.Score.risk_band, func.count(m.Score.id)).group_by(m.Score.risk_band).all()
    total_cases = db.query(func.count(m.Case.id)).scalar() or 0
    open_cases = db.query(func.count(m.Case.id)).filter(m.Case.status == "OPEN").scalar() or 0
    closed_cases = total_cases - open_cases
    avg_hybrid = db.query(func.avg(m.Score.hybrid_score)).scalar() or 0.0
    pending_ai_proposals = (
        db.query(func.count(m.AutoBlockLog.id))
        .filter(m.AutoBlockLog.review_status == "proposed")
        .scalar() or 0
    )
    active_policy = automation.get_active_policy(db)

    return s.MetricsOut(
        total_scored=total_scored,
        live_scored=total_scored - demo_scored,
        demo_scored=demo_scored,
        by_risk_band={band: count for band, count in band_rows},
        total_cases=total_cases,
        open_cases=open_cases,
        closed_cases=closed_cases,
        avg_hybrid_score=round(float(avg_hybrid), 2),
        model_version=settings.model_version,
        pending_ai_proposals=pending_ai_proposals,
        automation_mode=active_policy.mode if active_policy is not None else None,
    )


@app.get("/automation/status", response_model=s.AutomationStatusOut)
def get_automation_status(db: Session = Depends(get_db)):
    """Automation's aggregate view — read-only, no side effects (does NOT
    call check_circuit_breaker(), which can itself trip the breaker; that
    only ever runs from reject_ai_decision()). Shadow-mode's per-case
    detail never appears here or anywhere else — this reports only the
    aggregate shadow_agreement_stats(), which is what keeps shadow
    measurement blind."""
    policy = automation.get_active_policy(db)
    if policy is None:
        raise HTTPException(503, "No automation policy configured")

    last_trip = (
        db.query(m.AutomationPolicyVersion)
        .filter(m.AutomationPolicyVersion.auto_triggered.is_(True))
        .order_by(m.AutomationPolicyVersion.id.desc())
        .first()
    )

    return s.AutomationStatusOut(
        active_policy=s.ActivePolicyOut.model_validate(policy),
        shadow_agreement=automation.shadow_agreement_stats(db),
        reject_rate=automation.reject_rate_stats(db, policy_version_id=policy.id),
        bias_monitoring=automation.bias_monitoring_stats(db),
        circuit_breaker=s.CircuitBreakerStatusOut(
            tripped_recently=last_trip is not None,
            last_trip_policy_version=last_trip.version if last_trip is not None else None,
            last_trip_at=last_trip.created_at if last_trip is not None else None,
            last_trip_notes=last_trip.notes if last_trip is not None else None,
        ),
    )
