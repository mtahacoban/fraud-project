from __future__ import annotations

import csv
import io
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone

import xlsxwriter
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend import automation
from backend import db_models as m
from backend import findings
from backend import precedent
from backend import precedent_worker
from backend import report_worker
from backend import schemas as s
from backend.config import settings
from backend.database import (
    Base, engine, get_db,
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
    ensure_precedent_index_label_width,
    ensure_default_automation_policy,
    ensure_shap_explanation_direction_backfill,
    ensure_reason_code_rename,
)
from backend.scoring import ScoringEngine, load_scoring_engine
from backend.simulation_templates import SIMULATION_TEMPLATES


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_precedent_index_label_width()
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


app = FastAPI(title="Fraud Detection & Investigation System - API", lifespan=lifespan)

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
        if background_tasks is not None:
            report_worker.ensure_report_generation(case_row.id, background_tasks, db)
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
            hard_rule_hits=result["hard_rule_hits"],
            soft_rule_hits=result["soft_rule_hits"],
            band_reason=result["band_reason"],
        ))

    db.commit()

    return s.SimulationRunOut(
        requested=count,
        scored=len(results),
        band_counts=band_counts,
        results=results,
    )


_CASE_SORT_COLUMNS = {
    "hybrid_score": lambda: m.Score.hybrid_score,
    "created_at": lambda: m.Case.created_at,
}


def _apply_transaction_filters(
    query,
    *,
    txn_type: str | None,
    date_from: date | None,
    date_to: date | None,
    amount_min: float | None,
    amount_max: float | None,
    country: str | None,
):
    if txn_type:
        query = query.filter(m.Transaction.type == txn_type)
    if date_from:
        query = query.filter(
            m.Transaction.created_at >= datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
        )
    if date_to:
        query = query.filter(
            m.Transaction.created_at
            < datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)
        )
    if amount_min is not None:
        query = query.filter(m.Transaction.amount >= amount_min)
    if amount_max is not None:
        query = query.filter(m.Transaction.amount <= amount_max)
    if country:
        query = query.filter(m.Transaction.login_country == country)
    return query


def _top_rules_by_transaction_id(db: Session, transaction_ids: list[int]) -> dict[int, list[str]]:
    if not transaction_ids:
        return {}
    rows = (
        db.query(m.RuleHit.transaction_id, m.RuleHit.rule_name)
        .filter(m.RuleHit.transaction_id.in_(transaction_ids), m.RuleHit.rule_type.in_(["hard", "soft"]))
        .order_by(m.RuleHit.id.asc())
    )
    result: dict[int, list[str]] = {}
    for txn_id, rule_name in rows:
        result.setdefault(txn_id, []).append(rule_name)
    return result


def _auto_clean_base_query(
    db: Session,
    q: str | None,
    *,
    txn_type: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    amount_min: float | None = None,
    amount_max: float | None = None,
    country: str | None = None,
):
    query = (
        db.query(m.Score, m.Transaction)
        .join(m.Transaction, m.Transaction.id == m.Score.transaction_id)
        .outerjoin(m.Case, m.Case.transaction_id == m.Score.transaction_id)
        .filter(m.Score.risk_band == "GREEN", m.Case.id.is_(None))
    )
    if q and q.strip().isdigit():
        query = query.filter(m.Transaction.id == int(q.strip()))
    query = _apply_transaction_filters(
        query, txn_type=txn_type, date_from=date_from, date_to=date_to,
        amount_min=amount_min, amount_max=amount_max, country=country,
    )
    return query


def _auto_clean_virtual_rows(
    db: Session,
    q: str | None,
    limit: int,
    offset: int,
    *,
    txn_type: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    amount_min: float | None = None,
    amount_max: float | None = None,
    country: str | None = None,
) -> s.CaseListOut:
    query = _auto_clean_base_query(
        db, q, txn_type=txn_type, date_from=date_from, date_to=date_to,
        amount_min=amount_min, amount_max=amount_max, country=country,
    )
    total = query.count()
    rows = (
        query.order_by(m.Score.id.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )
    top_rules_by_txn = _top_rules_by_transaction_id(db, [txn.id for _, txn in rows])
    items = [
        s.CaseSummaryOut(
            case_id=None, transaction_id=txn.id, status="AUTO_CLEAN", priority="LOW",
            hybrid_score=score.hybrid_score, risk_band=score.risk_band, created_at=txn.created_at,
            amount=txn.amount, type=txn.type, top_rules=top_rules_by_txn.get(txn.id, []),
        )
        for score, txn in rows
    ]
    return s.CaseListOut(items=items, total=total)


def _all_rows_merged(
    db: Session,
    risk_band_upper: str | None,
    q: str | None,
    sort: str,
    order: str,
    limit: int,
    offset: int,
    *,
    txn_type: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    amount_min: float | None = None,
    amount_max: float | None = None,
    country: str | None = None,
) -> s.CaseListOut:
    case_query = (
        db.query(m.Case, m.Score, m.Transaction)
        .join(m.Score, m.Score.transaction_id == m.Case.transaction_id)
        .join(m.Transaction, m.Transaction.id == m.Case.transaction_id)
    )
    if risk_band_upper:
        case_query = case_query.filter(m.Score.risk_band == risk_band_upper)
    if q and q.strip().isdigit():
        qid = int(q.strip())
        case_query = case_query.filter((m.Case.id == qid) | (m.Case.transaction_id == qid))
    case_query = _apply_transaction_filters(
        case_query, txn_type=txn_type, date_from=date_from, date_to=date_to,
        amount_min=amount_min, amount_max=amount_max, country=country,
    )
    case_rows = case_query.all()

    case_ids = [case.id for case, _, _ in case_rows]
    pending_ids: set[int] = set()
    if case_ids:
        pending_ids = {
            row.case_id for row in
            db.query(m.AutoBlockLog.case_id)
            .filter(m.AutoBlockLog.review_status == "proposed", m.AutoBlockLog.case_id.in_(case_ids))
            .all()
        }
    top_rules_by_txn = _top_rules_by_transaction_id(db, [case.transaction_id for case, _, _ in case_rows])
    items = [
        s.CaseSummaryOut(
            case_id=case.id, transaction_id=case.transaction_id, status=case.status,
            priority=case.priority, hybrid_score=score.hybrid_score, risk_band=score.risk_band,
            created_at=case.created_at, pending_ai_proposal=case.id in pending_ids,
            amount=txn.amount, type=txn.type, top_rules=top_rules_by_txn.get(case.transaction_id, []),
        )
        for case, score, txn in case_rows
    ]

    auto_clean_total = 0
    if risk_band_upper is None or risk_band_upper == "GREEN":
        auto_clean_query = _auto_clean_base_query(
            db, q, txn_type=txn_type, date_from=date_from, date_to=date_to,
            amount_min=amount_min, amount_max=amount_max, country=country,
        )
        auto_clean_total = auto_clean_query.count()
        auto_clean_rows = auto_clean_query.order_by(m.Score.id.desc()).all()
        auto_clean_top_rules = _top_rules_by_transaction_id(db, [txn.id for _, txn in auto_clean_rows])
        items += [
            s.CaseSummaryOut(
                case_id=None, transaction_id=txn.id, status="AUTO_CLEAN", priority="LOW",
                hybrid_score=score.hybrid_score, risk_band=score.risk_band, created_at=txn.created_at,
                amount=txn.amount, type=txn.type, top_rules=auto_clean_top_rules.get(txn.id, []),
            )
            for score, txn in auto_clean_rows
        ]

    total = len(case_rows) + auto_clean_total

    sort_key = (lambda it: it.created_at) if sort == "created_at" else (lambda it: it.hybrid_score)
    items.sort(key=sort_key, reverse=(order != "asc"))

    return s.CaseListOut(items=items[offset:offset + limit], total=total)


@app.get("/cases", response_model=s.CaseListOut)
def list_cases(
    status: str | None = None,
    risk_band: str | None = None,
    q: str | None = None,
    sort: str = "hybrid_score",
    order: str = "desc",
    limit: int = 50,
    offset: int = 0,
    type: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    amount_min: float | None = None,
    amount_max: float | None = None,
    country: str | None = None,
    db: Session = Depends(get_db),
):
    status_upper = status.upper() if status else None
    risk_band_upper = risk_band.upper() if risk_band else None
    txn_type = type.upper() if type else None
    country_upper = country.upper() if country else None
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    if status_upper == "AUTO_CLEAN":
        return _auto_clean_virtual_rows(
            db, q, limit, offset, txn_type=txn_type, date_from=date_from, date_to=date_to,
            amount_min=amount_min, amount_max=amount_max, country=country_upper,
        )

    if status_upper is None:
        return _all_rows_merged(
            db, risk_band_upper, q, sort, order, limit, offset, txn_type=txn_type,
            date_from=date_from, date_to=date_to, amount_min=amount_min,
            amount_max=amount_max, country=country_upper,
        )

    query = (
        db.query(m.Case, m.Score, m.Transaction)
        .join(m.Score, m.Score.transaction_id == m.Case.transaction_id)
        .join(m.Transaction, m.Transaction.id == m.Case.transaction_id)
    )
    if status_upper:
        query = query.filter(m.Case.status == status_upper)
    if risk_band_upper:
        query = query.filter(m.Score.risk_band == risk_band_upper)
    if q and q.strip().isdigit():
        qid = int(q.strip())
        query = query.filter((m.Case.id == qid) | (m.Case.transaction_id == qid))
    query = _apply_transaction_filters(
        query, txn_type=txn_type, date_from=date_from, date_to=date_to,
        amount_min=amount_min, amount_max=amount_max, country=country_upper,
    )

    total = query.count()

    sort_col = _CASE_SORT_COLUMNS.get(sort, _CASE_SORT_COLUMNS["hybrid_score"])()
    sort_col = sort_col.asc() if order == "asc" else sort_col.desc()
    rows = query.order_by(sort_col).limit(limit).offset(offset).all()

    case_ids = [case.id for case, _, _ in rows]
    pending_ids: set[int] = set()
    if case_ids:
        pending_ids = {
            row.case_id for row in
            db.query(m.AutoBlockLog.case_id)
            .filter(m.AutoBlockLog.review_status == "proposed", m.AutoBlockLog.case_id.in_(case_ids))
            .all()
        }
    top_rules_by_txn = _top_rules_by_transaction_id(db, [case.transaction_id for case, _, _ in rows])

    items = [
        s.CaseSummaryOut(
            case_id=case.id, transaction_id=case.transaction_id, status=case.status,
            priority=case.priority, hybrid_score=score.hybrid_score, risk_band=score.risk_band,
            created_at=case.created_at, pending_ai_proposal=case.id in pending_ids,
            amount=txn.amount, type=txn.type, top_rules=top_rules_by_txn.get(case.transaction_id, []),
        )
        for case, score, txn in rows
    ]
    return s.CaseListOut(items=items, total=total)


_EXPORT_FIELDS = [
    "case_id", "transaction_id", "status", "risk_band", "hybrid_score",
    "type", "amount", "login_country", "source", "decision", "created_at",
]


def _export_case_rows(
    db: Session, status_upper: str | None, risk_band_upper: str | None, q: str | None,
    *, txn_type, date_from, date_to, amount_min, amount_max, country,
) -> list[dict]:
    query = (
        db.query(m.Case, m.Score, m.Transaction)
        .join(m.Score, m.Score.transaction_id == m.Case.transaction_id)
        .join(m.Transaction, m.Transaction.id == m.Case.transaction_id)
    )
    if status_upper:
        query = query.filter(m.Case.status == status_upper)
    if risk_band_upper:
        query = query.filter(m.Score.risk_band == risk_band_upper)
    if q and q.strip().isdigit():
        qid = int(q.strip())
        query = query.filter((m.Case.id == qid) | (m.Case.transaction_id == qid))
    query = _apply_transaction_filters(
        query, txn_type=txn_type, date_from=date_from, date_to=date_to,
        amount_min=amount_min, amount_max=amount_max, country=country,
    )
    triples = query.all()

    closed_ids = [case.id for case, _, _ in triples if case.status == "CLOSED"]
    decisions: dict[int, str] = {}
    if closed_ids:
        for ad in (
            db.query(m.AnalystDecision)
            .filter(m.AnalystDecision.case_id.in_(closed_ids))
            .order_by(m.AnalystDecision.id.desc())
        ):
            decisions.setdefault(ad.case_id, ad.action_taken)

    return [
        {
            "case_id": case.id, "transaction_id": txn.id, "status": case.status,
            "risk_band": score.risk_band, "hybrid_score": score.hybrid_score,
            "type": txn.type, "amount": txn.amount, "login_country": txn.login_country or "",
            "source": txn.source, "decision": decisions.get(case.id, ""),
            "created_at": case.created_at,
        }
        for case, score, txn in triples
    ]


def _export_auto_clean_rows(
    db: Session, q: str | None,
    *, txn_type, date_from, date_to, amount_min, amount_max, country,
) -> list[dict]:
    query = _auto_clean_base_query(
        db, q, txn_type=txn_type, date_from=date_from, date_to=date_to,
        amount_min=amount_min, amount_max=amount_max, country=country,
    )
    return [
        {
            "case_id": "", "transaction_id": txn.id, "status": "AUTO_CLEAN",
            "risk_band": score.risk_band, "hybrid_score": score.hybrid_score,
            "type": txn.type, "amount": txn.amount, "login_country": txn.login_country or "",
            "source": txn.source, "decision": "",
            "created_at": txn.created_at,
        }
        for score, txn in query.all()
    ]


def _export_rows(
    db: Session, status_upper: str | None, risk_band_upper: str | None, q: str | None,
    *, txn_type, date_from, date_to, amount_min, amount_max, country,
) -> list[dict]:
    kwargs = dict(
        txn_type=txn_type, date_from=date_from, date_to=date_to,
        amount_min=amount_min, amount_max=amount_max, country=country,
    )
    if status_upper == "AUTO_CLEAN":
        return _export_auto_clean_rows(db, q, **kwargs)

    if status_upper is None:
        rows = _export_case_rows(db, None, risk_band_upper, q, **kwargs)
        if risk_band_upper is None or risk_band_upper == "GREEN":
            rows += _export_auto_clean_rows(db, q, **kwargs)
        return rows

    return _export_case_rows(db, status_upper, risk_band_upper, q, **kwargs)


_DEMO_BANNER_TEXT = (
    "⚠ DEMO DATA - Synthetic PaySim-derived dataset, not a real fraud "
    "investigation or compliance record."
)
_RISK_BAND_COLORS = {"RED": "#FDECEA", "GRAY": "#F2F2F2", "GREEN": "#EAF7EE"}
_RISK_CHART_COLORS = {"RED": "#E57373", "GRAY": "#B0B0B0", "GREEN": "#81C784"}


def _build_xlsx_export(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    workbook = xlsxwriter.Workbook(buf, {"in_memory": True})

    fmt_banner = workbook.add_format({
        "bold": True, "font_color": "#B71C1C", "bg_color": "#FDECEA",
        "border": 1, "border_color": "#F5C6CB", "valign": "vcenter", "text_wrap": True,
    })
    fmt_header = workbook.add_format({"bold": True, "bg_color": "#EEEEEE", "border": 1})
    fmt_title = workbook.add_format({"bold": True, "font_size": 13})
    fmt_pct = workbook.add_format({"num_format": "0.0%"})
    fmt_money = workbook.add_format({"num_format": "#,##0.00"})

    col_props = {
        "case_id": {}, "transaction_id": {}, "status": {}, "risk_band": {},
        "hybrid_score": {"num_format": "0.00"}, "type": {},
        "amount": {"num_format": "#,##0.00"}, "login_country": {},
        "source": {}, "decision": {}, "created_at": {"num_format": "yyyy-mm-dd hh:mm"},
    }
    row_formats = {
        band: {
            field: workbook.add_format({**props, "bg_color": bg, "border": 1})
            for field, props in col_props.items()
        }
        for band, bg in _RISK_BAND_COLORS.items()
    }
    fallback_formats = {field: workbook.add_format({**props, "border": 1}) for field, props in col_props.items()}

    ws = workbook.add_worksheet("Cases")
    n_cols = len(_EXPORT_FIELDS)
    ws.merge_range(0, 0, 0, n_cols - 1, _DEMO_BANNER_TEXT, fmt_banner)
    ws.set_row(0, 30)
    for col, field in enumerate(_EXPORT_FIELDS):
        ws.write(1, col, field, fmt_header)
    for col, width in enumerate([10, 14, 10, 10, 12, 12, 16, 13, 9, 16, 18]):
        ws.set_column(col, col, width)

    for i, row in enumerate(rows):
        excel_row = i + 2
        fmts = row_formats.get(row["risk_band"], fallback_formats)
        for col, field in enumerate(_EXPORT_FIELDS):
            value = row[field]
            fmt = fmts[field]
            if field == "created_at":
                ws.write_datetime(excel_row, col, value, fmt)
            elif value == "" or value is None:
                ws.write_blank(excel_row, col, None, fmt)
            else:
                ws.write(excel_row, col, value, fmt)

    if rows:
        ws.autofilter(1, 0, len(rows) + 1, n_cols - 1)
    ws.freeze_panes(2, 0)

    ws2 = workbook.add_worksheet("Summary")
    ws2.merge_range(0, 0, 0, 3, _DEMO_BANNER_TEXT, fmt_banner)
    ws2.set_row(0, 30)
    ws2.set_column(0, 0, 24)
    ws2.set_column(1, 3, 12)

    total = len(rows)
    r = 2
    ws2.write(r, 0, "Filtered Export Summary", fmt_title)
    r += 2
    ws2.write(r, 0, "Total Cases", fmt_header)
    ws2.write(r, 1, total)
    r += 2

    def write_dist_table(title, counter, order=None):
        nonlocal r
        ws2.write(r, 0, title, fmt_title)
        r += 1
        ws2.write(r, 0, "Value", fmt_header)
        ws2.write(r, 1, "Count", fmt_header)
        ws2.write(r, 2, "%", fmt_header)
        r += 1
        first_data_row = r
        for key in order if order else sorted(counter, key=lambda k: -counter[k]):
            count = counter.get(key, 0)
            ws2.write(r, 0, key)
            ws2.write(r, 1, count)
            ws2.write(r, 2, (count / total) if total else 0, fmt_pct)
            r += 1
        last_data_row = r - 1
        r += 1
        return first_data_row, last_data_row

    risk_counter = Counter(row["risk_band"] for row in rows)
    status_counter = Counter(row["status"] for row in rows)
    type_counter = Counter(row["type"] for row in rows)
    country_counter = Counter((row["login_country"] or "Unknown") for row in rows)

    risk_first, risk_last = write_dist_table("Risk Band Distribution", risk_counter, order=["RED", "GRAY", "GREEN"])
    write_dist_table("Status Distribution", status_counter, order=["OPEN", "CLOSED", "AUTO_CLEAN"])
    write_dist_table("Transaction Type Distribution", type_counter)
    write_dist_table("Login Country Distribution", country_counter)

    amounts = [row["amount"] for row in rows]
    ws2.write(r, 0, "Amount Statistics", fmt_title)
    r += 1
    for label, value in [
        ("Min", min(amounts) if amounts else 0),
        ("Max", max(amounts) if amounts else 0),
        ("Average", (sum(amounts) / len(amounts)) if amounts else 0),
        ("Total", sum(amounts)),
    ]:
        ws2.write(r, 0, label, fmt_header)
        ws2.write(r, 1, value, fmt_money)
        r += 1

    if total:
        chart = workbook.add_chart({"type": "pie"})
        chart.add_series({
            "name": "Risk Band Distribution",
            "categories": ["Summary", risk_first, 0, risk_last, 0],
            "values": ["Summary", risk_first, 1, risk_last, 1],
            "points": [{"fill": {"color": _RISK_CHART_COLORS[b]}} for b in ["RED", "GRAY", "GREEN"]],
        })
        chart.set_title({"name": "Risk Band Distribution"})
        chart.set_size({"width": 380, "height": 260})
        ws2.insert_chart(risk_first, 5, chart)

    workbook.close()
    return buf.getvalue()


@app.get("/cases/export")
def export_cases(
    format: str = "csv",
    status: str | None = None,
    risk_band: str | None = None,
    q: str | None = None,
    type: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    amount_min: float | None = None,
    amount_max: float | None = None,
    country: str | None = None,
    db: Session = Depends(get_db),
):
    if format not in ("csv", "xlsx"):
        raise HTTPException(400, f"Unsupported export format: {format!r} (expected 'csv' or 'xlsx')")

    status_upper = status.upper() if status else None
    risk_band_upper = risk_band.upper() if risk_band else None
    txn_type = type.upper() if type else None
    country_upper = country.upper() if country else None

    rows = _export_rows(
        db, status_upper, risk_band_upper, q, txn_type=txn_type, date_from=date_from,
        date_to=date_to, amount_min=amount_min, amount_max=amount_max, country=country_upper,
    )
    rows.sort(key=lambda r: r["hybrid_score"], reverse=True)

    if format == "xlsx":
        return Response(
            content=_build_xlsx_export(rows),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=fraud_cases_demo_export.xlsx"},
        )

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_EXPORT_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "created_at": row["created_at"].isoformat()})

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fraud_cases_demo_export.csv"},
    )


@app.get("/cases/filter-options", response_model=s.CaseFilterOptionsOut)
def get_case_filter_options(db: Session = Depends(get_db)):
    countries = [
        row[0] for row in
        db.query(m.Transaction.login_country)
        .filter(m.Transaction.login_country.isnot(None))
        .distinct()
        .order_by(m.Transaction.login_country)
        .all()
    ]
    return s.CaseFilterOptionsOut(countries=countries)


@app.get("/cases/{case_id}", response_model=s.CaseDetailOut)
def get_case(case_id: int, db: Session = Depends(get_db)):
    case = _get_case_or_404(db, case_id)

    txn = db.get(m.Transaction, case.transaction_id)
    score = _latest_score(db, case.transaction_id)
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

    scaler = precedent.load_precedent_scaler()
    ai2_suggested_decision = None
    precedent_summary = None
    if scaler is not None:
        neighbors = precedent.find_precedents(case, db, scaler, k=precedent.DEFAULT_K)
        precedent_summary = precedent.summarize_precedents(neighbors)
        ai2_suggested_decision = precedent_summary["suggested_decision"]

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
    case = _get_case_or_404(db, case_id)
    if case.status == "CLOSED":
        raise HTTPException(409, "Case is already closed")

    proposal = automation.get_pending_proposal(case_id, db)
    if proposal is None:
        raise HTTPException(404, "No pending automation proposal for this case")

    direction = proposal.triggered_conditions.get("direction")
    if direction != "fraud":
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
    case = _get_case_or_404(db, case_id)
    if case.status == "CLOSED":
        raise HTTPException(409, "Case is already closed")

    proposal = automation.get_pending_proposal(case_id, db)
    if proposal is None:
        raise HTTPException(404, "No pending automation proposal for this case")

    proposal.review_status = "rejected"
    proposal.rejection_reason = payload.rejection_reason
    proposal.reviewed_at = datetime.now(timezone.utc)
    db.commit()

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
    case = _get_case_or_404(db, case_id)

    if case.status != "OPEN":
        return None

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


@app.get("/cases/{case_id}/automation-gates", response_model=s.AutomationGatesOut | None)
def get_case_automation_gates(case_id: int, db: Session = Depends(get_db)):
    case = _get_case_or_404(db, case_id)
    if case.status != "OPEN":
        return None

    policy = automation.get_active_policy(db)
    if policy is None:
        return None

    txn = db.get(m.Transaction, case.transaction_id)
    score = (
        db.query(m.Score)
        .filter(m.Score.transaction_id == case.transaction_id)
        .order_by(m.Score.id.desc())
        .first()
    )

    scaler = precedent.load_precedent_scaler()
    neighbors = precedent.find_precedents(case, db, scaler, k=precedent.DEFAULT_K) if scaler is not None else []
    precedent_summary = precedent.summarize_precedents(neighbors)

    hard_rule_hits = [
        h.rule_name for h in
        db.query(m.RuleHit).filter(m.RuleHit.transaction_id == txn.id, m.RuleHit.rule_type == "hard").all()
    ]
    clean_rule_hits = [
        h.rule_name for h in
        db.query(m.RuleHit).filter(m.RuleHit.transaction_id == txn.id, m.RuleHit.rule_type == "clean").all()
    ]

    result = automation.evaluate_auto_decision(
        precedent_summary, score.calibrated_proba, hard_rule_hits, policy, clean_rule_hits,
    )

    return s.AutomationGatesOut(
        case_id=case.id,
        eligible=result["eligible"],
        direction=result["direction"],
        policy_version=policy.version,
        gates=[s.AutomationGateOut(**g) for g in result["gates"]],
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


@app.get("/cases/{case_id}/report-findings", response_model=s.ReportFindingsOut)
def get_case_report_findings(case_id: int, db: Session = Depends(get_db)):
    case = _get_case_or_404(db, case_id)
    txn = db.get(m.Transaction, case.transaction_id)
    return s.ReportFindingsOut(
        case_id=case.id,
        transaction_summary=findings.txn_summary_text(txn),
        findings=findings.build_findings(case, db),
    )


def _attach_precedent_context(db: Session, neighbors: list[dict]) -> None:
    if not neighbors:
        return
    case_ids = [n["case_id"] for n in neighbors]
    rows = (
        db.query(m.Case.id, m.Transaction.type, m.Transaction.amount, m.Transaction.step_hour,
                  m.Score.risk_band, m.Score.hybrid_score)
        .join(m.Transaction, m.Transaction.id == m.Case.transaction_id)
        .join(m.Score, m.Score.transaction_id == m.Transaction.id)
        .filter(m.Case.id.in_(case_ids))
        .order_by(m.Score.id.desc())
    )
    context_by_case_id: dict[int, dict] = {}
    for row_case_id, txn_type, amount, step_hour, risk_band, hybrid_score in rows:
        context_by_case_id.setdefault(row_case_id, {
            "type": txn_type, "amount": amount, "step_hour": step_hour,
            "risk_band": risk_band, "hybrid_score": hybrid_score,
        })

    rule_rows = (
        db.query(m.Case.id, m.RuleHit.rule_name)
        .join(m.Transaction, m.Transaction.id == m.Case.transaction_id)
        .join(m.RuleHit, m.RuleHit.transaction_id == m.Transaction.id)
        .filter(m.Case.id.in_(case_ids), m.RuleHit.rule_type.in_(["hard", "soft"]))
    )
    rule_hits_by_case_id: dict[int, list[str]] = {}
    for row_case_id, rule_name in rule_rows:
        rule_hits_by_case_id.setdefault(row_case_id, []).append(rule_name)

    decision_rows = (
        db.query(m.AnalystDecision.case_id, m.AnalystDecision.analyst_reason_code)
        .filter(m.AnalystDecision.case_id.in_(case_ids))
        .order_by(m.AnalystDecision.id.desc())
    )
    reason_code_by_case_id: dict[int, str | None] = {}
    for row_case_id, reason_code in decision_rows:
        reason_code_by_case_id.setdefault(row_case_id, reason_code)

    for n in neighbors:
        n.update(context_by_case_id.get(n["case_id"], {}))
        n["rule_hits"] = rule_hits_by_case_id.get(n["case_id"], [])
        n["analyst_reason_code"] = reason_code_by_case_id.get(n["case_id"])


@app.get("/cases/{case_id}/precedents", response_model=s.PrecedentsOut)
def get_case_precedents(case_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    case = _get_case_or_404(db, case_id)

    txn = db.get(m.Transaction, case.transaction_id)
    scaler = precedent.load_precedent_scaler()

    if scaler is None:
        neighbors: list[dict] = []
    else:
        neighbors = precedent.find_precedents(case, db, scaler, k=precedent.DEFAULT_K)

    _attach_precedent_context(db, neighbors)

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


_AUDIT_ACTION_LABELS = {
    "confirm_fraud": "Confirmed as fraud",
    "approve_clean": "Closed as clean",
    "escalate": "Escalated",
    "reopened": "Reopened",
}
_AUDIT_DIRECTION_LABELS = {"fraud": "confirm fraud", "clean": "approve as clean", "escalate": "escalate"}
_RAPID_REDECISION_SECONDS = 60


@app.get("/cases/{case_id}/audit-trail", response_model=s.AuditTrailOut)
def get_case_audit_trail(case_id: int, db: Session = Depends(get_db)):
    case = _get_case_or_404(db, case_id)
    txn = db.get(m.Transaction, case.transaction_id)
    events: list[dict] = []

    score = _latest_score(db, txn.id)
    if score is not None:
        events.append({
            "timestamp": score.created_at, "event_type": "scored", "actor": "System",
            "summary": f"Transaction scored - {score.risk_band} ({score.hybrid_score:.0f}/100)",
            "detail": None, "before": None, "after": None, "anomaly_flags": [],
        })

    events.append({
        "timestamp": case.created_at, "event_type": "case_opened", "actor": "System",
        "summary": "Case opened", "detail": None, "before": None, "after": None, "anomaly_flags": [],
    })

    reports = db.query(m.LlmReport).filter(m.LlmReport.case_id == case.id).order_by(m.LlmReport.id.asc()).all()
    for r in reports:
        detail = "Generated by LLM (Groq)" if r.source and r.source != "fallback" else "Deterministic fallback - no LLM call"
        events.append({
            "timestamp": r.generated_at, "event_type": "report_generated", "actor": "AI",
            "summary": "Investigator Report generated",
            "detail": detail, "before": None, "after": None, "anomaly_flags": [],
        })

    precedent_row = db.query(m.PrecedentIndex).filter(m.PrecedentIndex.case_id == case.id).first()
    if precedent_row is not None:
        events.append({
            "timestamp": precedent_row.created_at, "event_type": "precedent_indexed", "actor": "System",
            "summary": f"Added to precedent pool as {_AUDIT_ACTION_LABELS.get(precedent_row.label, precedent_row.label)}",
            "detail": None, "before": None, "after": None, "anomaly_flags": [],
        })

    proposals = (
        db.query(m.AutoBlockLog)
        .filter(m.AutoBlockLog.case_id == case.id, m.AutoBlockLog.review_status != "shadow")
        .order_by(m.AutoBlockLog.id.asc())
        .all()
    )
    for p in proposals:
        direction = p.triggered_conditions.get("direction")
        direction_label = _AUDIT_DIRECTION_LABELS.get(direction, direction or "unknown")
        events.append({
            "timestamp": p.created_at, "event_type": "automation_proposed", "actor": "AI",
            "summary": f"Automation proposed to {direction_label}",
            "detail": "Rule-based gate evaluation - no LLM involved",
            "before": None, "after": None, "anomaly_flags": [],
        })
        if p.reviewed_at is not None:
            events.append({
                "timestamp": p.reviewed_at, "event_type": "automation_reviewed", "actor": "Analyst",
                "summary": f"Analyst {p.review_status} the automation proposal",
                "detail": p.rejection_reason,
                "before": f"proposed to {direction_label}", "after": p.review_status,
                "anomaly_flags": ["ai_human_conflict"] if p.review_status == "rejected" else [],
            })

    decisions = db.query(m.AnalystDecision).filter(m.AnalystDecision.case_id == case.id).order_by(m.AnalystDecision.id.asc()).all()
    prev_decided_at = None
    for d in decisions:
        is_reopen = d.action_taken == "reopened"
        flags = []
        if d.ai2_suggested_decision is not None and d.ai2_suggested_decision != d.action_taken:
            flags.append("ai_human_conflict")
        if not d.ai_proposed and d.analyst_reason_code is None and d.analyst_note is None:
            flags.append("no_reason")
        if prev_decided_at is not None and (d.decided_at - prev_decided_at).total_seconds() < _RAPID_REDECISION_SECONDS:
            flags.append("rapid_redecision")
        prev_decided_at = d.decided_at

        detail_parts = []
        if d.analyst_reason_code:
            detail_parts.append(d.analyst_reason_code)
        if d.analyst_note:
            detail_parts.append(f'"{d.analyst_note}"')
        if d.ai2_suggested_decision:
            detail_parts.append(f"Precedent Analysis had suggested: {_AUDIT_ACTION_LABELS.get(d.ai2_suggested_decision, d.ai2_suggested_decision)}")

        summary = _AUDIT_ACTION_LABELS.get(d.action_taken, d.action_taken)
        if d.ai_proposed:
            summary += " (automation-proposed, analyst-confirmed)"

        events.append({
            "timestamp": d.decided_at, "event_type": "decision", "actor": "Analyst",
            "summary": summary,
            "detail": " · ".join(detail_parts) if detail_parts else None,
            "before": "CLOSED" if is_reopen else "OPEN",
            "after": "OPEN" if is_reopen else "CLOSED",
            "anomaly_flags": flags,
        })

    events.sort(key=lambda e: e["timestamp"])

    return s.AuditTrailOut(case_id=case.id, events=[s.AuditEventOut(**e) for e in events])


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


@app.get("/metrics/trends", response_model=list[s.TrendPointOut])
def get_metrics_trends(days: int = 30, db: Session = Depends(get_db)):
    days = max(1, min(days, 90))

    rows = (
        db.query(m.Transaction.created_at, m.Score.risk_band, m.Score.hybrid_score)
        .join(m.Score, m.Score.transaction_id == m.Transaction.id)
        .all()
    )
    if not rows:
        return []

    latest_date = max(created_at.date() for created_at, _, _ in rows)
    start_date = latest_date - timedelta(days=days - 1)

    by_day: dict[date, dict] = {}
    for created_at, band, hybrid_score in rows:
        d = created_at.date()
        if d < start_date or d > latest_date:
            continue
        bucket = by_day.setdefault(d, {"case_count": 0, "red_count": 0, "scores": []})
        if band in ("RED", "GRAY"):
            bucket["case_count"] += 1
            if band == "RED":
                bucket["red_count"] += 1
        bucket["scores"].append(hybrid_score)

    points: list[s.TrendPointOut] = []
    for i in range(days):
        d = start_date + timedelta(days=i)
        bucket = by_day.get(d)
        if bucket is None:
            points.append(s.TrendPointOut(date=d.isoformat(), case_count=0, red_rate=None, avg_score=None, scored_count=0))
            continue
        case_count = bucket["case_count"]
        red_rate = round(bucket["red_count"] / case_count, 4) if case_count else None
        scored_count = len(bucket["scores"])
        avg_score = round(sum(bucket["scores"]) / scored_count, 2) if scored_count else None
        points.append(s.TrendPointOut(
            date=d.isoformat(), case_count=case_count, red_rate=red_rate, avg_score=avg_score,
            scored_count=scored_count,
        ))

    return points


@app.get("/automation/status", response_model=s.AutomationStatusOut)
def get_automation_status(db: Session = Depends(get_db)):
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
        gate_bottleneck=automation.gate_bottleneck_stats(db),
    )
