from __future__ import annotations

import threading

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from backend import db_models as m
from backend import precedent
from backend.database import SessionLocal

# Separate from report_worker's set/lock — an independent background job
# (precedent explanation vs. the SHAP/rules report), same pattern:
# process-local guard against generating the same thing twice at once.
_generating_case_ids: set[int] = set()
_lock = threading.Lock()


def generate_and_store_precedent_explanation(case_id: int) -> None:
    """Runs in a FastAPI BackgroundTask, after the triggering response has
    already been sent — opens its own db session, same reasoning as
    report_worker.generate_and_store_report(). Recomputes find_precedents()
    from scratch (cheap, deterministic) rather than trusting a summary
    passed in from before the background task was scheduled, in case the
    pool changed in between."""
    db = SessionLocal()
    try:
        case = db.get(m.Case, case_id)
        if case is None:
            return

        scaler = precedent.load_precedent_scaler()
        if scaler is None:
            return

        neighbors = precedent.find_precedents(case, db, scaler, k=precedent.DEFAULT_K)
        summary = precedent.summarize_precedents(neighbors)
        if summary["suggested_decision"] is None:
            return  # nothing to explain — this path shouldn't have been scheduled, but stay safe

        txn = db.get(m.Transaction, case.transaction_id)
        result = precedent.explain_precedents(summary, txn)

        db.add(m.PrecedentExplanation(
            case_id=case.id,
            explanation_text=result["text"],
            source=result["source"],
            precedent_count=summary["precedent_count"],
            suggested_decision=summary["suggested_decision"],
            pool_size_at_generation=db.query(m.PrecedentIndex).count(),
        ))
        db.commit()
    finally:
        db.close()
        with _lock:
            _generating_case_ids.discard(case_id)


def ensure_precedent_explanation(
    case_id: int, summary: dict, background_tasks: BackgroundTasks, db: Session,
) -> dict:
    """Only called when summary["suggested_decision"] is not None — the
    no-suggestion path is instant and doesn't need caching or a background
    task at all (see backend/precedent.explain_precedents).

    Returns {"status": "ready"|"generating", "text": str|None, "source": str|None}.
    A cached explanation is reused only if precedent_index hasn't grown
    since it was generated (pool_size_at_generation still matches the
    current total row count) — see PrecedentExplanation's docstring in
    db_models.py for why pool size, not this case's own aggregate numbers,
    is the invalidation key. Any decision anywhere invalidates every
    cached explanation; the next view regenerates it."""
    existing = (
        db.query(m.PrecedentExplanation)
        .filter(m.PrecedentExplanation.case_id == case_id)
        .order_by(m.PrecedentExplanation.id.desc())
        .first()
    )
    current_pool_size = db.query(m.PrecedentIndex).count()
    if existing is not None and existing.pool_size_at_generation == current_pool_size:
        return {"status": "ready", "text": existing.explanation_text, "source": existing.source}

    with _lock:
        if case_id in _generating_case_ids:
            return {"status": "generating", "text": None, "source": None}
        _generating_case_ids.add(case_id)

    background_tasks.add_task(generate_and_store_precedent_explanation, case_id)
    return {"status": "generating", "text": None, "source": None}
