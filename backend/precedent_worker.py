from __future__ import annotations

import threading

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from backend import db_models as m
from backend import precedent
from backend.database import SessionLocal

_generating_case_ids: set[int] = set()
_lock = threading.Lock()


def generate_and_store_precedent_explanation(case_id: int) -> None:
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
            return

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
