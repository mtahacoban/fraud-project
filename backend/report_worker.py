from __future__ import annotations

import threading

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from backend import db_models as m
from backend import llm_service
from backend.config import settings
from backend.database import SessionLocal
from backend.findings import build_findings, txn_summary_text

_generating_case_ids: set[int] = set()
_lock = threading.Lock()


def generate_and_store_report(case_id: int) -> None:
    db = SessionLocal()
    try:
        already = db.query(m.LlmReport).filter(m.LlmReport.case_id == case_id).first()
        if already is not None:
            return

        case = db.get(m.Case, case_id)
        if case is None:
            return

        findings = build_findings(case, db)
        txn = db.get(m.Transaction, case.transaction_id)
        result = llm_service.generate_report(findings, txn_summary_text(txn))

        model_name = settings.llm_model if result["source"] != "fallback" else "fallback"
        db.add(m.LlmReport(
            case_id=case.id,
            report_text=result["text"],
            model_name=model_name,
            source=result["source"],
        ))
        db.commit()
    finally:
        db.close()
        with _lock:
            _generating_case_ids.discard(case_id)


def ensure_report_generation(case_id: int, background_tasks: BackgroundTasks, db: Session) -> str:
    existing = db.query(m.LlmReport).filter(m.LlmReport.case_id == case_id).first()
    if existing is not None:
        return "ready"

    with _lock:
        if case_id in _generating_case_ids:
            return "generating"
        _generating_case_ids.add(case_id)

    background_tasks.add_task(generate_and_store_report, case_id)
    return "generating"
