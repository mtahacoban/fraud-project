"""
Backfills precedent_index from every currently-decided case (case.status ==
"CLOSED") — build_case_vector() + the case's latest analyst decision as the
label. Fits the precedent feature scaler (models/precedent_scaler.pkl) on
this same backfill population first, since a fresh scaler needs data to
fit on; every later vectorization (a brand-new case at query time, a case
added to the index right after a fresh decision) reuses these exact
parameters.

Idempotent: add_to_precedent_index() (backend/precedent.py) skips any
case_id already present in precedent_index, so re-running only adds cases
decided since the last run — this is also what happens automatically going
forward, via the same function wired into the decision endpoint.

Talks to the database directly (like clear_demo_data.py) — does not need
the backend server running.

Usage (from the project root, with venv):
    venv/Scripts/python.exe scripts/backfill_precedents.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.database import SessionLocal
from backend import db_models as m
from backend.precedent import (
    _raw_case_vector, add_to_precedent_index, fit_and_save_scaler, load_precedent_scaler,
)


def main() -> None:
    db = SessionLocal()

    closed_cases = db.query(m.Case).filter(m.Case.status == "CLOSED").all()
    print(f"Decided (CLOSED) cases available: {len(closed_cases)}")

    if not closed_cases:
        print("Nothing to backfill yet — precedent_index stays empty (cold start).")
        db.close()
        return

    already_indexed = {
        row.case_id for row in db.query(m.PrecedentIndex.case_id).all()
    }
    to_index = [c for c in closed_cases if c.id not in already_indexed]
    print(f"Already in precedent_index: {len(already_indexed)}; new to backfill: {len(to_index)}")

    if not to_index:
        print("Nothing new — precedent_index is already up to date.")
        db.close()
        return

    scaler = load_precedent_scaler()
    if scaler is None:
        print("No scaler yet — fitting a new one on this backfill population...")
        raw_vectors = []
        for c in to_index:
            txn = db.get(m.Transaction, c.transaction_id)
            score = (
                db.query(m.Score)
                .filter(m.Score.transaction_id == c.transaction_id)
                .order_by(m.Score.id.desc())
                .first()
            )
            raw_vectors.append(_raw_case_vector(txn, score.risk_band))
        scaler = fit_and_save_scaler(np.array(raw_vectors))
        print(f"Scaler fit on {len(raw_vectors)} cases, saved to models/precedent_scaler.pkl")

    added = 0
    labels: Counter[str] = Counter()
    for c in to_index:
        row = add_to_precedent_index(c, db, scaler)
        if row is not None:
            added += 1
            labels[row.label] += 1
    db.commit()

    print(f"\nBackfilled {added} precedents.")
    print("Label distribution:", dict(labels))

    example = db.query(m.PrecedentIndex).order_by(m.PrecedentIndex.id.desc()).first()
    if example is not None:
        print(f"\nExample precedent (case_id={example.case_id}, label={example.label!r}):")
        print(" ", [round(v, 4) for v in example.feature_vector])

    db.close()


if __name__ == "__main__":
    main()
