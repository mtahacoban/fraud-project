"""Recompute hybrid_score for existing rows after a scoring-formula change.

Hard-rule rows are recomputed via rule_engine.compute_hybrid_score();
high-confidence-override rows via scoring.py's promotion logic. Plain rows
(no hard rule, no promotion) are untouched. Recomputes from the stored
ml_score/rule_score rather than re-running the model - inference is
deterministic, so a reload would just reproduce the same value at higher
cost, and if it didn't, that would be model drift worth surfacing.

Idempotent: a second run reports 0 changes.

Safety: dry-run by default. --apply backs up the database first and
refuses to write if any row's risk_band would change (this script should
only redistribute scores within RED).

Usage:
    python -m scripts.rescore_hybrid            # dry run, prints the diff
    python -m scripts.rescore_hybrid --apply    # backs up, then writes
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import select

from backend import db_models as m
from backend.database import SessionLocal, engine
from backend.rule_engine import _THRESHOLD_RED, compute_hybrid_score
from backend.scoring import OVERRIDE_SPREAD


def _db_path() -> str | None:
    url = str(engine.url)
    return url.split("///")[-1] if url.startswith("sqlite") else None


def rescore(apply: bool = False) -> int:
    db = SessionLocal()
    try:
        hard_txn_ids = {
            row[0] for row in db.execute(
                select(m.RuleHit.transaction_id).where(m.RuleHit.rule_type == "hard")
            ).all()
        }
        scores = db.execute(select(m.Score)).scalars().all()

        def recompute(s: m.Score) -> dict | None:
            """Returns the row's new {hybrid_score, risk_band}, or None if
            this row's formula did not change."""
            if s.transaction_id in hard_txn_ids:
                return compute_hybrid_score(
                    ml_proba=s.ml_score / 100.0,
                    hard_rule_flag=True,
                    soft_score=s.rule_score,
                )
            if s.band_reason == "high_confidence_override":
                base = compute_hybrid_score(
                    ml_proba=s.ml_score / 100.0,
                    hard_rule_flag=False,
                    soft_score=s.rule_score,
                )
                return {
                    "hybrid_score": round(
                        _THRESHOLD_RED
                        + OVERRIDE_SPREAD * (base["hybrid_score"] / _THRESHOLD_RED),
                        2,
                    ),
                    "risk_band": "RED",
                }
            return None

        changes: list[tuple[int, float, float]] = []
        band_changes: list[tuple[int, str, str]] = []
        for s in scores:
            new = recompute(s)
            if new is None:
                continue
            if abs(new["hybrid_score"] - s.hybrid_score) > 0.005:
                changes.append((s.id, s.hybrid_score, new["hybrid_score"]))
            if new["risk_band"] != s.risk_band:
                band_changes.append((s.id, s.risk_band, new["risk_band"]))

        print(f"scores total        : {len(scores)}")
        print(f"hard-rule rows      : {sum(1 for s in scores if s.transaction_id in hard_txn_ids)}")
        print(f"override rows       : {sum(1 for s in scores if s.band_reason == 'high_confidence_override')}")
        print(f"hybrid_score changes: {len(changes)}")
        print(f"risk_band changes   : {len(band_changes)}")
        if changes[:5]:
            print("sample:")
            for sid, old, new_v in changes[:5]:
                print(f"  score#{sid}: {old} -> {new_v}")

        if band_changes:
            print("\nABORT: risk_band would change - this script only redistributes")
            print("scores within RED. Review the formula change before rescoring.")
            return 1

        if not apply:
            print("\nDry run - nothing written. Re-run with --apply to persist.")
            return 0

        path = _db_path()
        if path:
            backup = f"{path}.bak-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
            shutil.copy2(path, backup)
            print(f"\nbackup written: {backup}")
        else:
            print("\nWARNING: not a local SQLite file - no automatic backup made.")

        by_id = {sid: new_v for sid, _, new_v in changes}
        for s in scores:
            if s.id in by_id:
                s.hybrid_score = by_id[s.id]
        db.commit()
        print(f"applied: {len(changes)} rows updated")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    raise SystemExit(rescore(apply=parser.parse_args().apply))
