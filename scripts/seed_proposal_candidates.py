"""Adds N transactions matching the case-302 profile - TRANSFER + isFraud=1 +
drain (newbalanceOrig=0) + ghost destination (dest balance unchanged) - to
grow the pool of OPEN RED cases that clear the automation gates and
receive a proposal.

Unlike seed_additional_demo.py, this does not preserve the database's
overall fraud ratio - it targets this specific high-signal profile instead
of random real-world-shaped traffic.

Measured yield on this profile (20-row test run): 20/20 land in RED, ~6/20
also clear similarity + consensus + count gates to become proposals.

Otherwise identical to seed_additional_demo.py: /score path, demo
re-tagging, timestamp distribution, nameOrig exclusion, --delay for Groq
rate limits.

Usage (backend running, from the project root):
    python -m scripts.seed_proposal_candidates --count 100
    python -m scripts.seed_proposal_candidates --count 100 --dry-run
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import timedelta

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import func, select  # noqa: E402

from backend import db_models as m  # noqa: E402
from backend.database import SessionLocal  # noqa: E402

API_URL = "http://127.0.0.1:8000"
RANDOM_STATE = 42


def to_payload(row: pd.Series) -> dict:
    return {
        "step": int(row["step"]), "type": row["type"], "amount": float(row["amount"]),
        "oldbalanceOrg": float(row["oldbalanceOrg"]), "newbalanceOrig": float(row["newbalanceOrig"]),
        "oldbalanceDest": float(row["oldbalanceDest"]), "newbalanceDest": float(row["newbalanceDest"]),
        "nameOrig": str(row["nameOrig"]), "nameDest": str(row["nameDest"]),
        "device_id": str(row["device_id"]), "is_known_device": int(row["is_known_device"]),
        "login_country": str(row["login_country"]), "geo_velocity_flag": int(row["geo_velocity_flag"]),
        "channel": str(row["channel"]), "isFraud": int(row["isFraud"]),
    }


def pick_candidates(count: int) -> pd.DataFrame:
    """The case-302 profile filter is the whole point of this script - see
    module docstring. Rows this script has already seeded are excluded via
    name_orig, same discipline as seed_additional_demo.py."""
    df = pd.read_parquet("data/test.parquet")

    db = SessionLocal()
    try:
        seeded = {n for (n,) in db.execute(
            select(m.Transaction.name_orig).where(m.Transaction.name_orig.isnot(None))
        ).all()}
    finally:
        db.close()

    pool = df[
        (df["type"] == "TRANSFER") & (df["isFraud"] == 1)
        & (df["newbalanceOrig"] == 0)
        & (df["oldbalanceDest"] == df["newbalanceDest"])
        & (~df["nameOrig"].isin(seeded))
    ]
    if len(pool) < count:
        raise SystemExit(
            f"Only {len(pool)} unseen rows match the case-302 profile - asked for {count}."
        )

    print(f"  already seeded      : {len(seeded)} nameOrig values")
    print(f"  profile pool (unseen): {len(pool)} candidates")
    print(f"  drawing             : {count}")

    return pool.sample(n=count, random_state=RANDOM_STATE).reset_index(drop=True)


def mark_as_demo(after_id: int) -> int:
    """Same reason as seed_additional_demo.mark_as_demo - /score writes
    source="live", we re-tag afterward to keep the Overview KPI honest."""
    db = SessionLocal()
    try:
        rows = db.execute(
            select(m.Transaction).where(m.Transaction.id > after_id)
        ).scalars().all()
        for txn in rows:
            txn.source = "demo"
            txn.is_demo = True
        db.commit()
        return len(rows)
    finally:
        db.close()


def distribute_new_timestamps(after_id: int) -> None:
    """Identical to seed_additional_demo - places new rows within the
    existing demo window instead of stacking on today. See that script's
    docstring for why distribute_demo_timestamps.py cannot be re-run."""
    db = SessionLocal()
    try:
        window = db.execute(
            select(func.min(m.Transaction.created_at), func.max(m.Transaction.created_at))
            .where(m.Transaction.is_demo == 1, m.Transaction.id <= after_id)
        ).one()
        start, end = window
        if start is None or end is None:
            print("  no existing window to align to - leaving timestamps as written")
            return

        new_rows = db.execute(
            select(m.Transaction).where(m.Transaction.id > after_id).order_by(m.Transaction.id)
        ).scalars().all()
        if not new_rows:
            return

        steps = [t.step for t in new_rows]
        step_min, step_max = min(steps), max(steps)
        step_range = (step_max - step_min) or 1
        span_days = max((end - start).days, 1)
        rng = random.Random(RANDOM_STATE)

        for txn in new_rows:
            day_offset = round((txn.step - step_min) / step_range * span_days)
            instant = (
                start
                + timedelta(days=day_offset)
                + timedelta(hours=txn.step % 24, seconds=rng.randint(0, 3599))
            )
            if instant > end:
                instant = end - timedelta(minutes=rng.randint(1, 240))
            txn.created_at = instant

            for score in db.query(m.Score).filter(m.Score.transaction_id == txn.id).all():
                score.created_at = instant
            case = db.query(m.Case).filter(m.Case.transaction_id == txn.id).first()
            if case is None:
                continue
            case.created_at = instant
            for report in db.query(m.LlmReport).filter(m.LlmReport.case_id == case.id).all():
                report.generated_at = instant + timedelta(
                    minutes=rng.randint(1, 9), seconds=rng.randint(0, 59)
                )
            for log in db.query(m.AutoBlockLog).filter(m.AutoBlockLog.case_id == case.id).all():
                log.created_at = instant

        db.commit()
        print(f"  timestamps distributed across {start:%Y-%m-%d} .. {end:%Y-%m-%d}")
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="how many case-302-profile rows to add")
    parser.add_argument("--delay", type=float, default=6.5, help="seconds between /score calls (Groq rate-limit safety)")
    parser.add_argument("--dry-run", action="store_true", help="sample and report, send nothing")
    args = parser.parse_args()

    sample = pick_candidates(args.count)

    if args.dry_run:
        print(f"\nDry run - {len(sample)} rows selected, nothing sent.")
        print(sample[["step", "type", "amount", "isFraud"]].head().to_string(index=False))
        return

    db = SessionLocal()
    try:
        max_id_before = db.execute(select(func.max(m.Transaction.id))).scalar() or 0
    finally:
        db.close()

    ok = failed = 0
    t0 = time.time()
    for i, row in sample.iterrows():
        try:
            requests.post(f"{API_URL}/score", json=to_payload(row), timeout=15).raise_for_status()
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"  [ERROR] row {i}: {exc}")
        if args.delay:
            time.sleep(args.delay)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)} sent ({time.time() - t0:.1f}s)")

    print(f"\nsent: {ok} ok, {failed} failed, {time.time() - t0:.1f}s")
    print("Background Groq reports may still be generating - wait before checking counts.")
    print(f"  tagged as demo      : {mark_as_demo(max_id_before)} rows")
    distribute_new_timestamps(max_id_before)


if __name__ == "__main__":
    main()
