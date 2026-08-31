"""Adds more demo transactions to an already-seeded database.

seed_demo_data.py samples with a fixed RANDOM_STATE, so re-running it
would select the same rows again. This script excludes every nameOrig
already present and samples only from what's left, so it can be run
repeatedly without duplicating transactions.

Scores through the real POST /score path (backend must be running); ~70%
of sampled rows open a case with a real background Groq report - budget
accordingly.

New rows would otherwise all land on today's date. Since
distribute_demo_timestamps.py can't be safely re-run, this script instead
places only its own new rows across the existing demo window, using the
same step-derived ordering and offsets.

Usage (backend running, from the project root):
    python -m scripts.seed_additional_demo --count 136
    python -m scripts.seed_additional_demo --count 136 --dry-run
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import func, select  # noqa: E402

from backend import db_models as m  # noqa: E402
from backend.database import SessionLocal  # noqa: E402

API_URL = "http://127.0.0.1:8000"
RANDOM_STATE = 42
PAYLOAD_COLUMNS = (
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "nameOrig", "nameDest",
    "device_id", "is_known_device", "login_country", "geo_velocity_flag",
    "channel", "isFraud",
)


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


def pick_rows(count: int) -> pd.DataFrame:
    """Samples `count` unseen rows, preserving the fraud ratio already in
    the database rather than test.parquet's natural ~0.3% - the existing
    demo set is deliberately fraud-enriched, and drifting that ratio would
    silently change what every rate on the dashboard means."""
    df = pd.read_parquet("data/test.parquet")

    db = SessionLocal()
    try:
        seeded = {n for (n,) in db.execute(
            select(m.Transaction.name_orig).where(m.Transaction.name_orig.isnot(None))
        ).all()}
        total, frauds = db.execute(
            select(func.count(m.Transaction.id), func.sum(m.Transaction.is_fraud))
        ).one()
    finally:
        db.close()

    fraud_ratio = (frauds or 0) / total if total else 0.0
    n_fraud = round(count * fraud_ratio)
    n_clean = count - n_fraud

    unseen = df[~df["nameOrig"].isin(seeded)]
    fraud_pool = unseen[unseen["isFraud"] == 1]
    clean_pool = unseen[unseen["isFraud"] == 0]
    if len(fraud_pool) < n_fraud or len(clean_pool) < n_clean:
        raise SystemExit(
            f"Not enough unseen rows: need {n_fraud} fraud / {n_clean} clean, "
            f"pool has {len(fraud_pool)} / {len(clean_pool)}"
        )

    print(f"  already seeded      : {len(seeded)} nameOrig values")
    print(f"  current fraud ratio : {fraud_ratio:.1%} ({frauds}/{total})")
    print(f"  sampling            : {n_fraud} fraud + {n_clean} clean = {count}")

    return (
        pd.concat([
            fraud_pool.sample(n=n_fraud, random_state=RANDOM_STATE),
            clean_pool.sample(n=n_clean, random_state=RANDOM_STATE),
        ])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def mark_as_demo(after_id: int) -> int:
    """POST /score writes source="live", is_demo=False - the defaults in
    _write_score(), correct for a real incoming transaction and wrong for
    seeded ones. Left as-is, the Overview KPI's "N live" caption would
    claim this dataset contains live traffic it does not have. The rows
    are re-tagged here, after scoring, so nothing about the scoring path
    itself is special-cased for seeding."""
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
    """Spreads only the rows this run created across the window the
    existing demo data already occupies. Mirrors
    distribute_demo_timestamps.py: step drives the day, step % 24 the
    hour, with the same deterministic case/report offsets."""
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
                continue  # GREEN, auto-cleared - no case row to move
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
    parser.add_argument("--count", type=int, default=136, help="how many new transactions to add")
    parser.add_argument("--dry-run", action="store_true", help="sample and report, send nothing")
    parser.add_argument(
        "--delay", type=float, default=6.5,
        help="seconds between POST /score calls (default matches seed_demo_cases.py's own "
             "DEFAULT_DELAY_SECONDS - llm_service._generate_groq() catches a rate-limit error "
             "exactly like any other failure and falls back silently, no retry; sending this "
             "many requests with no delay risks quietly degrading most of the ~95 case-opening "
             "rows to source=\"fallback\" instead of a real Groq report)",
    )
    args = parser.parse_args()

    sample = pick_rows(args.count)

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
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)} sent ({time.time() - t0:.1f}s)")
        if args.delay:
            time.sleep(args.delay)

    print(f"\nsent: {ok} ok, {failed} failed, {time.time() - t0:.1f}s")
    print("Background Groq reports may still be generating - wait before checking counts.")
    print(f"  tagged as demo      : {mark_as_demo(max_id_before)} rows")
    distribute_new_timestamps(max_id_before)


if __name__ == "__main__":
    main()
