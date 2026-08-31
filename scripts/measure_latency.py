"""
Measures POST /score end-to-end latency (feature prep + XGBoost + rule
engine + SHAP + calibration + DB write) against a running backend. Only
TRANSFER/CASH_OUT rows are sampled so every request exercises the *full*
pipeline - PAYMENT/CASH_IN/DEBIT take the fast-path GREEN shortcut
(backend/scoring.py: is_model_relevant()) and would understate the number
this is meant to report.

LLM report generation runs in a background task and is deliberately NOT
part of this measurement - /score never waits on it, so timing the HTTP
round-trip already excludes it.

All sampled transactions are tagged (nameOrig="LATENCY_TEST_<i>") and
deleted, along with everything they produced (cases, scores, SHAP rows,
rule hits, reports, precedent/automation artifacts), at the end of the
run - this script leaves the database exactly as it found it.

Usage (with the backend running, from the project root):
    venv/Scripts/python.exe scripts/measure_latency.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API_URL = "http://127.0.0.1:8000"
RANDOM_STATE = 42
N_WARMUP = 10
N_MEASURED = 150
NAME_ORIG_TAG = "LATENCY_TEST"
POST_RUN_SETTLE_SECONDS = 8  # let background report/automation tasks finish before cleanup


def to_payload(row: pd.Series, tag: str) -> dict:
    return {
        "step": int(row["step"]),
        "type": row["type"],
        "amount": float(row["amount"]),
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
        "nameOrig": tag,
        "nameDest": str(row["nameDest"]),
        "device_id": str(row["device_id"]),
        "is_known_device": int(row["is_known_device"]),
        "login_country": str(row["login_country"]),
        "geo_velocity_flag": int(row["geo_velocity_flag"]),
        "channel": str(row["channel"]),
    }


def sample_rows(n: int) -> pd.DataFrame:
    df = pd.read_parquet("data/test.parquet")
    risky = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    return risky.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)


def cleanup(txn_ids: list[int]) -> None:
    from backend.database import SessionLocal
    from backend import db_models as m

    if not txn_ids:
        return

    print(f"\nWaiting {POST_RUN_SETTLE_SECONDS}s for background tasks to settle before cleanup...")
    time.sleep(POST_RUN_SETTLE_SECONDS)

    db = SessionLocal()
    try:
        case_ids = [
            c.id for c in
            db.query(m.Case).filter(m.Case.transaction_id.in_(txn_ids)).all()
        ]
        if case_ids:
            db.query(m.LlmReport).filter(m.LlmReport.case_id.in_(case_ids)).delete(synchronize_session=False)
            db.query(m.AutoBlockLog).filter(m.AutoBlockLog.case_id.in_(case_ids)).delete(synchronize_session=False)
            db.query(m.AnalystDecision).filter(m.AnalystDecision.case_id.in_(case_ids)).delete(synchronize_session=False)
        db.query(m.ShapExplanation).filter(m.ShapExplanation.transaction_id.in_(txn_ids)).delete(synchronize_session=False)
        db.query(m.RuleHit).filter(m.RuleHit.transaction_id.in_(txn_ids)).delete(synchronize_session=False)
        db.query(m.Score).filter(m.Score.transaction_id.in_(txn_ids)).delete(synchronize_session=False)
        if case_ids:
            db.query(m.Case).filter(m.Case.id.in_(case_ids)).delete(synchronize_session=False)
        db.query(m.Transaction).filter(m.Transaction.id.in_(txn_ids)).delete(synchronize_session=False)
        db.commit()
        print(f"Cleaned up {len(txn_ids)} test transaction(s), {len(case_ids)} test case(s).")
    finally:
        db.close()


def main() -> None:
    total_needed = N_WARMUP + N_MEASURED
    rows = sample_rows(total_needed)

    txn_ids: list[int] = []
    band_counts: dict[str, int] = {}

    print(f"Warming up ({N_WARMUP} requests, not measured)...")
    for i in range(N_WARMUP):
        payload = to_payload(rows.iloc[i], f"{NAME_ORIG_TAG}_warmup_{i}")
        resp = requests.post(f"{API_URL}/score", json=payload, timeout=30)
        resp.raise_for_status()
        txn_ids.append(resp.json()["txn_id"])

    print(f"Measuring {N_MEASURED} requests...")
    latencies_ms: list[float] = []
    for i in range(N_MEASURED):
        payload = to_payload(rows.iloc[N_WARMUP + i], f"{NAME_ORIG_TAG}_{i}")
        start = time.perf_counter()
        resp = requests.post(f"{API_URL}/score", json=payload, timeout=30)
        elapsed_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        body = resp.json()
        txn_ids.append(body["txn_id"])
        latencies_ms.append(elapsed_ms)
        band_counts[body["risk_band"]] = band_counts.get(body["risk_band"], 0) + 1

    latencies_ms.sort()
    avg = statistics.mean(latencies_ms)
    median = statistics.median(latencies_ms)
    p95 = latencies_ms[int(len(latencies_ms) * 0.95) - 1]
    p99 = latencies_ms[int(len(latencies_ms) * 0.99) - 1]

    print("\n=== POST /score latency (N={}, TRANSFER/CASH_OUT only, full pipeline) ===".format(N_MEASURED))
    print(f"  avg:    {avg:7.2f} ms")
    print(f"  median: {median:7.2f} ms")
    print(f"  p95:    {p95:7.2f} ms")
    print(f"  p99:    {p99:7.2f} ms")
    print(f"  min:    {latencies_ms[0]:7.2f} ms")
    print(f"  max:    {latencies_ms[-1]:7.2f} ms")
    print(f"  risk_band distribution: {band_counts}")

    cleanup(txn_ids)


if __name__ == "__main__":
    main()
