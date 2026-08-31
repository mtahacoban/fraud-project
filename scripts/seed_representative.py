"""
Feeds the dashboard with the full PaySim dataset's real transaction-type
distribution (alongside seed_demo_data.py, not instead of it). TRANSFER/CASH_OUT
is sampled only from the held-out test.parquet (no leakage); other types are
sampled from the raw CSV and pass through the fast-path type gate as auto-GREEN.
isFraud/nameOrig/nameDest are deliberately not sent - this simulates real-time
traffic.

Usage (with the backend running, from the project root):
    venv/Scripts/python.exe scripts/seed_representative.py
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API_URL = "http://127.0.0.1:8000/score"
FULL_CSV = "data/PS_20174392719_1491204439457_log.csv"
TEST_PARQUET = "data/test.parquet"
N_TOTAL = 5000
SEED = 42
RISKY_TYPES = {"TRANSFER", "CASH_OUT"}
RAW_COLS = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "isFraud",
]


def to_payload(row: pd.Series) -> dict:
    payload = {
        "step": int(row["step"]),
        "type": row["type"],
        "amount": float(row["amount"]),
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
    }
    # test.parquet rows (RISKY_TYPES) carry synthetic fields; rows from the raw
    # CSV (PAYMENT/CASH_IN/DEBIT) don't have these columns - skip them.
    device_id = row.get("device_id")
    if device_id is not None and not pd.isna(device_id):
        payload["device_id"] = str(device_id)
        payload["is_known_device"] = int(row["is_known_device"])
        payload["login_country"] = str(row["login_country"])
        payload["geo_velocity_flag"] = int(row["geo_velocity_flag"])
        payload["channel"] = str(row["channel"])
    return payload


def main() -> None:
    print(f"Loading raw data: {FULL_CSV} ...")
    t0 = time.time()
    full_df = pd.read_csv(FULL_CSV, usecols=RAW_COLS)
    print(f"  {len(full_df):,} rows loaded ({time.time() - t0:.1f}s)")

    type_share = full_df["type"].value_counts(normalize=True)
    print("\nFull data type distribution:")
    print((type_share * 100).round(1).to_string())

    counts = (type_share * N_TOTAL).round().astype(int).to_dict()

    # Risky types (TRANSFER/CASH_OUT): held-out test set only - leak-free.
    test_df = pd.read_parquet(TEST_PARQUET)
    risky_needed = sum(counts.get(t, 0) for t in RISKY_TYPES)
    risky_pool = test_df[test_df["type"].isin(RISKY_TYPES)]
    risky_sample = risky_pool.sample(n=min(risky_needed, len(risky_pool)), random_state=SEED)

    # Other types: from the raw CSV, proportional by type - these will pass
    # through the fast path, the model will never see them.
    other_samples = []
    for t, n in counts.items():
        if t in RISKY_TYPES:
            continue
        pool = full_df[full_df["type"] == t]
        other_samples.append(pool.sample(n=min(n, len(pool)), random_state=SEED))

    seed_df = pd.concat([risky_sample, *other_samples], ignore_index=True)
    seed_df = seed_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print(f"\nTotal seed: {len(seed_df)} | fraud (informational, not sent): {int(seed_df['isFraud'].sum())}")
    print("Seed type distribution:")
    print((seed_df["type"].value_counts(normalize=True) * 100).round(1).to_string())

    print(f"\nSending {len(seed_df)} transactions to /score...")
    ok, failed = 0, 0
    t0 = time.time()
    for i, row in seed_df.iterrows():
        try:
            res = requests.post(API_URL, json=to_payload(row), timeout=10)
            res.raise_for_status()
            ok += 1
        except Exception as e:
            failed += 1
            print(f"  [ERROR] row {i} ({row['type']}): {e}")
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(seed_df)} processed ({time.time() - t0:.1f}s)")

    print(f"\nDone: {ok} succeeded, {failed} failed, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
