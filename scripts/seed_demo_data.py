"""
Samples the held-out test set and sends it to the running backend via
POST /score to fill the dashboard with realistic data.

Usage (with the backend running, from the project root):
    venv/Scripts/python.exe scripts/seed_demo_data.py
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

API_URL = "http://127.0.0.1:8000"
RANDOM_STATE = 42
N_FRAUD = 30
N_CLEAN = 970
N_SHOWCASE_OVERRIDE = 5  # to make high_confidence_override visible on the dashboard


def to_payload(row: pd.Series) -> dict:
    return {
        "step": int(row["step"]),
        "type": row["type"],
        "amount": float(row["amount"]),
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
        "nameOrig": str(row["nameOrig"]),
        "nameDest": str(row["nameDest"]),
        "device_id": str(row["device_id"]),
        "is_known_device": int(row["is_known_device"]),
        "login_country": str(row["login_country"]),
        "geo_velocity_flag": int(row["geo_velocity_flag"]),
        "channel": str(row["channel"]),
        "isFraud": int(row["isFraud"]),
    }


def find_override_showcase(df: pd.DataFrame, n: int) -> pd.DataFrame:
    # high_confidence_override rarely shows up in a sample at the natural fraud rate;
    # to make the mechanism visible on the dashboard, we deliberately pick eligible
    # rows (scoring still goes through the real /score path).
    from backend.explain import ML_FEATURES
    from backend.rule_engine import check_hard_rules
    from backend.scoring import HIGH_CONFIDENCE_THRESHOLD, load_scoring_engine

    engine = load_scoring_engine()
    fraud_rows = df[df["isFraud"] == 1].copy()
    X = fraud_rows[ML_FEATURES].values.astype(float)
    raw_proba = engine.xgb_model.predict_proba(X)[:, 1]
    calibrated = engine.calibrator.predict_proba(raw_proba)
    hard_flag = [
        check_hard_rules({
            "type": r.type, "oldbalanceDest": r.oldbalanceDest, "newbalanceDest": r.newbalanceDest,
        })["hard_rule_flag"]
        for r in fraud_rows.itertuples()
    ]
    fraud_rows = fraud_rows.assign(calibrated_proba=calibrated, hard_rule_flag=hard_flag)
    eligible = fraud_rows[(fraud_rows["calibrated_proba"] >= HIGH_CONFIDENCE_THRESHOLD) & (~fraud_rows["hard_rule_flag"])]
    print(f"  Found {len(eligible)} real fraud rows eligible for override (in the test.parquet fraud subset)")
    return eligible.drop(columns=["calibrated_proba", "hard_rule_flag"]).sample(
        n=min(n, len(eligible)), random_state=RANDOM_STATE
    )


def main() -> None:
    df = pd.read_parquet("data/test.parquet")

    fraud = df[df["isFraud"] == 1]
    clean = df[df["isFraud"] == 0]
    fraud_sample = fraud.sample(n=min(N_FRAUD, len(fraud)), random_state=RANDOM_STATE)
    clean_sample = clean.sample(n=min(N_CLEAN, len(clean)), random_state=RANDOM_STATE)

    print("Selecting the high-confidence override showcase...")
    showcase = find_override_showcase(df, N_SHOWCASE_OVERRIDE)

    sample = (
        pd.concat([fraud_sample, clean_sample, showcase])
        .drop_duplicates()
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    print(f"Sending {len(sample)} transactions ({len(fraud_sample)} random fraud, "
          f"{len(clean_sample)} clean, {len(showcase)} high-confidence showcase)...")

    ok, failed = 0, 0
    t0 = time.time()
    for i, row in sample.iterrows():
        try:
            res = requests.post(f"{API_URL}/score", json=to_payload(row), timeout=10)
            res.raise_for_status()
            ok += 1
        except Exception as e:
            failed += 1
            print(f"  [ERROR] row {i}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(sample)} processed ({time.time() - t0:.1f}s)")

    print(f"Done: {ok} succeeded, {failed} failed, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
