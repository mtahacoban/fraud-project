"""
Cross-tabulation of the real isFraud label × the risk_band assigned by the backend.

MODE = "operational"  → looks up the results seed_representative.py wrote to
    the DB by natural key and cross-tabulates them. Small N — sanity check only.

MODE = "offline_full" → scores the entire test.parquet offline with the same
    model + rule engine as the backend and runs a serving-parity comparison
    against the test_metrics saved by train_models.py.

Usage (from the project root, with venv):
    venv/Scripts/python.exe scripts/analyze_confusion.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODE = "offline_full"  # "operational" | "offline_full"


def print_confusion(df: pd.DataFrame, label_col: str, band_col: str, title: str) -> None:
    ct = pd.crosstab(df[label_col], df[band_col], margins=True, margins_name="Total")
    for col in ["RED", "GRAY", "GREEN", "Total"]:
        if col not in ct.columns:
            ct[col] = 0
    print(f"\n=== {title} ===")
    print(ct[["RED", "GRAY", "GREEN", "Total"]].to_string())

    fraud = df[df[label_col] == "real fraud"]
    gray_total = int((df[band_col] == "GRAY").sum())
    gray_fraud = int((fraud[band_col] == "GRAY").sum())
    red_total = int((df[band_col] == "RED").sum())
    red_fraud = int((fraud[band_col] == "RED").sum())
    fn_green = int((fraud[band_col] == "GREEN").sum())

    print(f"\nFraud slipping into GREEN (false negative): {fn_green}/{len(fraud)}")
    print(f"GRAY fraud density (fraud within GRAY / all GRAY): "
          f"{gray_fraud}/{gray_total} = {gray_fraud / max(gray_total, 1):.4%}")
    if red_total:
        print(f"RED purity (share of RED that's real fraud): {red_fraud}/{red_total} = {red_fraud / red_total:.2%}")
    else:
        print("RED purity: no RED in this slice")


def run_operational() -> None:
    from seed_representative import FULL_CSV, N_TOTAL, RAW_COLS, RISKY_TYPES, SEED, TEST_PARQUET

    full_df = pd.read_csv(FULL_CSV, usecols=RAW_COLS)
    type_share = full_df["type"].value_counts(normalize=True)
    counts = (type_share * N_TOTAL).round().astype(int).to_dict()

    test_df = pd.read_parquet(TEST_PARQUET)
    risky_needed = sum(counts.get(t, 0) for t in RISKY_TYPES)
    risky_pool = test_df[test_df["type"].isin(RISKY_TYPES)]
    risky_sample = risky_pool.sample(n=min(risky_needed, len(risky_pool)), random_state=SEED).copy()
    print(f"Reproduced risky sample: {len(risky_sample)} rows "
          f"(fraud={int(risky_sample['isFraud'].sum())})")

    # The transactions.MAX(id)+1 right before this run — when multiple seed runs
    # share the same random_state, the natural-key join needs the range
    # restricted so it doesn't also pick up rows from earlier runs.
    MIN_TXN_ID = 1054

    con = sqlite3.connect("fraud.db")
    db = pd.read_sql(
        f"""
        SELECT t.id AS txn_id, t.type, t.amount, t.step,
               t.oldbalance_org, t.newbalance_orig, t.oldbalance_dest, t.newbalance_dest,
               s.risk_band, s.calibrated_proba, s.hybrid_score, s.model_version
        FROM transactions t
        JOIN scores s ON s.transaction_id = t.id
        WHERE t.type IN ('TRANSFER', 'CASH_OUT') AND t.id >= {MIN_TXN_ID}
        """,
        con,
    )
    con.close()

    merged = risky_sample.merge(
        db,
        left_on=["type", "amount", "step", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"],
        right_on=["type", "amount", "step", "oldbalance_org", "newbalance_orig", "oldbalance_dest", "newbalance_dest"],
        how="left",
    )

    n_matched = merged["risk_band"].notna().sum()
    n_dupe = len(merged) - len(risky_sample)
    print(f"Matched: {n_matched}/{len(risky_sample)}"
          + (f"  [WARNING: {n_dupe} extra rows — multiple matches found]" if n_dupe else ""))

    merged = merged.dropna(subset=["risk_band"])
    merged["label"] = merged["isFraud"].map({1: "real fraud", 0: "real clean"})

    print_confusion(merged, "label", "risk_band",
                     f"OPERATIONAL — risky types, seed_representative.py (N={len(merged)}, "
                     f"fraud={int((merged['label']=='real fraud').sum())}) — read from DB")
    print("\n[Note] This N carries no statistical power for a threshold decision — see MODE='offline_full'.")


def run_offline_full() -> None:
    from backend.explain import ML_FEATURES
    from backend.rule_engine import RULE_FIELDS, _THRESHOLD_RED
    from backend.rule_engine import score_transaction as rule_score_transaction
    from backend.scoring import HIGH_CONFIDENCE_THRESHOLD, load_scoring_engine

    print("Loading model/calibrator...")
    engine = load_scoring_engine()

    print("Loading test.parquet...")
    t0 = time.time()
    df = pd.read_parquet("data/test.parquet")
    print(f"  {len(df):,} rows, {int(df['isFraud'].sum())} fraud ({time.time() - t0:.1f}s)")

    off_types = set(df["type"].unique()) - {"TRANSFER", "CASH_OUT"}
    assert not off_types, f"test.parquet has unexpected types: {off_types}"

    X = df[ML_FEATURES].values.astype(float)
    t0 = time.time()
    raw_proba = engine.xgb_model.predict_proba(X)[:, 1]
    calibrated_proba = engine.calibrator.predict_proba(raw_proba)
    print(f"XGBoost + calibration (554K, batched): {time.time() - t0:.1f}s")

    t0 = time.time()
    hybrid = np.empty(len(df))
    band = np.empty(len(df), dtype=object)
    hard_flag = np.empty(len(df), dtype=bool)
    band_reason = np.full(len(df), None, dtype=object)
    cols = {k: df[k].to_numpy() for k in RULE_FIELDS}
    for i in range(len(df)):
        rule_input = {k: cols[k][i] for k in RULE_FIELDS}
        result = rule_score_transaction(rule_input, raw_proba[i])
        if result["risk_band"] != "RED" and calibrated_proba[i] >= HIGH_CONFIDENCE_THRESHOLD:
            result["risk_band"] = "RED"
            result["hybrid_score"] = round(max(result["hybrid_score"], _THRESHOLD_RED), 2)
            band_reason[i] = "high_confidence_override"
        hybrid[i] = result["hybrid_score"]
        band[i] = result["risk_band"]
        hard_flag[i] = result["hard_rule_flag"]
    print(f"Rule engine + high_confidence_override (554K, row-by-row): {time.time() - t0:.1f}s")
    print(f"  high_confidence_override triggered: {int((band_reason == 'high_confidence_override').sum())} rows")

    df = df.assign(
        raw_proba=raw_proba,
        calibrated_proba=calibrated_proba,
        band_reason=band_reason,
        hybrid_score=hybrid,
        risk_band=band,
        hard_rule_flag=hard_flag,
        label=df["isFraud"].map({1: "real fraud", 0: "real clean"}),
    )

    print_confusion(df, "label", "risk_band",
                     f"OFFLINE FULL TEST SET (N={len(df):,}, fraud={int(df['isFraud'].sum())}) — "
                     f"real label × risk_band")

    with open("models/xgb_v1_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    stored = meta["test_metrics"]
    thr = meta["working_threshold"]

    y_true = df["isFraud"].to_numpy()
    y_pred = (raw_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recomputed = {
        "PR-AUC": round(average_precision_score(y_true, raw_proba), 4),
        "ROC-AUC": round(roc_auc_score(y_true, raw_proba), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "FPR": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 6),
        "FNR": round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4),
    }

    print(f"\n=== Serving-parity (threshold={thr}, vs test_metrics in models/xgb_v1_meta.json) ===")
    print(f"{'metric':<12}{'offline (train)':>18}{'this run (online path)':>24}{'diff':>12}")
    max_abs_diff = 0.0
    for k in ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "FPR", "FNR"]:
        s, r = stored[k], recomputed[k]
        diff = r - s
        max_abs_diff = max(max_abs_diff, abs(diff))
        print(f"{k:<12}{s:>18}{r:>24}{diff:>+12.6f}")
    verdict = "PARITY HOLDS" if max_abs_diff < 0.001 else "DEVIATION — investigate"
    print(f"\nLargest absolute diff: {max_abs_diff:.6f} → {verdict}")

    # RED = hard_rule_flag OR hybrid_score>=85 — two different triggers,
    # so break down the RED band by hard-rule × calibrated p>=0.95.
    red = df[df["risk_band"] == "RED"].copy()
    red["p95"] = red["calibrated_proba"] >= 0.95

    print("\n=== RED band breakdown: hard-rule x calibrated p>=0.95 (N=%d) ===" % len(red))
    print(f"{'cell':<28}{'N':>8}{'fraud':>8}{'purity':>10}")
    for hard in (True, False):
        for p95 in (True, False):
            sub = red[(red["hard_rule_flag"] == hard) & (red["p95"] == p95)]
            n = len(sub)
            fraud = int((sub["label"] == "real fraud").sum())
            label = f"hard-rule={hard!s:<5} p>=0.95={p95!s:<5}"
            purity = f"{fraud / n:.2%}" if n else "—"
            print(f"{label:<28}{n:>8}{fraud:>8}{purity:>10}")

    p95_all = df[df["calibrated_proba"] >= 0.95]
    p95_purity = (p95_all["label"] == "real fraud").mean() if len(p95_all) else float("nan")
    print(f"\np>=0.95 purity across the full dataset (not restricted to RED): "
          f"{int((p95_all['label']=='real fraud').sum())}/{len(p95_all)} = {p95_purity:.2%}")

    hard_only = red[(red["hard_rule_flag"]) & (~red["p95"])]
    prob_only = red[(~red["hard_rule_flag"]) & (red["p95"])]
    print(f"\nRule-only RED (hard-rule, p<0.95): {len(hard_only)} txns, "
          f"purity {(hard_only['label']=='real fraud').mean():.2%}" if len(hard_only) else "\nRule-only RED: none")
    print(f"Probability-only RED (p>=0.95, no hard-rule): {len(prob_only)} txns, "
          f"purity {(prob_only['label']=='real fraud').mean():.2%}" if len(prob_only) else "Probability-only RED: none")


if __name__ == "__main__":
    if MODE == "operational":
        run_operational()
    elif MODE == "offline_full":
        run_offline_full()
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}")
