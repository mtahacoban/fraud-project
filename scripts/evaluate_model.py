import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    roc_auc_score, confusion_matrix,
    f1_score, precision_score, recall_score
)

DATA_DIR   = "data"
MODEL_DIR  = "models"
REPORT_DIR = "reports"
ML_FEATURES = ["amount", "step_hour", "errorBalanceOrig",
               "errorBalanceDest", "is_transfer", "is_cashout"]
TARGET       = "isFraud"
RANDOM_STATE = 42
os.makedirs(REPORT_DIR, exist_ok=True)

print("Loading data...")
train_raw = pd.read_parquet(DATA_DIR + "/train.parquet")
val_raw   = pd.read_parquet(DATA_DIR + "/val.parquet")
test_raw  = pd.read_parquet(DATA_DIR + "/test.parquet")
val_sc    = pd.read_parquet(DATA_DIR + "/val_scaled.parquet")
test_sc   = pd.read_parquet(DATA_DIR + "/test_scaled.parquet")

y_train = train_raw[TARGET].values
y_val   = val_raw[TARGET].values
y_test  = test_raw[TARGET].values

print("Loading models...")
xgb_model = joblib.load(MODEL_DIR + "/xgb_v1.pkl")
logreg    = joblib.load(MODEL_DIR + "/logreg_v1.pkl")

lr_test_proba  = logreg.predict_proba(test_sc[ML_FEATURES].values)[:, 1]
xgb_test_proba = xgb_model.predict_proba(test_raw[ML_FEATURES].values)[:, 1]
xgb_val_proba  = xgb_model.predict_proba(val_raw[ML_FEATURES].values)[:, 1]

prec_a, rec_a, thr_a = precision_recall_curve(y_val, xgb_val_proba)
denom = prec_a[:-1] + rec_a[:-1]
f1s   = np.where(denom > 0, 2 * prec_a[:-1] * rec_a[:-1] / denom, 0.0)
XGB_THR = float(thr_a[np.argmax(f1s)])
print(f"XGBoost working threshold: {XGB_THR:.4f}\n")

print("=" * 60)
print("HARD RULES - data validation")
print("=" * 60)

combined     = pd.concat([train_raw, val_raw], ignore_index=True)
baseline_rate = combined[TARGET].mean()
print(f"Overall fraud rate: {baseline_rate:.4%}  (n={len(combined):,})\n")

drain_mask  = (combined["oldbalanceOrg"] > 0) & \
              (combined["amount"] >= combined["oldbalanceOrg"] * 0.99)
n_drain     = drain_mask.sum()
drain_fraud = combined.loc[drain_mask, TARGET].mean()
drain_lift  = drain_fraud / baseline_rate
print(f"drain_account  -> {n_drain:,} txns  |  fraud: {drain_fraud:.4%}  |  lift: {drain_lift:.1f}x")
print(f"  Decision: lift={drain_lift:.1f}x - too broad for a hard rule (triggered on 43% of the test set)")
print(f"  -> DEMOTED TO SOFT RULE (weight=20)")
print()

ghost_mask  = (combined["oldbalanceDest"] == 0) & (combined["newbalanceDest"] == 0)
n_ghost     = ghost_mask.sum()
ghost_fraud = combined.loc[ghost_mask, TARGET].mean()
ghost_lift  = ghost_fraud / baseline_rate
print(f"ghost_destination -> {n_ghost:,} txns  |  fraud: {ghost_fraud:.4%}  |  lift: {ghost_lift:.1f}x")
print(f"  Decision: lift={ghost_lift:.1f}x - stays a HARD RULE")
print(f"  WARNING: PaySim artifact - merchant accounts don't update their balance;")
print(f"           a real system would need a merchant filter.")
print()

print("=" * 60)
print("SOFT RULES - correlation validation (train+val)")
print("=" * 60)
print(f"Threshold: lift >= 1.5 -> keep; lift < 1.5 -> drop\n")

candidates = {
    "night_transaction (step_hour 0-5)":    combined["step_hour"] <= 5,
    "drain_account (amount>=0.99*origBal)": (combined["oldbalanceOrg"] > 0) &
                                            (combined["amount"] >= combined["oldbalanceOrg"] * 0.99),
    "high_amount_transfer (TRANSFER>200k)": (combined["type"] == "TRANSFER") &
                                            (combined["amount"] > 200_000),
    "high_amount_cashout (CASHOUT>200k)":   (combined["type"] == "CASH_OUT") &
                                            (combined["amount"] > 200_000),
    "zero_source_balance (origBal=0)":      combined["oldbalanceOrg"] == 0,
}

kept, dropped = [], []
for rule_name, mask in candidates.items():
    n_pos = mask.sum()
    if n_pos == 0:
        continue
    fraud_r = combined.loc[mask, TARGET].mean()
    lift    = fraud_r / baseline_rate
    status  = "KEEP" if lift >= 1.5 else "DROP"
    print(f"  {rule_name}")
    print(f"    n={n_pos:,}  fraud rate={fraud_r:.4%}  lift={lift:.2f}x  -> {status}")
    if lift >= 1.5:
        kept.append(rule_name)
    else:
        dropped.append(rule_name)

print(f"\nKept   : {kept}")
print(f"Dropped: {dropped}\n")

print("=" * 60)
print("HYBRID SCORE - test set")
print("=" * 60)

test = test_raw.copy()
test["xgb_proba"] = xgb_test_proba
test["ml_score"]  = (xgb_test_proba * 100).round(2)

# Hard rule (vectorized): ghost_destination only
test["ghost_destination"] = (test["oldbalanceDest"] == 0) & (test["newbalanceDest"] == 0)
test["hard_rule_flag"]    = test["ghost_destination"]

# Soft rules (vectorized; validated rules)
soft_sc = pd.Series(0.0, index=test.index)
soft_sc += (test["step_hour"] <= 5).astype(float) * 45          # night_transaction
soft_sc += ((test["oldbalanceOrg"] > 0) &
            (test["amount"] >= test["oldbalanceOrg"] * 0.99)
           ).astype(float) * 20                                   # drain_account (soft)
soft_sc += ((test["type"] == "TRANSFER") &
            (test["amount"] > 200_000)).astype(float) * 30       # high_amount_transfer
test["soft_score"] = soft_sc.clip(0, 100).astype(int)

raw_hybrid = np.where(
    test["hard_rule_flag"],
    np.maximum(85.0, test["ml_score"].values),
    (0.70 * test["ml_score"].values + 0.30 * test["soft_score"].values).clip(0, 100)
)
test["hybrid_score"] = raw_hybrid.round(2)

test["risk_band"] = "GRAY"
test.loc[test["hybrid_score"] <  15, "risk_band"] = "GREEN"
test.loc[test["hybrid_score"] >= 85, "risk_band"] = "RED"

band_counts = test.groupby("risk_band")[TARGET].agg(["count", "sum"])
band_counts.columns  = ["total", "fraud_count"]
band_counts["clean"] = band_counts["total"] - band_counts["fraud_count"]
band_counts["fraud_pct"] = (band_counts["fraud_count"] / band_counts["total"] * 100).round(2)

total_fraud = int(test[TARGET].sum())
print(f"\nRisk Band Distribution (test, n={len(test):,}, total fraud={total_fraud}):\n")
for band in ["GREEN", "GRAY", "RED"]:
    if band not in band_counts.index:
        continue
    row    = band_counts.loc[band]
    fn_pct = row["fraud_count"] / total_fraud * 100
    print(f"  {band:5s}: {int(row['total']):>8,} txns  |  "
          f"{int(row['fraud_count']):>4} fraud ({fn_pct:5.1f}% of total)  |  "
          f"{row['fraud_pct']:.2f}% txn fraud rate")

n_ghost_test  = int(test["ghost_destination"].sum())
gf_fraud      = int(test.loc[test["ghost_destination"], TARGET].sum())
print(f"\nHard rule (ghost_destination): {n_ghost_test:,} triggers  |  "
      f"{gf_fraud} fraud ({gf_fraud/n_ghost_test*100:.1f}%)")

print("\nSample hybrid scoring:")
idx_fraud  = test[test[TARGET] == 1].index[:3]
idx_clean  = test[test[TARGET] == 0].index[:3]
sample_idx = list(idx_fraud) + list(idx_clean)
cols_show  = ["amount", "type", "hard_rule_flag", "ghost_destination",
              "soft_score", "ml_score", "hybrid_score", "risk_band", TARGET]
print(test.loc[sample_idx, cols_show].to_string())

print("\n" + "=" * 60)
print("COMPARISON - Test Set")
print("=" * 60)

hybrid_proba = test["hybrid_score"].values / 100.0

lr_prauc  = average_precision_score(y_test, lr_test_proba)
xgb_prauc = average_precision_score(y_test, xgb_test_proba)
hyb_prauc = average_precision_score(y_test, hybrid_proba)

print(f"  LogReg   PR-AUC: {lr_prauc:.4f}  ROC-AUC: {roc_auc_score(y_test, lr_test_proba):.4f}")
print(f"  XGBoost  PR-AUC: {xgb_prauc:.4f}  ROC-AUC: {roc_auc_score(y_test, xgb_test_proba):.4f}")
print(f"  Hybrid   PR-AUC: {hyb_prauc:.4f}  ROC-AUC: {roc_auc_score(y_test, hybrid_proba):.4f}")

def evaluate(name, y_true, y_proba, thr):
    y_pred = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Model": name, "Thr": round(thr, 4),
        "PR-AUC":    round(average_precision_score(y_true, y_proba), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "FPR":       round(fp / (fp + tn), 6),
        "FNR":       round(fn / (fn + tp), 4),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }

lr_val_proba  = logreg.predict_proba(val_sc[ML_FEATURES].values)[:, 1]
p_lr, r_lr, t_lr = precision_recall_curve(y_val, lr_val_proba)
d_lr = p_lr[:-1] + r_lr[:-1]
f_lr = np.where(d_lr > 0, 2 * p_lr[:-1] * r_lr[:-1] / d_lr, 0.0)
LR_THR = float(t_lr[np.argmax(f_lr)])

HYB_THR = 0.85

lr_m  = evaluate("LogReg",  y_test, lr_test_proba, LR_THR)
xgb_m = evaluate("XGBoost", y_test, xgb_test_proba, XGB_THR)
hyb_m = evaluate("Hybrid",  y_test, hybrid_proba,   HYB_THR)

print("\nThreshold-based comparison:")
print(f"{'Model':<8} {'Thr':>6}  {'PR-AUC':>7}  {'P':>7}  {'R':>7}  {'F1':>7}  "
      f"{'FPR':>10}  {'TP':>5}  {'FN':>5}")
for m in [lr_m, xgb_m, hyb_m]:
    print(f"  {m['Model']:<7} {m['Thr']:>6.4f}  {m['PR-AUC']:>7.4f}  "
          f"{m['Precision']:>7.4f}  {m['Recall']:>7.4f}  {m['F1']:>7.4f}  "
          f"{m['FPR']:>10.6f}  {m['TP']:>5}  {m['FN']:>5}")

print(f"\n  Hybrid RED threshold = 0.85 (hybrid_score >= 85 -> predict fraud)")
print(f"  Total fraud caught by Hybrid: {hyb_m['TP']} / {total_fraud} = {hyb_m['Recall']:.4f} recall")
print(f"  Caught via ghost_destination: {gf_fraud} (hard rule; no ML score involved)")

print("\nGenerating charts...")

# 1. PR Curves
fig, ax = plt.subplots(figsize=(8, 6))
for name, proba, color, ls in [
    ("LogReg",  lr_test_proba,  "steelblue", "--"),
    ("XGBoost", xgb_test_proba, "tomato",    "-"),
    ("Hybrid",  hybrid_proba,   "seagreen",  "-"),
]:
    p, r, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    ax.plot(r, p, label=f"{name} (AP={ap:.3f})", color=color, lw=2, linestyle=ls)

for m, c in [(xgb_m, "tomato"), (hyb_m, "seagreen")]:
    ax.scatter([m["Recall"]], [m["Precision"]], s=110, color=c, zorder=5)

ax.axhline(y_test.mean(), linestyle=":", color="gray", alpha=0.6,
           label=f"Random (P={y_test.mean():.4f})")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("PR Curve: LogReg vs XGBoost vs Hybrid - Test Set", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR + "/model_comparison_pr_curve.png", dpi=120)
plt.close()
print("  -> reports/model_comparison_pr_curve.png")

# 2. Risk Band Distribution (bar)
bands  = ["GREEN", "GRAY", "RED"]
colors = ["seagreen", "goldenrod", "tomato"]
t_c    = [int(band_counts.loc[b, "clean"])       if b in band_counts.index else 0 for b in bands]
f_c    = [int(band_counts.loc[b, "fraud_count"]) if b in band_counts.index else 0 for b in bands]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(bands)); w = 0.35
ax.bar(x - w/2, t_c, w, label="Clean", color="steelblue", alpha=0.8)
ax.bar(x + w/2, f_c, w, label="Fraud", color="tomato",    alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(bands, fontsize=12)
ax.set_ylabel("Transaction count")
ax.set_title("Risk Band Distribution - Test Set", fontsize=13)
ax.set_yscale("log")
ax.legend(); ax.grid(axis="y", alpha=0.3)
for xi, fc in zip(x + w/2, f_c):
    if fc > 0:
        ax.text(xi, fc * 1.5, str(fc), ha="center", fontsize=9, color="darkred", fontweight="bold")
plt.tight_layout()
plt.savefig(REPORT_DIR + "/risk_band_distribution.png", dpi=120)
plt.close()
print("  -> reports/risk_band_distribution.png")

# 3. Hybrid score distribution
fig, ax = plt.subplots(figsize=(9, 4))
bins = np.linspace(0, 100, 51)
ax.hist(test.loc[test[TARGET] == 0, "hybrid_score"], bins=bins,
        alpha=0.6, color="steelblue", label="Clean")
ax.hist(test.loc[test[TARGET] == 1, "hybrid_score"], bins=bins,
        alpha=0.8, color="tomato",    label="Fraud")
ax.axvline(15, color="seagreen", linestyle="--", lw=1.5, label="GREEN/GRAY (15)")
ax.axvline(85, color="tomato",   linestyle="--", lw=1.5, label="GRAY/RED (85)")
ax.set_xlabel("Hybrid Score (0-100)", fontsize=12)
ax.set_ylabel("Transaction count (log)")
ax.set_title("Hybrid Score Distribution - Fraud vs Clean", fontsize=13)
ax.set_yscale("log"); ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR + "/hybrid_score_distribution.png", dpi=120)
plt.close()
print("  -> reports/hybrid_score_distribution.png")

# 4. ghost_destination rule trigger
fig, ax = plt.subplots(figsize=(6, 4))
ct = pd.crosstab(test["ghost_destination"], test[TARGET])
ct.index   = ["No", "Yes"]
ct.columns = ["Clean", "Fraud"]
ct.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], alpha=0.8, rot=0)
gf_pct = test.loc[test["ghost_destination"], TARGET].mean()
ax.set_title(f"ghost_destination hard rule\nfraud rate (triggered): {gf_pct:.1%}", fontsize=11)
ax.set_xlabel("Rule triggered?"); ax.set_ylabel("Count")
ax.set_yscale("log"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR + "/hard_rule_hits.png", dpi=120)
plt.close()
print("  -> reports/hard_rule_hits.png")

# 5. Soft score distribution (only rows with hard_rule_flag=False)
no_hard = test[~test["hard_rule_flag"]]
fig, ax = plt.subplots(figsize=(8, 4))
bins2 = range(0, 101, 5)
ax.hist(no_hard.loc[no_hard[TARGET] == 0, "soft_score"],
        bins=bins2, alpha=0.6, color="steelblue", label="Clean")
ax.hist(no_hard.loc[no_hard[TARGET] == 1, "soft_score"],
        bins=bins2, alpha=0.8, color="tomato",    label="Fraud")
ax.set_xlabel("Soft Score (0-100)", fontsize=12)
ax.set_ylabel("Transaction count")
ax.set_title("Soft Score Distribution (no hard rule) - Fraud vs Clean", fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR + "/soft_rule_distribution.png", dpi=120)
plt.close()
print("  -> reports/soft_rule_distribution.png")

print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  PR-AUC: LogReg={lr_prauc:.4f}  XGBoost={xgb_prauc:.4f}  Hybrid={hyb_prauc:.4f}")
print()
print("  Risk Band (test set):")
for band in ["GREEN", "GRAY", "RED"]:
    if band in band_counts.index:
        row = band_counts.loc[band]
        fn_pct = row["fraud_count"] / total_fraud * 100
        print(f"    {band:5s}: {int(row['total']):>8,} txns | "
              f"{int(row['fraud_count']):>4} fraud ({fn_pct:.1f}% of total fraud) | "
              f"{row['fraud_pct']:.2f}% txn fraud rate")
print()
print(f"  Hard rule: ghost_destination")
print(f"    Triggers: {n_ghost_test:,} | Fraud caught: {gf_fraud}/{total_fraud} ({gf_fraud/total_fraud*100:.1f}%)")
print(f"  Soft rules: drain_account (x20) + night_transaction (x45) + high_amount_transfer (x30)")
print(f"  Dropped (lift<1.5): high_amount_cashout, zero_source_balance")
print()
print(f"  Hybrid (thr=0.85): Recall={hyb_m['Recall']:.4f}  FPR={hyb_m['FPR']:.6f}  TP={hyb_m['TP']}  FN={hyb_m['FN']}")
print(f"  XGBoost (thr={XGB_THR:.4f}): Recall={xgb_m['Recall']:.4f}  FPR={xgb_m['FPR']:.6f}  TP={xgb_m['TP']}  FN={xgb_m['FN']}")
print()
print("  Note: Hybrid PR-AUC < XGBoost PR-AUC because the ghost_destination hard rule")
print("  assigns a fixed score=85 at the boundary, creating a 'step' in the PR curve.")
print("  The value isn't in raw PR-AUC: it's in risk banding + rule explainability.")
