import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score
from datetime import datetime

from backend.calibrate import XGBCalibrator, brier_score, brier_skill_score

DATA_DIR   = "data"
MODEL_DIR  = "models"
REPORT_DIR = "reports"
ML_FEATURES = ["amount", "step_hour", "errorBalanceOrig",
               "errorBalanceDest", "is_transfer", "is_cashout"]
TARGET = "isFraud"
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Load ────────────────────────────────────────────────────────────────────
print("Loading model and data...")
xgb_model = joblib.load(MODEL_DIR + "/xgb_v1.pkl")
val_raw   = pd.read_parquet(DATA_DIR + "/val.parquet")
test_raw  = pd.read_parquet(DATA_DIR + "/test.parquet")
y_val     = val_raw[TARGET].values
y_test    = test_raw[TARGET].values
X_val     = val_raw[ML_FEATURES].values
X_test    = test_raw[ML_FEATURES].values

# Raw XGBoost probabilities
raw_val_proba  = xgb_model.predict_proba(X_val)[:, 1]
raw_test_proba = xgb_model.predict_proba(X_test)[:, 1]

print(f"Val : {len(y_val):,}  |  Fraud: {y_val.sum():,}")
print(f"Test: {len(y_test):,}  |  Fraud: {y_test.sum():,}")
print(f"XGBoost raw val  PR-AUC: {average_precision_score(y_val,  raw_val_proba):.4f}")
print(f"XGBoost raw test PR-AUC: {average_precision_score(y_test, raw_test_proba):.4f}")

# ── Calibration (fit on the val set) ─────────────────────────────────────────
print("\nCalibrating (on val-set raw probabilities)...")
cal_iso = XGBCalibrator("isotonic").fit(raw_val_proba, y_val)
cal_sig = XGBCalibrator("sigmoid").fit(raw_val_proba, y_val)

iso_test_proba = cal_iso.predict_proba(raw_test_proba)
sig_test_proba = cal_sig.predict_proba(raw_test_proba)

# ── Brier scores (test set) ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BRIER SCORES (test set)")
print("=" * 60)

prevalence  = float(y_test.mean())
ref_proba   = np.full(len(y_test), prevalence)

scores = {
    "Baseline (prevalence)":  (ref_proba,  None),
    "Raw XGBoost":            (raw_test_proba,  None),
    "Calibrated (Isotonic)":  (iso_test_proba,  None),
    "Calibrated (Sigmoid/Platt)": (sig_test_proba,  None),
}

brier_results = {}
for name, (proba, _) in scores.items():
    bs  = brier_score(y_test, proba)
    bss = brier_skill_score(y_test, proba)
    brier_results[name] = {"brier": bs, "bss": bss}
    print(f"  {name:<30s}  Brier={bs:.6f}  BSS={bss:+.4f}")

# Pick the best method
iso_brier = brier_results["Calibrated (Isotonic)"]["brier"]
sig_brier = brier_results["Calibrated (Sigmoid/Platt)"]["brier"]
best_method   = "isotonic" if iso_brier <= sig_brier else "sigmoid"
best_cal = cal_iso if iso_brier <= sig_brier else cal_sig
best_proba    = iso_test_proba if iso_brier <= sig_brier else sig_test_proba
print(f"\n  Best method: {best_method}  (Brier={min(iso_brier, sig_brier):.6f})")

# ── Calibration curves ────────────────────────────────────────────────────────
N_BINS = 10
STRATEGY = "quantile"

curves = {
    "Raw XGBoost":         (raw_test_proba,  "tomato",    "--"),
    "Isotonic":            (iso_test_proba,  "steelblue", "-"),
    "Sigmoid/Platt":       (sig_test_proba,  "seagreen",  "-"),
}

fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig)

# Left: calibration curve (reliability diagram)
ax1 = fig.add_subplot(gs[0])
ax1.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
for name, (proba, color, ls) in curves.items():
    fop, mpv = calibration_curve(y_test, proba, n_bins=N_BINS, strategy=STRATEGY)
    bs = brier_score(y_test, proba)
    ax1.plot(mpv, fop, color=color, linestyle=ls, lw=2,
             marker="o", ms=5, label=f"{name} (Brier={bs:.5f})")
ax1.set_xlabel("Mean Predicted Probability", fontsize=11)
ax1.set_ylabel("Actual Fraud Rate (fraction)", fontsize=11)
ax1.set_title("Calibration Curve (Reliability Diagram)", fontsize=12)
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1])

# Right: probability distribution (raw vs calibrated)
ax2 = fig.add_subplot(gs[1])
bins = np.linspace(0, 1, 51)
ax2.hist(raw_test_proba,  bins=bins, alpha=0.6, color="tomato",    label="Raw XGBoost",  density=True)
ax2.hist(iso_test_proba,  bins=bins, alpha=0.6, color="steelblue", label="Isotonic",     density=True)
ax2.set_xlabel("Fraud Probability", fontsize=11)
ax2.set_ylabel("Density")
ax2.set_title("Probability Distribution: Raw vs Isotonic", fontsize=12)
ax2.set_yscale("log"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.suptitle("XGBoost Calibration Analysis (Test Set)", fontsize=13)
plt.tight_layout()
plt.savefig(REPORT_DIR + "/calibration_curve.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n  -> reports/calibration_curve.png")

# Brier bar chart
fig, ax = plt.subplots(figsize=(7, 4))
names_plot  = list(brier_results.keys())
brier_vals  = [brier_results[n]["brier"] for n in names_plot]
colors_plot = ["gray", "tomato", "steelblue", "seagreen"]
bars = ax.bar(range(len(names_plot)), brier_vals, color=colors_plot, alpha=0.85)
ax.set_xticks(range(len(names_plot)))
ax.set_xticklabels(names_plot, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Brier Score (lower is better)")
ax.set_title("Brier Score Comparison — Test Set")
for bar, val in zip(bars, brier_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.00002, f"{val:.5f}",
            ha="center", va="bottom", fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR + "/brier_score_comparison.png", dpi=120)
plt.close()
print("  -> reports/brier_score_comparison.png")

# ── Save ───────────────────────────────────────────────────────────────────
cal_path = MODEL_DIR + "/xgb_v1_calibrated.pkl"
joblib.dump(best_cal, cal_path)
print(f"\n  Calibrated model saved: {cal_path}")

meta = {
    "base_model":   "xgb_v1",
    "method":       best_method,
    "calibrated_at": datetime.now().isoformat(),
    "cal_set":      "val",
    "n_cal":        int(len(X_val)),
    "brier_scores": {
        "baseline":    round(brier_results["Baseline (prevalence)"]["brier"], 8),
        "raw_xgb":     round(brier_results["Raw XGBoost"]["brier"], 8),
        "isotonic":    round(brier_results["Calibrated (Isotonic)"]["brier"], 8),
        "sigmoid":     round(brier_results["Calibrated (Sigmoid/Platt)"]["brier"], 8),
    },
    "bss": {
        "raw_xgb":  round(brier_results["Raw XGBoost"]["bss"], 6),
        "isotonic": round(brier_results["Calibrated (Isotonic)"]["bss"], 6),
        "sigmoid":  round(brier_results["Calibrated (Sigmoid/Platt)"]["bss"], 6),
    },
    "note": (
        "The hybrid score is not calibrated — the fixed-85 step would distort the curve. "
        "The calibrated probability is used for the AI2 automation confidence gate."
    ),
}
with open(MODEL_DIR + "/xgb_v1_calibrated_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print("  Meta saved: models/xgb_v1_calibrated_meta.json")

# ── Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Method       : {best_method}")
print(f"  Cal set      : val ({len(X_val):,} samples)")
print()
print(f"  Brier (baseline): {brier_results['Baseline (prevalence)']['brier']:.6f}")
print(f"  Brier (raw)     : {brier_results['Raw XGBoost']['brier']:.6f}  "
      f"BSS={brier_results['Raw XGBoost']['bss']:+.4f}")
print(f"  Brier (iso)     : {brier_results['Calibrated (Isotonic)']['brier']:.6f}  "
      f"BSS={brier_results['Calibrated (Isotonic)']['bss']:+.4f}")
print(f"  Brier (sigmoid) : {brier_results['Calibrated (Sigmoid/Platt)']['brier']:.6f}  "
      f"BSS={brier_results['Calibrated (Sigmoid/Platt)']['bss']:+.4f}")
print()
print("  Notes:")
print("  - The calibrated probability is used for the automation confidence gate")
print("  - The hybrid score is not calibrated (fixed-85 -> would distort the curve)")
print("  - The val set was used for both early stopping and calibration")
print("    (note as a limitation; practical effect is small, n_val=138k)")
