import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap
import warnings
warnings.filterwarnings("ignore")

from backend.explain import build_explainer, explain_transaction, compute_shap_matrix

DATA_DIR   = "data"
MODEL_DIR  = "models"
REPORT_DIR = "reports"
ML_FEATURES = ["amount", "step_hour", "errorBalanceOrig",
               "errorBalanceDest", "is_transfer", "is_cashout"]
TARGET = "isFraud"
os.makedirs(REPORT_DIR, exist_ok=True)

print("Loading model and data...")
xgb_model = joblib.load(MODEL_DIR + "/xgb_v1.pkl")
test_raw  = pd.read_parquet(DATA_DIR + "/test.parquet")
y_test    = test_raw[TARGET].values
X_test    = test_raw[ML_FEATURES].values

print(f"Test set: {len(test_raw):,}  |  Fraud: {y_test.sum():,}")

print("Building TreeExplainer...")
explainer = build_explainer(xgb_model)

# Fraud sample: NO hard rule (ghost_destination=False) → an interesting ML explanation
fraud_all  = test_raw.index[test_raw[TARGET] == 1].tolist()
ghost_mask = (test_raw["oldbalanceDest"] == 0) & (test_raw["newbalanceDest"] == 0)
gray_fraud = [i for i in fraud_all if not ghost_mask.loc[i]]
demo_fraud = gray_fraud[0]

demo_clean = test_raw.index[test_raw[TARGET] == 0][0]

print("\n" + "=" * 60)
print("SINGLE TRANSACTION EXPLANATION - explain_transaction() API")
print("=" * 60)

for label, idx in [("FRAUD (GRAY zone, caught by ML)", demo_fraud),
                   ("CLEAN", demo_clean)]:
    row = test_raw.loc[idx]
    fvals = {f: row[f] for f in ML_FEATURES}
    factors = explain_transaction(explainer, fvals, top_n=6)
    ml_p = float(xgb_model.predict_proba(
        np.array([[fvals[f] for f in ML_FEATURES]])
    )[:, 1][0])
    print(f"\n  [{label}]  idx={idx}  amount={row['amount']:,.0f}"
          f"  type={row['type']}  ml_proba={ml_p:.4f}")
    for f in factors:
        arrow = "↑" if f["direction"] == "increasing" else "↓"
        print(f"    {arrow} {f['feature']:22s}  val={f['value']:>12.2f}"
              f"  shap={f['shap_value']:+.4f}")

def plot_waterfall(idx, label, fname):
    row   = test_raw.loc[idx]
    fvals = {f: row[f] for f in ML_FEATURES}
    facts = explain_transaction(explainer, fvals, top_n=6)
    ml_p  = float(xgb_model.predict_proba(
        np.array([[fvals[f] for f in ML_FEATURES]])
    )[:, 1][0])

    fig, ax = plt.subplots(figsize=(9, 4))
    names   = [f"{f['feature']} = {f['value']:.2f}" for f in facts]
    svals   = [f["shap_value"] for f in facts]
    colors  = ["tomato" if s > 0 else "steelblue" for s in svals]

    ax.barh(names[::-1], svals[::-1], color=colors[::-1], height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (log-odds contribution)")
    ax.set_title(f"{label}\n(idx={idx}, ml_proba={ml_p:.4f}, isFraud={int(row[TARGET])})")
    red_p  = mpatches.Patch(color="tomato",    label="fraud ↑ increasing")
    blue_p = mpatches.Patch(color="steelblue", label="fraud ↓ decreasing")
    ax.legend(handles=[red_p, blue_p], fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR + f"/{fname}", dpi=120)
    plt.close()
    print(f"  -> reports/{fname}")

print("\nWaterfall charts...")
plot_waterfall(demo_fraud, "Fraud Sample - SHAP Explanation", "shap_waterfall_fraud.png")
plot_waterfall(demo_clean, "Clean Sample - SHAP Explanation", "shap_waterfall_clean.png")

print("\nGlobal SHAP (2000-sample)...")
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_test), size=2_000, replace=False)
X_sample   = X_test[sample_idx]
y_sample   = y_test[sample_idx]

shap_matrix = compute_shap_matrix(explainer, X_sample)
print(f"  SHAP matrix: {shap_matrix.shape}")

plt.figure(figsize=(9, 5))
shap.summary_plot(shap_matrix, X_sample, feature_names=ML_FEATURES, show=False)
plt.title("Global SHAP - Beeswarm (2,000-sample)", fontsize=13)
plt.tight_layout()
plt.savefig(REPORT_DIR + "/shap_beeswarm.png", dpi=120, bbox_inches="tight")
plt.close()
print("  -> reports/shap_beeswarm.png")

plt.figure(figsize=(7, 4))
shap.summary_plot(shap_matrix, X_sample, feature_names=ML_FEATURES,
                  plot_type="bar", show=False)
plt.title("Global SHAP - Mean |Contribution|", fontsize=13)
plt.tight_layout()
plt.savefig(REPORT_DIR + "/shap_feature_importance.png", dpi=120, bbox_inches="tight")
plt.close()
print("  -> reports/shap_feature_importance.png")

fraud_mask  = y_sample == 1
mean_abs_fraud = np.abs(shap_matrix[fraud_mask]).mean(axis=0)
mean_abs_clean = np.abs(shap_matrix[~fraud_mask]).mean(axis=0)

fig, ax = plt.subplots(figsize=(9, 4))
x  = np.arange(len(ML_FEATURES)); w = 0.35
ax.bar(x - w/2, mean_abs_fraud, w, label="Fraud", color="tomato",    alpha=0.85)
ax.bar(x + w/2, mean_abs_clean, w, label="Clean", color="steelblue", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(ML_FEATURES, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Mean |SHAP|")
ax.set_title("Fraud vs Clean - Mean SHAP Contribution")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR + "/shap_fraud_vs_clean.png", dpi=120)
plt.close()
print("  -> reports/shap_fraud_vs_clean.png")

mean_abs = np.abs(shap_matrix).mean(axis=0)
ranked   = sorted(zip(ML_FEATURES, mean_abs), key=lambda x: -x[1])

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("  Global feature importance (mean |SHAP|, 2000-sample):")
for feat, val in ranked:
    bar = "#" * int(val / max(mean_abs) * 30)
    print(f"    {feat:22s}: {val:.4f}  {bar}")
print()
print("  explain_transaction() output as DB-ready JSON:")
print("    [{feature, value, shap_value, direction}, ...]")
print()
print("  Saved charts:")
for f in ["shap_waterfall_fraud.png", "shap_waterfall_clean.png",
          "shap_beeswarm.png", "shap_feature_importance.png", "shap_fraud_vs_clean.png"]:
    print(f"    reports/{f}")
