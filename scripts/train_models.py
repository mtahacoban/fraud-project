import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd, numpy as np, json, joblib, warnings
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
    roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

DATA_DIR, MODEL_DIR, REPORT_DIR = 'data', 'models', 'reports'
ML_FEATURES = ['amount','step_hour','errorBalanceOrig','errorBalanceDest','is_transfer','is_cashout']
TARGET, RANDOM_STATE = 'isFraud', 42
os.makedirs(REPORT_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)

# --- Data ---
train_sc  = pd.read_parquet(DATA_DIR + '/train_scaled.parquet')
val_sc    = pd.read_parquet(DATA_DIR + '/val_scaled.parquet')
test_sc   = pd.read_parquet(DATA_DIR + '/test_scaled.parquet')
train_raw = pd.read_parquet(DATA_DIR + '/train.parquet')
val_raw   = pd.read_parquet(DATA_DIR + '/val.parquet')
test_raw  = pd.read_parquet(DATA_DIR + '/test.parquet')

y_train = train_raw[TARGET].values; y_val = val_raw[TARGET].values; y_test = test_raw[TARGET].values
X_train_sc = train_sc[ML_FEATURES].values
X_val_sc   = val_sc[ML_FEATURES].values
X_test_sc  = test_sc[ML_FEATURES].values
X_train = train_raw[ML_FEATURES].values
X_val   = val_raw[ML_FEATURES].values
X_test  = test_raw[ML_FEATURES].values

assert np.array_equal(train_sc[TARGET].values, y_train)
assert np.array_equal(val_sc[TARGET].values, y_val)
print(f"Train: {X_train.shape} | fraud: {y_train.mean():.4%}")
print(f"Val  : {X_val.shape}   | fraud: {y_val.mean():.4%}")
print(f"Test : {X_test.shape}  | fraud: {y_test.mean():.4%}")

# --- LogReg ---
print("\n=== 2.1 LogReg ===")
logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
logreg.fit(X_train_sc, y_train)
lr_val_proba  = logreg.predict_proba(X_val_sc)[:, 1]
lr_test_proba = logreg.predict_proba(X_test_sc)[:, 1]
print(f"LogReg Val  PR-AUC: {average_precision_score(y_val,  lr_val_proba):.4f}")
print(f"LogReg Test PR-AUC: {average_precision_score(y_test, lr_test_proba):.4f}")

# --- XGBoost ---
print("\n=== 2.2 XGBoost ===")
neg_count = int((y_train == 0).sum()); pos_count = int((y_train == 1).sum())
spw = neg_count / pos_count
print(f"scale_pos_weight = {spw:.2f}  ({neg_count:,} neg / {pos_count:,} pos)")

xgb_model = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, eval_metric='aucpr', early_stopping_rounds=30,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
)
print("Training...")
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
best_round     = xgb_model.best_iteration
xgb_val_proba  = xgb_model.predict_proba(X_val)[:, 1]
xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
print(f"Best round: {best_round}")
print(f"XGBoost Val  PR-AUC: {average_precision_score(y_val,  xgb_val_proba):.4f}")
print(f"XGBoost Test PR-AUC: {average_precision_score(y_test, xgb_test_proba):.4f}")

# --- Threshold selection ---
def f1max_threshold(y_true, y_proba):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    denom = prec[:-1] + rec[:-1]
    f1s   = np.where(denom > 0, 2*prec[:-1]*rec[:-1]/denom, 0.0)
    idx   = int(np.argmax(f1s))
    return float(thr[idx]), float(f1s[idx])

XGB_THR, xgb_val_f1 = f1max_threshold(y_val, xgb_val_proba)
LR_THR,  lr_val_f1  = f1max_threshold(y_val, lr_val_proba)
print(f"\nXGBoost F1-max threshold (val): {XGB_THR:.4f}  (val F1={xgb_val_f1:.4f})")
print(f"LogReg  F1-max threshold (val): {LR_THR:.4f}  (val F1={lr_val_f1:.4f})")

# --- Evaluation ---
def evaluate(name, y_true, y_proba, thr):
    y_pred = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Model': name, 'threshold': round(thr, 4),
        'PR-AUC':    round(average_precision_score(y_true, y_proba), 4),
        'ROC-AUC':   round(roc_auc_score(y_true, y_proba), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
        'FPR':       round(fp / (fp + tn) if (fp + tn) > 0 else 0, 6),
        'FNR':       round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4),
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
    }

lr_m  = evaluate('LogReg (F1-max)',  y_test, lr_test_proba,  LR_THR)
xgb_m = evaluate('XGBoost (F1-max)', y_test, xgb_test_proba, XGB_THR)

print("\n" + "="*65)
print("TEST SET FINAL RESULTS")
print("="*65)
for m in [lr_m, xgb_m]:
    print()
    print("  Model    :", m['Model'], "(thr=" + str(m['threshold']) + ")")
    print("  PR-AUC   :", m['PR-AUC'])
    print("  ROC-AUC  :", m['ROC-AUC'])
    print("  Precision:", m['Precision'])
    print("  Recall   :", m['Recall'])
    print("  F1       :", m['F1'])
    print("  FPR      :", m['FPR'])
    print("  FNR      :", m['FNR'])
    total_fraud = m['TP'] + m['FN']
    print(f"  Confusion: TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")
    print(f"  Fraud detected: {m['TP']} / {total_fraud}  |  Missed (FN): {m['FN']}")

print("\n" + "="*65)
print("FEATURE IMPORTANCES (XGBoost gain)")
print("="*65)
imp = dict(zip(ML_FEATURES, xgb_model.feature_importances_))
for k, v in sorted(imp.items(), key=lambda x: -x[1]):
    bar = '#' * int(v * 40)
    print(f"  {k:20s}: {v:.4f}  {bar}")

# Save models
joblib.dump(xgb_model, MODEL_DIR + '/xgb_v1.pkl')
joblib.dump(logreg,    MODEL_DIR + '/logreg_v1.pkl')

meta = {
    'model': 'xgb_v1', 'trained_at': datetime.now().isoformat(),
    'ml_features': ML_FEATURES, 'random_state': RANDOM_STATE,
    'scale_pos_weight': round(spw, 4), 'best_iteration': int(best_round),
    'working_threshold': round(XGB_THR, 6),
    'val_metrics': {
        'pr_auc':  round(average_precision_score(y_val,  xgb_val_proba),  6),
        'roc_auc': round(roc_auc_score(y_val, xgb_val_proba), 6)
    },
    'test_metrics': {k: xgb_m[k] for k in ['PR-AUC','ROC-AUC','Precision','Recall','F1','FPR','FNR']},
    'baseline_logreg_test': {k: lr_m[k] for k in ['PR-AUC','ROC-AUC','F1']}
}
with open(MODEL_DIR + '/xgb_v1_meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("\nSaved: models/xgb_v1.pkl  |  logreg_v1.pkl  |  xgb_v1_meta.json")
