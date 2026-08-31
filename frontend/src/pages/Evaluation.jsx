import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { BarChart2, Target, TrendingUp, Activity, GitCompareArrows, Cpu, Info, LineChart } from "lucide-react";
import { getModelInfo, reportAssetUrl } from "../api.js";
import PageHeader from "../components/PageHeader.jsx";

export default function Evaluation() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModelInfo().then(setInfo).catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="error">Error: {error}</p>;
  if (!info) return <p className="loading">Loading</p>;

  const xgb = info.xgb || {};
  const test = xgb.test_metrics || {};
  const val = xgb.val_metrics || {};
  const baseline = xgb.baseline_logreg_test || {};
  const cal = info.calibration || {};

  return (
    <div>
      <PageHeader
        icon={LineChart}
        eyebrow="System"
        title="Model Evaluation"
        subtitle="Calibration, comparison, and explainability metrics"
        tone="gray"
      />

      <p className="note honesty-banner">
        <Info size={13} /> All metrics on this page are from model training and calibration
        (held-out test set, dated per card) - not live operational performance. For live
        automation behavior, see <Link to="/automation">Automation Status</Link>.
      </p>

      <div className="card">
        <h2><Target size={16} /> Model Comparison (Test Set)</h2>
        <table className="kv-table">
          <thead>
            <tr><th></th><th>Logistic Regression</th><th>XGBoost</th></tr>
          </thead>
          <tbody>
            <tr><th>PR-AUC</th><td>{baseline["PR-AUC"]}</td><td>{test["PR-AUC"]}</td></tr>
            <tr><th>ROC-AUC</th><td>{baseline["ROC-AUC"]}</td><td>{test["ROC-AUC"]}</td></tr>
            <tr><th>F1</th><td>{baseline["F1"]}</td><td>{test["F1"]}</td></tr>
            <tr><th>Precision</th><td> - </td><td>{test["Precision"]}</td></tr>
            <tr><th>Recall</th><td> - </td><td>{test["Recall"]}</td></tr>
            <tr><th>FPR</th><td> - </td><td>{test["FPR"]}</td></tr>
            <tr><th>FNR</th><td> - </td><td>{test["FNR"]}</td></tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2><Cpu size={16} /> Model Info (XGBoost)</h2>
        <table className="kv-table">
          <tbody>
            <tr><th>Trained At</th><td>{xgb.trained_at && new Date(xgb.trained_at).toLocaleString("en-US")}</td></tr>
            <tr><th>scale_pos_weight</th><td>{xgb.scale_pos_weight}</td></tr>
            <tr><th>Best Iteration (early stopping)</th><td>{xgb.best_iteration}</td></tr>
            <tr><th>Working Threshold</th><td>{xgb.working_threshold}</td></tr>
            <tr><th>Val PR-AUC / ROC-AUC</th><td>{val.pr_auc} / {val.roc_auc}</td></tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2><Activity size={16} /> Calibration (Brier Score)</h2>
        <table className="kv-table">
          <tbody>
            <tr><th>Method</th><td>{cal.method}</td></tr>
            <tr><th>Calibration Set</th><td>{cal.cal_set} ({cal.n_cal?.toLocaleString("en-US")} samples)</td></tr>
            <tr><th>Calibrated At</th><td>{cal.calibrated_at && new Date(cal.calibrated_at).toLocaleString("en-US")}</td></tr>
            {Object.entries(cal.brier_scores || {}).map(([k, v]) => (
              <tr key={k}><th>Brier - {k}</th><td>{v}</td></tr>
            ))}
            {Object.entries(cal.bss || {}).map(([k, v]) => (
              <tr key={`bss-${k}`}><th>Brier Skill Score - {k}</th><td>{v >= 0 ? "+" : ""}{v}</td></tr>
            ))}
          </tbody>
        </table>
        {cal.note && <p className="note">{cal.note}</p>}
      </div>

      <div className="card">
        <h2><GitCompareArrows size={16} /> LogReg vs XGBoost vs Hybrid</h2>
        <img className="report-img" src={reportAssetUrl("model_comparison_pr_curve.png")} alt="Model comparison PR curve" />
      </div>

      <div className="card">
        <h2><TrendingUp size={16} /> Calibration Curve</h2>
        <img className="report-img" src={reportAssetUrl("calibration_curve.png")} alt="Calibration curve" />
      </div>

      <div className="card">
        <h2><BarChart2 size={16} /> Brier Score Comparison</h2>
        <img className="report-img" src={reportAssetUrl("brier_score_comparison.png")} alt="Brier score comparison" />
      </div>

      <div className="card">
        <h2><BarChart2 size={16} /> Global SHAP Summary</h2>
        <img className="report-img" src={reportAssetUrl("shap_feature_importance.png")} alt="Global SHAP feature importance ranking" />
      </div>

      <p className="note">
        Precedent Analysis suggestions and automation behavior are tracked on the{" "}
        <Link to="/automation">Automation Status</Link> page, not here - this page covers the
        scoring model only.
      </p>
    </div>
  );
}
