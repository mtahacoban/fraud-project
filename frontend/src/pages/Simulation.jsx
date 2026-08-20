import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, PlayCircle, Radar, Shuffle } from "lucide-react";
import { getSimulationTemplates, runSimulation } from "../api.js";
import RiskBadge from "../components/RiskBadge.jsx";
import ShapChart from "../components/ShapChart.jsx";

const TYPE_OPTIONS = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"];

// "Future Signals": illustrative-only fields (never read by
// scoring/rules/precedent — see backend/schemas.py's TransactionIn
// comment and the Simulation panel's own disclaimer below). Defaulted
// here so the inputs are always controlled, including right after a
// template reset (see applyTemplate).
const CHANNEL_OPTIONS = ["mobile", "web", "atm", "branch"];
const COUNTRY_OPTIONS = ["DE", "GB", "NL", "TR", "US"];

const EMPTY_FORM = {
  step: 12, type: "TRANSFER", amount: 1000,
  oldbalanceOrg: 1000, newbalanceOrig: 0, oldbalanceDest: 0, newbalanceDest: 0,
  device_id: "", is_known_device: 0, login_country: "DE", geo_velocity_flag: 0, channel: "mobile",
};

export default function Simulation() {
  const [templates, setTemplates] = useState(null);
  const [templatesError, setTemplatesError] = useState(null);
  const [templateId, setTemplateId] = useState("");
  const [form, setForm] = useState(EMPTY_FORM);
  const [count, setCount] = useState(1);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    getSimulationTemplates().then(setTemplates).catch((e) => setTemplatesError(e.message));
  }, []);

  function applyTemplate(id) {
    setTemplateId(id);
    const tpl = templates?.find((t) => t.id === id);
    if (tpl) {
      // Deliberately excluded from the prefilled form (no label leakage: never pre-fill isFraud/identity fields)
      const { nameOrig: _nameOrig, nameDest: _nameDest, isFraud: _isFraud, ...fields } = tpl.input;
      // Templates never carry Future Signals fields — merge over EMPTY_FORM
      // so those inputs reset to a defined default instead of going
      // uncontrolled (undefined).
      setForm({ ...EMPTY_FORM, ...fields });
    }
  }

  function setField(key, value) {
    setTemplateId("");
    setForm((f) => ({ ...f, [key]: value }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    setResult(null);
    try {
      const payload = {
        transaction: {
          step: Number(form.step),
          type: form.type,
          amount: Number(form.amount),
          oldbalanceOrg: Number(form.oldbalanceOrg),
          newbalanceOrig: Number(form.newbalanceOrig),
          oldbalanceDest: Number(form.oldbalanceDest),
          newbalanceDest: Number(form.newbalanceDest),
          // Future Signals — illustrative only (see the panel below and
          // backend/schemas.py's TransactionIn comment). Sent through so
          // the extensibility story is concrete end-to-end, but verified
          // to never reach scoring/rules/precedent similarity.
          device_id: form.device_id ? String(form.device_id) : undefined,
          is_known_device: Number(form.is_known_device),
          login_country: form.login_country,
          geo_velocity_flag: Number(form.geo_velocity_flag),
          channel: form.channel,
        },
        count: Number(count),
      };
      setResult(await runSimulation(payload));
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  const single = result && result.scored === 1 ? result.results[0] : null;

  return (
    <div>
      <h1>Simulation</h1>
      <p className="subtitle">Build or generate transactions and see how the model scores them in real time</p>

      <div className="grid-2">
        <div className="card">
          <h2><Shuffle size={16} /> Build a Transaction</h2>
          {templatesError && <p className="error">Error loading templates: {templatesError}</p>}

          <form onSubmit={handleSubmit}>
            <div className="decision-form" style={{ maxWidth: "none", marginBottom: 14 }}>
              <label>
                Template
                <select value={templateId} onChange={(e) => applyTemplate(e.target.value)}>
                  <option value="">Custom</option>
                  {templates?.map((t) => (
                    <option key={t.id} value={t.id}>{t.label}</option>
                  ))}
                </select>
              </label>
              {templateId && (
                <p className="note" style={{ marginTop: -6 }}>
                  {templates.find((t) => t.id === templateId)?.description}
                </p>
              )}
            </div>

            <div
              className="decision-form"
              style={{ maxWidth: "none", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 14 }}
            >
              <label>
                Type
                <select value={form.type} onChange={(e) => setField("type", e.target.value)}>
                  {TYPE_OPTIONS.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </label>
              <label>
                Hour of day (step)
                <input type="number" min="0" value={form.step} onChange={(e) => setField("step", e.target.value)} />
              </label>
              <label>
                Amount
                <input type="number" step="0.01" min="0" value={form.amount} onChange={(e) => setField("amount", e.target.value)} />
              </label>
              <label>
                Source Balance (before)
                <input type="number" step="0.01" min="0" value={form.oldbalanceOrg} onChange={(e) => setField("oldbalanceOrg", e.target.value)} />
              </label>
              <label>
                Source Balance (after)
                <input type="number" step="0.01" min="0" value={form.newbalanceOrig} onChange={(e) => setField("newbalanceOrig", e.target.value)} />
              </label>
              <label>
                Destination Balance (before)
                <input type="number" step="0.01" min="0" value={form.oldbalanceDest} onChange={(e) => setField("oldbalanceDest", e.target.value)} />
              </label>
              <label>
                Destination Balance (after)
                <input type="number" step="0.01" min="0" value={form.newbalanceDest} onChange={(e) => setField("newbalanceDest", e.target.value)} />
              </label>
              <label>
                Generate N (random variations)
                <input type="number" min="1" max="500" value={count} onChange={(e) => setCount(e.target.value)} />
              </label>
            </div>

            <div className="illustrative-panel">
              <p className="illustrative-panel-caption" style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
                <Radar size={13} /> <strong>Future Signals</strong> (illustrative — not scored)
              </p>
              <p className="illustrative-panel-caption">
                These signals are illustrative — a production system would incorporate device
                fingerprinting and geo-velocity; they are <strong>not used</strong> in the current
                model or score.
              </p>
              <div
                className="decision-form"
                style={{ maxWidth: "none", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}
              >
                <label>
                  Channel
                  <select value={form.channel} onChange={(e) => setField("channel", e.target.value)}>
                    {CHANNEL_OPTIONS.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </label>
                <label>
                  Login Country
                  <select value={form.login_country} onChange={(e) => setField("login_country", e.target.value)}>
                    {COUNTRY_OPTIONS.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </label>
                <label>
                  Device ID
                  <input
                    type="text" placeholder="e.g. 82931"
                    value={form.device_id} onChange={(e) => setField("device_id", e.target.value)}
                  />
                </label>
                <label className="illustrative-checkbox">
                  <input
                    type="checkbox" checked={!!form.is_known_device}
                    onChange={(e) => setField("is_known_device", e.target.checked ? 1 : 0)}
                  />
                  Known device
                </label>
                <label className="illustrative-checkbox">
                  <input
                    type="checkbox" checked={!!form.geo_velocity_flag}
                    onChange={(e) => setField("geo_velocity_flag", e.target.checked ? 1 : 0)}
                  />
                  Geo-velocity flag (impossible travel)
                </label>
              </div>
            </div>

            {error && <p className="error" style={{ marginBottom: 14 }}>Error: {error}</p>}

            <div className="action-btns" style={{ marginTop: 16 }}>
              <button type="submit" className="action-btn action-btn-neutral" disabled={submitting}>
                <PlayCircle size={15} />
                {submitting ? "Scoring…" : Number(count) > 1 ? `Generate ${count} & Score` : "Score / Submit"}
              </button>
            </div>
          </form>
        </div>

        <div className="card">
          <h2><PlayCircle size={16} /> Live Result</h2>

          {!result && !submitting && (
            <p className="note">Build a transaction (or pick a template) and submit to see the model's score here.</p>
          )}
          {submitting && <p className="loading">Scoring</p>}

          {single && (
            <>
              <div className="header-stats" style={{ margin: "0 0 18px", padding: 0, borderTop: "none" }}>
                <div className="header-stat" style={{ paddingLeft: 0 }}>
                  <span className="header-stat-label">Risk Band</span>
                  <RiskBadge band={single.risk_band} score={single.hybrid_score} />
                </div>
                <div className="header-stat">
                  <span className="header-stat-label">Calibrated Probability</span>
                  <span className="header-stat-value">{(single.calibrated_proba * 100).toFixed(2)}%</span>
                </div>
                <div className="header-stat">
                  <span className="header-stat-label">Hybrid Score</span>
                  <span className="header-stat-value">{single.hybrid_score.toFixed(0)}<small>/100</small></span>
                </div>
              </div>

              <ShapChart
                data={single.shap_factors}
                height={200}
                emptyMessage="SHAP explanation is not available for this result."
              />

              <p className="note" style={{ marginTop: 16 }}>
                <Link to={single.case_id ? `/triage?case=${single.case_id}` : "/triage"}>
                  Added to queue — view in Triage <ArrowRight size={12} style={{ verticalAlign: "middle" }} />
                </Link>
              </p>
              <p className="note">Device/location/channel signals above are illustrative and were not used in this score.</p>
            </>
          )}

          {result && result.scored > 1 && (
            <>
              <div className="header-stats" style={{ margin: "0 0 18px", padding: 0, borderTop: "none" }}>
                <div className="header-stat" style={{ paddingLeft: 0 }}>
                  <span className="header-stat-label">Scored</span>
                  <span className="header-stat-value">{result.scored}</span>
                </div>
                <div className="header-stat">
                  <span className="header-stat-label">Red</span>
                  <span className="header-stat-value" style={{ color: "var(--red)" }}>{result.band_counts.RED || 0}</span>
                </div>
                <div className="header-stat">
                  <span className="header-stat-label">Gray</span>
                  <span className="header-stat-value" style={{ color: "var(--gray)" }}>{result.band_counts.GRAY || 0}</span>
                </div>
              </div>
              <p className="note">
                Green (auto-clean, {result.band_counts.GREEN || 0}) transactions don't open a case.
              </p>
              <p className="note" style={{ marginTop: 6 }}>
                <Link to="/triage">
                  Added to queue — view in Triage <ArrowRight size={12} style={{ verticalAlign: "middle" }} />
                </Link>
              </p>
              <p className="note">Device/location/channel signals above are illustrative and were not used in scoring.</p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
