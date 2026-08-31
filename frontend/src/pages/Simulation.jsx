import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowRight, Bot, Check, FlaskConical, Info, ListChecks, PlayCircle, Radar, ShieldAlert, Shuffle, Users, X,
} from "lucide-react";
import {
  getCaseAutomationGates, getCasePrecedents, getCaseReport, getSimulationTemplates, runSimulation,
} from "../api.js";
import { automationGateValueText, automationGateVisualState } from "./triage/CaseDetailPanel.jsx";
import PageHeader from "../components/PageHeader.jsx";
import RiskBadge from "../components/RiskBadge.jsx";
import ShapChart from "../components/ShapChart.jsx";
import SourceBadge from "../components/SourceBadge.jsx";

const TYPE_OPTIONS = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"];

const REPORT_POLL_INTERVAL_MS = 2000;
const REPORT_POLL_MAX_ATTEMPTS = 10;
const PRECEDENT_POLL_INTERVAL_MS = 2000;
const PRECEDENT_POLL_MAX_ATTEMPTS = 10;

const AUTOMATION_GATE_LABELS = {
  direction_automatable: "Suggested Direction",
  similarity: "Precedent Similarity",
  precedent_count: "Precedent Count",
  consensus: "Precedent Consensus",
  calibrated_proba: "Model Confidence (Calibrated Probability)",
  hard_rule_conflict: "Hard-Rule Conflict Check",
};

const ACTION_LABELS = { confirm_fraud: "Confirmed as fraud", approve_clean: "Closed as clean", escalate: "Escalated" };
const DECISION_COLOR_VAR = { confirm_fraud: "--red", approve_clean: "--green", escalate: "--gray" };
const BAND_REASON_LABELS = {
  fast_path: "Fast-path type gate (PAYMENT/CASH_IN/DEBIT - model never called)",
  high_confidence_override: "High-confidence promotion (calibrated p≥0.95, no rule confirmation)",
};

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

  const [reportState, setReportState] = useState(null);
  const [reportPollGaveUp, setReportPollGaveUp] = useState(false);
  const reportPollTimeout = useRef(null);
  const [precedentsState, setPrecedentsState] = useState(null);
  const [precedentPollGaveUp, setPrecedentPollGaveUp] = useState(false);
  const precedentPollTimeout = useRef(null);
  const [automationGates, setAutomationGates] = useState(undefined);

  useEffect(() => {
    getSimulationTemplates().then(setTemplates).catch((e) => setTemplatesError(e.message));
  }, []);

  const singleCaseId = result && result.scored === 1 ? result.results[0].case_id : null;

  useEffect(() => {
    if (!singleCaseId) {
      setReportState(null);
      setPrecedentsState(null);
      setAutomationGates(undefined);
      return;
    }
    let cancelled = false;
    setReportPollGaveUp(false);
    setPrecedentPollGaveUp(false);

    async function pollReport(attempt) {
      let res;
      try {
        res = await getCaseReport(singleCaseId);
      } catch {
        return;
      }
      if (cancelled) return;
      setReportState(res);
      if (res.status === "generating") {
        if (attempt < REPORT_POLL_MAX_ATTEMPTS) {
          reportPollTimeout.current = setTimeout(() => pollReport(attempt + 1), REPORT_POLL_INTERVAL_MS);
        } else {
          setReportPollGaveUp(true);
        }
      }
    }

    async function pollPrecedents(attempt) {
      let res;
      try {
        res = await getCasePrecedents(singleCaseId);
      } catch {
        return;
      }
      if (cancelled) return;
      setPrecedentsState(res);
      if (res.explanation.status === "generating") {
        if (attempt < PRECEDENT_POLL_MAX_ATTEMPTS) {
          precedentPollTimeout.current = setTimeout(() => pollPrecedents(attempt + 1), PRECEDENT_POLL_INTERVAL_MS);
        } else {
          setPrecedentPollGaveUp(true);
        }
      }
    }

    getCaseAutomationGates(singleCaseId).then(setAutomationGates).catch(() => setAutomationGates(null));
    pollReport(1);
    pollPrecedents(1);

    return () => {
      cancelled = true;
      if (reportPollTimeout.current) clearTimeout(reportPollTimeout.current);
      if (precedentPollTimeout.current) clearTimeout(precedentPollTimeout.current);
    };
  }, [singleCaseId]);

  function applyTemplate(id) {
    setTemplateId(id);
    const tpl = templates?.find((t) => t.id === id);
    if (tpl) {
      const { nameOrig: _nameOrig, nameDest: _nameDest, isFraud: _isFraud, ...fields } = tpl.input;
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
      <PageHeader
        icon={FlaskConical}
        eyebrow="System"
        title="Simulation"
        subtitle="Build or generate transactions and see how the model scores them in real time"
        tone="green"
      />

      <p className="note honesty-banner">
        <Info size={13} /> Simulated transactions are written and scored exactly like real ones - a real case may open (visible in Triage) and a real Investigator Report may generate.
        Only the <code>source=simulator</code> tag distinguishes them from a live transaction.
      </p>

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

            <p
              className="note simulation-groq-note"
              style={Number(count) > 50 ? { color: "var(--accent)", fontWeight: 600 } : undefined}
            >
              Each RED/GRAY result generates a real background report (Groq) - high counts can use
              meaningful API quota.
              {Number(count) > 50 && " This run may trigger dozens of real report generations."}
            </p>

            <div className="illustrative-panel">
              <p className="illustrative-panel-caption" style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
                <Radar size={13} /> <strong>Future Signals</strong> (illustrative - not scored)
              </p>
              <p className="illustrative-panel-caption">
                These signals are illustrative - a production system would incorporate device
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

              {single.case_id ? (
                <p className="note" style={{ marginTop: 16 }}>
                  <Link to={`/triage?case=${single.case_id}`}>
                    Real case opened - Case #{single.case_id}, view in Triage <ArrowRight size={12} style={{ verticalAlign: "middle" }} />
                  </Link>
                </p>
              ) : (
                <p className="note" style={{ marginTop: 16 }}>GREEN - auto-cleared, no case opened, nothing further to show below.</p>
              )}
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
                  Added to queue - view in Triage <ArrowRight size={12} style={{ verticalAlign: "middle" }} />
                </Link>
              </p>
              <p className="note">Device/location/channel signals above are illustrative and were not used in scoring.</p>
            </>
          )}
        </div>
      </div>

      {single?.case_id && (
        <>
          <div className="card">
            <h2><ShieldAlert size={16} /> Triggered Rules</h2>
            {single.hard_rule_hits.length === 0 && single.soft_rule_hits.length === 0 && single.band_reason !== "high_confidence_override" ? (
              <p className="note">No rules triggered.</p>
            ) : (
              <ul className="rule-list">
                {single.hard_rule_hits.map((name) => (
                  <li key={name} className="rule-hard"><strong>{name}</strong> - critical risk indicator</li>
                ))}
                {single.soft_rule_hits.map((name) => (
                  <li key={name} className="rule-soft"><strong>{name}</strong> - soft rule</li>
                ))}
                {single.band_reason === "high_confidence_override" && (
                  <li className="rule-override">
                    <strong>high_confidence_override</strong> - no rule confirmation, calibrated probability ≥0.95 promoted to RED
                  </li>
                )}
              </ul>
            )}
            {single.band_reason && BAND_REASON_LABELS[single.band_reason] && (
              <p className="note" style={{ marginTop: 10 }}>{BAND_REASON_LABELS[single.band_reason]}</p>
            )}
          </div>

          <div className="card card-info">
            <div className="card-head-row">
              <h2><Bot size={16} /> Investigator Report</h2>
              <span className="card-caption">Informational only · not a decision</span>
            </div>
            {!reportState && <p className="loading">Loading</p>}
            {reportState?.status === "generating" && (
              <p className="note report-generating">
                <span className="report-spinner" />
                {reportPollGaveUp ? "Still generating - check this case in Triage later." : "Report is being generated…"}
              </p>
            )}
            {reportState?.status === "ready" && reportState.report && (
              <>
                <SourceBadge source={reportState.report.source} />
                <p className="note report-meta">
                  Model: <strong>{reportState.report.model_name}</strong> · Generated: {new Date(reportState.report.generated_at).toLocaleString("en-US")}
                </p>
                <p className="report-text">{reportState.report.report_text}</p>
              </>
            )}
          </div>

          <div className="card card-info">
            <div className="card-head-row">
              <h2><Users size={16} /> Precedent Analysis</h2>
              <span className="card-caption">Informational only · not a decision</span>
            </div>
            {!precedentsState && <p className="loading">Loading</p>}
            {precedentsState && (
              <>
                {precedentsState.summary.suggested_decision ? (
                  <div className="precedent-suggestion">
                    <span
                      className="status-pill"
                      style={{ color: `var(${DECISION_COLOR_VAR[precedentsState.summary.suggested_decision] || "--text-muted"})` }}
                    >
                      Suggested: {ACTION_LABELS[precedentsState.summary.suggested_decision] || precedentsState.summary.suggested_decision}
                    </span>
                    <span className="note">
                      {precedentsState.summary.precedent_count} similar cases · {(precedentsState.summary.consensus_ratio * 100).toFixed(0)}% consensus · {(precedentsState.summary.avg_similarity * 100).toFixed(0)}% avg. similarity
                    </span>
                  </div>
                ) : (
                  <p className="note">{precedentsState.summary.note || "insufficient precedent - use judgment"}</p>
                )}
                {precedentsState.summary.common_patterns?.length > 0 && (
                  <p className="note precedent-pattern">
                    Most common pattern: {precedentsState.summary.common_patterns
                      .map((p) => `${p.rule} (${p.count}/${p.total})`)
                      .join(", ")}
                  </p>
                )}
                {precedentsState.summary.common_reason_codes?.length > 0 && (
                  <p className="note precedent-pattern">
                    Reason codes: {precedentsState.summary.common_reason_codes
                      .map((c) => `${c.count} ${c.rule}`)
                      .join(", ")}
                  </p>
                )}
                {precedentsState.explanation.status === "generating" ? (
                  <p className="note report-generating">
                    <span className="report-spinner" />
                    {precedentPollGaveUp ? "Still generating - check this case in Triage later." : "Explanation is being generated…"}
                  </p>
                ) : precedentsState.explanation.text && (
                  <>
                    <SourceBadge source={precedentsState.explanation.source} />
                    <p className="report-text">{precedentsState.explanation.text}</p>
                  </>
                )}
              </>
            )}
          </div>

          <div className="card card-info">
            <div className="card-head-row">
              <h2><ListChecks size={16} /> Automation Eligibility</h2>
              <span className="card-caption">Informational only · not a decision</span>
            </div>
            {automationGates === undefined && <p className="loading">Loading</p>}
            {automationGates === null && (
              <p className="note">Automation is not active for this deployment (no active policy) - nothing to evaluate.</p>
            )}
            {automationGates && (
              <>
                <p className="note" style={{ marginBottom: 12 }}>
                  {automationGates.eligible ? "This case clears every automation gate." : "This case has not cleared every automation gate."}
                  {" "}Policy {automationGates.policy_version}.
                </p>
                <ul className="automation-gate-list">
                  {automationGates.gates.map((g) => {
                    const state = automationGateVisualState(g);
                    return (
                      <li
                        key={g.gate}
                        className={`automation-gate-row automation-gate-${state}`}
                        title={state === "close" ? "Close to the threshold - a visual hint, not a system classification. This gate did not pass." : undefined}
                      >
                        {g.passed ? <Check size={14} /> : <X size={14} />}
                        <span className="automation-gate-name">{AUTOMATION_GATE_LABELS[g.gate] || g.gate}</span>
                        <span className="automation-gate-value">{automationGateValueText(g)}</span>
                      </li>
                    );
                  })}
                </ul>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}
