import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle, ArrowUpCircle, BarChart3, Bot, Check, ClipboardCheck, ClipboardList, Cog,
  FileDown, FileText, Gauge, Gavel, Globe, History, ListChecks, Lock, LockOpen, RotateCcw,
  ShieldAlert, ShieldCheck, ShieldX, Smartphone, User, Users, X, Zap,
} from "lucide-react";
import {
  confirmAiDecision, getCase, getCaseAuditTrail, getCaseAutomationGates, getCasePrecedents,
  getCaseReport, getCaseReportFindings, getPendingAiDecision, postDecision, rejectAiDecision,
  reopenCase,
} from "../../api.js";
import CaseTimeline from "../../components/CaseTimeline.jsx";
import DecisionFlow from "../../components/DecisionFlow.jsx";
import ReasonCodeCombobox from "../../components/ReasonCodeCombobox.jsx";
import RiskBadge from "../../components/RiskBadge.jsx";
import ShapChart from "../../components/ShapChart.jsx";
import SourceBadge from "../../components/SourceBadge.jsx";
import TransactionFlow from "../../components/TransactionFlow.jsx";

export function countryFlagEmoji(code) {
  if (!code || code.length !== 2) return null;
  const points = [...code.toUpperCase()].map((c) => 0x1f1e6 - 65 + c.charCodeAt(0));
  return String.fromCodePoint(...points);
}

const REPORT_POLL_INTERVAL_MS = 2000;
const REPORT_POLL_MAX_ATTEMPTS = 10;

const PRECEDENT_POLL_INTERVAL_MS = 2000;
const PRECEDENT_POLL_MAX_ATTEMPTS = 10;

const DECISION_COLOR_VAR = {
  confirm_fraud: "--red",
  approve_clean: "--green",
  escalate: "--gray",
};

const ACTIONS = [
  { value: "confirm_fraud", label: "Mark as fraud", icon: ShieldX, cls: "action-btn-red" },
  { value: "approve_clean", label: "Close as clean", icon: ShieldCheck, cls: "action-btn-green" },
  { value: "escalate", label: "Escalate", icon: ArrowUpCircle, cls: "action-btn-neutral" },
];

const ACTION_LABELS = {
  confirm_fraud: "Confirmed as fraud",
  approve_clean: "Closed as clean",
  escalate: "Escalated",
  reopened: "Reopened",
};

const BAND_REASON_LABELS = {
  fast_path: "Fast-path type gate (PAYMENT/CASH_IN/DEBIT - model never called)",
  high_confidence_override: "High-confidence promotion (calibrated p≥0.95, no rule confirmation)",
};

const AUTOMATION_GATE_LABELS = {
  direction_automatable: "Suggested Direction",
  similarity: "Precedent Similarity",
  precedent_count: "Precedent Count",
  consensus: "Precedent Consensus",
  calibrated_proba: "Model Confidence (Calibrated Probability)",
  hard_rule_conflict: "Hard-Rule Conflict Check",
};

export function automationGateVisualState(g) {
  if (g.passed) return "pass";
  if (g.actual === null || g.threshold === null) return "fail";
  const gap = g.threshold - g.actual;
  const isClose = g.gate === "precedent_count" ? gap <= 2 : gap < 0.05;
  return isClose ? "close" : "fail";
}

export function automationGateValueText(g) {
  if (g.actual === null || g.threshold === null) return g.detail;
  if (g.gate === "precedent_count") return `${g.actual} (need ≥ ${g.threshold})`;
  return `${(g.actual * 100).toFixed(1)}% (need ≥ ${(g.threshold * 100).toFixed(0)}%)`;
}

const AUDIT_ACTOR_ICONS = { System: Cog, AI: Bot, Analyst: User };
const AUDIT_ANOMALY_LABELS = {
  ai_human_conflict: "Overrode AI suggestion",
  no_reason: "No reason recorded",
  rapid_redecision: "Rapid re-decision",
};

export default function CaseDetailPanel({ caseId, onCaseChanged }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);
  const [note, setNote] = useState({ analyst_reason_code: "", analyst_note: "" });
  const [submitting, setSubmitting] = useState(null);
  const [reopenNote, setReopenNote] = useState("");
  const [reopening, setReopening] = useState(false);
  const [reportState, setReportState] = useState(null);
  const [reportPollGaveUp, setReportPollGaveUp] = useState(false);
  const reportPollTimeout = useRef(null);
  const [precedentsState, setPrecedentsState] = useState(null);
  const [precedentPollGaveUp, setPrecedentPollGaveUp] = useState(false);
  const precedentPollTimeout = useRef(null);
  const [pendingAiDecision, setPendingAiDecision] = useState(undefined);
  const [aiActionSubmitting, setAiActionSubmitting] = useState(false);
  const [showAiRejectForm, setShowAiRejectForm] = useState(false);
  const [aiRejectReason, setAiRejectReason] = useState("");
  const [exporting, setExporting] = useState(false);
  const exportRootRef = useRef(null);
  const [automationGates, setAutomationGates] = useState(undefined);
  const [showFindings, setShowFindings] = useState(false);
  const [reportFindings, setReportFindings] = useState(undefined);
  const [auditTrail, setAuditTrail] = useState(undefined);

  const load = () => getCase(caseId).then(setDetail).catch((e) => setError(e.message));
  const loadPendingAiDecision = () =>
    getPendingAiDecision(caseId).then(setPendingAiDecision).catch(() => setPendingAiDecision(null));
  const loadAutomationGates = () =>
    getCaseAutomationGates(caseId).then(setAutomationGates).catch(() => setAutomationGates(null));
  const loadAuditTrail = () =>
    getCaseAuditTrail(caseId).then(setAuditTrail).catch(() => setAuditTrail(null));
  function toggleFindings() {
    setShowFindings((v) => !v);
    if (reportFindings === undefined) {
      getCaseReportFindings(caseId).then(setReportFindings).catch(() => setReportFindings(null));
    }
  }

  useEffect(() => {
    load();
    loadPendingAiDecision();
    loadAutomationGates();
    loadAuditTrail();
  }, [caseId]);

  useEffect(() => {
    let cancelled = false;

    async function poll(attempt) {
      let res;
      try {
        res = await getCaseReport(caseId);
      } catch {
        return;
      }
      if (cancelled) return;
      setReportState(res);
      if (res.status === "generating") {
        if (attempt < REPORT_POLL_MAX_ATTEMPTS) {
          reportPollTimeout.current = setTimeout(() => poll(attempt + 1), REPORT_POLL_INTERVAL_MS);
        } else {
          setReportPollGaveUp(true);
        }
      }
    }

    poll(1);

    return () => {
      cancelled = true;
      if (reportPollTimeout.current) clearTimeout(reportPollTimeout.current);
    };
  }, [caseId]);

  useEffect(() => {
    let cancelled = false;

    async function poll(attempt) {
      let res;
      try {
        res = await getCasePrecedents(caseId);
      } catch {
        return;
      }
      if (cancelled) return;
      setPrecedentsState(res);
      if (res.explanation.status === "generating") {
        if (attempt < PRECEDENT_POLL_MAX_ATTEMPTS) {
          precedentPollTimeout.current = setTimeout(() => poll(attempt + 1), PRECEDENT_POLL_INTERVAL_MS);
        } else {
          setPrecedentPollGaveUp(true);
        }
      }
    }

    poll(1);

    return () => {
      cancelled = true;
      if (precedentPollTimeout.current) clearTimeout(precedentPollTimeout.current);
    };
  }, [caseId]);

  async function handleDecision(action_taken) {
    if (!note.analyst_reason_code?.trim()) return;
    setSubmitting(action_taken);
    setError(null);
    try {
      await postDecision(caseId, { action_taken, ...note });
      await load();
      await loadAuditTrail();
      onCaseChanged?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(null);
    }
  }

  async function handleApproveAi() {
    setAiActionSubmitting(true);
    setError(null);
    try {
      await confirmAiDecision(caseId, {});
      await load();
      await loadPendingAiDecision();
      await loadAuditTrail();
      onCaseChanged?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setAiActionSubmitting(false);
    }
  }

  async function handleRejectAi() {
    if (!aiRejectReason.trim()) return;
    setAiActionSubmitting(true);
    setError(null);
    try {
      await rejectAiDecision(caseId, { rejection_reason: aiRejectReason.trim() });
      setShowAiRejectForm(false);
      setAiRejectReason("");
      await loadPendingAiDecision();
      await loadAuditTrail();
      onCaseChanged?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setAiActionSubmitting(false);
    }
  }

  async function handleReopen() {
    setReopening(true);
    setError(null);
    try {
      await reopenCase(caseId, { analyst_note: reopenNote || undefined });
      setReopenNote("");
      await load();
      await loadAutomationGates();
      await loadAuditTrail();
      onCaseChanged?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setReopening(false);
    }
  }

  async function handleExportPdf() {
    setExporting(true);
    try {
      const [{ default: html2canvas }, { default: jsPDF }] = await Promise.all([
        import("html2canvas"),
        import("jspdf"),
      ]);
      const el = exportRootRef.current;
      el.classList.add("exporting-pdf");
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      try {
        const canvas = await html2canvas(el, { backgroundColor: "#f4f4f7", scale: 3 });
        const pdf = new jsPDF({ unit: "px", format: [canvas.width, canvas.height] });
        pdf.addImage(canvas.toDataURL("image/jpeg", 1.0), "JPEG", 0, 0, canvas.width, canvas.height);
        pdf.save(`case-${detail.case_id}-demo-export.pdf`);
      } finally {
        el.classList.remove("exporting-pdf");
      }
    } finally {
      setExporting(false);
    }
  }

  if (error && !detail) {
    return <div className="card triage-detail-status"><p className="error">Error: {error}</p></div>;
  }
  if (!detail) {
    return <div className="card triage-detail-status"><p className="loading">Loading</p></div>;
  }

  const { transaction: txn, score, rule_hits, shap_explanations, decisions } = detail;
  const bandVar = `var(--${(score?.risk_band || "gray").toLowerCase()})`;
  const isOverride = score?.band_reason === "high_confidence_override";
  const hasGhostDestination = rule_hits.some((h) => h.rule_name === "ghost_destination");

  return (
    <div ref={exportRootRef}>
      <div className="pdf-export-header">
        <p className="pdf-export-banner">
          <AlertTriangle size={13} /> DEMO DATA - Synthetic PaySim-derived case, not a real fraud investigation or compliance record.
        </p>
        <h2>Fraud Case Report - Case #{detail.case_id}</h2>
        <p className="note">Exported {new Date().toLocaleString("en-US")}</p>
      </div>

      <div className="card detail-header-card">
        <div className="detail-header">
          <div>
            <h1>
              {score?.risk_band === "RED" && <AlertTriangle size={19} />}
              Case #{detail.case_id}
            </h1>
            <p className="detail-subtitle">
              Transaction <strong>#{txn.id}</strong> · {new Date(detail.created_at).toLocaleString("en-US")} · Priority <strong>{detail.priority}</strong>
              {detail.closed_at && <> · Closed <strong>{new Date(detail.closed_at).toLocaleString("en-US")}</strong></>}
            </p>
          </div>
          <div className="detail-header-badges">
            <RiskBadge band={score?.risk_band} score={score?.hybrid_score} />
            <span className="status-pill">
              {detail.status === "OPEN" ? <LockOpen size={12} /> : <Lock size={12} />}
              {detail.status === "OPEN" ? "Open" : "Closed"}
            </span>
            <button type="button" className="pagination-btn no-export" disabled={exporting} onClick={handleExportPdf}>
              <FileDown size={13} /> {exporting ? "Exporting…" : "Export PDF"}
            </button>
          </div>
        </div>
        <div className="header-stats">
          <div className="header-stat">
            <span className="header-stat-label">Hybrid Score</span>
            <span className="header-stat-value" style={{ color: bandVar }}>
              {score?.hybrid_score.toFixed(0)}<small>/100</small>
            </span>
          </div>
          <div className="header-stat">
            <span className="header-stat-label">Calibrated Probability</span>
            <span className="header-stat-value">{score?.calibrated_proba.toFixed(2)}</span>
          </div>
          <div className="header-stat">
            <span className="header-stat-label">Model</span>
            <span className="header-stat-value" style={{ fontSize: 15 }}>{score?.model_version}</span>
          </div>
        </div>
      </div>

      <div className="card">
        <h2><FileText size={16} /> Transaction Details</h2>
        <table className="kv-table">
          <tbody>
            <tr><th>Transaction No.</th><td>#{txn.id}</td></tr>
            <tr><th>Type</th><td>{txn.type}</td></tr>
            <tr><th>Time Step (step)</th><td>{txn.step}</td></tr>
          </tbody>
        </table>

        <TransactionFlow txn={txn} hasGhostDestination={hasGhostDestination} />

        <p className="note" style={{ marginTop: 18, marginBottom: 8 }}>Login context (synthetic - not derived from PaySim):</p>
        <div className="badge-row">
          <span className="status-pill">
            {countryFlagEmoji(txn.login_country) || <Globe size={12} />} {txn.login_country || "Unknown"}
          </span>
          <span className="status-pill">
            <Smartphone size={12} /> {txn.channel || "—"}{txn.device_id ? ` · ${txn.is_known_device ? "known device" : "new device"}` : ""}
          </span>
          {txn.geo_velocity_flag ? (
            <span className="status-pill" style={{ borderColor: "var(--red)", color: "var(--red)" }}>
              <Zap size={12} /> Geo-velocity flagged
            </span>
          ) : null}
        </div>
      </div>

      <div className="card">
        <h2><History size={16} /> Case Timeline</h2>
        <CaseTimeline detail={detail} />
      </div>

      <DecisionFlow
        detail={detail}
        reportState={reportState}
        precedentsState={precedentsState}
        automationGates={automationGates}
        auditTrail={auditTrail}
      />

      <div className="card" id="flow-score">
        <h2><Gauge size={16} /> Score Breakdown</h2>
        <table className="kv-table">
          <tbody>
            <tr><th>ML Score</th><td>{score?.ml_score.toFixed(2)}</td></tr>
            <tr><th>Rule (Soft) Score</th><td>{score?.soft_score.toFixed(2)}</td></tr>
            <tr><th>Hybrid Score</th><td>{score?.hybrid_score.toFixed(2)}</td></tr>
            <tr><th>Calibrated Probability</th><td>{(score?.calibrated_proba * 100).toFixed(4)}%</td></tr>
            <tr><th>Model Version</th><td>{score?.model_version}</td></tr>
            {score?.band_reason && (
              <tr><th>Band Reason</th><td>{BAND_REASON_LABELS[score.band_reason] || score.band_reason}</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2><ShieldAlert size={16} /> Triggered Rules</h2>
        {rule_hits.length === 0 && !isOverride ? <p className="note">No rules triggered.</p> : (
          <ul className="rule-list">
            {rule_hits.map((h, i) => (
              <li key={i} className={`rule-${h.rule_type}`}>
                <strong>{h.rule_name}</strong> - {h.rule_type === "hard" ? "critical risk indicator" : "soft rule"}
              </li>
            ))}
            {isOverride && (
              <li className="rule-override">
                <strong>high_confidence_override</strong> - no rule confirmation, calibrated probability ≥0.95 promoted to RED
              </li>
            )}
          </ul>
        )}
      </div>

      <div className="card" id="flow-shap">
        <h2><BarChart3 size={16} /> Risk Factors (SHAP)</h2>
        <p className="card-subtext">Factors that contributed most to the model's decision</p>
        <ShapChart
          data={shap_explanations}
          height={220}
          emptyMessage="SHAP explanation is not available for this case."
          showLegend
        />
      </div>

      <div className="card card-info" id="flow-report">
        <div className="card-head-row">
          <h2><Bot size={16} /> Investigator Report</h2>
          <span className="card-caption">Informational only · not a decision</span>
        </div>
        {!reportState && <p className="loading">Loading</p>}
        {reportState?.status === "generating" && (
          <p className="note report-generating">
            <span className="report-spinner" />
            {reportPollGaveUp
              ? "Still generating - refresh the page to check again."
              : "Report is being generated…"}
          </p>
        )}
        {reportState?.status === "ready" && reportState.report && (
          <>
            <SourceBadge source={reportState.report.source} />
            <p className="note report-meta">
              Model: <strong>{reportState.report.model_name}</strong> · Generated: {new Date(reportState.report.generated_at).toLocaleString("en-US")}
            </p>
            <p className="report-text">{reportState.report.report_text}</p>

            <button type="button" className="report-findings-toggle" onClick={toggleFindings}>
              {showFindings ? "Hide findings this report was based on" : "Show findings this report was based on"}
            </button>

            {showFindings && (
              <div className="report-findings">
                {reportFindings === undefined && <p className="loading">Loading</p>}
                {reportFindings === null && <p className="note">Findings could not be loaded.</p>}
                {reportFindings && (
                  <>
                    <p className="note" style={{ marginBottom: 8 }}>{reportFindings.transaction_summary}</p>
                    <ul className="report-findings-list">
                      {reportFindings.findings.map((f, i) => <li key={i}>{f}</li>)}
                    </ul>
                    <p className="note report-findings-freshness">
                      Computed live from this case's score, SHAP factors, and triggered rules - deterministic, not
                      stored. The report text above was generated on {new Date(reportState.report.generated_at).toLocaleString("en-US")}.
                    </p>
                  </>
                )}
              </div>
            )}
          </>
        )}
      </div>

      <div className="card card-info" id="flow-precedent">
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
                {precedentPollGaveUp
                  ? "Still generating - refresh the page to check again."
                  : "Explanation is being generated…"}
              </p>
            ) : precedentsState.explanation.text && (
              <>
                <SourceBadge source={precedentsState.explanation.source} />
                <p className="report-text">{precedentsState.explanation.text}</p>
              </>
            )}

            {precedentsState.precedents.length > 0 && (
              <div className="table-wrap precedent-table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Case</th>
                      <th>Transaction</th>
                      <th>Risk</th>
                      <th>Similarity</th>
                      <th>Analyst Decision</th>
                    </tr>
                  </thead>
                  <tbody>
                    {precedentsState.precedents.map((p) => (
                      <tr key={p.case_id}>
                        <td><Link to={`/triage?case=${p.case_id}`}>Case #{p.case_id}</Link></td>
                        <td className="note">
                          {p.type
                            ? `${p.type} · ${p.amount.toLocaleString("en-US")} · ${String(p.step_hour).padStart(2, "0")}:00`
                            : "—"}
                        </td>
                        <td>{p.risk_band ? <RiskBadge band={p.risk_band} score={p.hybrid_score} /> : "—"}</td>
                        <td>{(p.similarity * 100).toFixed(0)}%</td>
                        <td>{ACTION_LABELS[p.analyst_decision] || p.analyst_decision}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>

      {detail.status === "OPEN" && (
        <div className="card card-info" id="flow-automation">
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
                {automationGates.eligible
                  ? "This case clears every automation gate."
                  : "This case has not cleared every automation gate."}
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
              <p className="note automation-gates-freshness">
                Computed live from the current precedent pool - if this case has a pending automation proposal above, these
                numbers can drift slightly from the ones it was created with as more cases get decided over time.
              </p>
            </>
          )}
        </div>
      )}

      {pendingAiDecision && (
        <div className="card card-pending">
          <div className="card-head-row">
            <h2><ShieldAlert size={16} /> Automation Proposed Decision - Awaiting Your Confirmation</h2>
            <span className="card-caption">Policy {pendingAiDecision.policy_version}</span>
          </div>
          <p className="pending-directive">
            Proposed action: <strong>{ACTION_LABELS.confirm_fraud}</strong>
          </p>
          <p className="note" style={{ marginBottom: 10 }}>
            Generated by a deterministic multi-condition gate - not the LLM, not the
            Precedent Analysis suggestion above. This is a distinct, higher-bar mechanism: nothing here ever
            finalizes without your explicit Approve.
          </p>
          <ul className="pending-reasoning">
            {pendingAiDecision.triggered_conditions.reason.map((r, i) => (
              <li key={i}><Check size={13} /> {r}</li>
            ))}
          </ul>
          {error && <p className="error" style={{ marginBottom: 14 }}>Error: {error}</p>}
          <div className="action-btns">
            <button
              type="button"
              className="action-btn action-btn-green"
              disabled={aiActionSubmitting}
              onClick={handleApproveAi}
            >
              <Check size={15} />
              {aiActionSubmitting ? "Submitting…" : "Approve"}
            </button>
            <button
              type="button"
              className="action-btn action-btn-red"
              disabled={aiActionSubmitting}
              onClick={() => setShowAiRejectForm((v) => !v)}
            >
              <X size={15} />
              Reject
            </button>
          </div>
          {showAiRejectForm && (
            <div className="decision-form" style={{ maxWidth: 480, marginTop: 14 }}>
              <label>
                <span>Reason for rejection <span className="field-required">(required)</span></span>
                <textarea
                  placeholder="e.g. manual review shows the destination account has prior legitimate history"
                  value={aiRejectReason}
                  onChange={(e) => setAiRejectReason(e.target.value)}
                />
              </label>
              <button
                type="button"
                className="action-btn action-btn-neutral"
                style={{ alignSelf: "flex-start" }}
                disabled={aiActionSubmitting || !aiRejectReason.trim()}
                onClick={handleRejectAi}
              >
                {aiActionSubmitting ? "Submitting…" : "Submit Rejection"}
              </button>
            </div>
          )}
          <p className="decision-note">
            Approve closes this case as confirmed fraud. Reject leaves it open for your own decision below - it does not close the case. Doing nothing leaves this proposal pending indefinitely; nothing here ever auto-finalizes.
          </p>
        </div>
      )}

      <div className="card" id="flow-decision">
        <h2><Gavel size={16} /> Analyst Decisions</h2>
        {decisions.length === 0 ? <p className="note">No decision yet.</p> : (
          <ul className="decision-list">
            {decisions.map((d, i) => (
              <li key={i}>
                <strong>{ACTION_LABELS[d.action_taken] || d.action_taken}</strong> - {d.analyst_reason_code || "—"}
                {d.analyst_note && ` · "${d.analyst_note}"`}
                {" "}<time>({new Date(d.decided_at).toLocaleString("en-US")})</time>
                {d.auto_processed && <> · <span className="note">auto-processed</span></>}
              </li>
            ))}
          </ul>
        )}
      </div>

      {detail.status === "OPEN" && (
        <div className="card no-export analyst-decision-card">
          <div className="analyst-decision-head">
            <span className="analyst-decision-badge"><ClipboardCheck size={16} /></span>
            <div>
              <h2>Analyst Decision</h2>
              <p className="analyst-decision-sub">
                Case #{detail.case_id} · {score?.risk_band} · {score?.hybrid_score.toFixed(0)}/100
              </p>
            </div>
          </div>
          <div className="decision-form" style={{ maxWidth: 440, marginBottom: 16 }}>
            <label>
              <span>Reason Code <span className="field-required">(required)</span></span>
              <ReasonCodeCombobox
                value={note.analyst_reason_code}
                onChange={(v) => setNote({ ...note, analyst_reason_code: v })}
              />
            </label>
            <label>
              Decision Rationale (optional)
              <textarea
                placeholder="e.g. suspected account takeover - sender balance fully drained"
                value={note.analyst_note}
                onChange={(e) => setNote({ ...note, analyst_note: e.target.value })}
              />
            </label>
          </div>
          {error && <p className="error" style={{ marginBottom: 14 }}>Error: {error}</p>}
          <div className="action-btns">
            {ACTIONS.map(({ value, label, icon: Icon, cls }) => {
              const reasonMissing = !note.analyst_reason_code?.trim();
              return (
                <button
                  key={value}
                  type="button"
                  className={`action-btn ${cls}`}
                  disabled={submitting !== null || reasonMissing}
                  title={reasonMissing ? "Reason code required" : undefined}
                  onClick={() => handleDecision(value)}
                >
                  <Icon size={15} />
                  {submitting === value ? "Saving…" : label}
                </button>
              );
            })}
          </div>
          <p className="decision-note analyst-decision-footnote"><Lock size={12} /> This decision moves the case to "CLOSED".</p>
        </div>
      )}

      {detail.status === "CLOSED" && (
        <div className="card no-export">
          <h2><RotateCcw size={16} /> Reopen Case</h2>
          <p className="note" style={{ marginTop: 0, marginBottom: 14 }}>
            This case is closed - you can reopen it for further review if needed. The previous decision stays in history and is not deleted.
          </p>
          <div className="decision-form" style={{ maxWidth: 440, marginBottom: 16 }}>
            <label>
              Reason for reopening (optional)
              <textarea
                placeholder="e.g. decision was made in error, new evidence surfaced"
                value={reopenNote}
                onChange={(e) => setReopenNote(e.target.value)}
              />
            </label>
          </div>
          {error && <p className="error" style={{ marginBottom: 14 }}>Error: {error}</p>}
          <div className="action-btns">
            <button
              type="button"
              className="action-btn action-btn-neutral"
              disabled={reopening}
              onClick={handleReopen}
            >
              <RotateCcw size={15} />
              {reopening ? "Reopening…" : "Reopen Case"}
            </button>
          </div>
        </div>
      )}

      <div className="card" id="flow-audit">
        <div className="card-head-row">
          <h2><ClipboardList size={16} /> Audit Trail</h2>
          <span className="card-caption">Informational only · not a decision</span>
        </div>

        {auditTrail === undefined && <p className="loading">Loading</p>}
        {auditTrail === null && <p className="note">Audit trail could not be loaded.</p>}
        {auditTrail && (
          <>
            <ul className="audit-trail-list">
              {auditTrail.events.map((e, i) => {
                const ActorIcon = AUDIT_ACTOR_ICONS[e.actor] || User;
                return (
                  <li key={i} className="audit-trail-row">
                    <span className={`audit-trail-actor audit-trail-actor-${e.actor.toLowerCase()}`} title={e.actor}>
                      <ActorIcon size={13} />
                    </span>
                    <div className="audit-trail-body">
                      <div className="audit-trail-top">
                        <span className="audit-trail-summary">{e.summary}</span>
                        <time className="audit-trail-time">{new Date(e.timestamp).toLocaleString("en-US")}</time>
                      </div>
                      {e.before && e.after && (
                        <span className="audit-trail-transition">{e.before} → {e.after}</span>
                      )}
                      {e.detail && <span className="audit-trail-detail">{e.detail}</span>}
                      {e.anomaly_flags.length > 0 && (
                        <div className="audit-trail-flags">
                          {e.anomaly_flags.map((f) => (
                            <span
                              key={f}
                              className="audit-trail-flag"
                              title="An observation, not a verdict - worth a second look, not necessarily a problem."
                            >
                              <AlertTriangle size={11} /> {AUDIT_ANOMALY_LABELS[f] || f}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          </>
        )}
      </div>

      <p className="pdf-export-footer">
        DEMO DATA - synthetic case, not a real fraud investigation record · Fraud Investigation Decision Support Platform
      </p>
    </div>
  );
}
