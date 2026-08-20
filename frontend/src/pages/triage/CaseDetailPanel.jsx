import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle, ArrowUpCircle, BarChart3, Bot, Check, FileText, Gauge,
  Gavel, Lock, LockOpen, RotateCcw, ShieldAlert, ShieldCheck, ShieldX, Users, X,
} from "lucide-react";
import {
  confirmAiDecision, getCase, getCasePrecedents, getCaseReport, getPendingAiDecision,
  postDecision, rejectAiDecision, reopenCase,
} from "../../api.js";
import RiskBadge from "../../components/RiskBadge.jsx";
import ShapChart from "../../components/ShapChart.jsx";
import SourceBadge from "../../components/SourceBadge.jsx";

// Ported from CaseDetail.jsx (Aşama 4.2.3) — identical polling/decision logic.
// Case Detail polls GET /cases/{id}/report while a report is "generating"
// (proactive generation on case creation usually beats the analyst to it,
// but this covers the lazy-trigger / cold-start case). Bounded — 10 tries
// at 2s is a ~20s ceiling, not an indefinite poll.
const REPORT_POLL_INTERVAL_MS = 2000;
const REPORT_POLL_MAX_ATTEMPTS = 10;

// Same bounded-poll shape for AI #2's precedent explanation — only ever
// actually "generating" when there's a suggested_decision (an LLM call was
// scheduled); the no-suggestion path returns "ready" on the first request.
const PRECEDENT_POLL_INTERVAL_MS = 2000;
const PRECEDENT_POLL_MAX_ATTEMPTS = 10;

// Suggested-decision pill color — reuses the same red/green/gray vocabulary
// as the decision-action buttons below, not the AI-blue used for source
// badges, so "what AI #2 suggests" and "who/what generated this text" stay
// visually distinct.
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
  fast_path: "Fast-path type gate (PAYMENT/CASH_IN/DEBIT — model never called)",
  high_confidence_override: "High-confidence promotion (calibrated p≥0.95, no rule confirmation)",
};

// caseId is used together with key={caseId} in TriageLayout — a case switch
// fully unmounts/remounts this component, so every effect below runs at
// most once per instance (caseId cannot change within one mounted
// instance's lifetime). That's why the manual "reset state for the new id"
// calls CaseDetail.jsx needed (it stayed mounted across id changes, since
// react-router doesn't remount on a param-only change within the same
// route) are gone here — the remount already guarantees fresh state.
// Polling cleanup (clearTimeout in each effect's return) still runs on
// unmount either way, which is what actually prevents a poll from one case
// leaking into the next.
export default function CaseDetailPanel({ caseId, onCaseChanged }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);
  const [note, setNote] = useState({ analyst_reason_code: "", analyst_note: "" });
  const [submitting, setSubmitting] = useState(null);
  const [reopenNote, setReopenNote] = useState("");
  const [reopening, setReopening] = useState(false);
  const [reportState, setReportState] = useState(null); // null = not fetched yet
  const [reportPollGaveUp, setReportPollGaveUp] = useState(false);
  const reportPollTimeout = useRef(null);
  const [precedentsState, setPrecedentsState] = useState(null);
  const [precedentPollGaveUp, setPrecedentPollGaveUp] = useState(false);
  const precedentPollTimeout = useRef(null);
  const [pendingAiDecision, setPendingAiDecision] = useState(undefined); // undefined = not fetched yet, null = none
  const [aiActionSubmitting, setAiActionSubmitting] = useState(false);
  const [showAiRejectForm, setShowAiRejectForm] = useState(false);
  const [aiRejectReason, setAiRejectReason] = useState("");

  const load = () => getCase(caseId).then(setDetail).catch((e) => setError(e.message));
  const loadPendingAiDecision = () =>
    getPendingAiDecision(caseId).then(setPendingAiDecision).catch(() => setPendingAiDecision(null));

  useEffect(() => {
    load();
    loadPendingAiDecision();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseId]);

  useEffect(() => {
    let cancelled = false;

    async function poll(attempt) {
      let res;
      try {
        res = await getCaseReport(caseId);
      } catch {
        return; // report status is secondary to the case itself — fail quietly
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
        return; // precedent status is secondary to the case itself — fail quietly
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
    setSubmitting(action_taken);
    setError(null);
    try {
      await postDecision(caseId, { action_taken, ...note });
      await load();
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
      onCaseChanged?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setReopening(false);
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

  return (
    <div>
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

      {pendingAiDecision && (
        <div className="card card-pending">
          <div className="card-head-row">
            <h2><ShieldAlert size={16} /> AI #2 Proposed Decision — Awaiting Your Confirmation</h2>
            <span className="card-caption">Policy {pendingAiDecision.policy_version}</span>
          </div>
          <p className="pending-directive">
            Proposed action: <strong>{ACTION_LABELS.confirm_fraud}</strong>
          </p>
          <p className="note" style={{ marginBottom: 10 }}>
            Generated by a deterministic multi-condition gate — not the LLM, not the
            AI #2 suggestion above. This is a distinct, higher-bar mechanism: nothing here ever
            finalizes without your explicit Approve.
          </p>
          {/* A proposal only ever exists when eligible=true — every gate below
              passed by construction, so failed_gates is always empty here.
              Shown as a plain checklist, not conditionally styled. */}
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
                Reason for rejection (required)
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
            Approve closes this case as confirmed fraud. Reject leaves it open for your own decision below —
            it does not close the case. Doing nothing leaves this proposal pending indefinitely; nothing here ever auto-finalizes.
          </p>
        </div>
      )}

      <div className="grid-2">
        <div className="card">
          <h2><FileText size={16} /> Transaction Details</h2>
          <table className="kv-table">
            <tbody>
              <tr><th>Transaction No.</th><td>#{txn.id}</td></tr>
              <tr><th>Type</th><td>{txn.type}</td></tr>
              <tr><th>Time Step (step)</th><td>{txn.step}</td></tr>
              <tr><th>Amount</th><td>{txn.amount.toLocaleString("en-US")}</td></tr>
              <tr><th>Source Balance (before → after)</th><td>{txn.oldbalanceOrg.toLocaleString("en-US")} → {txn.newbalanceOrig.toLocaleString("en-US")}</td></tr>
              <tr><th>Destination Balance (before → after)</th><td>{txn.oldbalanceDest.toLocaleString("en-US")} → {txn.newbalanceDest.toLocaleString("en-US")}</td></tr>
              <tr><th>Device / Channel (synthetic)</th><td>{txn.device_id || "—"} / {txn.channel || "—"}</td></tr>
              <tr><th>Known Device? (synthetic)</th><td>{txn.is_known_device ? "Yes" : "No"}</td></tr>
              <tr><th>Login Country (synthetic)</th><td>{txn.login_country || "—"}</td></tr>
              <tr><th>Geo-Velocity Flag (synthetic)</th><td>{txn.geo_velocity_flag ? "Triggered" : "—"}</td></tr>
            </tbody>
          </table>
        </div>

        <div className="card">
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
      </div>

      <div className="card">
        <h2><ShieldAlert size={16} /> Triggered Rules</h2>
        {rule_hits.length === 0 && !isOverride ? <p className="note">No rules triggered.</p> : (
          <ul className="rule-list">
            {rule_hits.map((h, i) => (
              <li key={i} className={`rule-${h.rule_type}`}>
                <strong>{h.rule_name}</strong> — {h.rule_type === "hard" ? "critical risk indicator" : "soft rule"}
              </li>
            ))}
            {isOverride && (
              <li className="rule-override">
                <strong>high_confidence_override</strong> — no rule confirmation, calibrated probability ≥0.95 promoted to RED
              </li>
            )}
          </ul>
        )}
      </div>

      <div className="card">
        <h2><BarChart3 size={16} /> Risk Factors (SHAP)</h2>
        <p className="card-subtext">Factors that contributed most to the model's decision</p>
        <ShapChart
          data={shap_explanations}
          height={220}
          emptyMessage="SHAP explanation is not available for this case."
          showLegend
        />
      </div>

      <div className="card card-info">
        <div className="card-head-row">
          <h2><Bot size={16} /> AI Investigator Report</h2>
          <span className="card-caption">Explanation only · not a decision</span>
        </div>
        {!reportState && <p className="loading">Loading</p>}
        {reportState?.status === "generating" && (
          <p className="note report-generating">
            <span className="report-spinner" />
            {reportPollGaveUp
              ? "Still generating — refresh the page to check again."
              : "Report is being generated…"}
          </p>
        )}
        {reportState?.status === "ready" && reportState.report && (
          <>
            <SourceBadge source={reportState.report.source} />
            <p className="report-text">{reportState.report.report_text}</p>
          </>
        )}
      </div>

      <div className="card card-info">
        <div className="card-head-row">
          <h2><Users size={16} /> AI #2 — Similar Past Cases</h2>
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
              <p className="note">{precedentsState.summary.note || "insufficient precedent — use judgment"}</p>
            )}

            {precedentsState.explanation.status === "generating" ? (
              <p className="note report-generating">
                <span className="report-spinner" />
                {precedentPollGaveUp
                  ? "Still generating — refresh the page to check again."
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
                      <th>Similarity</th>
                      <th>Analyst Decision</th>
                    </tr>
                  </thead>
                  <tbody>
                    {precedentsState.precedents.map((p) => (
                      <tr key={p.case_id}>
                        <td><Link to={`/triage?case=${p.case_id}`}>Case #{p.case_id}</Link></td>
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

      <div className="card">
        <h2><Gavel size={16} /> Analyst Decisions</h2>
        {decisions.length === 0 ? <p className="note">No decision yet.</p> : (
          <ul className="decision-list">
            {decisions.map((d, i) => (
              <li key={i}>
                <strong>{ACTION_LABELS[d.action_taken] || d.action_taken}</strong> — {d.analyst_reason_code || "—"}
                {d.analyst_note && ` · "${d.analyst_note}"`}
                {" "}<time>({new Date(d.decided_at).toLocaleString("en-US")})</time>
                {d.auto_processed && <> · <span className="note">auto-processed</span></>}
              </li>
            ))}
          </ul>
        )}
      </div>

      {detail.status === "OPEN" && (
        <div className="card">
          <h2>Analyst Decision</h2>
          <div className="decision-form" style={{ maxWidth: 440, marginBottom: 16 }}>
            <label>
              Reason Code (optional)
              <input
                type="text"
                placeholder="e.g. account_takeover"
                value={note.analyst_reason_code}
                onChange={(e) => setNote({ ...note, analyst_reason_code: e.target.value })}
              />
            </label>
            <label>
              Decision Rationale (optional)
              <textarea
                placeholder="e.g. suspected account takeover — sender balance fully drained"
                value={note.analyst_note}
                onChange={(e) => setNote({ ...note, analyst_note: e.target.value })}
              />
            </label>
          </div>
          {error && <p className="error" style={{ marginBottom: 14 }}>Error: {error}</p>}
          <div className="action-btns">
            {ACTIONS.map(({ value, label, icon: Icon, cls }) => (
              <button
                key={value}
                type="button"
                className={`action-btn ${cls}`}
                disabled={submitting !== null}
                onClick={() => handleDecision(value)}
              >
                <Icon size={15} />
                {submitting === value ? "Saving…" : label}
              </button>
            ))}
          </div>
          <p className="decision-note">This decision moves the case to "CLOSED".</p>
        </div>
      )}

      {detail.status === "CLOSED" && (
        <div className="card">
          <h2><RotateCcw size={16} /> Reopen Case</h2>
          <p className="note" style={{ marginTop: 0, marginBottom: 14 }}>
            This case is closed — you can reopen it for further review if needed. The previous decision stays in history and is not deleted.
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
    </div>
  );
}
