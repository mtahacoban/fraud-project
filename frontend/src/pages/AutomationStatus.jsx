import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle, ArrowUpRight, ChevronDown, Filter, Gauge, GitBranch, ShieldAlert, ShieldQuestion, Users,
} from "lucide-react";
import { getAutomationStatus } from "../api.js";
import PageHeader from "../components/PageHeader.jsx";

const MODE_LABELS = {
  off: "Off - not observing",
  shadow: "Shadow - observing, nothing surfaced",
  propose: "Propose - eligible cases get a pending proposal",
};

const MODE_COLOR_VAR = {
  off: "--text-muted",
  shadow: "--info",
  propose: "--accent",
};

const GATE_LABELS = {
  direction_automatable: "Suggested Direction",
  similarity: "Precedent Similarity",
  precedent_count: "Precedent Count",
  consensus: "Precedent Consensus",
  calibrated_proba: "Model Confidence (Calibrated Probability)",
  hard_rule_conflict: "Hard-Rule Conflict Check",
};

function CaseListStatRow({ label, count, caseIds, isOpen, onToggle }) {
  const hasCases = count > 0 && caseIds.length > 0;
  return (
    <>
      <tr>
        <th>{label}</th>
        <td>
          {hasCases ? (
            <button type="button" className="kv-reveal-pill" onClick={onToggle} aria-expanded={isOpen}>
              <span>{count}</span>
              <ChevronDown size={13} className="kv-reveal-chevron" />
            </button>
          ) : (
            count
          )}
        </td>
      </tr>
      {hasCases && (
        <tr className="kv-expand-row">
          <td colSpan={2}>
            <div className={`kv-reveal${isOpen ? " is-open" : ""}`}>
              <div className="kv-reveal-inner">
                <ul className="kv-case-list">
                  {caseIds.map((id) => (
                    <li key={id}>
                      <Link to={`/triage?case=${id}`}>
                        Case #{id}
                        <ArrowUpRight size={11} />
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function GateRing({ label, passed, failed, isBottleneck }) {
  const total = passed + failed;
  const rate = total > 0 ? passed / total : 0;
  const pct = Math.round(rate * 100);

  const color = pct < 40 ? "#A32D2D" : pct <= 60 ? "#BA7517" : "#0F6E56";

  const circumference = 2 * Math.PI * 34;
  const offset = circumference * (1 - rate);

  return (
    <div className={`gate-ring-card${isBottleneck ? " gate-ring-bottleneck" : ""}`}>
      <div className="gate-ring-svg-wrap">
        <svg viewBox="0 0 80 80" className="gate-ring-svg">
          <circle cx="40" cy="40" r="34" fill="none" stroke="var(--border)" strokeWidth="6" />
          <circle
            cx="40" cy="40" r="34"
            fill="none"
            stroke={color}
            strokeWidth="6"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
          />
        </svg>
        <span className="gate-ring-pct" style={{ color }}>{pct}%</span>
      </div>
      <span className="gate-ring-label" style={isBottleneck ? { color } : undefined}>
        {label}
      </span>
      <span className="gate-ring-detail">{passed} passed · {failed} failed</span>
      {isBottleneck && <span className="gate-ring-tag">bottleneck</span>}
    </div>
  );
}

export default function AutomationStatus() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState({});

  function toggleExpanded(key) {
    setExpanded((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  useEffect(() => {
    getAutomationStatus().then(setStatus).catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="error">Error: {error}</p>;
  if (!status) return <p className="loading">Loading</p>;

  const { active_policy: policy, shadow_agreement, reject_rate, bias_monitoring, circuit_breaker, gate_bottleneck } = status;
  const worstGate = gate_bottleneck.gates.length
    ? gate_bottleneck.gates.reduce((worst, g) => (g.failed_count > worst.failed_count ? g : worst))
    : null;
  const directionBreakdown = gate_bottleneck.direction_breakdown;

  return (
    <div>
      <PageHeader
        icon={ShieldAlert}
        eyebrow="System"
        title="Automation Status"
        subtitle="Human-confirmed automation built on Precedent Analysis. Aggregate view only; shadow-mode detail is never shown per-case (that's what keeps it a blind measurement - see Case Detail)."
        tone="info"
      />

      <div className="card">
        <div className="card-head-row">
          <h2><GitBranch size={16} /> Active Policy</h2>
          <span className="card-caption">{policy.version}</span>
        </div>
        <div className="pending-directive" style={{ marginBottom: 14 }}>
          <span
            className="status-pill"
            style={{ color: `var(${MODE_COLOR_VAR[policy.mode] || "--text-muted"})` }}
          >
            {MODE_LABELS[policy.mode] || policy.mode}
          </span>
        </div>
        <table className="kv-table">
          <tbody>
            <tr><th>Active since</th><td>{new Date(policy.created_at).toLocaleString("en-US")}</td></tr>
            <tr><th>Min similarity (fraud)</th><td>{policy.fraud_similarity_threshold ?? "—"}</td></tr>
            <tr><th>Min precedent count</th><td>{policy.min_precedent_count ?? "—"}</td></tr>
            <tr><th>Min consensus ratio</th><td>{policy.min_consensus_ratio ?? "—"}</td></tr>
            <tr><th>Min calibrated probability (model confidence)</th><td>{policy.min_calibrated_proba ?? "—"}</td></tr>
            <tr><th>Hard-rule required (Gate B strict mode)</th><td>{policy.hard_rule_required ? "Yes" : "No - conflict-only"}</td></tr>
            <tr><th>Auto-clean enabled</th><td>{policy.auto_clean_enabled ? "Yes" : "No - fraud direction only"}</td></tr>
            <tr><th>Circuit breaker: max reject rate</th><td>{policy.circuit_breaker_max_reversal_rate != null ? `${(policy.circuit_breaker_max_reversal_rate * 100).toFixed(0)}%` : "—"}</td></tr>
            <tr><th>Circuit breaker: min confirmations</th><td>{policy.circuit_breaker_min_confirmations ?? "—"}</td></tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <div className="card-head-row">
          <h2><Filter size={16} /> Gate Bottleneck</h2>
          <span className="card-caption">single shadow period, not a reliability measure</span>
        </div>
        <p className="card-subtext">
          Why so few cases qualify, not just how many do - {gate_bottleneck.n_evaluations} logged
          evaluations, overwhelmingly from one shadow period against a still-largely-synthetic
          precedent pool. Shows what the mechanism did on this sample, not a claim about
          automation's reliability.
        </p>
        <div className="gate-ring-grid">
          {gate_bottleneck.gates.map((g) => (
            <GateRing
              key={g.gate}
              label={GATE_LABELS[g.gate] || g.gate}
              passed={g.passed_count}
              failed={g.failed_count}
              isBottleneck={worstGate && g.gate === worstGate.gate && g.failed_count > 0}
            />
          ))}
        </div>

        <div className="gate-direction-breakdown">
          <span className="gate-direction-title">Direction breakdown</span>
          <div className="gate-direction-row">
            <span className="gate-dir-item">
              <span className="gate-dir-dot" style={{ background: "#A32D2D" }} />
              {directionBreakdown.clean_blocked} clean-blocked
            </span>
            <span className="gate-dir-item">
              <span className="gate-dir-dot" style={{ background: "#BA7517" }} />
              {directionBreakdown.no_suggestion} insufficient
            </span>
            <span className="gate-dir-item">
              <span className="gate-dir-dot" style={{ background: "var(--text-muted)" }} />
              {directionBreakdown.escalate} escalate
            </span>
            <span className="gate-dir-item">
              <span className="gate-dir-dot" style={{ background: "#0F6E56" }} />
              {directionBreakdown.passed} passed
            </span>
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h2><Users size={16} /> Shadow Agreement (blind)</h2>
          <p className="card-subtext">
            What automation would have decided vs. what the analyst independently decided - the
            analyst never saw the shadow verdict. Distinct from the suggestion-agreement measured
            for Precedent Analysis, which the analyst could see while deciding.
          </p>
          {shadow_agreement.shadow_accuracy === null ? (
            <p className="note">
              Insufficient eligible cases yet ({shadow_agreement.n_eligible} eligible of{" "}
              {shadow_agreement.n_shadow_evaluations} shadow evaluations logged).
            </p>
          ) : (
            <p className="header-stat-value">{(shadow_agreement.shadow_accuracy * 100).toFixed(0)}%</p>
          )}
          <table className="kv-table">
            <tbody>
              <CaseListStatRow
                label="Shadow evaluations logged"
                count={shadow_agreement.n_shadow_evaluations}
                caseIds={shadow_agreement.shadow_evaluation_case_ids || []}
                isOpen={!!expanded.shadow_evaluations}
                onToggle={() => toggleExpanded("shadow_evaluations")}
              />
              <CaseListStatRow
                label="Eligible (comparable)"
                count={shadow_agreement.n_eligible}
                caseIds={shadow_agreement.eligible_case_ids || []}
                isOpen={!!expanded.shadow_eligible}
                onToggle={() => toggleExpanded("shadow_eligible")}
              />
              <CaseListStatRow
                label="Not eligible"
                count={shadow_agreement.n_not_eligible}
                caseIds={shadow_agreement.not_eligible_case_ids || []}
                isOpen={!!expanded.shadow_not_eligible}
                onToggle={() => toggleExpanded("shadow_not_eligible")}
              />
              <CaseListStatRow
                label="Would have confirmed correctly"
                count={shadow_agreement.would_have_confirmed_correctly}
                caseIds={shadow_agreement.would_have_confirmed_correctly_case_ids || []}
                isOpen={!!expanded.shadow_would_confirm}
                onToggle={() => toggleExpanded("shadow_would_confirm")}
              />
              <CaseListStatRow
                label="Would have been wrong"
                count={shadow_agreement.would_have_been_wrong}
                caseIds={shadow_agreement.would_have_been_wrong_case_ids || []}
                isOpen={!!expanded.shadow_would_wrong}
                onToggle={() => toggleExpanded("shadow_would_wrong")}
              />
            </tbody>
          </table>
        </div>

        <div className="card">
          <h2><Gauge size={16} /> Reject Rate (active policy)</h2>
          <p className="card-subtext">
            Confirmed vs. rejected among this policy version's own proposals - what the circuit
            breaker itself evaluates.
          </p>
          {reject_rate.reject_rate === null ? (
            <p className="note">{reject_rate.note}</p>
          ) : (
            <p className="header-stat-value" style={{ color: "var(--red)" }}>
              {(reject_rate.reject_rate * 100).toFixed(0)}%
            </p>
          )}
          <table className="kv-table">
            <tbody>
              <CaseListStatRow
                label="Confirmed"
                count={reject_rate.confirmed}
                caseIds={reject_rate.confirmed_case_ids || []}
                isOpen={!!expanded.reject_confirmed}
                onToggle={() => toggleExpanded("reject_confirmed")}
              />
              <CaseListStatRow
                label="Rejected"
                count={reject_rate.rejected}
                caseIds={reject_rate.rejected_case_ids || []}
                isOpen={!!expanded.reject_rejected}
                onToggle={() => toggleExpanded("reject_rejected")}
              />
              <CaseListStatRow
                label="Pending (not yet reviewed)"
                count={reject_rate.pending}
                caseIds={reject_rate.pending_case_ids || []}
                isOpen={!!expanded.reject_pending}
                onToggle={() => toggleExpanded("reject_pending")}
              />
              <tr><th>Total judged (n)</th><td>{reject_rate.n}</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <div className="card-head-row">
          <h2><AlertTriangle size={16} /> Circuit Breaker</h2>
          <span className="card-caption">always on</span>
        </div>
        {circuit_breaker.tripped_recently ? (
          <>
            <p className="note" style={{ color: "var(--red)", fontWeight: 600, marginBottom: 8 }}>
              Last tripped at {circuit_breaker.last_trip_policy_version}
              {circuit_breaker.last_trip_at && ` · ${new Date(circuit_breaker.last_trip_at).toLocaleString("en-US")}`}
            </p>
            <p className="note">{circuit_breaker.last_trip_notes}</p>
          </>
        ) : (
          <p className="note">No trip on record. Mode only downgrades automatically if the active policy's own reject rate crosses its threshold with enough sample size.</p>
        )}
      </div>

      <div className="card">
        <div className="card-head-row">
          <h2><ShieldQuestion size={16} /> Bias Monitoring</h2>
          <span className="card-caption">smoke check, not an audit</span>
        </div>
        <p className="card-subtext">
          Confirmed vs. rejected counts by transaction type - a systematic gap would be a concrete
          trace of Precedent Analysis imitating (and potentially amplifying) an analyst's own blind
          spot. One dimension only; not a substitute for a real fairness audit.
        </p>
        {Object.keys(bias_monitoring.by_transaction_type).length === 0 ? (
          <p className="note">No confirmed/rejected proposals yet.</p>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr><th>Transaction Type</th><th>Confirmed</th><th>Rejected</th></tr>
              </thead>
              <tbody>
                {Object.entries(bias_monitoring.by_transaction_type).map(([type, counts]) => (
                  <tr key={type}>
                    <td>{type}</td>
                    <td>{counts.confirmed}</td>
                    <td>{counts.rejected}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
