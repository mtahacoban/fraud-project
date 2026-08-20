import { useEffect, useState } from "react";
import {
  AlertTriangle, Gauge, GitBranch, ShieldQuestion, Users,
} from "lucide-react";
import { getAutomationStatus } from "../api.js";

const MODE_LABELS = {
  off: "Off — not observing",
  shadow: "Shadow — observing, nothing surfaced",
  propose: "Propose — eligible cases get a pending proposal",
};

const MODE_COLOR_VAR = {
  off: "--text-muted",
  shadow: "--info",
  propose: "--accent",
};

export default function AutomationStatus() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getAutomationStatus().then(setStatus).catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="error">Error: {error}</p>;
  if (!status) return <p className="loading">Loading</p>;

  const { active_policy: policy, shadow_agreement, reject_rate, bias_monitoring, circuit_breaker } = status;

  return (
    <div>
      <h1>Automation Status</h1>
      <p className="subtitle">
        AI #2's human-confirmed automation. Aggregate view only; shadow-mode detail is
        never shown per-case (that's what keeps it a blind measurement — see Case Detail).
      </p>

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
            <tr><th>Min similarity (fraud)</th><td>{policy.fraud_similarity_threshold ?? "—"}</td></tr>
            <tr><th>Min precedent count</th><td>{policy.min_precedent_count ?? "—"}</td></tr>
            <tr><th>Min consensus ratio</th><td>{policy.min_consensus_ratio ?? "—"}</td></tr>
            <tr><th>Min calibrated probability (AI #1 agreement)</th><td>{policy.min_calibrated_proba ?? "—"}</td></tr>
            <tr><th>Hard-rule required (Gate B strict mode)</th><td>{policy.hard_rule_required ? "Yes" : "No — conflict-only"}</td></tr>
            <tr><th>Auto-clean enabled</th><td>{policy.auto_clean_enabled ? "Yes" : "No — fraud direction only"}</td></tr>
            <tr><th>Circuit breaker: max reject rate</th><td>{policy.circuit_breaker_max_reversal_rate != null ? `${(policy.circuit_breaker_max_reversal_rate * 100).toFixed(0)}%` : "—"}</td></tr>
            <tr><th>Circuit breaker: min confirmations</th><td>{policy.circuit_breaker_min_confirmations ?? "—"}</td></tr>
          </tbody>
        </table>
        {policy.notes && <p className="note" style={{ marginTop: 12 }}>{policy.notes}</p>}
      </div>

      <div className="grid-2">
        <div className="card">
          <h2><Users size={16} /> Shadow Agreement (blind)</h2>
          <p className="card-subtext">
            What automation would have decided vs. what the analyst independently decided — the
            analyst never saw the shadow verdict. Distinct from AI #2's suggestion-agreement,
            which the analyst could see while deciding.
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
              <tr><th>Shadow evaluations logged</th><td>{shadow_agreement.n_shadow_evaluations}</td></tr>
              <tr><th>Eligible (comparable)</th><td>{shadow_agreement.n_eligible}</td></tr>
              <tr><th>Not eligible</th><td>{shadow_agreement.n_not_eligible}</td></tr>
              <tr><th>Would have confirmed correctly</th><td>{shadow_agreement.would_have_confirmed_correctly}</td></tr>
              <tr><th>Would have been wrong</th><td>{shadow_agreement.would_have_been_wrong}</td></tr>
            </tbody>
          </table>
        </div>

        <div className="card">
          <h2><Gauge size={16} /> Reject Rate (active policy)</h2>
          <p className="card-subtext">
            Confirmed vs. rejected among this policy version's own proposals — what the circuit
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
              <tr><th>Confirmed</th><td>{reject_rate.confirmed}</td></tr>
              <tr><th>Rejected</th><td>{reject_rate.rejected}</td></tr>
              <tr><th>Total (n)</th><td>{reject_rate.n}</td></tr>
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
          Confirmed vs. rejected counts by transaction type — a systematic gap would be a concrete
          trace of AI #2 imitating (and potentially amplifying) an analyst's own blind spot. One
          dimension only; not a substitute for a real fairness audit.
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
