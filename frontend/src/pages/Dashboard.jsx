import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Bot, PieChart, Radar } from "lucide-react";
import { getMetrics, getCases } from "../api.js";
import RiskBadge from "../components/RiskBadge.jsx";

const AUTOMATION_MODE_LABELS = {
  off: "Off",
  shadow: "Shadow (observing)",
  propose: "Propose (active)",
};

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [topCases, setTopCases] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getMetrics().then(setMetrics).catch((e) => setError(e.message));
    getCases({ sort: "hybrid_score", order: "desc", limit: 5 })
      .then((res) => setTopCases(res.items))
      .catch(() => {});
  }, []);

  if (error) return <p className="error">Error: {error}</p>;
  if (!metrics) return <p className="loading">Loading</p>;

  const bands = metrics.by_risk_band;
  const bandTotal = (bands.RED || 0) + (bands.GRAY || 0) + (bands.GREEN || 0);
  const pct = (n) => (bandTotal ? (n / bandTotal) * 100 : 0);

  return (
    <div>
      <h1>Overview</h1>
      <p className="subtitle">Live risk posture and the highest-priority cases</p>

      {/* Three cards, not a generic map — a fixed, deliberately-chosen set
          (see Aşama 4.3 plan): Open Cases first as the single most
          actionable number, Pending AI Proposals kept as its own Link (the
          one card with different behavior), then volume context. Every
          value comes straight from GET /metrics — no field here is
          invented; latency/agreement-rate live on /automation/status and
          are out of this endpoint's scope. */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <span className="kpi-label">Open Cases</span>
          <span className="kpi-value">{metrics.open_cases}</span>
          <span className="kpi-caption">{metrics.closed_cases} closed · {metrics.total_cases} total</span>
        </div>
        <Link
          to="/automation"
          className="kpi-card kpi-card-link"
          style={metrics.pending_ai_proposals > 0 ? { borderColor: "var(--accent)" } : undefined}
        >
          <span className="kpi-label">Pending AI Proposals</span>
          <span className="kpi-value" style={metrics.pending_ai_proposals > 0 ? { color: "var(--accent)" } : undefined}>
            {metrics.pending_ai_proposals}
          </span>
          <span className="kpi-caption">
            automation: {AUTOMATION_MODE_LABELS[metrics.automation_mode] || "not configured"}
          </span>
        </Link>
        <div className="kpi-card">
          <span className="kpi-label">Scored Transactions</span>
          <span className="kpi-value">{metrics.total_scored}</span>
          <span className="kpi-caption">{metrics.live_scored} live · {metrics.demo_scored} simulated</span>
        </div>
      </div>

      <div className="card">
        <h2><PieChart size={15} /> Risk Band Distribution</h2>
        <div className="band-bar">
          <span className="seg-red" style={{ width: `${pct(bands.RED || 0)}%` }} />
          <span className="seg-gray" style={{ width: `${pct(bands.GRAY || 0)}%` }} />
          <span className="seg-green" style={{ width: `${pct(bands.GREEN || 0)}%` }} />
        </div>
        <div className="band-legend">
          <span className="band-legend-item">
            <span className="band-legend-dot" style={{ background: "var(--red)" }} />
            Red <strong>{bands.RED || 0}</strong>
          </span>
          <span className="band-legend-item">
            <span className="band-legend-dot" style={{ background: "var(--gray)" }} />
            Gray <strong>{bands.GRAY || 0}</strong>
          </span>
          <span className="band-legend-item">
            <span className="band-legend-dot" style={{ background: "var(--green)" }} />
            Green <strong>{bands.GREEN || 0}</strong>
          </span>
        </div>
        <p className="note">
          Green auto-clears with no case opened; Gray and Red open a case for review.
        </p>
      </div>

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <div style={{ padding: "20px 22px 4px" }}>
          <h2 style={{ marginBottom: 0 }}><Radar size={15} /> Recent High-Risk Cases</h2>
        </div>
        {!topCases && <p style={{ padding: "8px 22px 20px" }} className="loading">Loading</p>}
        {topCases && topCases.length === 0 && (
          <p style={{ padding: "8px 22px 20px" }} className="note">No cases found.</p>
        )}
        {topCases && topCases.length > 0 && (
          <>
            <table className="table" style={{ marginTop: 12 }}>
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Transaction</th>
                  <th>Priority</th>
                  <th>Risk</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {topCases.map((c) => (
                  <tr key={c.case_id}>
                    <td><Link to={`/triage?case=${c.case_id}`}>#{c.case_id}</Link></td>
                    <td>#{c.transaction_id}</td>
                    <td>{c.priority}</td>
                    <td><RiskBadge band={c.risk_band} score={c.hybrid_score} /></td>
                    <td>
                      {c.pending_ai_proposal && (
                        <span className="status-pill status-pill-accent"><Bot size={11} /> AI proposal</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <Link to="/triage" className="table-foot-link">View all →</Link>
          </>
        )}
      </div>
    </div>
  );
}
