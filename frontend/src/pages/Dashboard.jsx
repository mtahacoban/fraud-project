import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Bot, Gauge, LayoutDashboard, PieChart, Radar, RotateCcw, TrendingUp } from "lucide-react";
import { Cell, Pie, PieChart as RePieChart, ResponsiveContainer } from "recharts";
import { getMetrics, getMetricsTrends, getCases } from "../api.js";
import KpiCards, { DEFAULT_ORDER, loadOrder, saveOrder } from "../components/KpiCards.jsx";
import PageHeader from "../components/PageHeader.jsx";
import RiskBadge from "../components/RiskBadge.jsx";
import TrendChart from "../components/TrendChart.jsx";

const STATUS_LABELS = { OPEN: "Open", CLOSED: "Closed", AUTO_CLEAN: "Auto-Clean" };

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [topCases, setTopCases] = useState(null);
  const [error, setError] = useState(null);
  const [trendDays, setTrendDays] = useState(30);
  const [trends, setTrends] = useState(null);
  const [kpiOrder, setKpiOrder] = useState(loadOrder);
  const isDefaultKpiOrder = kpiOrder.join(",") === DEFAULT_ORDER.join(",");
  function handleResetKpiOrder() {
    setKpiOrder(DEFAULT_ORDER);
    saveOrder(DEFAULT_ORDER);
  }

  const loadMetricsAndCases = useCallback(() => {
    getMetrics().then(setMetrics).catch((e) => setError(e.message));
    getCases({ status: "OPEN", sort: "hybrid_score", order: "desc", limit: 5 })
      .then((res) => setTopCases(res.items))
      .catch(() => {});
  }, []);

  const loadTrends = useCallback(() => {
    getMetricsTrends(trendDays).then(setTrends).catch(() => setTrends([]));
  }, [trendDays]);

  useEffect(() => {
    loadMetricsAndCases();
  }, [loadMetricsAndCases]);

  useEffect(() => {
    setTrends(null);
    loadTrends();
  }, [loadTrends]);

  if (error) return <p className="error">Error: {error}</p>;
  if (!metrics) return <p className="loading">Loading</p>;

  const bands = metrics.by_risk_band;
  const autoCleanCount = Math.max(metrics.total_scored - metrics.total_cases, 0);
  const trendSummary = trends && trends.length > 0 ? {
    caseTotal: trends.reduce((sum, d) => sum + (d.case_count || 0), 0),
    redRate: (() => {
      const withData = trends.filter((d) => d.red_rate != null && d.case_count);
      const n = withData.reduce((sum, d) => sum + d.case_count, 0);
      return n ? withData.reduce((sum, d) => sum + d.red_rate * d.case_count, 0) / n : null;
    })(),
    scoreAvg: (() => {
      const withData = trends.filter((d) => d.avg_score != null && d.scored_count);
      const n = withData.reduce((sum, d) => sum + d.scored_count, 0);
      return n ? withData.reduce((sum, d) => sum + d.avg_score * d.scored_count, 0) / n : null;
    })(),
  } : null;

  return (
    <div>
      <PageHeader
        icon={LayoutDashboard}
        eyebrow="Operations"
        title="Overview"
        subtitle="Live risk posture and the highest-priority cases"
        tone="accent"
        actions={
          <button
            type="button"
            className="pagination-btn"
            onClick={handleResetKpiOrder}
            disabled={isDefaultKpiOrder}
          >
            <RotateCcw size={12} /> Reset Order
          </button>
        }
      />

      <div className="section-label"><Gauge size={15} /> Key Indicators</div>

      <KpiCards metrics={metrics} order={kpiOrder} onOrderChange={setKpiOrder} />

      <div className="section-label"><PieChart size={15} /> Risk &amp; Case Overview</div>

      <div className="card">
        <div className="donut-split">
          <div>
            <p className="donut-half-label">By Risk Band</p>
            <div className="donut-layout">
              <div className="donut-chart-wrapper">
                <ResponsiveContainer width={220} height={220}>
                  <RePieChart>
                    <Pie
                      data={[
                        { name: "Red", value: bands.RED || 0 },
                        { name: "Gray", value: bands.GRAY || 0 },
                        { name: "Green", value: bands.GREEN || 0 },
                      ]}
                      dataKey="value"
                      innerRadius={65}
                      outerRadius={100}
                      paddingAngle={2}
                      stroke="none"
                    >
                      <Cell fill="var(--red)" />
                      <Cell fill="var(--gray)" />
                      <Cell fill="var(--green)" />
                    </Pie>
                  </RePieChart>
                </ResponsiveContainer>
                <div className="donut-center-label">
                  <span className="donut-center-title">Total</span>
                  <span className="donut-center-value">{metrics.total_scored?.toLocaleString()}</span>
                </div>
              </div>

              <div className="donut-legend">
                {[
                  { label: "Red (high risk)", value: bands.RED || 0, color: "var(--red)" },
                  { label: "Gray (medium)", value: bands.GRAY || 0, color: "var(--gray)" },
                  { label: "Green (low risk)", value: bands.GREEN || 0, color: "var(--green)" },
                ].map((item) => (
                  <div className="donut-legend-row" key={item.label}>
                    <span className="donut-legend-dot" style={{ background: item.color }} />
                    <span className="donut-legend-label">{item.label}</span>
                    <span className="donut-legend-value">{item.value}</span>
                    <span className="donut-legend-pct">
                      {metrics.total_scored ? `${((item.value / metrics.total_scored) * 100).toFixed(1)}%` : "—"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="donut-split-divider" />

          <div>
            <p className="donut-half-label">By Status</p>
            <div className="donut-layout">
              <div className="donut-chart-wrapper">
                <ResponsiveContainer width={220} height={220}>
                  <RePieChart>
                    <Pie
                      data={[
                        { name: "Open", value: metrics.open_cases || 0 },
                        { name: "Closed", value: metrics.closed_cases || 0 },
                        { name: "Auto-Clean", value: autoCleanCount },
                      ]}
                      dataKey="value"
                      innerRadius={65}
                      outerRadius={100}
                      paddingAngle={2}
                      stroke="none"
                    >
                      <Cell fill="var(--info)" />
                      <Cell fill="var(--green)" />
                      <Cell fill="var(--gray)" />
                    </Pie>
                  </RePieChart>
                </ResponsiveContainer>
                <div className="donut-center-label">
                  <span className="donut-center-title">Total</span>
                  <span className="donut-center-value">{metrics.total_scored?.toLocaleString()}</span>
                </div>
              </div>

              <div className="donut-legend">
                {[
                  { label: "Open", value: metrics.open_cases || 0, color: "var(--info)" },
                  { label: "Closed", value: metrics.closed_cases || 0, color: "var(--green)" },
                  { label: "Auto-Clean", value: autoCleanCount, color: "var(--gray)" },
                ].map((item) => (
                  <div className="donut-legend-row" key={item.label}>
                    <span className="donut-legend-dot" style={{ background: item.color }} />
                    <span className="donut-legend-label">{item.label}</span>
                    <span className="donut-legend-value">{item.value}</span>
                    <span className="donut-legend-pct">
                      {metrics.total_scored ? `${((item.value / metrics.total_scored) * 100).toFixed(1)}%` : "—"}
                    </span>
                  </div>
                ))}
                <div className="donut-legend-row donut-legend-row-extra">
                  <span className="donut-legend-dot" style={{ background: "var(--accent)" }} />
                  <span className="donut-legend-label">Pending AI Proposals</span>
                  <span className="donut-legend-value">{metrics.pending_ai_proposals || 0}</span>
                  <span className="donut-legend-pct">
                    {metrics.total_scored ? `${((metrics.pending_ai_proposals / metrics.total_scored) * 100).toFixed(1)}%` : "—"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="section-label"><TrendingUp size={15} /> Activity Trends</div>

      <div className="card">
        <div className="card-head-row" style={{ justifyContent: "flex-end" }}>
          <div style={{ display: "flex", gap: 6 }}>
            {[7, 30].map((n) => (
              <button
                key={n}
                type="button"
                className="pagination-btn"
                style={trendDays === n ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}
                onClick={() => setTrendDays(n)}
              >
                {n}d
              </button>
            ))}
          </div>
        </div>

        {!trends && <p className="loading">Loading</p>}

        {trends && (
          <>
            <div className="trend-grid">
              <div>
                <div className="card-head-row">
                  <h3>Cases Opened</h3>
                  <span className="trend-stat">
                    {trendSummary ? `${trendDays}-day total: ${trendSummary.caseTotal}` : ""}
                  </span>
                </div>
                <TrendChart data={trends} dataKey="case_count" color="var(--accent)" tooltipLabel="cases" />
              </div>
              <div>
                <div className="card-head-row">
                  <h3>High-Risk Rate</h3>
                  <span className="trend-stat">
                    {trendSummary?.redRate != null ? `${trendDays}-day: ${Math.round(trendSummary.redRate * 100)}% of cases opened` : ""}
                  </span>
                </div>
                <TrendChart
                  data={trends}
                  dataKey="red_rate"
                  color="var(--red)"
                  tooltipLabel="% of opened cases flagged RED - not of all scored"
                  valueFormatter={(v) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`}
                />
              </div>
              <div>
                <div className="card-head-row">
                  <h3>Avg Risk Score</h3>
                  <span className="trend-stat">
                    {trendSummary?.scoreAvg != null ? `${trendDays}-day avg: ${trendSummary.scoreAvg.toFixed(1)}` : ""}
                  </span>
                </div>
                <TrendChart data={trends} dataKey="avg_score" color="var(--info)" tooltipLabel="avg hybrid score" />
              </div>
            </div>
            <p className="note">Demo data · distributed timeline - period summaries, not day-over-day change.</p>
          </>
        )}
      </div>

      <div className="section-label"><Radar size={15} /> Priority Queue</div>

      <div className="card" style={{ padding: "20px 0 0", overflow: "hidden" }}>
        {!topCases && <p style={{ padding: "0 22px 20px" }} className="loading">Loading</p>}
        {topCases && topCases.length === 0 && (
          <p style={{ padding: "8px 22px 20px" }} className="note">No cases found.</p>
        )}
        {topCases && topCases.length > 0 && (
          <>
            <table className="table" style={{ marginTop: 8 }}>
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Transaction</th>
                  <th>Type</th>
                  <th>Amount</th>
                  <th>Priority</th>
                  <th>Risk</th>
                  <th>Status</th>
                  <th>Signals</th>
                  <th>Opened</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {topCases.map((c) => (
                  <tr key={c.case_id}>
                    <td><Link to={`/triage?case=${c.case_id}`}>#{c.case_id}</Link></td>
                    <td>#{c.transaction_id}</td>
                    <td>{c.type}</td>
                    <td className="cell-num">
                      {c.amount != null
                        ? c.amount.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })
                        : "—"}
                    </td>
                    <td>{c.priority}</td>
                    <td><RiskBadge band={c.risk_band} score={c.hybrid_score} showLabel={false} /></td>
                    <td>{STATUS_LABELS[c.status] || c.status}</td>
                    <td className="cell-signals">{c.top_rules?.length ? c.top_rules.join(", ") : "—"}</td>
                    <td className="cell-num">{new Date(c.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric" })}</td>
                    <td>
                      {c.pending_ai_proposal && (
                        <span className="status-pill status-pill-accent"><Bot size={11} /> AI proposal</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="table-foot-row">
              <span className="note">Showing top {topCases.length} open cases by risk score, out of {metrics.open_cases}</span>
              <Link to="/triage?status=OPEN" className="table-foot-link">View all →</Link>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
