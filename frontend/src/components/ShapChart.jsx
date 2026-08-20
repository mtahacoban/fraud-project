import {
  Bar, BarChart, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

const FEATURE_LABELS = {
  amount: "Transaction Amount",
  errorBalanceOrig: "Source Balance Inconsistency",
  errorBalanceDest: "Destination Balance Inconsistency",
  is_transfer: "Transfer Type",
  is_cashout: "Cash-Out Type",
  step_hour: "Hour of Day (0-24)",
};

// Positions the SHAP value label INSIDE the bar, anchored at the zero end
// and offset toward the bar's own tip — not outside the tip (the previous
// approach). Anchoring at the tip meant the label's distance from the
// YAxis shrank as the bar grew, so the single largest bar (by definition
// reaching furthest) always had the least room and could still collide
// with the YAxis feature-name column no matter how much XAxis domain
// padding was added (padding is proportional; label width is a fixed
// pixel count — no percentage survives every plot-width/value
// combination). Anchoring at zero instead sidesteps the problem
// structurally: per recharts 3.10's Bar source (cartesian/Bar.js,
// horizontal-layout branch), x is ALWAYS xAxisScale(baseValue=0)
// regardless of sign — the plot's horizontal center, i.e. the point
// furthest from the YAxis on either side. A label anchored there and
// growing into the bar can never reach the YAxis, independent of the
// bar's magnitude. width is no longer needed (only x/y/height matter).
// Small bars may spill past their own short bar toward the empty area
// near zero — harmless, since that direction is never the YAxis.
function shapValueLabel({ x, y, height, value }) {
  const isNegative = value < 0;
  return (
    <text
      x={isNegative ? x - 6 : x + 6}
      y={y + height / 2}
      dy={4}
      textAnchor={isNegative ? "end" : "start"}
      className="shap-label"
      fill={isNegative ? "var(--text)" : "#ffffff"}
    >
      {`${value >= 0 ? "+" : ""}${value.toFixed(2)}`}
    </text>
  );
}

// Leaf component (like RiskBadge) — the outer card/heading stays with each
// consumer, only the "given raw SHAP factors, render the chart or an
// empty-state message" part lives here. See Aşama 4.5.1 plan: the two
// pre-extraction copies (Simulation.jsx, CaseDetailPanel.jsx) had identical
// FEATURE_LABELS/shapValueLabel/sort+label/domain logic — only height,
// empty-state wording, and the post-chart legend note differed by context,
// which is why those three are props instead of being hardcoded here.
export default function ShapChart({ data, height = 220, emptyMessage = "SHAP explanation is not available.", showLegend = false }) {
  const shapData = [...data]
    .sort((a, b) => a.shap_value - b.shap_value)
    .map((d) => ({ ...d, label: FEATURE_LABELS[d.feature] || d.feature }));
  // Symmetric domain around zero (SHAP convention: left = decreases risk,
  // right = increases risk) instead of recharts' default auto-scaled,
  // usually positive-skewed domain — keeps the zero line centered so
  // negative bars get real room instead of being squeezed against the
  // YAxis. Floor avoids a degenerate [0,0] domain if every value is 0.
  const shapMaxAbs = shapData.length
    ? Math.max(...shapData.map((d) => Math.abs(d.shap_value)), 0.01)
    : 1;

  if (shapData.length === 0) {
    return (
      <div className="triage-detail-status">
        <p className="note">{emptyMessage}</p>
      </div>
    );
  }

  return (
    <>
      <div className="shap-axis-caption">
        <span className="lo">← Decreases risk</span>
        <span className="hi">Increases risk →</span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={shapData} layout="vertical" margin={{ left: 30, right: 44 }}>
          <XAxis
            type="number"
            domain={[-shapMaxAbs, shapMaxAbs]}
            tickFormatter={(v) => v.toFixed(2)}
            tick={{ fontSize: 11, fill: "var(--text-faint)" }}
            stroke="var(--border)"
          />
          <YAxis
            type="category"
            dataKey="label"
            width={150}
            tick={{ fontSize: 12, fill: "var(--text-muted)" }}
            stroke="var(--border)"
          />
          <Tooltip
            formatter={(v) => v.toFixed(4)}
            contentStyle={{
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              borderRadius: 7,
              fontSize: 12.5,
              color: "var(--text)",
            }}
            cursor={{ fill: "var(--surface-hover)" }}
          />
          <Bar dataKey="shap_value" isAnimationActive={false} radius={[4, 4, 4, 4]} barSize={16}>
            {shapData.map((d, i) => (
              <Cell key={i} fill={d.shap_value >= 0 ? "#f2555a" : "#34d399"} />
            ))}
            <LabelList dataKey="shap_value" content={shapValueLabel} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {showLegend && <p className="note">Red: increases risk · Green: decreases risk</p>}
    </>
  );
}
