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

export default function ShapChart({ data, height = 220, emptyMessage = "SHAP explanation is not available.", showLegend = false }) {
  const shapData = [...data]
    .sort((a, b) => a.shap_value - b.shap_value)
    .map((d) => ({ ...d, label: FEATURE_LABELS[d.feature] || d.feature }));
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
