import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

export default function TrendChart({
  data, dataKey, color, height = 160, emptyMessage = "No data for this period.",
  valueFormatter = (v) => v, tooltipLabel, tickCount = 4,
}) {
  const hasData = data.some((d) => d[dataKey] !== null && d[dataKey] !== undefined);

  if (!hasData) {
    return (
      <div className="triage-detail-status">
        <p className="note">{emptyMessage}</p>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 12, left: -12, bottom: 0 }}>
        <CartesianGrid stroke="var(--border-soft)" vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={(d) => d.slice(5)}
          tick={{ fontSize: 10.5, fill: "var(--text-faint)" }}
          stroke="var(--border)"
          interval="preserveStartEnd"
          minTickGap={28}
        />
        <YAxis
          tick={{ fontSize: 10.5, fill: "var(--text-faint)" }}
          stroke="var(--border)"
          width={34}
          tickFormatter={valueFormatter}
          tickCount={tickCount}
        />
        <Tooltip
          formatter={(v) => [v === null || v === undefined ? "—" : valueFormatter(v), tooltipLabel || dataKey]}
          contentStyle={{
            background: "var(--surface-2)",
            border: "1px solid var(--border)",
            borderRadius: 7,
            fontSize: 12.5,
            color: "var(--text)",
          }}
          cursor={{ stroke: "var(--border)" }}
        />
        <Line
          type="monotone"
          dataKey={dataKey}
          stroke={color}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
          connectNulls
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
