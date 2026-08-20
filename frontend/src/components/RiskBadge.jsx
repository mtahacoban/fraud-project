const LABELS = { RED: "High", GRAY: "Medium", GREEN: "Low" };

export default function RiskBadge({ band, score }) {
  const cls = (band || "").toLowerCase();
  return (
    <span className={`score-dot score-dot-${cls}`}>
      {typeof score === "number" && (
        <>
          {score.toFixed(0)}
          <span className="score-dot-sep">·</span>
        </>
      )}
      {LABELS[band] || band}
    </span>
  );
}
