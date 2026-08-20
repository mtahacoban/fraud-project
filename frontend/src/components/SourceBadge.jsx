import { Sparkles } from "lucide-react";

// Leaf component (like RiskBadge, ShapChart) — which "source" produced a
// piece of AI-adjacent text (LLM report / precedent explanation). Never
// styled as an error/warning — a fallback (deterministic, no LLM call) is
// a fully valid output on its own, just a different badge, not a lesser
// one. source is undefined/null (not yet generated) or "fallback" ->
// Auto-summary; any other provider name (e.g. "groq") -> AI-generated.
export default function SourceBadge({ source }) {
  return source && source !== "fallback" ? (
    <span className="report-badge report-badge-ai"><Sparkles size={11} /> AI-generated</span>
  ) : (
    <span className="report-badge report-badge-auto">Auto-summary</span>
  );
}
