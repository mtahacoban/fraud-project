import { Sparkles } from "lucide-react";

export default function SourceBadge({ source }) {
  return source && source !== "fallback" ? (
    <span className="report-badge report-badge-ai"><Sparkles size={11} /> AI-generated</span>
  ) : (
    <span className="report-badge report-badge-auto">Auto-summary</span>
  );
}
