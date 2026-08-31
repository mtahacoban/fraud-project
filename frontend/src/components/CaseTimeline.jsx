import { Bot, FileText, Gavel } from "lucide-react";

const ACTION_LABELS = {
  confirm_fraud: "Confirmed as fraud",
  approve_clean: "Closed as clean",
  escalate: "Escalated",
};

export default function CaseTimeline({ detail }) {
  const report = detail.llm_reports?.[0];
  const decision = detail.status === "CLOSED" ? detail.decisions?.[detail.decisions.length - 1] : null;

  return (
    <div className="case-timeline">
      <div className="case-timeline-step">
        <div className="case-timeline-dot"><FileText size={13} /></div>
        <div className="case-timeline-body">
          <span className="case-timeline-label">Transaction Scored &amp; Case Opened</span>
          <span className="case-timeline-time">{new Date(detail.created_at).toLocaleString("en-US")}</span>
        </div>
      </div>

      <div className="case-timeline-connector" />

      <div className={`case-timeline-step${report ? "" : " case-timeline-step-pending"}`}>
        <div className="case-timeline-dot"><Bot size={13} /></div>
        <div className="case-timeline-body">
          <span className="case-timeline-label">AI Report Generated</span>
          <span className="case-timeline-time">
            {report ? new Date(report.generated_at).toLocaleString("en-US") : "pending"}
          </span>
        </div>
      </div>

      <div className="case-timeline-connector" style={decision ? undefined : { opacity: 0.35 }} />

      <div className={`case-timeline-step${decision ? "" : " case-timeline-step-pending"}`}>
        <div className={`case-timeline-dot${decision ? " case-timeline-dot-final" : ""}`}><Gavel size={13} /></div>
        <div className="case-timeline-body">
          <span className="case-timeline-label">
            {decision ? (ACTION_LABELS[decision.action_taken] || decision.action_taken) : "Decision"}
          </span>
          <span className="case-timeline-time">
            {decision ? new Date(decision.decided_at).toLocaleString("en-US") : "awaiting analyst"}
          </span>
        </div>
      </div>
    </div>
  );
}
