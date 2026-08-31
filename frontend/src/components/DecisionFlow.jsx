import {
  Activity, BarChart3, Bot, ChevronDown, Gavel, ListChecks, Users,
} from "lucide-react";

const SHAP_FEATURE_LABELS = {
  amount: "Transaction Amount",
  errorBalanceOrig: "Source Balance Inconsistency",
  errorBalanceDest: "Destination Balance Inconsistency",
  is_transfer: "Transfer Type",
  is_cashout: "Cash-Out Type",
  step_hour: "Hour of Day",
};
const ACTION_LABELS = {
  confirm_fraud: "Confirmed as fraud",
  approve_clean: "Closed as clean",
  escalate: "Escalated",
  reopened: "Reopened",
};

function scrollToNode(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function FlowNode({ targetId, icon: Icon, title, summary, tone, span2, dim }) {
  return (
    <button
      type="button"
      className={`flow-node flow-node-${tone}${span2 ? " flow-node-span2" : ""}${dim ? " flow-node-dim" : ""}`}
      onClick={() => scrollToNode(targetId)}
    >
      <span className="flow-node-icon"><Icon size={15} /></span>
      <span className="flow-node-text">
        <span className="flow-node-title">{title}</span>
        <span className="flow-node-summary">{summary}</span>
      </span>
    </button>
  );
}

function BranchConnector() {
  return (
    <div className="flow-connector flow-connector-branch" aria-hidden="true">
      <span className="fc-elbow fc-elbow-branch-left" />
      <span className="fc-elbow fc-elbow-branch-right" />
      <span className="fc-chevron fc-chevron-left"><ChevronDown size={12} /></span>
      <span className="fc-chevron fc-chevron-right"><ChevronDown size={12} /></span>
    </div>
  );
}

function MergeConnector() {
  return (
    <div className="flow-connector flow-connector-merge" aria-hidden="true">
      <span className="fc-elbow fc-elbow-merge-left" />
      <span className="fc-elbow fc-elbow-merge-right" />
      <span className="fc-dot" />
      <span className="fc-stem" />
      <span className="fc-chevron fc-chevron-center"><ChevronDown size={12} /></span>
    </div>
  );
}

function ColumnArrow() {
  return (
    <div className="flow-column-arrow"><ChevronDown size={14} /></div>
  );
}

function ColumnPassthrough() {
  return <div className="flow-column-passthrough" />;
}

export default function DecisionFlow({ detail, reportState, precedentsState, automationGates, auditTrail }) {
  const score = detail.score;
  const isOpen = detail.status === "OPEN";
  const decisions = detail.decisions || [];
  const lastDecision = decisions.length ? decisions[decisions.length - 1] : null;

  const coreTone = score ? score.risk_band.toLowerCase() : "neutral";
  const coreSummary = score ? `${score.risk_band} · ${score.hybrid_score.toFixed(0)}/100` : "No score";

  const topShap = detail.shap_explanations?.length
    ? [...detail.shap_explanations].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))[0]
    : null;
  const shapSummary = topShap
    ? `Top factor: ${SHAP_FEATURE_LABELS[topShap.feature] || topShap.feature} (${topShap.shap_value >= 0 ? "+" : ""}${topShap.shap_value.toFixed(2)})`
    : "No SHAP data";

  const reportSummary = reportState?.status === "generating"
    ? "Generating…"
    : reportState?.status === "ready" && reportState.report
      ? `Generated (${reportState.report.source || "unknown"})`
      : "Not yet generated";
  const reportDim = reportState?.status !== "ready";

  let precSummary = "Loading…";
  let precTone = "neutral";
  let precDim = true;
  if (precedentsState) {
    precDim = false;
    if (precedentsState.summary.suggested_decision) {
      precSummary = `${ACTION_LABELS[precedentsState.summary.suggested_decision]} · ${(precedentsState.summary.consensus_ratio * 100).toFixed(0)}% consensus`;
      precTone = "green";
    } else {
      precSummary = "Insufficient precedent";
      precTone = "amber";
    }
  }

  let autoSummary = "Loading…";
  let autoTone = "neutral";
  let autoDim = true;
  let autoTargetId = "flow-automation";
  if (isOpen) {
    if (automationGates === null) {
      autoSummary = "Not configured";
    } else if (automationGates) {
      autoDim = false;
      const passed = automationGates.gates.filter((g) => g.passed).length;
      autoSummary = automationGates.eligible
        ? `Eligible · ${passed}/6 gates passed`
        : `Not eligible · ${passed}/6 gates passed`;
      autoTone = automationGates.eligible ? "green" : "gray";
    }
  } else {
    autoTargetId = "flow-audit";
    if (auditTrail) {
      autoDim = false;
      const reviewed = auditTrail.events.find((e) => e.event_type === "automation_reviewed");
      const proposed = auditTrail.events.find((e) => e.event_type === "automation_proposed");
      if (reviewed) {
        autoSummary = `Analyst ${reviewed.after} the proposal`;
        autoTone = reviewed.after === "confirmed" ? "green" : "gray";
      } else if (proposed) {
        autoSummary = "Proposed, not reviewed";
        autoTone = "amber";
      } else {
        autoSummary = "Not proposed";
        autoTone = "neutral";
      }
    }
  }

  const humanSummary = isOpen
    ? "Awaiting decision"
    : lastDecision
      ? `${ACTION_LABELS[lastDecision.action_taken] || lastDecision.action_taken}${lastDecision.ai_proposed ? " · AI-confirmed" : ""}`
      : "No decision recorded";
  const humanTone = isOpen ? "neutral" : (lastDecision?.action_taken === "confirm_fraud" ? "red" : lastDecision?.action_taken === "approve_clean" ? "green" : "gray");
  const mechanicalHandoff = !isOpen && !!lastDecision?.ai_proposed;

  return (
    <div className="card decision-flow-card">
      <div className="card-head-row">
        <h2><Activity size={16} /> Decision Flow</h2>
        <span className="card-caption">Summary view · click a step to jump to its detail</span>
      </div>

      <div className="decision-flow-grid">
        <FlowNode targetId="flow-score" icon={Activity} title="Transaction & Scoring" summary={coreSummary} tone={coreTone} span2 />

        <BranchConnector />

        <FlowNode targetId="flow-shap" icon={BarChart3} title="SHAP" summary={shapSummary} tone="neutral" />
        <FlowNode targetId="flow-precedent" icon={Users} title="Precedent Analysis" summary={precSummary} tone={precTone} dim={precDim} />

        <ColumnArrow />
        <ColumnPassthrough />

        <FlowNode targetId="flow-report" icon={Bot} title="Investigator Report" summary={reportSummary} tone="neutral" dim={reportDim} />
        <ColumnPassthrough />

        <MergeConnector />

        <FlowNode
          targetId={autoTargetId}
          icon={ListChecks}
          title="Automation Eligibility"
          summary={autoSummary}
          tone={autoTone}
          dim={autoDim}
          span2
        />

        <div className={`flow-connector-single${mechanicalHandoff ? " flow-connector-single-solid" : " flow-connector-single-informs"}`}>
          {mechanicalHandoff ? <ChevronDown size={14} /> : <span className="flow-informs-label">informs</span>}
        </div>

        <FlowNode targetId="flow-decision" icon={Gavel} title="Analyst Decision" summary={humanSummary} tone={humanTone} dim={isOpen} span2 />
      </div>
    </div>
  );
}
