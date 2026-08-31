import { AlertTriangle, ArrowRight } from "lucide-react";

export function deltaColor(oldVal, newVal) {
  if (newVal < oldVal) return "var(--red)";
  if (newVal > oldVal) return "var(--green)";
  return "var(--text-muted)";
}

function Box({ title, oldVal, newVal, alert }) {
  const fullyDrained = oldVal > 0 && newVal === 0;
  return (
    <div className={`txn-flow-box${alert ? " txn-flow-box-alert" : ""}`}>
      <span className="txn-flow-box-title">
        {alert && <AlertTriangle size={12} />}
        {title}
      </span>
      <div className="txn-flow-balance-row">
        <span className="txn-flow-balance-tag">Before</span>
        <span className="txn-flow-balance-num">{oldVal.toLocaleString("en-US")}</span>
      </div>
      <div className="txn-flow-balance-row">
        <span className="txn-flow-balance-tag">After</span>
        <span className="txn-flow-balance-num txn-flow-balance-num-after" style={{ color: deltaColor(oldVal, newVal) }}>
          {newVal.toLocaleString("en-US")}
        </span>
      </div>
      {alert ? (
        <p className="txn-flow-footnote txn-flow-footnote-alert">
          <AlertTriangle size={11} /> Ghost destination - zero balance before and after
        </p>
      ) : fullyDrained ? (
        <p className="txn-flow-footnote">Fully drained</p>
      ) : null}
    </div>
  );
}

export default function TransactionFlow({ txn, hasGhostDestination }) {
  return (
    <div className="txn-flow">
      <Box title="Sender" oldVal={txn.oldbalanceOrg} newVal={txn.newbalanceOrig} />

      <div className="txn-flow-connector">
        <span className="txn-flow-connector-caption">Transferred Amount</span>
        <div className="txn-flow-connector-line">
          <ArrowRight size={18} />
        </div>
        <span className="txn-flow-connector-amount">{txn.amount.toLocaleString("en-US")}</span>
      </div>

      <Box title="Receiver" oldVal={txn.oldbalanceDest} newVal={txn.newbalanceDest} alert={hasGhostDestination} />
    </div>
  );
}
