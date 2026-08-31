import { Link } from "react-router-dom";
import { AlertTriangle, FileText, ShieldCheck, Users } from "lucide-react";
import {
  DndContext, KeyboardSensor, PointerSensor, closestCenter, useSensor, useSensors,
} from "@dnd-kit/core";
import {
  SortableContext, arrayMove, rectSortingStrategy, sortableKeyboardCoordinates, useSortable,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";

const AUTOMATION_MODE_LABELS = {
  off: "Off",
  shadow: "Shadow (observing)",
  propose: "Propose (active)",
};

export const DEFAULT_ORDER = ["scored_transactions", "high_risk", "open_cases", "pending_proposals"];
export const STORAGE_KEY = "dashboard_kpi_order";

export function loadOrder() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_ORDER;
    const saved = JSON.parse(raw);
    if (!Array.isArray(saved)) return DEFAULT_ORDER;
    const known = saved.filter((k) => DEFAULT_ORDER.includes(k));
    const missing = DEFAULT_ORDER.filter((k) => !known.includes(k));
    return [...known, ...missing];
  } catch {
    return DEFAULT_ORDER;
  }
}

export function saveOrder(order) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(order));
  } catch {
  }
}

function ScoredTransactionsCard({ metrics }) {
  return (
    <Link to="/triage" className="kpi-card kpi-card-link">
      <div className="kpi-icon-badge kpi-icon-purple">
        <Users size={22} strokeWidth={2} />
      </div>
      <div className="kpi-body">
        <span className="kpi-label">Scored Transactions</span>
        <span className="kpi-value">{metrics.total_scored}</span>
        <span className="kpi-caption">{metrics.live_scored} live · {metrics.demo_scored} simulated</span>
      </div>
    </Link>
  );
}

function HighRiskCard({ metrics }) {
  const redCount = metrics.by_risk_band?.RED || 0;
  return (
    <Link to="/triage?status=OPEN&risk_band=RED" className="kpi-card kpi-card-link">
      <div className="kpi-icon-badge kpi-icon-red">
        <AlertTriangle size={22} strokeWidth={2} />
      </div>
      <div className="kpi-body">
        <span className="kpi-label">High-Risk (RED)</span>
        <span className="kpi-value">{redCount}</span>
        <span className="kpi-caption">of {metrics.total_scored} scored</span>
      </div>
    </Link>
  );
}

function OpenCasesCard({ metrics }) {
  return (
    <Link to="/triage?status=OPEN" className="kpi-card kpi-card-link">
      <div className="kpi-icon-badge kpi-icon-info">
        <FileText size={22} strokeWidth={2} />
      </div>
      <div className="kpi-body">
        <span className="kpi-label">Open Cases</span>
        <span className="kpi-value">{metrics.open_cases}</span>
        <span className="kpi-caption">{metrics.closed_cases} closed · {metrics.total_cases} total</span>
      </div>
    </Link>
  );
}

function PendingProposalsCard({ metrics }) {
  return (
    <Link to="/automation" className="kpi-card kpi-card-link">
      <div className="kpi-icon-badge kpi-icon-green">
        <ShieldCheck size={22} strokeWidth={2} />
      </div>
      <div className="kpi-body">
        <span className="kpi-label">Pending AI Proposals</span>
        <span className="kpi-value" style={metrics.pending_ai_proposals > 0 ? { color: "var(--accent)" } : undefined}>
          {metrics.pending_ai_proposals}
        </span>
        <span className="kpi-caption">
          automation: {AUTOMATION_MODE_LABELS[metrics.automation_mode] || "not configured"}
        </span>
      </div>
    </Link>
  );
}

const KPI_CARDS = {
  scored_transactions: ScoredTransactionsCard,
  high_risk: HighRiskCard,
  open_cases: OpenCasesCard,
  pending_proposals: PendingProposalsCard,
};

function DraggableKpiCard({ id, children }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id });
  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
    boxShadow: isDragging ? "var(--shadow-sm)" : undefined,
    cursor: isDragging ? "grabbing" : "grab",
    touchAction: "none",
  };
  return (
    <div ref={setNodeRef} style={style} className="kpi-draggable" {...attributes} {...listeners}>
      {children}
    </div>
  );
}

export default function KpiCards({ metrics, order, onOrderChange }) {
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  function handleDragEnd(event) {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      const next = arrayMove(order, order.indexOf(active.id), order.indexOf(over.id));
      saveOrder(next);
      onOrderChange(next);
    }
  }

  return (
    <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
      <SortableContext items={order} strategy={rectSortingStrategy}>
        <div className="kpi-grid">
          {order.map((key) => {
            const Card = KPI_CARDS[key];
            return (
              <DraggableKpiCard key={key} id={key}>
                <Card metrics={metrics} />
              </DraggableKpiCard>
            );
          })}
        </div>
      </SortableContext>
    </DndContext>
  );
}
