import { Navigate, NavLink, Route, Routes, useParams } from "react-router-dom";
import { LayoutDashboard, ListChecks, LineChart, FlaskConical, ShieldAlert } from "lucide-react";
import Dashboard from "./pages/Dashboard.jsx";
import Evaluation from "./pages/Evaluation.jsx";
import Simulation from "./pages/Simulation.jsx";
import AutomationStatus from "./pages/AutomationStatus.jsx";
import TriageLayout from "./pages/triage/TriageLayout.jsx";

const NAV_ITEMS = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/triage", label: "Triage", icon: ListChecks, end: false },
  { to: "/automation", label: "Automation", icon: ShieldAlert, end: false },
  { to: "/simulation", label: "Simulation", icon: FlaskConical, end: false },
  { to: "/evaluation", label: "Evaluation", icon: LineChart, end: false },
];

// Retired in Aşama 4.2.4 — CaseQueue.jsx/CaseDetail.jsx are superseded by
// Triage (pages/triage/) but stay on disk, unimported, as an archival
// reference (deletion is its own separate, later decision). These two
// routes exist only to carry old bookmarks/links forward, param included.
function CaseDetailRedirect() {
  const { id } = useParams();
  return <Navigate to={`/triage?case=${id}`} replace />;
}

function App() {
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-dot" />
          <span className="brand-text">
            <span className="brand-name">Fraud Investigation</span>
            <span className="brand-sub">Decision Support Platform</span>
          </span>
        </div>
        <nav>
          {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className="nav-link">
              <Icon size={16} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">AI-Powered Fraud Detection &amp; Investigation System</div>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/triage" element={<TriageLayout />} />
          {/* Old bookmarks/links still work — redirected, param preserved. */}
          <Route path="/cases" element={<Navigate to="/triage" replace />} />
          <Route path="/cases/:id" element={<CaseDetailRedirect />} />
          <Route path="/simulation" element={<Simulation />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/automation" element={<AutomationStatus />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
