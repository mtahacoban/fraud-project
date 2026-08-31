import { useEffect, useState } from "react";
import { Navigate, NavLink, Route, Routes, useParams } from "react-router-dom";
import { LayoutDashboard, ListChecks, LineChart, FlaskConical, ShieldAlert, Sun, Moon, Fingerprint } from "lucide-react";
import Dashboard from "./pages/Dashboard.jsx";
import Evaluation from "./pages/Evaluation.jsx";
import Simulation from "./pages/Simulation.jsx";
import AutomationStatus from "./pages/AutomationStatus.jsx";
import TriageLayout from "./pages/triage/TriageLayout.jsx";

const NAV_GROUPS = [
  {
    label: "Operations",
    items: [
      { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
      { to: "/triage", label: "Triage", icon: ListChecks, end: false },
    ],
  },
  {
    label: "System",
    items: [
      { to: "/automation", label: "Automation", icon: ShieldAlert, end: false },
      { to: "/simulation", label: "Simulation", icon: FlaskConical, end: false },
      { to: "/evaluation", label: "Evaluation", icon: LineChart, end: false },
    ],
  },
];

function CaseDetailRedirect() {
  const { id } = useParams();
  return <Navigate to={`/triage?case=${id}`} replace />;
}

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "light");
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-logo">
            <Fingerprint size={20} strokeWidth={2.25} />
          </div>
          <div>
            <div className="sidebar-brand-title">Fraud Investigation</div>
            <div className="sidebar-brand-sub">Decision support</div>
          </div>
        </div>

        <div className="sidebar-divider" />

        <nav className="sidebar-nav">
          {NAV_GROUPS.map(({ label: groupLabel, items }) => (
            <div key={groupLabel}>
              <div className="sidebar-group-label">{groupLabel}</div>
              {items.map(({ to, label, icon: Icon, end }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={end}
                  className={({ isActive }) => `sidebar-item${isActive ? " active" : ""}`}
                >
                  <Icon size={19} strokeWidth={2.25} />
                  <span>{label}</span>
                </NavLink>
              ))}
            </div>
          ))}
        </nav>

        <div style={{ flex: 1 }} />

        <div className="sidebar-divider" />

        <button
          className="sidebar-theme-toggle"
          onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
          aria-label={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
        >
          {theme === "light" ? <Moon size={14} /> : <Sun size={14} />}
          <span>{theme === "light" ? "Dark mode" : "Light mode"}</span>
        </button>

        <div className="sidebar-divider" />

        <div className="sidebar-user">
          <div className="sidebar-user-avatar">TÇ</div>
          <div className="sidebar-user-name">Taha Çoban</div>
        </div>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/triage" element={<TriageLayout />} />
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
