import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Bot, ChevronLeft, ChevronRight, Filter, Search } from "lucide-react";
import { getCases } from "../../api.js";
import RiskBadge from "../../components/RiskBadge.jsx";

// Ported byte-for-byte from CaseQueue.jsx (Aşama 4.2.2) — same state shape,
// same GET /cases params, same pagination math. Only the rendering below is
// adapted for a narrow column instead of a full-page table.
const STATUS_LABELS = {
  OPEN: "Open",
  CLOSED: "Closed",
  AUTO_CLEAN: "Auto-Clean",
};

const PAGE_SIZE = 50;

export default function CaseList({ selectedCaseId, refreshToken }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState("");
  const [riskBand, setRiskBand] = useState("");
  const [sort, setSort] = useState("hybrid_score");
  const [search, setSearch] = useState("");
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  const listRef = useRef(null);

  useEffect(() => {
    setOffset(0);
  }, [status, riskBand, sort, q]);

  // Scroll the selected row into view when it's already on the loaded page
  // (e.g. arriving via a deep link or a Dashboard/Simulation link) —
  // "nearest" only moves the pane if the row is actually out of view, so
  // it never fights a scroll position the analyst set themselves.
  useEffect(() => {
    if (!selectedCaseId || !result) return;
    listRef.current
      ?.querySelector(`[data-case-id="${selectedCaseId}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [selectedCaseId, result]);

  useEffect(() => {
    let cancelled = false;
    setResult((prev) => (prev ? { ...prev, loading: true } : null));
    getCases({ status, risk_band: riskBand, sort, order: "desc", q, limit: PAGE_SIZE, offset })
      .then((res) => { if (!cancelled) setResult(res); })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
    // refreshToken bumps after a decision/confirm/reject/reopen succeeds in
    // CaseDetailPanel (via TriageLayout's onCaseChanged) — refetches this
    // list without a manual page reload.
  }, [status, riskBand, sort, q, offset, refreshToken]);

  function handleSearchSubmit(e) {
    e.preventDefault();
    setQ(search.trim());
  }

  const cases = result?.items;
  const total = result?.total ?? 0;
  const page = Math.floor(offset / PAGE_SIZE) + 1;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const rangeStart = total === 0 ? 0 : offset + 1;
  const rangeEnd = Math.min(offset + PAGE_SIZE, total);

  return (
    <div className="card triage-list-card">
      <div className="triage-toolbar">
        <label>
          <span className="triage-toolbar-label-text"><Filter size={13} /> Status</span>
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">All</option>
            <option value="OPEN">Open</option>
            <option value="CLOSED">Closed</option>
            <option value="AUTO_CLEAN">Auto-Clean (Green)</option>
          </select>
        </label>
        <label>
          <span className="triage-toolbar-label-text">Risk</span>
          <select value={riskBand} onChange={(e) => setRiskBand(e.target.value)}>
            <option value="">All</option>
            <option value="RED">Red</option>
            <option value="GRAY">Gray</option>
            <option value="GREEN">Green</option>
          </select>
        </label>
        <label>
          <span className="triage-toolbar-label-text">Sort by</span>
          <select value={sort} onChange={(e) => setSort(e.target.value)}>
            <option value="hybrid_score">Highest risk</option>
            <option value="created_at">Newest</option>
          </select>
        </label>
        <form onSubmit={handleSearchSubmit} className="triage-toolbar-search">
          <Search size={13} />
          <input
            type="text"
            placeholder="Search case or transaction #"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </form>
      </div>

      {error && <p className="error">Error: {error}</p>}
      {!cases && !error && <p className="loading">Loading</p>}
      {cases && cases.length === 0 && <p className="note">No cases found.</p>}

      {cases && cases.length > 0 && (
        <>
          <div className="triage-row-list" ref={listRef}>
            {cases.map((c) => {
              const isSelected = c.case_id != null && String(c.case_id) === String(selectedCaseId);
              const rowContent = (
                <>
                  <div className="triage-row-top">
                    <span className="triage-row-id">{c.case_id ? `#${c.case_id}` : `txn #${c.transaction_id}`}</span>
                    <RiskBadge band={c.risk_band} score={c.hybrid_score} />
                  </div>
                  <div className="triage-row-meta">
                    <span>{c.priority} · {STATUS_LABELS[c.status] || c.status}</span>
                    <span>{new Date(c.created_at).toLocaleDateString("en-US")}</span>
                  </div>
                  {c.pending_ai_proposal && (
                    <span className="status-pill status-pill-accent" style={{ marginTop: 6 }}>
                      <Bot size={11} /> AI proposal
                    </span>
                  )}
                </>
              );

              return c.case_id ? (
                <Link
                  key={c.case_id}
                  data-case-id={c.case_id}
                  to={`/triage?case=${c.case_id}`}
                  className={`triage-row${isSelected ? " triage-row-selected" : ""}`}
                >
                  {rowContent}
                </Link>
              ) : (
                <div key={`txn-${c.transaction_id}`} className="triage-row triage-row-disabled">
                  {rowContent}
                </div>
              );
            })}
          </div>
          <div className="pagination">
            <span className="pagination-summary">
              {rangeStart}–{rangeEnd} of {total.toLocaleString("en-US")}
            </span>
            <div className="pagination-controls">
              <button
                type="button"
                className="pagination-btn"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              >
                <ChevronLeft size={14} /> Prev
              </button>
              <span className="pagination-page">Page {page} of {pageCount}</span>
              <button
                type="button"
                className="pagination-btn"
                disabled={offset + PAGE_SIZE >= total}
                onClick={() => setOffset(offset + PAGE_SIZE)}
              >
                Next <ChevronRight size={14} />
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
