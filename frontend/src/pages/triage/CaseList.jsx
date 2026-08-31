import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  Bot, ChevronLeft, ChevronRight, Download, FileSpreadsheet, Filter, RotateCcw, Search,
  SlidersHorizontal,
} from "lucide-react";
import { exportCasesCsvUrl, exportCasesXlsxUrl, getCaseFilterOptions, getCases } from "../../api.js";
import RiskBadge from "../../components/RiskBadge.jsx";

const STATUS_LABELS = {
  OPEN: "Open",
  CLOSED: "Closed",
  AUTO_CLEAN: "Auto-Clean",
};

const PAGE_SIZE = 50;
const DEFAULT_SORT = "hybrid_score";
const AMOUNT_DEBOUNCE_MS = 400;

const PANEL_FILTER_KEYS = ["type", "country", "date_from", "date_to", "amount_min", "amount_max"];

export default function CaseList({ selectedCaseId, refreshToken }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [offset, setOffset] = useState(0);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [countryOptions, setCountryOptions] = useState([]);
  const listRef = useRef(null);

  const status = searchParams.get("status") || "";
  const riskBand = searchParams.get("risk_band") || "";
  const sort = searchParams.get("sort") || DEFAULT_SORT;
  const q = searchParams.get("q") || "";
  const type = searchParams.get("type") || "";
  const country = searchParams.get("country") || "";
  const dateFrom = searchParams.get("date_from") || "";
  const dateTo = searchParams.get("date_to") || "";
  const amountMin = searchParams.get("amount_min") || "";
  const amountMax = searchParams.get("amount_max") || "";

  const [search, setSearch] = useState(q);
  const [amountMinDraft, setAmountMinDraft] = useState(amountMin);
  const [amountMaxDraft, setAmountMaxDraft] = useState(amountMax);

  useEffect(() => setSearch(q), [q]);
  useEffect(() => setAmountMinDraft(amountMin), [amountMin]);
  useEffect(() => setAmountMaxDraft(amountMax), [amountMax]);

  useEffect(() => {
    getCaseFilterOptions().then((res) => setCountryOptions(res.countries)).catch(() => setCountryOptions([]));
  }, []);

  const filterKey = [status, riskBand, sort, q, type, country, dateFrom, dateTo, amountMin, amountMax].join("|");
  useEffect(() => {
    setOffset(0);
  }, [filterKey]);

  useEffect(() => {
    if (!selectedCaseId || !result) return;
    listRef.current
      ?.querySelector(`[data-case-id="${selectedCaseId}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [selectedCaseId, result]);

  useEffect(() => {
    let cancelled = false;
    setResult((prev) => (prev ? { ...prev, loading: true } : null));
    getCases({
      status, risk_band: riskBand, sort, order: "desc", q, limit: PAGE_SIZE, offset,
      type, country, date_from: dateFrom, date_to: dateTo, amount_min: amountMin, amount_max: amountMax,
    })
      .then((res) => { if (!cancelled) setResult(res); })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [filterKey, offset, refreshToken]);

  useEffect(() => {
    if (amountMinDraft === amountMin) return;
    const t = setTimeout(() => setParam("amount_min", amountMinDraft), AMOUNT_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [amountMinDraft]);
  useEffect(() => {
    if (amountMaxDraft === amountMax) return;
    const t = setTimeout(() => setParam("amount_max", amountMaxDraft), AMOUNT_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [amountMaxDraft]);

  function setParam(key, value, defaultValue = "") {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      if (value && value !== defaultValue) next.set(key, value); else next.delete(key);
      return next;
    }, { replace: true });
  }

  function handleSearchSubmit(e) {
    e.preventDefault();
    setParam("q", search.trim());
  }

  function clearPanelFilters() {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      for (const key of PANEL_FILTER_KEYS) next.delete(key);
      return next;
    }, { replace: true });
  }

  function caseHref(caseId) {
    const next = new URLSearchParams(searchParams);
    next.set("case", caseId);
    return `/triage?${next.toString()}`;
  }

  const activeFilterCount = PANEL_FILTER_KEYS.filter((k) =>
    k === "amount_min" ? amountMinDraft : k === "amount_max" ? amountMaxDraft : searchParams.get(k)
  ).length;

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
          <select value={status} onChange={(e) => setParam("status", e.target.value)}>
            <option value="">All</option>
            <option value="OPEN">Open</option>
            <option value="CLOSED">Closed</option>
            <option value="AUTO_CLEAN">Auto-Clean (Green)</option>
          </select>
        </label>
        <label>
          <span className="triage-toolbar-label-text">Risk</span>
          <select value={riskBand} onChange={(e) => setParam("risk_band", e.target.value)}>
            <option value="">All</option>
            <option value="RED">Red</option>
            <option value="GRAY">Gray</option>
            <option value="GREEN">Green</option>
          </select>
        </label>
        <label>
          <span className="triage-toolbar-label-text">Sort by</span>
          <select value={sort} onChange={(e) => setParam("sort", e.target.value, DEFAULT_SORT)}>
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
        <button
          type="button"
          className="pagination-btn triage-filters-toggle"
          style={filtersOpen || activeFilterCount > 0 ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}
          onClick={() => setFiltersOpen((v) => !v)}
        >
          <SlidersHorizontal size={13} />
          More Filters{activeFilterCount > 0 ? ` (${activeFilterCount})` : ""}
        </button>
        <a
          className="pagination-btn"
          href={exportCasesCsvUrl({
            status, risk_band: riskBand, q, type, country,
            date_from: dateFrom, date_to: dateTo, amount_min: amountMin, amount_max: amountMax,
          })}
        >
          <Download size={13} /> Export CSV
        </a>
        <a
          className="pagination-btn"
          href={exportCasesXlsxUrl({
            status, risk_band: riskBand, q, type, country,
            date_from: dateFrom, date_to: dateTo, amount_min: amountMin, amount_max: amountMax,
          })}
        >
          <FileSpreadsheet size={13} /> Export Excel
        </a>
      </div>

      {filtersOpen && (
        <div className="triage-filters-panel">
          <label>
            <span className="triage-toolbar-label-text">Transaction Type</span>
            <select value={type} onChange={(e) => setParam("type", e.target.value)}>
              <option value="">All</option>
              <option value="TRANSFER">Transfer</option>
              <option value="CASH_OUT">Cash-Out</option>
            </select>
          </label>
          <label>
            <span className="triage-toolbar-label-text">Login Country</span>
            <select value={country} onChange={(e) => setParam("country", e.target.value)}>
              <option value="">All</option>
              {countryOptions.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label>
            <span className="triage-toolbar-label-text">Date From</span>
            <input type="date" value={dateFrom} onChange={(e) => setParam("date_from", e.target.value)} />
          </label>
          <label>
            <span className="triage-toolbar-label-text">Date To</span>
            <input type="date" value={dateTo} onChange={(e) => setParam("date_to", e.target.value)} />
          </label>
          <label>
            <span className="triage-toolbar-label-text">Amount Min</span>
            <input
              type="number"
              min="0"
              placeholder="e.g. 100000"
              value={amountMinDraft}
              onChange={(e) => setAmountMinDraft(e.target.value)}
            />
          </label>
          <label>
            <span className="triage-toolbar-label-text">Amount Max</span>
            <input
              type="number"
              min="0"
              placeholder="e.g. 5000000"
              value={amountMaxDraft}
              onChange={(e) => setAmountMaxDraft(e.target.value)}
            />
          </label>
          <button type="button" className="pagination-btn triage-filters-clear" onClick={clearPanelFilters} disabled={activeFilterCount === 0}>
            <RotateCcw size={12} /> Clear Filters
          </button>
        </div>
      )}

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
                  to={caseHref(c.case_id)}
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
