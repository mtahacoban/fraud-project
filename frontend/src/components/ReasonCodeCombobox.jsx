import { useEffect, useRef, useState } from "react";

export const REASON_CODE_GROUPS = [
  {
    label: "Fraud", tone: "red",
    codes: ["account_takeover", "unauthorized_transaction", "identity_theft", "phishing_victim", "card_fraud", "social_engineering"],
  },
  {
    label: "AML", tone: "accent",
    codes: ["money_laundering", "structuring", "mule_account", "suspicious_pattern", "sanctions_concern", "terrorist_financing"],
  },
  {
    label: "Clean", tone: "green",
    codes: ["verified_legitimate", "false_positive", "customer_confirmed"],
  },
  {
    label: "Other", tone: "gray",
    codes: ["insufficient_evidence", "escalated_to_compliance", "pending_investigation"],
  },
];

const optionId = (code) => `reason-code-opt-${code}`;

export default function ReasonCodeCombobox({ value, onChange, placeholder = "Select or type a reason code…" }) {
  const [open, setOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(null);
  const containerRef = useRef(null);
  const optionRefs = useRef({});

  const query = (value || "").trim().toLowerCase();
  const filteredGroups = REASON_CODE_GROUPS
    .map((g) => ({ ...g, codes: query ? g.codes.filter((c) => c.includes(query)) : g.codes }))
    .filter((g) => g.codes.length > 0);
  const flatOptions = filteredGroups.flatMap((g) => g.codes);
  const noMatches = query.length > 0 && flatOptions.length === 0;

  useEffect(() => {
    if (!open) return;
    function handlePointerDown(e) {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, [open]);

  useEffect(() => {
    if (highlightedIndex === null) return;
    const code = flatOptions[highlightedIndex];
    optionRefs.current[code]?.scrollIntoView({ block: "nearest" });
  }, [highlightedIndex]);

  function selectCode(code) {
    onChange(code);
    setOpen(false);
    setHighlightedIndex(null);
  }

  function handleKeyDown(e) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (!open) { setOpen(true); return; }
      if (flatOptions.length === 0) return;
      setHighlightedIndex((i) => (i === null ? 0 : Math.min(i + 1, flatOptions.length - 1)));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (!open || flatOptions.length === 0) return;
      setHighlightedIndex((i) => (i === null ? flatOptions.length - 1 : Math.max(i - 1, 0)));
    } else if (e.key === "Enter") {
      if (!open) return;
      e.preventDefault();
      if (highlightedIndex !== null) {
        selectCode(flatOptions[highlightedIndex]);
      } else {
        setOpen(false);
      }
    } else if (e.key === "Escape") {
      if (!open) return;
      e.preventDefault();
      setOpen(false);
      setHighlightedIndex(null);
    }
  }

  const activeDescendant = highlightedIndex !== null ? optionId(flatOptions[highlightedIndex]) : undefined;
  const listboxId = "reason-code-listbox";

  return (
    <div className="reason-combobox" ref={containerRef}>
      <input
        type="text"
        role="combobox"
        aria-expanded={open}
        aria-controls={listboxId}
        aria-autocomplete="list"
        aria-activedescendant={activeDescendant}
        autoComplete="off"
        placeholder={placeholder}
        value={value || ""}
        onChange={(e) => { onChange(e.target.value); setHighlightedIndex(null); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onKeyDown={handleKeyDown}
      />
      {open && (
        <div className="reason-combobox-dropdown">
          {noMatches ? (
            <p className="note reason-combobox-hint">Will be used as a custom reason code</p>
          ) : (
            <div id={listboxId} role="listbox" className="reason-combobox-listbox">
              {filteredGroups.map((group) => (
                <div key={group.label} role="group" aria-label={group.label} className="reason-combobox-group">
                  <div className={`reason-combobox-group-label reason-combobox-group-label-${group.tone}`}>
                    {group.label}
                  </div>
                  {group.codes.map((code) => {
                    const highlighted = flatOptions[highlightedIndex] === code;
                    return (
                      <div
                        key={code}
                        id={optionId(code)}
                        role="option"
                        aria-selected={highlighted}
                        ref={(el) => { optionRefs.current[code] = el; }}
                        className={`reason-combobox-option${highlighted ? " reason-combobox-option-highlighted" : ""}`}
                        onMouseEnter={() => setHighlightedIndex(flatOptions.indexOf(code))}
                        onClick={() => selectCode(code)}
                      >
                        {code}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
