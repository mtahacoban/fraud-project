import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import CaseList from "./CaseList.jsx";
import CaseDetailPanel from "./CaseDetailPanel.jsx";

export default function TriageLayout() {
  const [searchParams] = useSearchParams();
  const selectedCaseId = searchParams.get("case");
  // Bumped by CaseDetailPanel's onCaseChanged after a decision/confirm/
  // reject/reopen succeeds — included in CaseList's GET /cases effect
  // dependencies, so the list refetches without a manual page reload.
  const [refreshToken, setRefreshToken] = useState(0);

  return (
    <div>
      <h1>Triage</h1>
      <p className="subtitle">Review the case queue and investigate a case side by side</p>

      <div className="triage-layout">
        <div className="triage-list-pane">
          <CaseList selectedCaseId={selectedCaseId} refreshToken={refreshToken} />
        </div>
        <div className="triage-detail-pane">
          {selectedCaseId ? (
            <CaseDetailPanel
              caseId={selectedCaseId}
              key={selectedCaseId}
              onCaseChanged={() => setRefreshToken((t) => t + 1)}
            />
          ) : (
            <div className="triage-empty-state">
              <p className="note">Select a case from the list to view its details.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
