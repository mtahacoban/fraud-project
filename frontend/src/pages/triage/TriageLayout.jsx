import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ListChecks } from "lucide-react";
import PageHeader from "../../components/PageHeader.jsx";
import CaseList from "./CaseList.jsx";
import CaseDetailPanel from "./CaseDetailPanel.jsx";

export default function TriageLayout() {
  const [searchParams] = useSearchParams();
  const selectedCaseId = searchParams.get("case");
  const [refreshToken, setRefreshToken] = useState(0);

  return (
    <div>
      <PageHeader
        icon={ListChecks}
        eyebrow="Operations"
        title="Triage"
        subtitle="Review the case queue and investigate a case side by side"
        tone="red"
      />

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
