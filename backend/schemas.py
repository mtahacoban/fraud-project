from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

TxnType = Literal["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]


class TransactionIn(BaseModel):
    step: int = Field(ge=0, description="PaySim time step (hour)")
    type: TxnType
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)
    nameOrig: str | None = None
    nameDest: str | None = None

    device_id: str | None = None
    is_known_device: int | None = None
    login_country: str | None = None
    geo_velocity_flag: int | None = None
    channel: str | None = None

    isFraud: int | None = None


class ShapFactorOut(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str


class ScoreOut(BaseModel):
    txn_id: int
    case_id: int | None
    ml_score: float
    soft_score: float
    hybrid_score: float
    risk_band: str
    calibrated_proba: float
    hard_rule_hits: list[str]
    soft_rule_hits: list[str]
    shap_factors: list[ShapFactorOut]
    model_version: str
    band_reason: str | None = None


class CaseSummaryOut(BaseModel):
    case_id: int | None
    transaction_id: int
    status: str
    priority: str
    hybrid_score: float
    risk_band: str
    created_at: datetime
    pending_ai_proposal: bool = False
    amount: float
    type: str
    top_rules: list[str] = []

    model_config = {"from_attributes": True}


class CaseListOut(BaseModel):
    items: list[CaseSummaryOut]
    total: int


class CaseFilterOptionsOut(BaseModel):
    countries: list[str]


class TransactionOut(BaseModel):
    id: int
    step: int
    type: str
    amount: float
    oldbalanceOrg: float = Field(validation_alias="oldbalance_org")
    newbalanceOrig: float = Field(validation_alias="newbalance_orig")
    oldbalanceDest: float = Field(validation_alias="oldbalance_dest")
    newbalanceDest: float = Field(validation_alias="newbalance_dest")
    device_id: str | None
    is_known_device: int | None
    login_country: str | None
    geo_velocity_flag: int | None
    channel: str | None

    model_config = {"from_attributes": True, "populate_by_name": True}


class RuleHitOut(BaseModel):
    rule_name: str
    rule_type: str
    severity: str

    model_config = {"from_attributes": True}


class LlmReportOut(BaseModel):
    report_text: str
    model_name: str
    source: str | None = None
    generated_at: datetime

    model_config = {"from_attributes": True}


class ReportStatusOut(BaseModel):
    status: Literal["ready", "generating"]
    report: LlmReportOut | None = None


class ReportFindingsOut(BaseModel):
    case_id: int
    transaction_summary: str
    findings: list[str]


class PrecedentNeighborOut(BaseModel):
    case_id: int
    similarity: float
    analyst_decision: str
    type: str | None = None
    amount: float | None = None
    step_hour: int | None = None
    risk_band: str | None = None
    hybrid_score: float | None = None
    rule_hits: list[str] = []
    analyst_reason_code: str | None = None


class RulePatternOut(BaseModel):
    rule: str
    count: int
    total: int


class PrecedentSummaryOut(BaseModel):
    precedent_count: int
    avg_similarity: float | None
    decision_distribution: dict[str, int]
    consensus_ratio: float | None
    suggested_decision: str | None
    note: str | None
    common_patterns: list[RulePatternOut] = []
    common_reason_codes: list[RulePatternOut] = []


class PrecedentExplanationOut(BaseModel):
    status: Literal["ready", "generating"]
    text: str | None = None
    source: str | None = None


class PrecedentsOut(BaseModel):
    precedents: list[PrecedentNeighborOut]
    summary: PrecedentSummaryOut
    explanation: PrecedentExplanationOut


class AnalystDecisionOut(BaseModel):
    action_taken: str
    analyst_reason_code: str | None
    analyst_note: str | None
    auto_processed: bool
    ai2_suggested_decision: str | None = None
    ai_proposed: bool = False
    ai_proposal_id: int | None = None
    decided_at: datetime

    model_config = {"from_attributes": True}


class AuditEventOut(BaseModel):
    timestamp: datetime
    event_type: str
    actor: Literal["System", "AI", "Analyst"]
    summary: str
    detail: str | None = None
    before: str | None = None
    after: str | None = None
    anomaly_flags: list[str] = []


class AuditTrailOut(BaseModel):
    case_id: int
    events: list[AuditEventOut]


class PendingAiDecisionOut(BaseModel):
    auto_block_log_id: int
    case_id: int
    direction: str
    triggered_conditions: dict
    policy_version: str
    proposed_at: datetime


class AutomationGateOut(BaseModel):
    gate: str
    passed: bool
    actual: float | int | None
    threshold: float | int | None
    detail: str


class AutomationGatesOut(BaseModel):
    case_id: int
    eligible: bool
    direction: str | None
    policy_version: str
    gates: list[AutomationGateOut]


class ConfirmAiDecisionIn(BaseModel):
    analyst_note: str | None = None


class ConfirmAiDecisionOut(BaseModel):
    case_id: int
    status: str
    action_taken: str
    ai_proposal_id: int
    decided_at: datetime


class RejectAiDecisionIn(BaseModel):
    rejection_reason: str = Field(min_length=1)


class RejectAiDecisionOut(BaseModel):
    case_id: int
    status: str
    auto_block_log_id: int
    rejection_reason: str
    rejected_at: datetime


class ActivePolicyOut(BaseModel):
    version: str
    mode: str
    fraud_similarity_threshold: float | None
    min_precedent_count: int | None
    min_consensus_ratio: float | None
    min_calibrated_proba: float | None
    hard_rule_required: bool
    auto_clean_enabled: bool
    circuit_breaker_max_reversal_rate: float | None
    circuit_breaker_min_confirmations: int | None
    notes: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class CircuitBreakerStatusOut(BaseModel):
    tripped_recently: bool
    last_trip_policy_version: str | None = None
    last_trip_at: datetime | None = None
    last_trip_notes: str | None = None


class AutomationStatusOut(BaseModel):
    active_policy: ActivePolicyOut
    shadow_agreement: dict
    reject_rate: dict
    bias_monitoring: dict
    circuit_breaker: CircuitBreakerStatusOut
    gate_bottleneck: dict


class CaseDetailOut(BaseModel):
    case_id: int
    status: str
    priority: str
    created_at: datetime
    closed_at: datetime | None
    transaction: TransactionOut
    score: ScoreOut | None = None
    rule_hits: list[RuleHitOut]
    shap_explanations: list[ShapFactorOut]
    llm_reports: list[LlmReportOut]
    decisions: list[AnalystDecisionOut]


class DecisionIn(BaseModel):
    action_taken: Literal["approve_clean", "confirm_fraud", "escalate"]
    analyst_reason_code: str = Field(min_length=1)
    analyst_note: str | None = None

    @field_validator("analyst_reason_code")
    @classmethod
    def _reason_code_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("analyst_reason_code is required and cannot be blank")
        return v


class DecisionOut(BaseModel):
    case_id: int
    status: str
    action_taken: str
    decided_at: datetime


class ReopenIn(BaseModel):
    analyst_reason_code: str | None = None
    analyst_note: str | None = None


class ReopenOut(BaseModel):
    case_id: int
    status: str
    reopened_at: datetime


class SimulationTemplateOut(BaseModel):
    id: str
    label: str
    description: str
    input: TransactionIn


class SimulationRunIn(BaseModel):
    transaction: TransactionIn
    count: int = Field(default=1, ge=1, le=500)


class SimulationResultOut(BaseModel):
    txn_id: int
    case_id: int | None
    risk_band: str
    hybrid_score: float
    calibrated_proba: float
    shap_factors: list[ShapFactorOut] = []
    hard_rule_hits: list[str] = []
    soft_rule_hits: list[str] = []
    band_reason: str | None = None


class SimulationRunOut(BaseModel):
    requested: int
    scored: int
    band_counts: dict[str, int]
    results: list[SimulationResultOut]


class MetricsOut(BaseModel):
    total_scored: int
    live_scored: int
    demo_scored: int
    by_risk_band: dict[str, int]
    total_cases: int
    open_cases: int
    closed_cases: int
    avg_hybrid_score: float
    model_version: str
    pending_ai_proposals: int
    automation_mode: str | None


class TrendPointOut(BaseModel):
    date: str
    case_count: int
    red_rate: float | None
    avg_score: float | None
    scored_count: int
