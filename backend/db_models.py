from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    step: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(16))
    amount: Mapped[float] = mapped_column(Float)
    name_orig: Mapped[str | None] = mapped_column(String(32), nullable=True)
    oldbalance_org: Mapped[float] = mapped_column(Float)
    newbalance_orig: Mapped[float] = mapped_column(Float)
    name_dest: Mapped[str | None] = mapped_column(String(32), nullable=True)
    oldbalance_dest: Mapped[float] = mapped_column(Float)
    newbalance_dest: Mapped[float] = mapped_column(Float)

    # Derived (ML_FEATURES)
    error_balance_orig: Mapped[float] = mapped_column(Float)
    error_balance_dest: Mapped[float] = mapped_column(Float)
    step_hour: Mapped[int] = mapped_column(Integer)
    is_transfer: Mapped[int] = mapped_column(Integer)
    is_cashout: Mapped[int] = mapped_column(Integer)

    # Synthetic demo fields — never enter scoring/rules/automation/similarity, UI/LLM only
    device_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_known_device: Mapped[int | None] = mapped_column(Integer, nullable=True)
    login_country: Mapped[str | None] = mapped_column(String(8), nullable=True)
    geo_velocity_flag: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channel: Mapped[str | None] = mapped_column(String(16), nullable=True)

    is_fraud: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # "live" (default) vs "simulator" — lets simulator-generated traffic be
    # filtered out of real metrics/analysis without touching scoring logic.
    source: Mapped[str] = mapped_column(String(16), default="live", server_default="live")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    scores: Mapped[list["Score"]] = relationship(back_populates="transaction")
    rule_hits: Mapped[list["RuleHit"]] = relationship(back_populates="transaction")
    shap_explanations: Mapped[list["ShapExplanation"]] = relationship(back_populates="transaction")
    cases: Mapped[list["Case"]] = relationship(back_populates="transaction")


class Score(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"))
    ml_score: Mapped[float] = mapped_column(Float)
    rule_score: Mapped[float] = mapped_column(Float)
    hybrid_score: Mapped[float] = mapped_column(Float)
    risk_band: Mapped[str] = mapped_column(String(8))
    calibrated_proba: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64))
    # set when a triage rule promoted the band beyond the hybrid formula (see scoring.py)
    band_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    transaction: Mapped["Transaction"] = relationship(back_populates="scores")


class RuleHit(Base):
    __tablename__ = "rule_hits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"))
    rule_name: Mapped[str] = mapped_column(String(64))
    rule_type: Mapped[str] = mapped_column(String(8))
    severity: Mapped[str] = mapped_column(String(16))

    transaction: Mapped["Transaction"] = relationship(back_populates="rule_hits")


class ShapExplanation(Base):
    __tablename__ = "shap_explanations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"))
    feature_name: Mapped[str] = mapped_column(String(64))
    shap_value: Mapped[float] = mapped_column(Float)
    feature_value: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(16))

    transaction: Mapped["Transaction"] = relationship(back_populates="shap_explanations")


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"))
    status: Mapped[str] = mapped_column(String(16), default="OPEN")
    priority: Mapped[str] = mapped_column(String(8), default="NORMAL")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    transaction: Mapped["Transaction"] = relationship(back_populates="cases")
    llm_reports: Mapped[list["LlmReport"]] = relationship(back_populates="case")
    decisions: Mapped[list["AnalystDecision"]] = relationship(back_populates="case")


class LlmReport(Base):
    __tablename__ = "llm_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    report_text: Mapped[str] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(String(64))
    # "groq" (or another future provider) vs. "fallback" — nullable so
    # rows written before this column existed stay valid.
    source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    case: Mapped["Case"] = relationship(back_populates="llm_reports")


class AnalystDecision(Base):
    __tablename__ = "analyst_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    action_taken: Mapped[str] = mapped_column(String(32))
    analyst_reason_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    analyst_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    auto_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    # What the precedent engine suggested for this case at the moment it
    # was decided (captured in main.py's decide_case(), before this
    # decision is added to precedent_index — self-exclusion is automatic
    # since the case isn't indexed yet at that point). NULL means no
    # suggestion existed (cold start, or the confidence gates weren't met)
    # — not "not measured". Rows from before this column existed stay NULL
    # rather than being guessed retroactively.
    ai2_suggested_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Provenance: was this decision the analyst's own call, or a
    # human-confirmed automation proposal? auto_processed deliberately
    # stays False in both cases — nothing in this system finalizes without
    # a human clicking Confirm, so auto_processed=True would misrepresent
    # every decision this system can produce. ai_proposed is the real
    # signal: True only via POST /cases/{id}/confirm-ai-decision.
    # ai_proposal_id links back to the exact auto_block_log row (and, via
    # its policy_version_id, the exact policy) that produced the proposal —
    # needed to distinguish AI-confirmed from analyst-original decisions
    # and for a full audit trail.
    ai_proposed: Mapped[bool] = mapped_column(Boolean, default=False)
    ai_proposal_id: Mapped[int | None] = mapped_column(ForeignKey("auto_block_log.id"), nullable=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    case: Mapped["Case"] = relationship(back_populates="decisions")


class PrecedentIndex(Base):
    __tablename__ = "precedent_index"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"))
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    feature_vector: Mapped[dict] = mapped_column(JSON)
    # Raw analyst_decisions.action_taken value, stored as-is (no shorthand
    # mapping) so precedent_index.label and analyst_decisions.action_taken
    # are always identical strings.
    label: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PrecedentExplanation(Base):
    """Cached LLM explanation for a case's precedent summary. Only ever
    written when summarize_precedents() found a suggested_decision — the
    no-suggestion path is instant/deterministic and never calls the LLM,
    so it has nothing worth caching.

    precedent_count/suggested_decision record what this explanation was
    about, for display/debugging. The actual cache-invalidation key is
    pool_size_at_generation: the total precedent_index row count at
    generation time. precedent_index grows live as analysts decide cases,
    and a query case's own neighbor count/suggestion can stay unchanged
    (still capped at k) even while its actual *set* of neighbors shifts
    underneath — so "the pool changed at all" is the reliable,
    easy-to-reason-about signal, not "this case's aggregate numbers
    happened to change." A new decision anywhere invalidates every cached
    explanation; a new row is appended on regeneration, not overwritten —
    same append-only, "take latest" pattern as llm_reports."""
    __tablename__ = "precedent_explanations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    explanation_text: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(16))
    precedent_count: Mapped[int] = mapped_column(Integer)
    suggested_decision: Mapped[str] = mapped_column(String(32))
    pool_size_at_generation: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AutoBlockLog(Base):
    """The automation event log — one row per evaluate_auto_decision()
    call, whatever the outcome. review_status tracks the row's lifecycle:
    "shadow" (mode=shadow — logged, never surfaced to the analyst),
    "proposed" (mode=propose, eligible — pending in Case Detail),
    "confirmed" / "rejected" (analyst acted on a proposal), "withdrawn"
    (proposal invalidated before the analyst acted, e.g. the pool moved
    on). No column default: both construction sites (log_shadow_evaluation,
    propose_auto_decision) always set review_status explicitly, and it
    should stay that way — a silent default here would be a stale-looking
    value no code actually chose.

    case_id gives direct access without a join through transaction_id;
    policy_version_id records exactly which policy thresholds produced
    this row, for an audit trail across policy changes."""
    __tablename__ = "auto_block_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"))
    case_id: Mapped[int | None] = mapped_column(ForeignKey("cases.id"), nullable=True)
    policy_version_id: Mapped[int | None] = mapped_column(ForeignKey("automation_policy_versions.id"), nullable=True)
    triggered_conditions: Mapped[dict] = mapped_column(JSON)
    review_status: Mapped[str] = mapped_column(String(16))
    # Required whenever review_status becomes "rejected" — rubber-stamping
    # friction (a reject with no reason is rejected at the API level) and
    # the raw material for the reject-rate metric.
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AutomationPolicyVersion(Base):
    """Versioned policy for the human-confirmed automation gate. Every
    threshold change is a NEW row (never edit one in place) —
    get_active_policy() always reads whichever single row has active=True,
    so switching policy is "deactivate old, activate new", fully auditable
    via created_at/notes history."""
    __tablename__ = "automation_policy_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(32))
    clean_similarity_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)  # reserved — auto_clean is disabled in v1
    fraud_similarity_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)  # Gate: min similarity (~0.95)
    hard_rule_required: Mapped[bool] = mapped_column(Boolean, default=False)  # Gate B strictness switch — see evaluate_auto_decision()

    mode: Mapped[str] = mapped_column(String(16), default="off")  # "off" | "shadow" | "propose"
    min_precedent_count: Mapped[int | None] = mapped_column(Integer, nullable=True)  # Gate: min precedent count (~10)
    min_consensus_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)  # Gate: min consensus (~0.90)
    min_calibrated_proba: Mapped[float | None] = mapped_column(Float, nullable=True)  # Gate: min calibrated_proba (~0.95) — scoring engine / precedent engine agreement
    auto_clean_enabled: Mapped[bool] = mapped_column(Boolean, default=False)  # v1: always False — fraud-direction only
    circuit_breaker_max_reversal_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    circuit_breaker_min_confirmations: Mapped[int | None] = mapped_column(Integer, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)  # why this version was created (tuning vs. circuit-breaker trip)
    # True only when check_circuit_breaker() itself called
    # activate_new_policy_version() — distinguishes an automatic trip from
    # a deliberate manual policy change, without string-matching `notes`.
    auto_triggered: Mapped[bool] = mapped_column(Boolean, default=False)

    active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
