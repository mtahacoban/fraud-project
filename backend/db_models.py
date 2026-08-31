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

    error_balance_orig: Mapped[float] = mapped_column(Float)
    error_balance_dest: Mapped[float] = mapped_column(Float)
    step_hour: Mapped[int] = mapped_column(Integer)
    is_transfer: Mapped[int] = mapped_column(Integer)
    is_cashout: Mapped[int] = mapped_column(Integer)

    device_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_known_device: Mapped[int | None] = mapped_column(Integer, nullable=True)
    login_country: Mapped[str | None] = mapped_column(String(8), nullable=True)
    geo_velocity_flag: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channel: Mapped[str | None] = mapped_column(String(16), nullable=True)

    is_fraud: Mapped[int | None] = mapped_column(Integer, nullable=True)

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
    ai2_suggested_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
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
    label: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PrecedentExplanation(Base):
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
    __tablename__ = "auto_block_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"))
    case_id: Mapped[int | None] = mapped_column(ForeignKey("cases.id"), nullable=True)
    policy_version_id: Mapped[int | None] = mapped_column(ForeignKey("automation_policy_versions.id"), nullable=True)
    triggered_conditions: Mapped[dict] = mapped_column(JSON)
    review_status: Mapped[str] = mapped_column(String(16))
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AutomationPolicyVersion(Base):
    __tablename__ = "automation_policy_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(32))
    clean_similarity_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    fraud_similarity_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    hard_rule_required: Mapped[bool] = mapped_column(Boolean, default=False)

    mode: Mapped[str] = mapped_column(String(16), default="off")
    min_precedent_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    min_consensus_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_calibrated_proba: Mapped[float | None] = mapped_column(Float, nullable=True)
    auto_clean_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    circuit_breaker_max_reversal_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    circuit_breaker_min_confirmations: Mapped[int | None] = mapped_column(Integer, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    auto_triggered: Mapped[bool] = mapped_column(Boolean, default=False)

    active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
