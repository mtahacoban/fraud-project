from __future__ import annotations

import logging

from sqlalchemy import bindparam, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import settings

logger = logging.getLogger(__name__)

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _add_columns_if_missing(table: str, column_ddls: dict[str, str]) -> None:
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns(table)}
    missing = {name: ddl for name, ddl in column_ddls.items() if name not in existing}
    if not missing:
        return
    with engine.begin() as conn:
        for name, ddl in missing.items():
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def _drop_columns_if_present(table: str, columns: list[str]) -> None:
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns(table)}
    to_drop = [c for c in columns if c in existing]
    if not to_drop:
        return
    with engine.begin() as conn:
        for col in to_drop:
            conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {col}"))


def ensure_llm_reports_source_column() -> None:
    _add_columns_if_missing("llm_reports", {"source": "VARCHAR(16)"})


def ensure_precedent_index_label_width() -> None:
    inspector = inspect(engine)
    if "precedent_index" not in inspector.get_table_names():
        return
    columns = {c["name"]: c for c in inspector.get_columns("precedent_index")}
    label_col = columns.get("label")
    if label_col is None:
        return
    current_length = getattr(label_col["type"], "length", None)
    if current_length is None or current_length >= 32:
        return

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            row_count = conn.execute(text("SELECT COUNT(*) FROM precedent_index")).scalar()
            if row_count:
                return
            conn.execute(text("DROP TABLE precedent_index"))
        else:
            conn.execute(text("ALTER TABLE precedent_index ALTER COLUMN label TYPE VARCHAR(32)"))


def ensure_precedent_explanation_pool_size_column() -> None:
    _add_columns_if_missing("precedent_explanations", {"pool_size_at_generation": "INTEGER DEFAULT 0"})


def ensure_analyst_decisions_ai2_suggestion_column() -> None:
    _add_columns_if_missing("analyst_decisions", {"ai2_suggested_decision": "VARCHAR(32)"})


_AUTOMATION_POLICY_NEW_COLUMNS: dict[str, str] = {
    "mode": "VARCHAR(16) DEFAULT 'off'",
    "min_precedent_count": "INTEGER",
    "min_consensus_ratio": "FLOAT",
    "min_calibrated_proba": "FLOAT",
    "auto_clean_enabled": "BOOLEAN DEFAULT 0",
    "circuit_breaker_max_reversal_rate": "FLOAT",
    "circuit_breaker_min_confirmations": "INTEGER",
    "notes": "TEXT",
}


def ensure_automation_policy_columns() -> None:
    _add_columns_if_missing("automation_policy_versions", _AUTOMATION_POLICY_NEW_COLUMNS)


def ensure_default_automation_policy() -> None:
    from backend import db_models as m

    inspector = inspect(engine)
    if "automation_policy_versions" not in inspector.get_table_names():
        return

    db = SessionLocal()
    try:
        if db.query(m.AutomationPolicyVersion).count():
            return
        db.add(m.AutomationPolicyVersion(
            version="v1",
            mode="shadow",
            active=True,
            fraud_similarity_threshold=0.95,
            clean_similarity_threshold=None,
            min_precedent_count=10,
            min_consensus_ratio=0.90,
            min_calibrated_proba=0.95,
            hard_rule_required=False,
            auto_clean_enabled=False,
            circuit_breaker_max_reversal_rate=0.20,
            circuit_breaker_min_confirmations=5,
            notes=(
                "Initial policy - starts in shadow mode (observe only, no "
                "proposals). Thresholds intentionally stricter than the "
                "precedent engine's own suggestion gates."
            ),
        ))
        db.commit()
    finally:
        db.close()


def ensure_auto_block_log_columns() -> None:
    _add_columns_if_missing("auto_block_log", {"case_id": "INTEGER", "policy_version_id": "INTEGER"})


def ensure_auto_block_log_review_columns() -> None:
    _add_columns_if_missing("auto_block_log", {"rejection_reason": "TEXT", "reviewed_at": "DATETIME"})


def ensure_analyst_decisions_ai_proposal_columns() -> None:
    _add_columns_if_missing(
        "analyst_decisions", {"ai_proposed": "BOOLEAN DEFAULT 0", "ai_proposal_id": "INTEGER"},
    )


def ensure_automation_policy_auto_triggered_column() -> None:
    _add_columns_if_missing("automation_policy_versions", {"auto_triggered": "BOOLEAN DEFAULT 0"})


def ensure_automation_policy_legacy_columns_dropped() -> None:
    _drop_columns_if_present(
        "automation_policy_versions", ["green_threshold", "red_threshold", "block_threshold"],
    )


def ensure_case_assigned_to_dropped() -> None:
    _drop_columns_if_present("cases", ["assigned_to"])


def ensure_rule_hit_score_impact_dropped() -> None:
    _drop_columns_if_present("rule_hits", ["score_impact"])


_SHAP_DIRECTION_TRANSLATIONS = {
    "artıran": "increasing",
    "azaltan": "decreasing",
}


def ensure_shap_explanation_direction_backfill() -> None:
    inspector = inspect(engine)
    if "shap_explanations" not in inspector.get_table_names():
        return

    with engine.begin() as conn:
        for legacy, current in _SHAP_DIRECTION_TRANSLATIONS.items():
            conn.execute(
                text("UPDATE shap_explanations SET direction = :current WHERE direction = :legacy"),
                {"current": current, "legacy": legacy},
            )

        known = set(_SHAP_DIRECTION_TRANSLATIONS.values())
        unexpected = conn.execute(
            text(
                "SELECT DISTINCT direction FROM shap_explanations "
                "WHERE direction NOT IN :known"
            ).bindparams(bindparam("known", expanding=True)),
            {"known": list(known)},
        ).scalars().all()
        if unexpected:
            logger.warning(
                "shap_explanations.direction backfill: %d unexpected value(s) left untouched: %r",
                len(unexpected), unexpected,
            )


_REASON_CODE_RENAMES = {
    "phase9_seed_decision": "seed_cluster_decision",
    "adim6_live_decision": "precedent_learning_test",
    "adim6_redecision_test": "precedent_redecision_test",
    "adim7_live_decision": "agreement_capture_test",
    "adim8_e2e_proof": "end_to_end_proof",
    "adim8_cold_start_test": "cold_start_test",
    "adim10_shadow_test": "automation_shadow_test",
    "adim10_latency_test": "automation_latency_test",
    "adim10_reject_then_own_decision": "automation_reject_test",
}


def ensure_reason_code_rename() -> None:
    inspector = inspect(engine)
    if "analyst_decisions" not in inspector.get_table_names():
        return

    with engine.begin() as conn:
        for old, new in _REASON_CODE_RENAMES.items():
            conn.execute(
                text("UPDATE analyst_decisions SET analyst_reason_code = :new WHERE analyst_reason_code = :old"),
                {"new": new, "old": old},
            )
