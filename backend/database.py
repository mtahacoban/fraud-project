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
    """Adds each {column: ddl} pair to `table` via ALTER TABLE ... ADD
    COLUMN, skipping columns that already exist. No-op if the table itself
    doesn't exist yet (nothing to migrate — Base.metadata.create_all() will
    create it with the column already in db_models.py). Idempotent: safe to
    call on every startup, a no-op once every column exists. Plain ADD
    COLUMN, works on SQLite and PostgreSQL. Shared by every ensure_*_column
    migration below; each caller's column_ddls dict is the single source of
    truth for that column's exact type/default — this helper never invents
    a DDL of its own."""
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
    """Drops each named column from `table` via ALTER TABLE ... DROP
    COLUMN, skipping columns that are already gone. No-op if the table
    itself doesn't exist. Idempotent: safe to call on every startup. Plain
    DROP COLUMN — supported by SQLite >= 3.35 and any current PostgreSQL.
    Shared by every ensure_*_dropped migration below."""
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
    """Adds llm_reports.source if missing."""
    _add_columns_if_missing("llm_reports", {"source": "VARCHAR(16)"})


def ensure_precedent_index_label_width() -> None:
    """Widens precedent_index.label to VARCHAR(32) — it was declared
    VARCHAR(8), too narrow for values like "confirm_fraud" (13 chars).
    SQLite ignores declared VARCHAR length, so this was silently fine
    there, but PostgreSQL enforces it and would reject/truncate inserts.

    Must run before Base.metadata.create_all(): SQLite has no ALTER
    COLUMN, so a too-narrow existing table is dropped here and rebuilt
    fresh by create_all() right after — only safe because the table is
    verified empty first, never assumed. PostgreSQL uses ALTER COLUMN TYPE
    directly, no drop needed."""
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
                # Real data present — don't drop it, leave the column as-is
                # rather than guess at a migration.
                return
            conn.execute(text("DROP TABLE precedent_index"))
        else:
            conn.execute(text("ALTER TABLE precedent_index ALTER COLUMN label TYPE VARCHAR(32)"))


def ensure_precedent_explanation_pool_size_column() -> None:
    """Adds precedent_explanations.pool_size_at_generation if missing.
    DEFAULT 0 backfills existing rows as immediately stale (0 is always
    less than any real pool size), which is the correct self-healing
    behavior — they simply regenerate on next view."""
    _add_columns_if_missing("precedent_explanations", {"pool_size_at_generation": "INTEGER DEFAULT 0"})


def ensure_analyst_decisions_ai2_suggestion_column() -> None:
    """Adds analyst_decisions.ai2_suggested_decision — what the precedent
    engine suggested for a case at the moment it was decided (NULL = no
    suggestion existed, whether cold start or the confidence gates weren't
    met). Captured going forward in main.py's decide_case(); older
    decisions stay NULL rather than being retroactively guessed, since the
    suggestion pool's state at that time can't be reconstructed."""
    _add_columns_if_missing("analyst_decisions", {"ai2_suggested_decision": "VARCHAR(32)"})


# automation_policy_versions predates the multi-gate automation design.
# fraud_similarity_threshold/clean_similarity_threshold/hard_rule_required
# are reused from that original design (their names already fit); everything
# below is genuinely new. (Its other original columns — green_threshold,
# red_threshold, block_threshold — were dropped entirely; see
# ensure_automation_policy_legacy_columns_dropped().)
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
    """Adds every automation_policy_versions column the multi-gate policy
    design needs, one ADD COLUMN per missing column."""
    _add_columns_if_missing("automation_policy_versions", _AUTOMATION_POLICY_NEW_COLUMNS)


def ensure_default_automation_policy() -> None:
    """Seeds automation_policy_versions v1 if the table is completely
    empty — cold-start safety so a fresh deploy always has an active
    policy to read. Never touches the table if any row already exists;
    this is a one-time bootstrap, not something that overwrites deliberate
    policy changes. Starts in "shadow" mode (observe only, never propose);
    thresholds are intentionally stricter than the precedent engine's own
    suggestion gates, since automating a decision demands a higher bar
    than merely suggesting one.

    Uses the ORM (not raw SQL) so columns without an explicit value here
    (created_at, auto_triggered) get their Python-side defaults applied
    automatically. Imports db_models locally to avoid a circular import
    (db_models imports Base from this module)."""
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
                "Initial policy — starts in shadow mode (observe only, no "
                "proposals). Thresholds intentionally stricter than the "
                "precedent engine's own suggestion gates."
            ),
        ))
        db.commit()
    finally:
        db.close()


def ensure_auto_block_log_columns() -> None:
    """Adds auto_block_log.case_id and .policy_version_id if missing."""
    _add_columns_if_missing("auto_block_log", {"case_id": "INTEGER", "policy_version_id": "INTEGER"})


def ensure_auto_block_log_review_columns() -> None:
    """Adds auto_block_log.rejection_reason and .reviewed_at if missing."""
    _add_columns_if_missing("auto_block_log", {"rejection_reason": "TEXT", "reviewed_at": "DATETIME"})


def ensure_analyst_decisions_ai_proposal_columns() -> None:
    """Adds analyst_decisions.ai_proposed and .ai_proposal_id if missing."""
    _add_columns_if_missing(
        "analyst_decisions", {"ai_proposed": "BOOLEAN DEFAULT 0", "ai_proposal_id": "INTEGER"},
    )


def ensure_automation_policy_auto_triggered_column() -> None:
    """Adds automation_policy_versions.auto_triggered if missing — flags a
    policy version created automatically by the circuit breaker rather
    than a manual change."""
    _add_columns_if_missing("automation_policy_versions", {"auto_triggered": "BOOLEAN DEFAULT 0"})


def ensure_automation_policy_legacy_columns_dropped() -> None:
    """Drops automation_policy_versions.green_threshold/red_threshold/
    block_threshold — leftover columns from this table's original,
    single-threshold design, superseded entirely by the multi-gate policy
    fields above. Confirmed zero readers anywhere in the codebase and
    excluded from automation.py's _POLICY_FIELDS, so they were never even
    copied forward onto a new policy version — every row got the same
    static column default, permanently."""
    _drop_columns_if_present(
        "automation_policy_versions", ["green_threshold", "red_threshold", "block_threshold"],
    )


def ensure_case_assigned_to_dropped() -> None:
    """Drops cases.assigned_to. Confirmed zero readers/writers anywhere:
    not in any Pydantic schema (never exposed via the API), not rendered
    in the frontend, not used in any query/filter — no case-assignment
    feature exists in this system. Nullable, so no data-loss risk beyond
    the column itself (every value was already NULL — verified before
    dropping)."""
    _drop_columns_if_present("cases", ["assigned_to"])


def ensure_rule_hit_score_impact_dropped() -> None:
    """Drops rule_hits.score_impact. Confirmed always NULL in practice:
    none of the three RuleHit(...) construction sites in main.py ever set
    it, and every existing row in the live database had NULL here too
    (verified before dropping). Was exposed via RuleHitOut — a live API
    field that could only ever return null — removed from the schema at
    the same time this column was dropped."""
    _drop_columns_if_present("rule_hits", ["score_impact"])


# A one-time refactor translated explain.py's SHAP direction labels from
# Turkish ("artıran"/"azaltan") to English but never backfilled rows
# already written under the old labels, leaving the two vocabularies mixed
# in the same column.
_SHAP_DIRECTION_TRANSLATIONS = {
    "artıran": "increasing",
    "azaltan": "decreasing",
}


def ensure_shap_explanation_direction_backfill() -> None:
    """One-time data backfill: translates legacy Turkish
    shap_explanations.direction values to their English equivalents. Maps
    only the two known legacy values (exact match) — anything else is left
    untouched rather than guessed, and logged so it doesn't disappear
    silently. Safe to re-run: each UPDATE's WHERE clause only matches rows
    still in the old vocabulary, so once the backfill has run once, every
    subsequent call is a no-op."""
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


# Renames old internal-development reason codes to plain, descriptive
# labels. Exact match only — anything else is left untouched.
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
    """One-time data cleanup: renames analyst_decisions.analyst_reason_code
    values still using old internal-development labels. Exact match only;
    anything outside the map is left untouched. Safe to re-run — each
    UPDATE's WHERE clause only matches rows still carrying the old label."""
    inspector = inspect(engine)
    if "analyst_decisions" not in inspector.get_table_names():
        return

    with engine.begin() as conn:
        for old, new in _REASON_CODE_RENAMES.items():
            conn.execute(
                text("UPDATE analyst_decisions SET analyst_reason_code = :new WHERE analyst_reason_code = :old"),
                {"new": new, "old": old},
            )
