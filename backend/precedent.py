from __future__ import annotations

import os

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from backend import db_models as m
from backend import llm_service
from backend.findings import txn_summary_text

RISK_BANDS = ["RED", "GRAY", "GREEN"]

SCALER_PATH = "models/precedent_scaler.pkl"


def _raw_case_vector(txn: m.Transaction, risk_band: str) -> np.ndarray:
    ml_values = [
        txn.amount,
        txn.step_hour,
        txn.error_balance_orig,
        txn.error_balance_dest,
        txn.is_transfer,
        txn.is_cashout,
    ]
    band_one_hot = [1.0 if risk_band == b else 0.0 for b in RISK_BANDS]
    return np.array(ml_values + band_one_hot, dtype=float)


def load_precedent_scaler() -> StandardScaler | None:
    if not os.path.exists(SCALER_PATH):
        return None
    return joblib.load(SCALER_PATH)


def fit_and_save_scaler(raw_vectors: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(raw_vectors)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def build_case_vector(case: m.Case, db: Session, scaler: StandardScaler) -> np.ndarray:
    txn = db.get(m.Transaction, case.transaction_id)
    score = (
        db.query(m.Score)
        .filter(m.Score.transaction_id == case.transaction_id)
        .order_by(m.Score.id.desc())
        .first()
    )
    raw = _raw_case_vector(txn, score.risk_band)
    return scaler.transform(raw.reshape(1, -1))[0]


def latest_decision_label(case: m.Case, db: Session) -> str | None:
    latest = (
        db.query(m.AnalystDecision)
        .filter(m.AnalystDecision.case_id == case.id)
        .order_by(m.AnalystDecision.id.desc())
        .first()
    )
    return latest.action_taken if latest is not None else None


def add_to_precedent_index(case: m.Case, db: Session, scaler: StandardScaler) -> m.PrecedentIndex | None:
    label = latest_decision_label(case, db)
    if label is None:
        return None

    existing = db.query(m.PrecedentIndex).filter(m.PrecedentIndex.case_id == case.id).first()
    if existing is not None:
        if existing.label != label:
            existing.label = label
        return existing

    vector = build_case_vector(case, db, scaler)
    row = m.PrecedentIndex(
        transaction_id=case.transaction_id,
        case_id=case.id,
        feature_vector=vector.tolist(),
        label=label,
    )
    db.add(row)
    return row


DEFAULT_K = 15


def find_precedents(case: m.Case, db: Session, scaler: StandardScaler, k: int = DEFAULT_K) -> list[dict]:
    rows = db.query(m.PrecedentIndex).filter(m.PrecedentIndex.case_id != case.id).all()
    if not rows:
        return []

    vectors = np.array([r.feature_vector for r in rows])
    n_neighbors = min(k, len(rows))

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(vectors)

    query_vector = build_case_vector(case, db, scaler).reshape(1, -1)
    distances, indices = nn.kneighbors(query_vector)

    return [
        {
            "case_id": rows[idx].case_id,
            "similarity": 1.0 - float(dist),
            "analyst_decision": rows[idx].label,
        }
        for dist, idx in zip(distances[0], indices[0])
    ]


DECISION_LABELS = ("confirm_fraud", "approve_clean", "escalate")

SIMILARITY_FLOOR = 0.5

MIN_PRECEDENT_COUNT = 5
MIN_AVG_SIMILARITY = 0.85
MIN_CONSENSUS_RATIO = 0.70

KNOWN_AML_REASON_CODES = frozenset({
    "account_takeover", "unauthorized_transaction", "identity_theft",
    "phishing_victim", "card_fraud", "social_engineering",
    "money_laundering", "structuring", "mule_account", "suspicious_pattern",
    "sanctions_concern", "terrorist_financing",
    "verified_legitimate", "false_positive", "customer_confirmed",
    "insufficient_evidence", "escalated_to_compliance", "pending_investigation",
})


def summarize_precedents(neighbors: list[dict]) -> dict:
    counted = [n for n in neighbors if n["similarity"] >= SIMILARITY_FLOOR]
    precedent_count = len(counted)

    insufficient = {
        "precedent_count": precedent_count,
        "avg_similarity": None,
        "decision_distribution": {label: 0 for label in DECISION_LABELS},
        "consensus_ratio": None,
        "suggested_decision": None,
        "note": "insufficient precedent - use judgment",
        "common_patterns": [],
        "common_reason_codes": [],
    }
    if precedent_count == 0:
        return insufficient

    avg_similarity = sum(n["similarity"] for n in counted) / precedent_count

    distribution = {label: 0 for label in DECISION_LABELS}
    for n in counted:
        distribution[n["analyst_decision"]] = distribution.get(n["analyst_decision"], 0) + 1

    top_label, top_count = max(distribution.items(), key=lambda kv: kv[1])
    consensus_ratio = top_count / precedent_count

    gates_passed = (
        precedent_count >= MIN_PRECEDENT_COUNT
        and avg_similarity >= MIN_AVG_SIMILARITY
        and consensus_ratio >= MIN_CONSENSUS_RATIO
    )

    rule_counts: dict[str, int] = {}
    for n in counted:
        for rule in set(n.get("rule_hits") or []):
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
    common_patterns = [
        {"rule": rule, "count": count, "total": precedent_count}
        for rule, count in sorted(rule_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ][:2]

    reason_code_counts: dict[str, int] = {}
    for n in counted:
        code = n.get("analyst_reason_code")
        if code in KNOWN_AML_REASON_CODES:
            reason_code_counts[code] = reason_code_counts.get(code, 0) + 1
    common_reason_codes = [
        {"rule": code, "count": count, "total": precedent_count}
        for code, count in sorted(reason_code_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ][:2]

    return {
        "precedent_count": precedent_count,
        "avg_similarity": round(avg_similarity, 4),
        "decision_distribution": distribution,
        "consensus_ratio": round(consensus_ratio, 4),
        "suggested_decision": top_label if gates_passed else None,
        "note": None if gates_passed else "insufficient precedent - use judgment",
        "common_patterns": common_patterns,
        "common_reason_codes": common_reason_codes,
    }


PRECEDENT_SYSTEM_PROMPT = (
    "You are a fraud investigation assistant helping an analyst understand "
    "how similar past cases were decided. Using ONLY the precedent data "
    "provided below - do not invent any numbers or facts, and do not add "
    "information that isn't there - write a 2-3 sentence explanation of "
    "which past cases this transaction resembles and how analysts decided "
    "them. You explain the precedent, you never make or recommend a "
    "decision yourself: the suggested decision, if any, is already "
    "determined by the precedent statistics, not by you."
)


def build_precedent_findings(summary: dict) -> list[str]:
    if summary["precedent_count"] == 0:
        return ["No similar past cases were found in the precedent history."]

    count = summary["precedent_count"]
    lines = [
        f"{count} similar past case{'s' if count != 1 else ''} found, "
        f"average similarity {summary['avg_similarity']:.2f}.",
    ]

    dist_text = ", ".join(
        f"{n} {label}" for label, n in summary["decision_distribution"].items() if n > 0
    )
    lines.append(f"Decision breakdown: {dist_text}.")

    if summary.get("common_patterns"):
        patterns_text = ", ".join(p["rule"] for p in summary["common_patterns"])
        counts_text = "; ".join(f"{p['rule']} in {p['count']}/{p['total']}" for p in summary["common_patterns"])
        lines.append(f"Most share: {patterns_text} ({counts_text}).")

    if summary.get("common_reason_codes"):
        codes_text = ", ".join(f"{c['count']} {c['rule']}" for c in summary["common_reason_codes"])
        lines.append(f"Reason codes: {codes_text}.")

    if summary["suggested_decision"] is not None:
        lines.append(
            f"Consensus: {summary['consensus_ratio'] * 100:.0f}% "
            f"{summary['suggested_decision']} - meets the confidence threshold "
            f"(>= {MIN_PRECEDENT_COUNT} precedents, >= {MIN_AVG_SIMILARITY:.2f} avg "
            f"similarity, >= {MIN_CONSENSUS_RATIO:.0%} consensus)."
        )
    else:
        lines.append(f"{summary['note']} (does not meet the confidence threshold for a suggestion).")

    return lines


def explain_precedents(summary: dict, txn: m.Transaction) -> dict:
    findings = build_precedent_findings(summary)

    if summary["suggested_decision"] is None:
        return {"text": " ".join(findings), "source": "fallback"}

    return llm_service.generate_report(findings, txn_summary_text(txn), system_prompt=PRECEDENT_SYSTEM_PROMPT)


AGREEMENT_CATEGORIES = ("agree", "disagree", "analyst_escalated", "analyst_decisive", "no_suggestion")


def classify_agreement(suggested: str | None, actual: str) -> str:
    if suggested is None:
        return "no_suggestion"
    if suggested == actual:
        return "agree"
    if actual == "escalate":
        return "analyst_escalated"
    if suggested == "escalate":
        return "analyst_decisive"
    return "disagree"


def compute_agreement_stats(db: Session, *, retrospective: bool = False, scaler: StandardScaler | None = None) -> dict:
    counts = {cat: 0 for cat in AGREEMENT_CATEGORIES}
    closed_cases = db.query(m.Case).filter(m.Case.status == "CLOSED").all()

    for case in closed_cases:
        actual = latest_decision_label(case, db)
        if actual is None:
            continue

        if retrospective:
            if scaler is None:
                raise ValueError("retrospective=True requires a fitted scaler")
            neighbors = find_precedents(case, db, scaler, k=DEFAULT_K)
            suggested = summarize_precedents(neighbors)["suggested_decision"]
        else:
            latest = (
                db.query(m.AnalystDecision)
                .filter(m.AnalystDecision.case_id == case.id)
                .order_by(m.AnalystDecision.id.desc())
                .first()
            )
            suggested = latest.ai2_suggested_decision if latest is not None else None

        counts[classify_agreement(suggested, actual)] += 1

    comparable = counts["agree"] + counts["disagree"] + counts["analyst_escalated"] + counts["analyst_decisive"]
    agreement_rate = round(counts["agree"] / comparable, 4) if comparable > 0 else None

    return {"n": len(closed_cases), "agreement_rate": agreement_rate, "counts": counts}
