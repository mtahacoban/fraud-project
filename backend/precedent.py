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

# Fraud only ever occurs in TRANSFER/CASH_OUT, and GREEN transactions never
# open a Case (see scoring.py / _write_score) — so in practice every decided
# case is RED or GRAY. GREEN is still included in the one-hot for a stable,
# future-proof vector shape; it is currently always 0 and contributes no
# variance to the scaler (harmless — StandardScaler leaves zero-variance
# columns unscaled rather than dividing by zero).
RISK_BANDS = ["RED", "GRAY", "GREEN"]

SCALER_PATH = "models/precedent_scaler.pkl"


def _raw_case_vector(txn: m.Transaction, risk_band: str) -> np.ndarray:
    """Unscaled vector: ML_FEATURES (the same 6 features the XGBoost model
    and SHAP use — see explain.py) + risk-band one-hot. Fixed order and
    dimension for every case, decided or not, so it can be reused unchanged
    both to backfill precedent_index and to vectorize a brand-new case at
    query time."""
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
    """None means "no scaler fit yet" — callers must treat this the same as
    an empty precedent pool (cold start), not as an error."""
    if not os.path.exists(SCALER_PATH):
        return None
    return joblib.load(SCALER_PATH)


def fit_and_save_scaler(raw_vectors: np.ndarray) -> StandardScaler:
    """Fit once, on the decided-case backfill population, and persist —
    every later vectorization (new cases at query time, cases added to the
    index after a fresh decision) reuses these exact parameters. Refitting
    per-query would let the scaling drift out from under existing precedent
    vectors and silently invalidate their stored (already-scaled) values."""
    scaler = StandardScaler()
    scaler.fit(raw_vectors)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def build_case_vector(case: m.Case, db: Session, scaler: StandardScaler) -> np.ndarray:
    """Scaled, fixed-length feature vector for a case. Raises if the case
    has no transaction/score — callers only call this for cases that are
    known to have both (every case does, by construction of _write_score)."""
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
    """The case's current, terminal decision — None if the case was never
    decided. For a CLOSED case this is guaranteed to be a real action
    (approve_clean/confirm_fraud/escalate), never "reopened": reopening
    always flips status back to OPEN, so a case can only be CLOSED again
    after a fresh real decision (see main.py's decide_case/reopen_case)."""
    latest = (
        db.query(m.AnalystDecision)
        .filter(m.AnalystDecision.case_id == case.id)
        .order_by(m.AnalystDecision.id.desc())
        .first()
    )
    return latest.action_taken if latest is not None else None


def add_to_precedent_index(case: m.Case, db: Session, scaler: StandardScaler) -> m.PrecedentIndex | None:
    """Upserts a single decided case into precedent_index — vectorize +
    write if it isn't there yet, or update its label in place if it is.
    Returns None only if the case isn't actually decided (label is None);
    otherwise returns the (created or updated) row. Does not commit; caller
    controls the transaction.

    The update path matters for the decide -> reopen -> re-decide cycle: a
    reopened case's OLD label stays valid precedent until it's genuinely
    re-decided, at which point this must overwrite
    the existing row rather than insert a second one — one case, one
    precedent entry, always reflecting its current decision. The feature
    vector itself never needs recomputing on update: it's derived from the
    transaction, which is immutable once scored."""
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
    """Up to k nearest precedents to `case`, most similar first — each
    {case_id, similarity, analyst_decision}, similarity = 1 - cosine_distance
    (1.0 = identical direction, 0.0 = orthogonal, can go negative for
    opposite vectors, though that's not expected with these features).

    Self-exclusion: the row for `case` itself (if it happens to already be
    in the pool — e.g. querying precedents for a case that was itself used
    to seed the index) is filtered out *before* fitting the index, so it
    can never be its own neighbor and can never inflate similarity to 1.0.

    Cold start: an empty pool (nothing decided yet, or nothing left after
    self-exclusion) returns [] rather than raising — callers must treat
    that as "no precedent available", not an error.

    k is clamped to the pool size — asking for k=15 in a 3-case pool
    returns 3 neighbors, not an error."""
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

# A neighbor below this isn't "precedent" just because k-NN had to pad out
# to k results — it doesn't get a vote. Set from observed test data: the
# tight RED cluster sat at 0.88-0.91 similarity, a neighbor pulled in from
# an unrelated cluster landed at 0.53 — clearly not the same kind of case.
# Tunable; revisit once there's more precedent volume to calibrate against.
SIMILARITY_FLOOR = 0.5

MIN_PRECEDENT_COUNT = 5
MIN_AVG_SIMILARITY = 0.85
MIN_CONSENSUS_RATIO = 0.70


def summarize_precedents(neighbors: list[dict]) -> dict:
    """Deterministic-only summary of a find_precedents() result — no LLM,
    no decision-making, just arithmetic over what k-NN found.

    Similarity floor: only neighbors with similarity >= SIMILARITY_FLOOR
    count toward precedent_count / avg_similarity / decision_distribution /
    consensus_ratio. find_precedents() returns up to k neighbors regardless
    of how relevant they are (it pads out to k from the nearest available,
    even if the pool's actual similar cases run out first) — voting on a
    neighbor that isn't meaningfully similar would let a barely-related
    case quietly participate in the recommendation.

    Consensus is plurality by raw count (the most common decision's share
    of *counted* neighbors) — not similarity-weighted. Every counted vote
    is equal; a neighbor either clears the floor and gets one vote, or it's
    excluded entirely. This keeps the number auditable ("8 of 11
    confirm_fraud, 73%") instead of an opaque weighted score that's harder
    to explain to an analyst.

    suggested_decision is set ONLY if all three gates pass:
      precedent_count  >= MIN_PRECEDENT_COUNT   (enough neighbors)
      avg_similarity   >= MIN_AVG_SIMILARITY    (they're actually close)
      consensus_ratio  >= MIN_CONSENSUS_RATIO   (they mostly agree)
    Any gate failing -> suggested_decision=None, note="insufficient
    precedent — use judgment". escalate is a full third label here — a
    suggestion of "escalate" is valid output, though it can never become
    an automatic action (see automation.py)."""
    counted = [n for n in neighbors if n["similarity"] >= SIMILARITY_FLOOR]
    precedent_count = len(counted)

    insufficient = {
        "precedent_count": precedent_count,
        "avg_similarity": None,
        "decision_distribution": {label: 0 for label in DECISION_LABELS},
        "consensus_ratio": None,
        "suggested_decision": None,
        "note": "insufficient precedent — use judgment",
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

    return {
        "precedent_count": precedent_count,
        "avg_similarity": round(avg_similarity, 4),
        "decision_distribution": distribution,
        "consensus_ratio": round(consensus_ratio, 4),
        "suggested_decision": top_label if gates_passed else None,
        "note": None if gates_passed else "insufficient precedent — use judgment",
    }


# Deliberately its own prompt, not llm_service's SYSTEM_PROMPT — that one
# talks about SHAP factors and rule hits, which don't exist in this
# context and would invite the model to hallucinate them. Reuses
# llm_service's call layer (same Groq client, async safety,
# fallback/timeout/key handling); only the instruction text and the
# "findings" content differ.
PRECEDENT_SYSTEM_PROMPT = (
    "You are a fraud investigation assistant helping an analyst understand "
    "how similar past cases were decided. Using ONLY the precedent data "
    "provided below — do not invent any numbers or facts, and do not add "
    "information that isn't there — write a 2-3 sentence explanation of "
    "which past cases this transaction resembles and how analysts decided "
    "them. You explain the precedent, you never make or recommend a "
    "decision yourself: the suggested decision, if any, is already "
    "determined by the precedent statistics, not by you."
)


def build_precedent_findings(summary: dict) -> list[str]:
    """Deterministic, human-readable sentences from summarize_precedents()'s
    output — no LLM. This list is itself a valid explanation on its own
    (used verbatim as the fallback/no-LLM-call text) and is what the LLM is
    restricted to rephrasing, exactly like findings.build_findings()."""
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

    if summary["suggested_decision"] is not None:
        lines.append(
            f"Consensus: {summary['consensus_ratio'] * 100:.0f}% "
            f"{summary['suggested_decision']} — meets the confidence threshold "
            f"(>= {MIN_PRECEDENT_COUNT} precedents, >= {MIN_AVG_SIMILARITY:.2f} avg "
            f"similarity, >= {MIN_CONSENSUS_RATIO:.0%} consensus)."
        )
    else:
        lines.append(f"{summary['note']} (does not meet the confidence threshold for a suggestion).")

    return lines


def explain_precedents(summary: dict, txn: m.Transaction) -> dict:
    """Provider-abstract precedent explanation. Returns {"text": str,
    "source": "<provider>" | "fallback"} — same contract as
    llm_service.generate_report(), which this calls directly for the actual
    LLM work (no separate integration).

    No suggestion -> no LLM call. If summarize_precedents() didn't clear
    its confidence gates (including the true cold-start case of 0
    precedents), there's nothing resembling a pattern to explain — calling
    the LLM would just be cost for no benefit. The deterministic
    "insufficient precedent" sentence is returned directly, tagged
    source="fallback" for a consistent contract with every other report in
    this system."""
    findings = build_precedent_findings(summary)

    if summary["suggested_decision"] is None:
        return {"text": " ".join(findings), "source": "fallback"}

    return llm_service.generate_report(findings, txn_summary_text(txn), system_prompt=PRECEDENT_SYSTEM_PROMPT)


# Agreement logging. Measurement only: no automation, no action is ever
# taken from this. It exists so the automation gate (automation.py) can be
# scoped with real evidence instead of a guess about how often AI #2 and
# analysts agree.
AGREEMENT_CATEGORIES = ("agree", "disagree", "analyst_escalated", "analyst_decisive", "no_suggestion")


def classify_agreement(suggested: str | None, actual: str) -> str:
    """Four-way classification, not a flat match/no-match — escalate isn't
    a fraud/clean verdict, it's "send to a human", so treating a
    suggested="confirm_fraud" vs. actual="escalate" pair the same as a
    suggested="confirm_fraud" vs. actual="approve_clean" pair (a direct,
    opposite-verdict contradiction) would blur two very different signals:
      agree             — suggested == actual
      analyst_escalated — analyst chose escalate, AI #2 suggested a verdict
                           (analyst was more cautious than the precedent data)
      analyst_decisive  — AI #2 suggested escalate, analyst gave a verdict
                           (analyst had context/confidence beyond precedent)
      disagree          — both are verdicts (confirm_fraud / approve_clean)
                           and they're opposite — the real contradiction
      no_suggestion     — AI #2 had nothing to say (excluded from the rate,
                           not counted as either agreeing or disagreeing)
    """
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
    """Aggregates AI #2 vs. analyst agreement over every CLOSED case.

    retrospective=False (default): uses each decision's *recorded*
    ai2_suggested_decision (captured live, at decision time, in
    decide_case()). This is the methodologically honest number — what the
    analyst could actually have seen — but is NULL, and therefore excluded
    entirely, for every decision made before this column existed (there
    was nothing to record yet). It will be empty/small until enough new
    decisions accumulate.

    retrospective=True: ignores the recorded column and instead
    recomputes find_precedents()/summarize_precedents() for every CLOSED
    case against TODAY's full precedent pool (self-excluding each case from
    its own comparison). This covers historical decisions the live capture
    never saw, including the seed batch — but it is NOT "what the analyst
    was shown": the pool has grown since, so it answers a different
    question ("does AI #2's current judgment agree with past human calls?")
    rather than "did analysts follow the suggestion they saw". Requires a
    fitted scaler; raises if retrospective=True and scaler is None.

    Returns {"n": int, "agreement_rate": float | None, "counts": {category: n}}.
    agreement_rate is agree / (agree+disagree+analyst_escalated+analyst_decisive)
    — no_suggestion cases have nothing to agree or disagree with, so they're
    excluded from the rate's denominator entirely, not counted as failures."""
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
