"""
Seeds the dashboard with a permanent demo dataset: samples PaySim's
held-out test set, scores it through the real pipeline (ScoringEngine +
rule engine + SHAP, identical to POST /score), closes a subset of cases on
PaySim's ground-truth label so precedent analysis has a real pool to work
from, and generates a real Groq report for every RED/GRAY case (open or
closed), synchronously and rate-limited.

Runs in three phases: score (bypassing the normal BackgroundTask report
path so ~380 concurrent Groq calls don't blow through the free-tier
rate limit), ground-truth close (a sized RED/GRAY subset closed through
the real decide_case() path, using PaySim's isFraud label as the decision -
tagged analyst_reason_code="seed_ground_truth_paysim", synthetic
precedent, not a real analyst decision), then a real Groq report for
every RED/GRAY case at a rate-limit-safe pace.

WARNING: the full 600-row set (open and closed) is the permanent demo
baseline. Do not run scripts/clear_demo_data.py to clean up later test
cases - it deletes every is_demo=1 row, including this whole seeded set
and its precedent pool. One-off test cases should delete their own
transaction_id/case_id directly instead.

WARNING: CASH_OUT-type RED cases will mostly keep showing "insufficient
precedent" regardless of --n-fraud - the model rarely reaches RED
confidently on CASH_OUT fraud, so that sub-population stays thin by
construction. That's the similarity threshold working as intended, not a
sizing bug; inflating --n-fraud further won't fix it.

Usage (from the project root, backend NOT required to be running):
    python scripts/seed_demo_cases.py --dry-run                  # preview only, no writes, no Groq calls
    python scripts/seed_demo_cases.py --yes                      # actually seed (clears prior is_demo=1 rows first)
    python scripts/seed_demo_cases.py --yes --n-transactions 600 --n-fraud 100 --delay 2.5

Defaults: 600 transactions, 100 fraud (~16.7%, well above PaySim's natural
~0.3% rate, for a visible RED cluster and precedent depth), 40 CLOSED-RED /
100 CLOSED-GRAY.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from backend import db_models as m
from backend import schemas as s
from backend.config import settings
from backend.database import SessionLocal
from backend.findings import build_findings, txn_summary_text
from backend.llm_service import GENERATION_CONFIG, REQUEST_TIMEOUT_SECONDS, SYSTEM_PROMPT, _build_user_content
from backend.main import _write_score, decide_case
from backend.precedent import (
    MIN_AVG_SIMILARITY, MIN_CONSENSUS_RATIO, MIN_PRECEDENT_COUNT,
    _raw_case_vector, fit_and_save_scaler, load_precedent_scaler, summarize_precedents,
)
from backend.scoring import load_scoring_engine

RANDOM_STATE = 42
MAX_GROQ_RETRIES = 5
DEFAULT_RETRY_WAIT_SECONDS = 65  # last-resort guess, only when Groq gives no usable reset hint at all

# Measured via a direct probe (see conversation/commit history): this
# model's real Groq free-tier ceiling is x-ratelimit-limit-tokens=8000/min
# - NOT the 30 req/min figure this script was originally paced against.
# At ~550-600 tokens/call (system prompt + findings + up to 800 output
# tokens), 2.5s spacing (~24 calls/min) overshoots the token budget by
# roughly 2x, which means most calls 429 and pay a retry penalty - a
# self-inflicted slowdown far worse than pacing correctly from the start.
# 6.5s (~9.2 calls/min, ~5300-5500 tokens/min) sits comfortably under the
# measured ceiling with margin for longer-than-average findings lists.
DEFAULT_DELAY_SECONDS = 6.5

# Sized from a real precedent.py similarity-gate measurement using the
# scaler decide_case() actually uses (fit only on the closed pool, not the
# full sample - see the module docstring above for why that distinction
# mattered). At these sizes, measured pass rates for a real analyst
# opening an OPEN case are ~67% RED (~80% for the dominant TRANSFER
# sub-type, ~0% for the structurally-thin CASH_OUT sub-type - expected,
# see the WARNING above) and ~84% GRAY. Re-verify with --dry-run after
# changing any of these three.
DEFAULT_N_FRAUD = 100
N_CLOSED_RED = 40
N_CLOSED_GRAY = 100

ANALYST_REASON_CODE = "seed_ground_truth_paysim"


def to_payload(row: pd.Series) -> dict:
    return {
        "step": int(row["step"]),
        "type": row["type"],
        "amount": float(row["amount"]),
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
        "nameOrig": str(row["nameOrig"]),
        "nameDest": str(row["nameDest"]),
        "device_id": str(row["device_id"]),
        "is_known_device": int(row["is_known_device"]),
        "login_country": str(row["login_country"]),
        "geo_velocity_flag": int(row["geo_velocity_flag"]),
        "channel": str(row["channel"]),
        "isFraud": int(row["isFraud"]),
    }


def sample_transactions(n_transactions: int, n_fraud: int) -> pd.DataFrame:
    df = pd.read_parquet("data/test.parquet")
    fraud = df[df["isFraud"] == 1]
    clean = df[df["isFraud"] == 0]
    n_clean = n_transactions - n_fraud

    fraud_sample = fraud.sample(n=min(n_fraud, len(fraud)), random_state=RANDOM_STATE)
    clean_sample = clean.sample(n=min(n_clean, len(clean)), random_state=RANDOM_STATE)

    return (
        pd.concat([fraud_sample, clean_sample])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def score_sample(sample: pd.DataFrame, engine) -> list[dict]:
    """Scores every row through the real engine, in sample order (fixed by
    RANDOM_STATE). Pure computation - no DB writes. Shared by --dry-run
    (preview) and Phase 1 (persist), so both see identical band
    assignments and identical RED/GRAY encounter order - which is exactly
    the order Phase 2 uses to pick the CLOSED subset, so a --dry-run
    preview and a real --yes run select the same rows."""
    scored = []
    for _, row in sample.iterrows():
        payload = to_payload(row)
        result = engine.score(payload)
        scored.append({"row": row, "payload": payload, "result": result,
                        "band": result["risk_band"], "is_fraud": int(row["isFraud"])})
    return scored


def _fake_txn_for_vector(result: dict) -> object:
    class _FakeTxn:
        pass
    t = _FakeTxn()
    t.amount = result["_amount"]
    t.step_hour = result["step_hour"]
    t.error_balance_orig = result["errorBalanceOrig"]
    t.error_balance_dest = result["errorBalanceDest"]
    t.is_transfer = result["is_transfer"]
    t.is_cashout = result["is_cashout"]
    return t


def simulate_precedent_gates(scored: list[dict], n_closed_red: int, n_closed_gray: int) -> dict:
    """Dry-run-only precedent feasibility check, using the SAME selection
    order Phase 2 will use (first n_closed_red RED-encountered, first
    n_closed_gray GRAY-encountered) and the real precedent.py vector +
    gate logic (_raw_case_vector, summarize_precedents, the real MIN_*
    constants) - just without touching the DB or an on-disk scaler.

    Crucially, the scaler is fit ONLY on the closed pool (mirroring
    backfill_precedents.py's actual "fit on the decided-case backfill
    population" behavior - decide_case() never refits, it only reuses
    whatever's on disk), not on the full sample. A scaler fit on the full
    RED+GRAY pool is a materially different (and misleadingly easier)
    test: RED is rare pool-wide, so its one-hot dimension gets a huge
    standardized magnitude that dominates cosine similarity between any
    two RED cases regardless of their other features - that was this
    design's first (too optimistic) probe; this is the faithful one.

    Tests EVERY OPEN case of each band, not just one - a single sample
    proved noisy in practice (RED splits into TRANSFER/CASH_OUT
    sub-populations with very different feature profiles; which specific
    case gets tested changes the single-case verdict from run to run).
    Reports a pass rate and breaks failures down by transaction type."""
    for entry in scored:
        entry["result"]["_amount"] = entry["payload"]["amount"]

    red = [e for e in scored if e["band"] == "RED"]
    gray = [e for e in scored if e["band"] == "GRAY"]

    closed_red = red[:n_closed_red]
    closed_gray = gray[:n_closed_gray]
    open_red = red[n_closed_red:]
    open_gray = gray[n_closed_gray:]

    closed_pool = closed_red + closed_gray
    for e in closed_pool:
        e["label"] = "confirm_fraud" if e["is_fraud"] else "approve_clean"

    if not closed_pool:
        return {"error": "no closed pool - nothing to test"}

    raw_vectors = np.array([
        _raw_case_vector(_fake_txn_for_vector(e["result"]), e["band"]) for e in closed_pool
    ])
    scaler = StandardScaler()
    scaler.fit(raw_vectors)
    for e, raw in zip(closed_pool, raw_vectors):
        e["scaled_vector"] = scaler.transform(raw.reshape(1, -1))[0]

    vectors = np.array([e["scaled_vector"] for e in closed_pool])
    n_neighbors = min(15, len(closed_pool))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(vectors)

    def query_gate(query_entry):
        raw = _raw_case_vector(_fake_txn_for_vector(query_entry["result"]), query_entry["band"])
        query_vec = scaler.transform(raw.reshape(1, -1))[0]
        distances, indices = nn.kneighbors(query_vec.reshape(1, -1))
        neighbors = [
            {"similarity": 1.0 - float(dist), "analyst_decision": closed_pool[idx]["label"]}
            for dist, idx in zip(distances[0], indices[0])
        ]
        return summarize_precedents(neighbors)

    def band_report(open_cases):
        if not open_cases:
            return {"n": 0, "passed": 0, "by_type": {}}
        by_type: dict[str, dict] = {}
        passed = 0
        for e in open_cases:
            summary = query_gate(e)
            ok = summary["suggested_decision"] is not None
            passed += int(ok)
            t = e["payload"]["type"]
            by_type.setdefault(t, {"n": 0, "passed": 0})
            by_type[t]["n"] += 1
            by_type[t]["passed"] += int(ok)
        return {"n": len(open_cases), "passed": passed, "by_type": by_type}

    return {
        "closed_red": len(closed_red), "closed_gray": len(closed_gray),
        "open_red": len(open_red), "open_gray": len(open_gray),
        "red_report": band_report(open_red),
        "gray_report": band_report(open_gray),
    }


def clear_prior_demo_rows(db) -> tuple[int, int]:
    """Same dependency-ordered delete as scripts/clear_demo_data.py, scoped
    to is_demo=1 (i.e. every row this script or any prior demo seed ever
    wrote - including the synthetic precedent_index/analyst_decisions rows
    Phase 2 creates), run inline so this script is fully idempotent on its
    own."""
    demo_txn_ids = [r[0] for r in db.query(m.Transaction.id).filter(m.Transaction.is_demo == True).all()]  # noqa: E712
    if not demo_txn_ids:
        return 0, 0

    demo_case_ids = [
        r[0] for r in db.query(m.Case.id).filter(m.Case.transaction_id.in_(demo_txn_ids)).all()
    ]

    db.query(m.PrecedentIndex).filter(m.PrecedentIndex.case_id.in_(demo_case_ids)).delete(synchronize_session=False)
    db.query(m.AnalystDecision).filter(m.AnalystDecision.case_id.in_(demo_case_ids)).delete(synchronize_session=False)
    db.query(m.LlmReport).filter(m.LlmReport.case_id.in_(demo_case_ids)).delete(synchronize_session=False)
    db.query(m.Case).filter(m.Case.id.in_(demo_case_ids)).delete(synchronize_session=False)
    db.query(m.ShapExplanation).filter(m.ShapExplanation.transaction_id.in_(demo_txn_ids)).delete(synchronize_session=False)
    db.query(m.RuleHit).filter(m.RuleHit.transaction_id.in_(demo_txn_ids)).delete(synchronize_session=False)
    db.query(m.Score).filter(m.Score.transaction_id.in_(demo_txn_ids)).delete(synchronize_session=False)
    db.query(m.Transaction).filter(m.Transaction.id.in_(demo_txn_ids)).delete(synchronize_session=False)
    db.commit()
    return len(demo_txn_ids), len(demo_case_ids)


def phase1_score_and_write(db, sample: pd.DataFrame, engine) -> list[dict]:
    """Scores every sampled row through the real engine and writes it
    through the exact same _write_score() path /score and /simulation/run
    use - background_tasks=None is the only difference, and it is what
    keeps this from ever scheduling the async (unpaced) Groq path.
    Returns per-case bookkeeping (case_id, band, is_fraud) in encounter
    order, which Phase 2 uses to pick the CLOSED subset."""
    case_infos: list[dict] = []
    for i, row in sample.iterrows():
        txn_in = s.TransactionIn(**to_payload(row))
        result = engine.score(txn_in.model_dump())
        _txn_row, case_row = _write_score(
            db, txn_in, result, source="demo", is_demo=True, background_tasks=None,
        )
        db.commit()
        if case_row is not None:
            case_infos.append({
                "case_id": case_row.id, "band": result["risk_band"], "is_fraud": int(row["isFraud"]),
            })
        if (i + 1) % 100 == 0:
            print(f"  Phase 1: {i + 1}/{len(sample)} transactions scored ({len(case_infos)} cases opened so far)")
    return case_infos


def phase2_close_for_precedent(db, case_infos: list[dict], n_closed_red: int, n_closed_gray: int) -> set[int]:
    """Closes the first n_closed_red RED cases and first n_closed_gray GRAY
    cases (encounter order - same order simulate_precedent_gates() used in
    --dry-run) on PaySim's ground-truth label, via the real decide_case()
    endpoint function called in-process. RED first, then GRAY, so each
    band's precedent pool is warm before later same-band closes query it."""
    red = [c for c in case_infos if c["band"] == "RED"]
    gray = [c for c in case_infos if c["band"] == "GRAY"]
    to_close = red[:n_closed_red] + gray[:n_closed_gray]

    closed_ids: set[int] = set()
    for info in to_close:
        action = "confirm_fraud" if info["is_fraud"] else "approve_clean"
        decision_in = s.DecisionIn(action_taken=action, analyst_reason_code=ANALYST_REASON_CODE)
        decide_case(case_id=info["case_id"], decision=decision_in, db=db)
        closed_ids.add(info["case_id"])

    print(f"  Phase 2: closed {len(to_close)} cases on ground truth "
          f"({len(red[:n_closed_red])} RED/confirm_fraud, {len(gray[:n_closed_gray])} GRAY/approve_clean)")
    return closed_ids


def _parse_groq_duration(value: str) -> float | None:
    """Parses Groq's own human-readable rate-limit-reset duration strings
    (e.g. '5h8m9.6s', '630ms', '12.4s') into seconds - a format the
    standard HTTP Retry-After header never uses (plain integer seconds),
    so a naive float() on one of these silently fails and falls through
    to the flat default wait. Confirmed by direct probe: this model's
    real constraint is x-ratelimit-limit-tokens=8000/min (a *far* tighter
    ceiling than the 1000/window request bucket), so its own reset-tokens
    hint is a much better wait estimate than a flat guess."""
    matches = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", value or "")
    if not matches:
        return None
    total = 0.0
    for num, unit in matches:
        n = float(num)
        total += n / 1000 if unit == "ms" else n * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    value = response.headers.get("retry-after")
    if value is not None:
        try:
            return float(value)
        except ValueError:
            pass
    # Fall back to Groq's own per-minute TOKEN bucket reset hint (the
    # actual binding constraint here) rather than a flat guess - but cap
    # it: x-ratelimit-reset-requests reflects a much longer (~daily)
    # window and must never be used for a single retry's sleep, and this
    # field can occasionally be stale/odd right after a burst.
    reset_tokens = response.headers.get("x-ratelimit-reset-tokens")
    parsed = _parse_groq_duration(reset_tokens) if reset_tokens else None
    return min(parsed + 1.0, 90.0) if parsed is not None else None


def _call_groq_sync(client, model: str, findings: list[str], txn_summary: str) -> str:
    """Same request shape as llm_service._generate_groq(), but never
    swallows an error into a silent fallback: transient errors are
    retried with backoff, and everything else propagates to the caller,
    which stops the script instead of writing a fake report."""
    from groq import APIConnectionError, APITimeoutError, RateLimitError

    user_content = _build_user_content(findings, txn_summary)
    attempt = 0
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=GENERATION_CONFIG["temperature"],
                max_tokens=GENERATION_CONFIG["max_output_tokens"],
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            text = (completion.choices[0].message.content or "").strip()
            if not text:
                finish_reason = completion.choices[0].finish_reason
                raise RuntimeError(f"Groq returned empty content (finish_reason={finish_reason})")
            return text
        except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
            attempt += 1
            if attempt > MAX_GROQ_RETRIES:
                raise
            wait = _retry_after_seconds(exc) or DEFAULT_RETRY_WAIT_SECONDS
            print(f"    [retry {attempt}/{MAX_GROQ_RETRIES}] {type(exc).__name__} - waiting {wait:.0f}s before retrying...")
            time.sleep(wait)


def phase3_generate_reports(db, case_ids: list[int], delay: float) -> None:
    """Every RED/GRAY case (closed or open) gets a real report - closing a
    case in Phase 2 does not exempt it."""
    from groq import Groq

    if not settings.llm_api_key or not settings.llm_model:
        print("FATAL: LLM_API_KEY or LLM_MODEL is not set in .env - cannot generate real Groq reports.")
        print("Nothing was written in Phase 3 (Phase 1/2's transactions/cases/decisions from this run remain committed).")
        sys.exit(1)

    client = Groq(api_key=settings.llm_api_key)
    model = settings.llm_model
    total = len(case_ids)
    t0 = time.time()

    for i, case_id in enumerate(case_ids, 1):
        case = db.get(m.Case, case_id)
        txn = db.get(m.Transaction, case.transaction_id)
        findings = build_findings(case, db)
        txn_summary = txn_summary_text(txn)

        try:
            text = _call_groq_sync(client, model, findings, txn_summary)
        except Exception as exc:
            print(f"\nFATAL: Groq report generation permanently failed on case {case_id} "
                  f"({type(exc).__name__}). Stopping - no fallback report was written for this "
                  f"or any remaining case.")
            print(f"  ({i - 1}/{total} cases already have real Groq reports and are safe as-is; "
                  f"run with --resume --yes to retry only what's missing, without losing them.)")
            sys.exit(1)

        db.add(m.LlmReport(case_id=case_id, report_text=text, model_name=model, source="groq"))
        db.commit()

        if i % 20 == 0 or i == total:
            elapsed = time.time() - t0
            remaining = (total - i) * delay
            print(f"  Phase 3: {i}/{total} cases, source=groq, {elapsed:.0f}s elapsed, "
                  f"~{remaining / 60:.1f} min remaining")

        if i < total:
            time.sleep(delay)


def find_cases_needing_report(db) -> tuple[list[int], list[int]]:
    """Returns (missing_case_ids, stale_fallback_case_ids) among every
    is_demo=1 case (RED/GRAY only - GREEN never opens a case, so every
    Case row already implies one of these two bands). 'missing' has zero
    llm_reports rows (e.g. an interrupted --yes run never reached it);
    'stale' has a source='fallback' row (e.g. the live app's own
    background report path raced the script and wrote one independently - see the WARNING in the module docstring about running the backend
    concurrently with seeding). Both need a real Groq generation."""
    demo_case_ids = [
        r[0] for r in db.query(m.Case.id)
        .join(m.Transaction, m.Transaction.id == m.Case.transaction_id)
        .filter(m.Transaction.is_demo == True)  # noqa: E712
        .all()
    ]
    missing: list[int] = []
    stale: list[int] = []
    for case_id in demo_case_ids:
        reports = db.query(m.LlmReport).filter(m.LlmReport.case_id == case_id).all()
        if not reports:
            missing.append(case_id)
        elif any(r.source == "fallback" for r in reports):
            stale.append(case_id)
    return missing, stale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-transactions", type=int, default=600)
    parser.add_argument("--n-fraud", type=int, default=DEFAULT_N_FRAUD)
    parser.add_argument("--n-closed-red", type=int, default=N_CLOSED_RED)
    parser.add_argument("--n-closed-gray", type=int, default=N_CLOSED_GRAY)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS,
                         help="seconds between Groq calls (measured free-tier ceiling: 8000 tokens/min, "
                              "not the 30 req/min this was first paced against)")
    parser.add_argument("--dry-run", action="store_true", help="preview only - no DB writes, no Groq calls")
    parser.add_argument("--yes", action="store_true", help="actually seed the database")
    parser.add_argument("--resume", action="store_true",
                         help="fill in missing/stale-fallback llm_reports for the EXISTING is_demo=1 seed - "
                              "does not re-sample, re-score, re-close, or clear anything. Use after an "
                              "interrupted --yes run so already-generated real reports aren't lost. "
                              "Combine with --yes to actually write; without it, previews what's missing.")
    args = parser.parse_args()

    do_write = args.yes and not args.dry_run

    if args.resume:
        db = SessionLocal()
        try:
            missing, stale = find_cases_needing_report(db)
            to_process = missing + stale
            print(f"Resume check (is_demo=1 seed only): {len(missing)} cases with no report, "
                  f"{len(stale)} cases with a stale fallback report that needs replacing.")
            if not to_process:
                print("Nothing to do - every is_demo=1 case already has a real Groq report.")
                return
            est_seconds = len(to_process) * args.delay
            print(f"{len(to_process)} reports to generate, {args.delay}s apart "
                  f"(~{est_seconds / 60:.1f} min estimated).")
            if not do_write:
                print("Preview only - pass --resume --yes to actually generate these.")
                return
            for case_id in stale:
                db.query(m.LlmReport).filter(
                    m.LlmReport.case_id == case_id, m.LlmReport.source == "fallback",
                ).delete(synchronize_session=False)
            db.commit()
            phase3_generate_reports(db, to_process, args.delay)
            print(f"\nResume done: {len(to_process)} reports generated "
                  f"({len(missing)} previously missing, {len(stale)} stale fallbacks replaced).")
        finally:
            db.close()
        return

    fraud_rate = args.n_fraud / args.n_transactions
    print(f"Sampling {args.n_transactions} transactions total ({args.n_fraud} fraud, "
          f"{args.n_transactions - args.n_fraud} clean = {fraud_rate:.1%} fraud rate - "
          f"deliberately far above PaySim's natural ~0.3%, for a visible RED cluster and "
          f"enough same-type CLOSED precedent depth) from data/test.parquet (RANDOM_STATE={RANDOM_STATE})...")
    sample = sample_transactions(args.n_transactions, args.n_fraud)
    engine = load_scoring_engine()

    if not do_write:
        print("\n--- DRY RUN: scoring through the real engine, writing nothing, calling nothing ---")
        scored = score_sample(sample, engine)
        bands = {"RED": 0, "GRAY": 0, "GREEN": 0}
        for e in scored:
            bands[e["band"]] += 1
        total_cases = bands["RED"] + bands["GRAY"]
        est_seconds = total_cases * args.delay

        print(f"\nRisk band distribution: {bands}")
        print(f"Cases that would open (RED+GRAY): {total_cases}")
        if bands["RED"] == 0:
            print("WARNING: 0 RED cases in this sample - increase --n-fraud and re-run --dry-run.")
        elif bands["RED"] < args.n_closed_red:
            print(f"WARNING: only {bands['RED']} RED cases available, but --n-closed-red={args.n_closed_red} "
                  f" - not enough RED supply for both the CLOSED precedent pool and an OPEN demo case. "
                  f"Increase --n-fraud.")
        else:
            print(f"OK: {bands['RED']} RED cases present (>= --n-closed-red={args.n_closed_red}).")

        print(f"\nPlanned CLOSED/OPEN split: CLOSED={args.n_closed_red + args.n_closed_gray} "
              f"({args.n_closed_red} RED/confirm_fraud + {args.n_closed_gray} GRAY/approve_clean), "
              f"OPEN={total_cases - args.n_closed_red - args.n_closed_gray}")

        print("\n--- Precedent gate simulation (real precedent.py vector/gate logic, scaler fit ONLY on the "
              "simulated closed pool - this is what decide_case() will actually use, not a full-sample scaler) ---")
        gates = simulate_precedent_gates(scored, args.n_closed_red, args.n_closed_gray)
        if "error" in gates:
            print(f"  {gates['error']}")
        else:
            print(f"  Simulated CLOSED pool: {gates['closed_red']} RED + {gates['closed_gray']} GRAY "
                  f"= {gates['closed_red'] + gates['closed_gray']} total")
            print(f"  OPEN remaining: {gates['open_red']} RED, {gates['open_gray']} GRAY "
                  f" - testing ALL of them (not just one sample), since which single case gets tested "
                  f"proved noisy in practice")
            print(f"  Gates required: precedent_count>={MIN_PRECEDENT_COUNT}, "
                  f"avg_similarity>={MIN_AVG_SIMILARITY}, consensus_ratio>={MIN_CONSENSUS_RATIO}")
            for label, report in [("OPEN RED", gates["red_report"]), ("OPEN GRAY", gates["gray_report"])]:
                if report["n"] == 0:
                    print(f"  [{label}] no open cases of this band to test")
                    continue
                rate = report["passed"] / report["n"]
                print(f"  [{label}] {report['passed']}/{report['n']} open cases get a suggestion ({rate:.0%})")
                for t, stats in sorted(report["by_type"].items()):
                    t_rate = stats["passed"] / stats["n"] if stats["n"] else 0.0
                    note = ""
                    if label == "OPEN RED" and t == "CASH_OUT" and t_rate < 0.3:
                        note = "  (expected - CASH_OUT-fraud rarely reaches RED confidently; AI #2 " \
                               "correctly declines to guess from a thin, dissimilar neighbor set)"
                    print(f"      {t}: {stats['passed']}/{stats['n']} ({t_rate:.0%}){note}")

        print(f"\nEstimated Phase 3 (Groq) duration at {args.delay}s/call: "
              f"{est_seconds:.0f}s (~{est_seconds / 60:.1f} min) for all {total_cases} RED/GRAY cases "
              f"(closed and open alike).")
        print("\nNo database rows were written. No Groq calls were made. Pass --yes to actually seed.")
        return

    db = SessionLocal()
    try:
        print("\nClearing any prior is_demo=1 rows (idempotent re-seed)...")
        cleared_txns, cleared_cases = clear_prior_demo_rows(db)
        print(f"  Cleared {cleared_txns} prior demo transactions, {cleared_cases} prior demo cases.")

        print("\n--- Phase 1: scoring + writing transactions/cases (no async report scheduling) ---")
        case_infos = phase1_score_and_write(db, sample, engine)
        red_count = sum(1 for c in case_infos if c["band"] == "RED")
        gray_count = sum(1 for c in case_infos if c["band"] == "GRAY")
        print(f"Phase 1 done: {len(sample)} transactions scored, {len(case_infos)} RED/GRAY cases opened "
              f"({red_count} RED, {gray_count} GRAY).")

        if not case_infos:
            print("No cases opened - nothing for Phase 2/3 to do.")
            return

        print(f"\n--- Phase 2: closing {args.n_closed_red} RED + {args.n_closed_gray} GRAY cases on "
              f"PaySim ground truth (source of precedent for AI #2; analyst_reason_code={ANALYST_REASON_CODE!r}) ---")
        phase2_close_for_precedent(db, case_infos, args.n_closed_red, args.n_closed_gray)

        all_case_ids = [c["case_id"] for c in case_infos]
        print(f"\n--- Phase 3: generating {len(all_case_ids)} real Groq reports for every RED/GRAY case "
              f"(closed and open alike, sequential, {args.delay}s apart, zero silent fallback) ---")
        phase3_generate_reports(db, all_case_ids, args.delay)
        print(f"\nDone: {len(all_case_ids)} cases, all with source=\"groq\" reports; "
              f"{args.n_closed_red + args.n_closed_gray} closed (precedent pool), "
              f"{len(all_case_ids) - args.n_closed_red - args.n_closed_gray} open (active queue).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
