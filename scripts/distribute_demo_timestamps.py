"""
One-time migration: spreads demo (is_demo=1) transaction/case/report/
decision timestamps across the last 30 real calendar days, using PaySim's
own `step` field (simulation hour) as the ordering signal instead of a
uniform random shuffle, so day-by-day trend charts show a shape instead of
a spike on the one or two days the seeding scripts actually ran on.

NOT IDEMPOTENT - DO NOT RE-RUN. Dates are anchored to date.today() at run
time; running it twice computes a different 30-day window and re-shifts
everything again, discarding the first distribution.

Shifts: Transaction.created_at from a linear step -> day mapping plus
deterministic jitter (RANDOM_STATE=42); Score/Case.created_at to the same
instant (they're written together); LlmReport.generated_at with a
deterministic 1-9 minute offset; Case.closed_at / AnalystDecision.decided_at
(closed cases) with a deterministic 0-3 day offset; PrecedentIndex.created_at
matching decided_at (a precedent entry is only ever created at decision
time); AutoBlockLog.created_at matching either the decision or case-open
instant depending on review_status. The one real non-demo baseline row is
untouched.

Usage (from the project root, backend NOT required to be running):
    python scripts/distribute_demo_timestamps.py --dry-run   # preview only, no writes
    python scripts/distribute_demo_timestamps.py --yes       # actually shift timestamps
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend import db_models as m
from backend.database import SessionLocal

RANDOM_STATE = 42
WINDOW_DAYS = 30


def build_plan(transactions: list[m.Transaction], today: date, rng: random.Random) -> list[dict]:
    """Pure computation, no DB access - same function drives both --dry-run
    (preview) and --yes (apply), so what's previewed is exactly what gets
    written. Iterates transactions in a fixed order (caller passes them
    sorted by id) and draws from `rng` in a fixed sequence per transaction,
    so --dry-run and --yes produce byte-identical timestamps given the
    same RANDOM_STATE."""
    steps = [t.step for t in transactions]
    step_min, step_max = min(steps), max(steps)
    step_range = (step_max - step_min) or 1  # guard: a single-step dataset would divide by zero

    plan = []
    for txn in transactions:
        day_offset = round((txn.step - step_min) / step_range * (WINDOW_DAYS - 1))
        day = today - timedelta(days=(WINDOW_DAYS - 1 - day_offset))
        hour = txn.step % 24
        minute = rng.randint(0, 59)
        second = rng.randint(0, 59)
        new_created_at = datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=timezone.utc)

        llm_offset = timedelta(minutes=rng.randint(1, 9), seconds=rng.randint(0, 59))
        decision_offset = timedelta(days=rng.randint(0, 3), hours=rng.randint(0, 23), minutes=rng.randint(0, 59))

        plan.append({
            "transaction_id": txn.id,
            "step": txn.step,
            "day_offset": day_offset,
            "new_created_at": new_created_at,
            "llm_offset": llm_offset,
            "decision_offset": decision_offset,
        })
    return plan


def apply_plan(db, plan: list[dict]) -> dict:
    """Applies one transaction's computed instant to it and every row
    reachable from it, per the anchoring rules in the module docstring.
    Returns per-table update counts for the final report."""
    counts = {
        "transactions": 0, "scores": 0, "cases": 0, "llm_reports": 0,
        "analyst_decisions": 0, "precedent_index": 0, "auto_block_log_shadow": 0,
        "auto_block_log_proposed": 0,
    }
    for entry in plan:
        txn = db.get(m.Transaction, entry["transaction_id"])
        txn.created_at = entry["new_created_at"]
        counts["transactions"] += 1

        for score in db.query(m.Score).filter(m.Score.transaction_id == txn.id).all():
            score.created_at = entry["new_created_at"]
            counts["scores"] += 1

        case = db.query(m.Case).filter(m.Case.transaction_id == txn.id).first()
        if case is None:
            continue  # GREEN, auto-clean - no Case row, nothing further to shift

        case.created_at = entry["new_created_at"]
        counts["cases"] += 1

        for report in db.query(m.LlmReport).filter(m.LlmReport.case_id == case.id).all():
            report.generated_at = entry["new_created_at"] + entry["llm_offset"]
            counts["llm_reports"] += 1

        for abl in db.query(m.AutoBlockLog).filter(m.AutoBlockLog.case_id == case.id, m.AutoBlockLog.review_status == "proposed").all():
            abl.created_at = entry["new_created_at"]
            counts["auto_block_log_proposed"] += 1

        if case.status == "CLOSED":
            decided_at = entry["new_created_at"] + entry["decision_offset"]
            case.closed_at = decided_at

            for ad in db.query(m.AnalystDecision).filter(m.AnalystDecision.case_id == case.id).all():
                ad.decided_at = decided_at
                counts["analyst_decisions"] += 1

            for pi in db.query(m.PrecedentIndex).filter(m.PrecedentIndex.case_id == case.id).all():
                pi.created_at = decided_at
                counts["precedent_index"] += 1

            for abl in db.query(m.AutoBlockLog).filter(m.AutoBlockLog.case_id == case.id, m.AutoBlockLog.review_status == "shadow").all():
                abl.created_at = decided_at
                counts["auto_block_log_shadow"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="preview only - no DB writes")
    parser.add_argument("--yes", action="store_true", help="actually shift timestamps")
    args = parser.parse_args()
    do_write = args.yes and not args.dry_run

    today = date.today()
    db = SessionLocal()
    try:
        transactions = (
            db.query(m.Transaction)
            .filter(m.Transaction.is_demo.is_(True))
            .order_by(m.Transaction.id)
            .all()
        )
        print(f"Demo (is_demo=1) transactions to shift: {len(transactions)}")
        if not transactions:
            print("Nothing to do.")
            return

        rng = random.Random(RANDOM_STATE)
        plan = build_plan(transactions, today, rng)
        steps = [t.step for t in transactions]
        step_min, step_max = min(steps), max(steps)
        print(f"step range observed: [{step_min}, {step_max}]  ->  mapped onto "
              f"{WINDOW_DAYS} days ending {today.isoformat()}")

        if not do_write:
            print("\n--- DRY RUN: preview only, no database writes ---")

            print("\nSample step -> new date mapping (first 5, last 5 by id):")
            for entry in plan[:5] + plan[-5:]:
                print(f"  txn #{entry['transaction_id']}: step={entry['step']} -> "
                      f"{entry['new_created_at'].date().isoformat()} "
                      f"{entry['new_created_at'].strftime('%H:%M:%S')} UTC "
                      f"(day {entry['day_offset']}/{WINDOW_DAYS - 1})")

            histogram: dict[str, int] = {}
            for entry in plan:
                d = entry["new_created_at"].date().isoformat()
                histogram[d] = histogram.get(d, 0) + 1
            all_days = [(today - timedelta(days=WINDOW_DAYS - 1 - i)).isoformat() for i in range(WINDOW_DAYS)]
            empty_days = [d for d in all_days if d not in histogram]

            print(f"\nPer-day histogram ({WINDOW_DAYS} days, {len(empty_days)} empty):")
            for d in all_days:
                n = histogram.get(d, 0)
                bar = "#" * n
                marker = "  <- EMPTY" if n == 0 else ""
                print(f"  {d}: {n:3d} {bar}{marker}")

            # Consistency preview: walk one real CLOSED case's full chain,
            # read-only, showing what every dependent row's date WOULD
            # become without writing anything.
            sample_case = (
                db.query(m.Case)
                .join(m.Transaction, m.Transaction.id == m.Case.transaction_id)
                .filter(m.Case.status == "CLOSED", m.Transaction.is_demo.is_(True))
                .first()
            )
            if sample_case is not None:
                entry = next(e for e in plan if e["transaction_id"] == sample_case.transaction_id)
                decided_at = entry["new_created_at"] + entry["decision_offset"]
                report = db.query(m.LlmReport).filter(m.LlmReport.case_id == sample_case.id).first()
                decision = db.query(m.AnalystDecision).filter(m.AnalystDecision.case_id == sample_case.id).first()
                pindex = db.query(m.PrecedentIndex).filter(m.PrecedentIndex.case_id == sample_case.id).first()
                print(f"\nConsistency preview - case {sample_case.id} (CLOSED, txn #{sample_case.transaction_id}):")
                print(f"  transaction.created_at (= score, case.created_at) -> {entry['new_created_at']}")
                if report is not None:
                    print(f"  llm_report.generated_at                          -> "
                          f"{entry['new_created_at'] + entry['llm_offset']}  "
                          f"(+{entry['llm_offset']})")
                print(f"  case.closed_at (= analyst_decision.decided_at)   -> {decided_at}  "
                      f"(+{entry['decision_offset']})")
                if pindex is not None:
                    print(f"  precedent_index.created_at                       -> {decided_at}  (= decision instant)")
                ordering_ok = entry["new_created_at"] <= decided_at
                print(f"  ordering transaction <= score <= report <= decision: {'OK' if ordering_ok else 'BROKEN'}")

            print("\nNo database rows were written. Pass --yes to actually shift timestamps.")
            return

        counts = apply_plan(db, plan)
        db.commit()
        print("\n--- Applied ---")
        for label, n in counts.items():
            print(f"  {label}: {n} rows updated")
    finally:
        db.close()


if __name__ == "__main__":
    main()
