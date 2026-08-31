"""
Backfills automation proposals for existing OPEN cases.

evaluate_and_propose_for_case() (backend/automation.py) only runs at
case-creation time, so switching the active policy from "shadow" to
"propose" does not retroactively evaluate cases that were already OPEN.
This script closes that gap by running the same evaluation against every
currently-OPEN case, using the active policy (automation.get_active_policy()).

Idempotent: propose_auto_decision() returns an existing "proposed" row
unchanged rather than creating a second one, so re-running is always safe.
No Groq calls - automation evaluation is precedent k-NN + policy-gate
arithmetic only.

Usage (from the project root, backend NOT required to be running):
    python scripts/backfill_auto_proposals.py --dry-run   # preview only, no writes
    python scripts/backfill_auto_proposals.py --yes       # actually create proposals
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend import automation
from backend import db_models as m
from backend import precedent
from backend.database import SessionLocal


def preview_eligible(db, policy) -> list[dict]:
    """Read-only preview: composes the exact same pure, side-effect-free
    functions evaluate_and_propose_for_case() calls internally
    (find_precedents, summarize_precedents, evaluate_auto_decision - all
    documented as no-DB-write, no-side-effect) without ever calling
    propose_auto_decision(), so --dry-run never writes anything. This is
    not a reimplementation of the app's decision logic - it's the same
    three functions, just stopped one step short of the write."""
    scaler = precedent.load_precedent_scaler()
    if scaler is None:
        return []

    open_cases = db.query(m.Case).filter(m.Case.status == "OPEN").all()
    eligible = []
    for case in open_cases:
        neighbors = precedent.find_precedents(case, db, scaler, k=precedent.DEFAULT_K)
        psum = precedent.summarize_precedents(neighbors)
        score = (
            db.query(m.Score)
            .filter(m.Score.transaction_id == case.transaction_id)
            .order_by(m.Score.id.desc())
            .first()
        )
        hard_hits = [
            h.rule_name for h in
            db.query(m.RuleHit)
            .filter(m.RuleHit.transaction_id == case.transaction_id, m.RuleHit.rule_type == "hard")
            .all()
        ]
        clean_hits = [
            h.rule_name for h in
            db.query(m.RuleHit)
            .filter(m.RuleHit.transaction_id == case.transaction_id, m.RuleHit.rule_type == "clean")
            .all()
        ]
        result = automation.evaluate_auto_decision(psum, score.calibrated_proba, hard_hits, policy, clean_hits)
        if result["eligible"] and result["direction"] == "fraud":
            eligible.append({
                "case_id": case.id, "transaction_id": case.transaction_id,
                "calibrated_proba": score.calibrated_proba,
                "avg_similarity": psum["avg_similarity"],
                "consensus_ratio": psum["consensus_ratio"],
                "precedent_count": psum["precedent_count"],
            })
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="preview only - no DB writes")
    parser.add_argument("--yes", action="store_true", help="actually create proposals")
    args = parser.parse_args()
    do_write = args.yes and not args.dry_run

    db = SessionLocal()
    try:
        policy = automation.get_active_policy(db)
        if policy is None:
            print("No active automation policy - nothing to do.")
            return

        print(f"Active policy: {policy.version} (mode={policy.mode})")
        print(f"Gates: similarity>={policy.fraud_similarity_threshold}, "
              f"precedent_count>={policy.min_precedent_count}, "
              f"consensus>={policy.min_consensus_ratio}, "
              f"calibrated_proba>={policy.min_calibrated_proba}, "
              f"hard_rule_required={policy.hard_rule_required}, "
              f"auto_clean_enabled={policy.auto_clean_enabled}")
        if policy.mode != "propose":
            print(f"\nWARNING: active policy mode is {policy.mode!r}, not 'propose' - "
                  f"propose_auto_decision() is a no-op outside propose mode (see its own "
                  f"docstring), so a real run right now would create zero proposals "
                  f"regardless of how many cases clear the gates. Switch the policy to "
                  f"propose first.")

        open_cases = db.query(m.Case).filter(m.Case.status == "OPEN").all()
        print(f"\nOPEN cases to scan: {len(open_cases)}")

        if not do_write:
            eligible = preview_eligible(db, policy)
            print(f"\n--- DRY RUN: {len(eligible)} case(s) would get a proposal ---")
            for e in eligible:
                print(f"  case {e['case_id']} (txn #{e['transaction_id']}): "
                      f"proba={e['calibrated_proba']:.4f} sim={e['avg_similarity']} "
                      f"consensus={e['consensus_ratio']} count={e['precedent_count']}")
            print("\nNo database rows were written. Pass --yes to actually create these proposals.")
            return

        proposed_before = db.query(m.AutoBlockLog).filter(m.AutoBlockLog.review_status == "proposed").count()

        for i, case in enumerate(open_cases, 1):
            automation.evaluate_and_propose_for_case(case.id)
            if i % 50 == 0 or i == len(open_cases):
                print(f"  scanned {i}/{len(open_cases)}")

        db.expire_all()
        proposed_after = db.query(m.AutoBlockLog).filter(m.AutoBlockLog.review_status == "proposed").count()

        print(f"\nDone: {len(open_cases)} OPEN cases scanned, "
              f"{proposed_after - proposed_before} new proposal(s) created "
              f"({proposed_before} already existed, {proposed_after} total now).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
