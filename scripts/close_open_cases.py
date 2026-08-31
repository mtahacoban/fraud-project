"""Closes existing OPEN cases on PaySim's own ground-truth label, growing
precedent_index without touching anything else in the database.

Unlike seed_demo_cases.py, this only closes cases that already exist -
nothing is scored or deleted. Selects eligible OPEN cases with a real
PaySim ground-truth label (excludes the baseline case and Simulation-built
cases), RED first then GRAY, oldest case id first for determinism.

Closes through the real decide_case() path (backend/main.py), called
in-process, so precedent indexing and shadow-mode automation logging run
exactly as they would for a real analyst decision. Reuses whatever
precedent scaler is already fit on disk without refitting (refitting would
move the geometry existing precedent vectors were indexed under); requires
a scaler to already exist rather than triggering a fit itself.

Every decision is tagged analyst_reason_code="seed_ground_truth_paysim",
same as seed_demo_cases.py's ground-truth batch - synthetic, not a real
analyst's judgment call (see the README's Known Limitations).

Usage (from the project root, with venv - backend does NOT need to be
running, this talks to the database directly):
    python scripts/close_open_cases.py --dry-run
    python scripts/close_open_cases.py --count 100 --yes
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend import db_models as m  # noqa: E402
from backend import precedent  # noqa: E402
from backend import schemas as s  # noqa: E402
from backend.database import SessionLocal  # noqa: E402
from backend.main import decide_case  # noqa: E402

ANALYST_REASON_CODE = "seed_ground_truth_paysim"


def _latest_risk_band(db, transaction_id: int) -> str:
    score = (
        db.query(m.Score)
        .filter(m.Score.transaction_id == transaction_id)
        .order_by(m.Score.id.desc())
        .first()
    )
    return score.risk_band if score else "UNKNOWN"


def pick_cases(db, count: int) -> list[dict]:
    """Selects up to `count` OPEN cases with a real PaySim ground-truth
    label, RED first (encounter order = case id ascending), GRAY filling
    the rest."""
    open_cases = db.query(m.Case).filter(m.Case.status == "OPEN").order_by(m.Case.id.asc()).all()

    eligible: list[dict] = []
    excluded_no_label = 0
    for case in open_cases:
        txn = db.query(m.Transaction).filter(m.Transaction.id == case.transaction_id).first()
        if txn.is_fraud is None:
            excluded_no_label += 1
            continue
        eligible.append({
            "case_id": case.id,
            "band": _latest_risk_band(db, case.transaction_id),
            "is_fraud": int(txn.is_fraud),
        })

    red = [c for c in eligible if c["band"] == "RED"]
    gray = [c for c in eligible if c["band"] == "GRAY"]
    other = [c for c in eligible if c["band"] not in ("RED", "GRAY")]

    n_red = min(len(red), count)
    n_gray = min(len(gray), max(count - n_red, 0))
    selected = red[:n_red] + gray[:n_gray]

    print(f"  open cases total        : {len(open_cases)}")
    print(f"  excluded (no ground truth label - non-demo baseline / simulator): {excluded_no_label}")
    print(f"  eligible RED / GRAY / other: {len(red)} / {len(gray)} / {len(other)}")
    print(f"  selecting                : {n_red} RED + {n_gray} GRAY = {len(selected)}")
    if len(selected) < count:
        print(f"  NOTE: only {len(selected)} eligible cases available - fewer than the requested {count}")

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=100, help="how many OPEN cases to close")
    parser.add_argument("--yes", action="store_true", help="actually close the cases (default: dry run)")
    parser.add_argument("--dry-run", action="store_true", help="explicit alias for the default (no --yes) behavior")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        pool_before = db.query(m.PrecedentIndex).count()
        scaler = precedent.load_precedent_scaler()
        print(f"  precedent pool now       : {pool_before}")
        print(f"  scaler on disk           : {'yes' if scaler is not None else 'NO'}")
        if scaler is None:
            print("\nNo precedent scaler on disk - this is a true cold start. Run "
                  "scripts/backfill_precedents.py first (it fits one from whatever's already "
                  "decided), then re-run this script. Stopping.")
            return 1

        selected = pick_cases(db, args.count)
        if not selected:
            print("\nNothing eligible to close.")
            return 0

        n_fraud = sum(1 for c in selected if c["is_fraud"])
        n_clean = len(selected) - n_fraud
        print(f"  decision split           : {n_fraud} confirm_fraud + {n_clean} approve_clean")

        if not args.yes:
            print(f"\nDry run - would close {len(selected)} cases. Re-run with --count {args.count} --yes to apply.")
            return 0

        print(f"\nClosing {len(selected)} cases via the real decide_case() path...")
        closed = 0
        for info in selected:
            action = "confirm_fraud" if info["is_fraud"] else "approve_clean"
            decision_in = s.DecisionIn(action_taken=action, analyst_reason_code=ANALYST_REASON_CODE)
            decide_case(case_id=info["case_id"], decision=decision_in, db=db)
            closed += 1
            if closed % 25 == 0:
                print(f"  {closed}/{len(selected)} closed")

        pool_after = db.query(m.PrecedentIndex).count()
        print(f"\nclosed: {closed}")
        print(f"precedent pool: {pool_before} -> {pool_after}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
