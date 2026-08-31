"""Regenerates every llm_reports row that fell back to the deterministic
template, replacing it with a real Groq report. Only touches
source=fallback rows.

Deletes the fallback row, then calls generate_report() fresh with the same
findings the original call would have seen (findings.build_findings() is
pure over stored rows, so the prompt is reproduced exactly). generated_at
moves to now and source becomes "groq" - the timestamp isn't preserved,
since claiming the original fallback moment would misrepresent when the
row was actually written.

Idempotent: after a successful run there are no fallback rows left, so a
second run finds nothing to do. --delay 6.5s by default for Groq rate limits.

Usage (backend NOT required to be running - calls the LLM module directly):
    python -m scripts.regenerate_fallback_reports --dry-run
    python -m scripts.regenerate_fallback_reports --yes
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select  # noqa: E402

from backend import db_models as m  # noqa: E402
from backend import findings, llm_service  # noqa: E402
from backend.database import SessionLocal  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=6.5, help="seconds between Groq calls")
    parser.add_argument("--dry-run", action="store_true", help="list targets, do nothing")
    parser.add_argument("--yes", action="store_true", help="required to actually rewrite")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        targets = db.execute(
            select(m.LlmReport).where(m.LlmReport.source == "fallback").order_by(m.LlmReport.id)
        ).scalars().all()

        print(f"  fallback reports found: {len(targets)}")
        for r in targets[:5]:
            print(f"    id={r.id} case_id={r.case_id} generated_at={r.generated_at:%Y-%m-%d %H:%M}")
        if len(targets) > 5:
            print(f"    ... and {len(targets)-5} more")

        if args.dry_run or not args.yes:
            print("\nDry run - nothing changed. Re-run with --yes to rewrite.")
            return

        regenerated = failed_kept = 0
        t0 = time.time()
        for row in targets:
            case = db.get(m.Case, row.case_id)
            if case is None:
                continue  # case was deleted since - leave orphan row alone
            txn = db.get(m.Transaction, case.transaction_id)

            fs = findings.build_findings(case, db)
            summary = findings.txn_summary_text(txn)
            result = llm_service.generate_report(findings=fs, txn_summary=summary)

            if result["source"] == "fallback":
                # Groq still refused this call - leave the existing row
                # untouched rather than write an identical fallback with a
                # newer timestamp (would erase the original evidence of
                # when the fallback actually happened, for zero gain).
                failed_kept += 1
                continue

            db.delete(row)
            db.add(m.LlmReport(
                case_id=row.case_id,
                report_text=result["text"],
                model_name=llm_service.settings.llm_model,
                source=result["source"],
            ))
            db.commit()
            regenerated += 1
            if args.delay:
                time.sleep(args.delay)

        print(f"\nregenerated: {regenerated}, still-fallback (kept): {failed_kept}, {time.time()-t0:.1f}s")
    finally:
        db.close()


if __name__ == "__main__":
    main()
