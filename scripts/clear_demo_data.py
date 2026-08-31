"""
Deletes all simulator/demo-tagged rows (transactions.is_demo=True) and
everything that references them (scores, rule_hits, shap_explanations,
cases, analyst_decisions, llm_reports) in dependency order, inside a
single transaction.

SQLite foreign keys aren't enforced in this project (no ondelete=CASCADE,
no PRAGMA foreign_keys=ON), so a bare `DELETE FROM transactions WHERE
is_demo=true` would leave orphaned child rows - this script deletes
children first instead.

Usage (from the project root, with venv):
    venv/Scripts/python.exe scripts/clear_demo_data.py          # dry run, shows counts only
    venv/Scripts/python.exe scripts/clear_demo_data.py --yes    # actually deletes
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="actually delete (default is a dry run)")
    args = parser.parse_args()

    if not settings.database_url.startswith("sqlite"):
        raise SystemExit(f"This script only supports the sqlite path used in development ({settings.database_url}).")

    db_path = settings.database_url.replace("sqlite:///", "", 1)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("SELECT COUNT(*) FROM transactions WHERE is_demo=1")
    n_txn = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(*) FROM cases
        WHERE transaction_id IN (SELECT id FROM transactions WHERE is_demo=1)
    """)
    n_cases = cur.fetchone()[0]

    print(f"Demo transactions: {n_txn}")
    print(f"Demo-linked cases: {n_cases}")

    if not args.yes:
        print("\nDry run only - pass --yes to actually delete.")
        con.close()
        return

    if n_txn == 0:
        print("Nothing to delete.")
        con.close()
        return

    try:
        cur.execute("""
            DELETE FROM analyst_decisions WHERE case_id IN (
                SELECT id FROM cases WHERE transaction_id IN (
                    SELECT id FROM transactions WHERE is_demo=1))
        """)
        cur.execute("""
            DELETE FROM llm_reports WHERE case_id IN (
                SELECT id FROM cases WHERE transaction_id IN (
                    SELECT id FROM transactions WHERE is_demo=1))
        """)
        cur.execute("DELETE FROM cases WHERE transaction_id IN (SELECT id FROM transactions WHERE is_demo=1)")
        cur.execute("DELETE FROM shap_explanations WHERE transaction_id IN (SELECT id FROM transactions WHERE is_demo=1)")
        cur.execute("DELETE FROM rule_hits WHERE transaction_id IN (SELECT id FROM transactions WHERE is_demo=1)")
        cur.execute("DELETE FROM scores WHERE transaction_id IN (SELECT id FROM transactions WHERE is_demo=1)")
        cur.execute("DELETE FROM transactions WHERE is_demo=1")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    print(f"\nDeleted {n_txn} demo transactions and {n_cases} linked cases (+ dependents).")


if __name__ == "__main__":
    main()
