"""
One-command clone-to-running setup: checks prerequisites, then runs the
seeding pipeline in the one order that's safe on a fresh database:
1) seed_demo_cases.py (scores + closes a ground-truth subset + Groq reports)
2) distribute_demo_timestamps.py (spreads timestamps across 30 days, not idempotent)
3) backfill_auto_proposals.py (evaluates Automation Eligibility for the new OPEN cases)

Does not train a model (models/*.pkl are committed) or download the PaySim
dataset (not needed for seeding), and does not start the backend/frontend
itself - start both yourself afterward.

Checks before writing: .env exists (copied from .env.example if missing),
model files present, LLM_API_KEY set (seed_demo_cases.py requires a real
Groq key - a free one from https://console.groq.com/keys works), and
fraud.db has no existing demo data (pass --force only after clearing it
with scripts/clear_demo_data.py).

Usage (from the project root, with venv active):
    python scripts/setup.py --dry-run   # checks everything, writes nothing
    python scripts/setup.py --yes       # runs the pipeline (~30-45 min, mostly Groq calls)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REQUIRED_MODEL_FILES = [
    "models/xgb_v1.pkl",
    "models/xgb_v1_calibrated.pkl",
    "models/precedent_scaler.pkl",
]

PIPELINE = [
    ("scripts/seed_demo_cases.py", "600 PaySim transactions, ground-truth-closed subset, real Groq reports"),
    ("scripts/distribute_demo_timestamps.py", "spreads timestamps across a 30-day window (trend charts)"),
    ("scripts/backfill_auto_proposals.py", "evaluates Automation Eligibility for the newly-opened OPEN cases"),
]


def _check_env_file() -> list[str]:
    problems = []
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            shutil.copy(".env.example", ".env")
            print("  .env did not exist - copied from .env.example.")
        else:
            problems.append(".env is missing and .env.example isn't present to copy from.")
    # frontend/ doesn't exist at all inside the Docker backend image (see
    # Dockerfile.backend - only backend/, models/, reports/, scripts/ are
    # copied in) - this script runs there too (`docker compose exec backend
    # python scripts/setup.py`), so a missing frontend/ dir is expected in
    # that context and must not be treated as a problem. Bare-metal, where
    # frontend/ always exists, this still copies frontend/.env as before.
    if os.path.isdir("frontend") and not os.path.exists("frontend/.env"):
        if os.path.exists("frontend/.env.example"):
            shutil.copy("frontend/.env.example", "frontend/.env")
            print("  frontend/.env did not exist - copied from frontend/.env.example.")
        else:
            problems.append("frontend/.env is missing and frontend/.env.example isn't present to copy from.")
    return problems


def _check_models() -> list[str]:
    missing = [p for p in REQUIRED_MODEL_FILES if not os.path.exists(p)]
    if not missing:
        return []
    return [
        f"Missing model file(s): {', '.join(missing)}. These are committed to the repo - "
        "if you're seeing this, either the clone is incomplete or they were deleted. "
        "See the README's \"Train from scratch\" section to regenerate them (requires "
        "downloading the Kaggle PaySim dataset into data/ first)."
    ]


def _check_llm_key() -> list[str]:
    # Imported here, not at module level - backend.config reads .env at
    # import time, and _check_env_file() above may have only just created it.
    from backend.config import settings
    if settings.llm_api_key:
        return []
    return [
        "LLM_API_KEY is blank in .env. scripts/seed_demo_cases.py requires a real "
        "Groq key and FATALs without one (by design - see its own module docstring). "
        "Get a free key at https://console.groq.com/keys and set LLM_API_KEY= in .env, "
        "then re-run."
    ]


def _check_fresh_db(force: bool) -> list[str]:
    from backend.database import SessionLocal
    from backend import db_models as m
    db = SessionLocal()
    try:
        demo_count = db.query(m.Transaction).filter(m.Transaction.is_demo.is_(True)).count()
    finally:
        db.close()
    if demo_count == 0 or force:
        return []
    return [
        f"fraud.db already has {demo_count} demo-tagged transaction(s). Running this "
        "pipeline again would re-shift already-distributed timestamps a second time "
        "(distribute_demo_timestamps.py is explicitly not idempotent). If you really "
        "want to reseed from scratch: python scripts/clear_demo_data.py --yes, then "
        "re-run this script - or pass --force to skip this check (not recommended)."
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="check prerequisites only, run nothing")
    parser.add_argument("--yes", action="store_true", help="actually run the seeding pipeline")
    parser.add_argument("--force", action="store_true", help="skip the fresh-database check (not recommended)")
    args = parser.parse_args()

    if not args.dry_run and not args.yes:
        print("Pass --dry-run to check prerequisites only, or --yes to actually run the pipeline.")
        return 1

    print("--- Checking prerequisites ---")
    problems = _check_env_file()
    problems += _check_models()
    problems += _check_llm_key()
    problems += _check_fresh_db(args.force)

    if problems:
        print("\nCan't proceed:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("  .env, model files, Groq key, and database state all look good.")

    if args.dry_run:
        print("\n--dry-run: would run, in order:")
        for script, desc in PIPELINE:
            print(f"  python {script} --yes   ({desc})")
        print("\nThis takes roughly 30-45 minutes end to end - almost all of it real Groq "
              "report generation, paced to stay under the free-tier rate limit, not something "
              "any --dry-run flag can skip. Re-run with --yes to actually do it.")
        return 0

    print(f"\n--- Running {len(PIPELINE)} steps - expect ~30-45 minutes, mostly Groq report generation ---")
    for script, desc in PIPELINE:
        print(f"\n>>> {script}  ({desc})")
        result = subprocess.run([sys.executable, script, "--yes"])
        if result.returncode != 0:
            print(f"\n{script} exited with code {result.returncode} - stopping here.")
            print("Fix the reported issue and re-run scripts/setup.py; the scripts above are "
                  "each safe to re-run individually except distribute_demo_timestamps.py "
                  "(see its own docstring).")
            return result.returncode

    print(
        "\n--- Done ---\n"
        "Start the backend:  uvicorn backend.main:app --reload --port 8000\n"
        "Start the frontend: cd frontend && npm install && npm run dev\n"
        "Then open http://localhost:5173"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
