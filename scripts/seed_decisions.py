"""
Seeds synthetic-but-realistic analyst decisions on existing OPEN RED/GRAY
cases through the real POST /cases/{id}/decision endpoint, to bootstrap
the precedent engine (there is otherwise only one real analyst decision in
the database).

Picks four clusters of similar OPEN cases (same risk_band + rule set) and
assigns each a decision policy - two fraud-consensus clusters, one mixed/
low-consensus cluster, one clean-consensus cluster - so the precedent
engine's k-NN logic has something meaningful to agree or disagree about.

All decisions are tagged analyst_reason_code="seed_cluster_decision" so
they're distinguishable from real ones. Idempotent: only targets OPEN
cases, and the API rejects deciding an already-CLOSED case.

Usage (with the backend running, from the project root):
    venv/Scripts/python.exe scripts/seed_decisions.py
"""
from __future__ import annotations

import os
import random
import sys
import time
from collections import defaultdict

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.database import SessionLocal
from backend import db_models as m

API_URL = "http://127.0.0.1:8000"
SEED_REASON_CODE = "seed_cluster_decision"
RANDOM_STATE = 42

# (risk_band, sorted triggered-rule tuple) -> which OPEN cases are "similar"
# enough (before a real feature vector exists) to seed as one cluster.
CLUSTERS = [
    {
        "key": ("RED", ("drain_account", "ghost_destination", "high_amount_transfer", "night_transaction")),
        "n": 15,
        "weights": {"confirm_fraud": 0.85, "escalate": 0.15},
        "note": "Full account drain to a zero-history destination at night - matches confirmed fraud pattern.",
    },
    {
        "key": ("RED", ("ghost_destination", "high_amount_transfer", "night_transaction")),
        "n": 15,
        "weights": {"confirm_fraud": 0.80, "escalate": 0.20},
        "note": "Ghost destination account, large transfer, night timing - consistent with prior confirmed fraud.",
    },
    {
        "key": ("GRAY", ("drain_account", "high_amount_transfer")),
        "n": 15,
        "weights": {"confirm_fraud": 0.34, "approve_clean": 0.33, "escalate": 0.33},
        "note": "Large transfer, no hard-rule confirmation - ambiguous; analysts have gone different ways on this pattern.",
    },
    {
        "key": ("GRAY", ()),
        "n": 15,
        "weights": {"approve_clean": 0.85, "escalate": 0.15},
        "note": "No rule hits, borderline model score only - usually resolves clean on review.",
    },
]


def pick_action(weights: dict[str, float], rng: random.Random) -> str:
    actions, probs = zip(*weights.items())
    return rng.choices(actions, weights=probs, k=1)[0]


def main() -> None:
    rng = random.Random(RANDOM_STATE)
    db = SessionLocal()

    open_cases = db.query(m.Case).filter(m.Case.status == "OPEN").all()
    by_cluster: dict[tuple, list[int]] = defaultdict(list)
    for c in open_cases:
        score = (
            db.query(m.Score)
            .filter(m.Score.transaction_id == c.transaction_id)
            .order_by(m.Score.id.desc())
            .first()
        )
        if score is None:
            continue
        hits = db.query(m.RuleHit).filter(m.RuleHit.transaction_id == c.transaction_id).all()
        rule_key = tuple(sorted(h.rule_name for h in hits))
        by_cluster[(score.risk_band, rule_key)].append(c.id)
    db.close()

    ok, skipped, failed = 0, 0, 0
    dist: dict[str, int] = defaultdict(int)
    t0 = time.time()

    for spec in CLUSTERS:
        candidates = list(by_cluster.get(spec["key"], []))
        rng.shuffle(candidates)
        chosen = candidates[: spec["n"]]
        print(f"Cluster {spec['key']}: {len(candidates)} candidates, seeding {len(chosen)} decisions")

        for case_id in chosen:
            action = pick_action(spec["weights"], rng)
            payload = {
                "action_taken": action,
                "analyst_reason_code": SEED_REASON_CODE,
                "analyst_note": spec["note"],
            }
            try:
                res = requests.post(f"{API_URL}/cases/{case_id}/decision", json=payload, timeout=10)
                if res.status_code == 409:
                    skipped += 1
                    continue
                res.raise_for_status()
                ok += 1
                dist[action] += 1
            except Exception as e:
                failed += 1
                print(f"  [ERROR] case {case_id}: {e}")

    print(f"\nDone in {time.time() - t0:.1f}s: {ok} decided, {skipped} already closed, {failed} failed")
    print("Decision distribution:", dict(dist))


if __name__ == "__main__":
    main()
