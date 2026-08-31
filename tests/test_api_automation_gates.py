from __future__ import annotations

import numpy as np
import pytest

from backend import db_models as m
from backend import precedent as precedent_module

BASE_AMOUNT = 4_321_000.0


def _ghost_dest_payload(amount):
    return {
        "step": 3, "type": "TRANSFER", "amount": amount,
        "oldbalanceOrg": amount, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }


@pytest.fixture(autouse=True)
def clean_precedent_index(db_session):
    db_session.query(m.PrecedentIndex).delete()
    db_session.commit()
    yield
    db_session.query(m.PrecedentIndex).delete()
    db_session.commit()


def _seed_confirm_fraud_pool(db_session, scaler, base_amount, count, band_one_hot=(1.0, 0.0, 0.0)):
    for i in range(count):
        raw = np.array([base_amount + i, 3.0, 0.0, base_amount + i, 1.0, 0.0] + list(band_one_hot))
        scaled = scaler.transform(raw.reshape(1, -1))[0]
        db_session.add(m.PrecedentIndex(
            transaction_id=7_000_000 + i, case_id=7_000_000 + i,
            feature_vector=scaled.tolist(), label="confirm_fraud",
        ))
    db_session.commit()


def test_all_six_gates_pass_for_a_well_supported_case(client, db_session):
    scaler = precedent_module.load_precedent_scaler()
    assert scaler is not None
    _seed_confirm_fraud_pool(db_session, scaler, BASE_AMOUNT, count=12)

    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 1))
    case_id = score_res.json()["case_id"]
    assert score_res.json()["calibrated_proba"] >= 0.95

    res = client.get(f"/cases/{case_id}/automation-gates")
    assert res.status_code == 200
    body = res.json()
    assert body["case_id"] == case_id
    assert body["eligible"] is True
    assert body["direction"] == "fraud"
    assert len(body["gates"]) == 6
    assert all(g["passed"] for g in body["gates"])


def test_gates_structure_has_actual_and_threshold_per_gate(client, db_session):
    scaler = precedent_module.load_precedent_scaler()
    _seed_confirm_fraud_pool(db_session, scaler, BASE_AMOUNT + 100, count=12)

    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 101))
    case_id = score_res.json()["case_id"]

    res = client.get(f"/cases/{case_id}/automation-gates")
    body = res.json()
    gate_names = {g["gate"] for g in body["gates"]}
    assert gate_names == {
        "direction_automatable", "similarity", "precedent_count",
        "consensus", "calibrated_proba", "hard_rule_conflict",
    }
    similarity_gate = next(g for g in body["gates"] if g["gate"] == "similarity")
    assert similarity_gate["threshold"] == 0.95
    assert similarity_gate["actual"] is not None


def test_insufficient_precedent_case_is_not_eligible(client):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 200))
    case_id = score_res.json()["case_id"]

    res = client.get(f"/cases/{case_id}/automation-gates")
    body = res.json()
    assert body["eligible"] is False
    direction_gate = next(g for g in body["gates"] if g["gate"] == "direction_automatable")
    assert direction_gate["passed"] is False


def test_automation_gates_reads_the_live_active_policy_threshold(client, db_session):
    from backend import automation

    original_policy = automation.get_active_policy(db_session)
    original_threshold = original_policy.fraud_similarity_threshold
    try:
        original_policy.fraud_similarity_threshold = 0.999999
        db_session.commit()

        scaler = precedent_module.load_precedent_scaler()
        db_session.query(m.PrecedentIndex).delete()
        db_session.commit()
        _seed_confirm_fraud_pool(db_session, scaler, BASE_AMOUNT + 300, count=12)
        score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 301))
        case_id = score_res.json()["case_id"]

        res = client.get(f"/cases/{case_id}/automation-gates")
        body = res.json()
        similarity_gate = next(g for g in body["gates"] if g["gate"] == "similarity")
        assert similarity_gate["threshold"] == 0.999999
    finally:
        original_policy.fraud_similarity_threshold = original_threshold
        db_session.commit()


def test_closed_case_returns_null_not_gate_data(client):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 400))
    case_id = score_res.json()["case_id"]
    client.post(f"/cases/{case_id}/decision", json={"action_taken": "approve_clean", "analyst_reason_code": "false_positive"})

    res = client.get(f"/cases/{case_id}/automation-gates")
    assert res.status_code == 200
    assert res.json() is None


def test_automation_gates_404s_for_unknown_case(client):
    res = client.get("/cases/999999999/automation-gates")
    assert res.status_code == 404


def test_pending_ai_decision_hides_proposal_on_a_closed_case(client, db_session):
    from backend.main import decide_case
    from backend import schemas as s

    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 44))
    case_id, txn_id = score_res.json()["case_id"], score_res.json()["txn_id"]

    db_session.add(m.AutoBlockLog(
        transaction_id=txn_id, case_id=case_id,
        triggered_conditions={"eligible": True, "direction": "fraud", "reason": [], "failed_gates": []},
        review_status="proposed",
    ))
    db_session.commit()

    assert client.get(f"/cases/{case_id}/pending-ai-decision").json() is not None

    decide_case(
        case_id=case_id,
        decision=s.DecisionIn(action_taken="confirm_fraud", analyst_reason_code="test_close"),
        db=db_session,
    )
    db_session.commit()

    assert client.get(f"/cases/{case_id}/pending-ai-decision").json() is None
    assert db_session.query(m.AutoBlockLog).filter_by(case_id=case_id, review_status="proposed").count() == 1


def test_no_active_policy_returns_null_not_error(client, db_session):
    from backend import automation

    original_policy = automation.get_active_policy(db_session)
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 500))
    case_id = score_res.json()["case_id"]
    try:
        original_policy.active = False
        db_session.commit()

        res = client.get(f"/cases/{case_id}/automation-gates")
        assert res.status_code == 200
        assert res.json() is None
    finally:
        original_policy.active = True
        db_session.commit()
