from __future__ import annotations

import numpy as np
import pytest

from backend import db_models as m
from backend import precedent as precedent_module

BASE_AMOUNT = 5_432_100.0


def _ghost_dest_payload(amount, country="TR"):
    return {
        "step": 3, "type": "TRANSFER", "amount": amount,
        "oldbalanceOrg": amount, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        "login_country": country,
    }


@pytest.fixture(autouse=True)
def clean_precedent_index(db_session):
    db_session.query(m.PrecedentIndex).delete()
    db_session.commit()
    yield
    db_session.query(m.PrecedentIndex).delete()
    db_session.commit()


def test_events_are_chronological_and_multi_decision_history_is_preserved(client):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 1))
    case_id = score_res.json()["case_id"]
    assert score_res.json()["risk_band"] == "RED"

    dec1 = client.post(f"/cases/{case_id}/decision", json={"action_taken": "confirm_fraud", "analyst_reason_code": "account_takeover"})
    assert dec1.status_code == 200

    reopen_res = client.post(f"/cases/{case_id}/reopen", json={"analyst_reason_code": "test_reopen"})
    assert reopen_res.status_code == 200
    dec2 = client.post(f"/cases/{case_id}/decision", json={"action_taken": "approve_clean", "analyst_reason_code": "false_positive"})
    assert dec2.status_code == 200

    trail = client.get(f"/cases/{case_id}/audit-trail")
    assert trail.status_code == 200
    events = trail.json()["events"]

    timestamps = [e["timestamp"] for e in events]
    assert timestamps == sorted(timestamps)

    event_types = [e["event_type"] for e in events]
    assert event_types.count("decision") == 3
    assert "scored" in event_types
    assert "case_opened" in event_types

    decision_events = [e for e in events if e["event_type"] == "decision"]
    assert decision_events[0]["summary"] == "Confirmed as fraud"
    assert decision_events[0]["before"] == "OPEN" and decision_events[0]["after"] == "CLOSED"
    assert "no_reason" not in decision_events[0]["anomaly_flags"]

    assert decision_events[1]["summary"] == "Reopened"
    assert decision_events[1]["before"] == "CLOSED" and decision_events[1]["after"] == "OPEN"

    assert decision_events[2]["summary"] == "Closed as clean"
    assert "no_reason" not in decision_events[2]["anomaly_flags"]
    assert "rapid_redecision" in decision_events[2]["anomaly_flags"]


def test_no_reason_flag_fires_for_a_historical_manual_decision_without_one(client, db_session):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 40))
    case_id = score_res.json()["case_id"]

    case = db_session.get(m.Case, case_id)
    db_session.add(m.AnalystDecision(
        case_id=case_id, action_taken="confirm_fraud",
        analyst_reason_code=None, analyst_note=None, ai_proposed=False,
    ))
    case.status = "CLOSED"
    db_session.commit()

    trail = client.get(f"/cases/{case_id}/audit-trail")
    decision_event = next(e for e in trail.json()["events"] if e["event_type"] == "decision")
    assert "no_reason" in decision_event["anomaly_flags"]


def test_no_reason_flag_never_fires_for_an_ai_confirmed_decision(client, db_session):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 41))
    case_id = score_res.json()["case_id"]

    case = db_session.get(m.Case, case_id)
    db_session.add(m.AnalystDecision(
        case_id=case_id, action_taken="confirm_fraud",
        analyst_reason_code=None, analyst_note=None, ai_proposed=True,
    ))
    case.status = "CLOSED"
    db_session.commit()

    trail = client.get(f"/cases/{case_id}/audit-trail")
    decision_event = next(e for e in trail.json()["events"] if e["event_type"] == "decision")
    assert "no_reason" not in decision_event["anomaly_flags"]


def test_decision_endpoint_rejects_a_missing_reason_code(client):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 42))
    case_id = score_res.json()["case_id"]
    res = client.post(f"/cases/{case_id}/decision", json={"action_taken": "confirm_fraud"})
    assert res.status_code == 422


def test_decision_endpoint_rejects_a_whitespace_only_reason_code(client):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 43))
    case_id = score_res.json()["case_id"]
    res = client.post(f"/cases/{case_id}/decision", json={"action_taken": "confirm_fraud", "analyst_reason_code": "   "})
    assert res.status_code == 422


def test_decision_endpoint_accepts_a_custom_reason_code_not_in_the_known_list(client):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 44))
    case_id = score_res.json()["case_id"]
    res = client.post(f"/cases/{case_id}/decision", json={"action_taken": "confirm_fraud", "analyst_reason_code": "custom_reason_xyz"})
    assert res.status_code == 200


def test_clean_case_has_no_anomaly_flags_anywhere(client):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 2))
    case_id = score_res.json()["case_id"]
    client.post(f"/cases/{case_id}/decision", json={
        "action_taken": "confirm_fraud", "analyst_reason_code": "manual_review", "analyst_note": "clear fraud pattern",
    })
    trail = client.get(f"/cases/{case_id}/audit-trail")
    events = trail.json()["events"]
    assert all(e["anomaly_flags"] == [] for e in events)


def test_shadow_mode_rows_never_appear_in_the_audit_trail(client, db_session):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 3))
    case_id = score_res.json()["case_id"]

    db_session.add(m.AutoBlockLog(
        transaction_id=score_res.json()["txn_id"], case_id=case_id,
        triggered_conditions={"eligible": True, "direction": "fraud", "reason": [], "failed_gates": []},
        review_status="shadow",
    ))
    db_session.commit()

    trail = client.get(f"/cases/{case_id}/audit-trail")
    event_types = [e["event_type"] for e in trail.json()["events"]]
    assert "automation_proposed" not in event_types
    assert "automation_reviewed" not in event_types


def test_ai_human_conflict_flag_when_analyst_overrides_the_precedent_suggestion(client, db_session):
    scaler = precedent_module.load_precedent_scaler()
    assert scaler is not None, "models/precedent_scaler.pkl must exist for this test - see precedent.py"

    band_one_hot = [1.0, 0.0, 0.0]
    for i in range(5):
        raw = np.array([BASE_AMOUNT + 4 + i, 3.0, 0.0, BASE_AMOUNT + 4 + i, 1.0, 0.0] + band_one_hot)
        scaled = scaler.transform(raw.reshape(1, -1))[0]
        db_session.add(m.PrecedentIndex(
            transaction_id=8_000_000 + i, case_id=8_000_000 + i,
            feature_vector=scaled.tolist(), label="confirm_fraud",
        ))
    db_session.commit()

    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 4))
    case_id = score_res.json()["case_id"]

    precedents_res = client.get(f"/cases/{case_id}/precedents")
    summary = precedents_res.json()["summary"]
    assert summary["suggested_decision"] == "confirm_fraud", (
        f"precedent pool didn't clear the suggestion gates: {summary}"
    )

    client.post(f"/cases/{case_id}/decision", json={
        "action_taken": "approve_clean", "analyst_reason_code": "override", "analyst_note": "AI missed context",
    })

    trail = client.get(f"/cases/{case_id}/audit-trail")
    decision_event = next(e for e in trail.json()["events"] if e["event_type"] == "decision")
    assert "ai_human_conflict" in decision_event["anomaly_flags"]
    assert "Precedent Analysis had suggested: Confirmed as fraud" in decision_event["detail"]


def test_automation_reject_is_also_flagged_as_ai_human_conflict(client, db_session):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 10))
    case_id = score_res.json()["case_id"]
    txn_id = score_res.json()["txn_id"]

    db_session.add(m.AutoBlockLog(
        transaction_id=txn_id, case_id=case_id,
        triggered_conditions={"eligible": True, "direction": "fraud", "reason": [], "failed_gates": []},
        review_status="proposed",
    ))
    db_session.commit()

    reject_res = client.post(f"/cases/{case_id}/reject-ai-decision", json={"rejection_reason": "insufficient evidence"})
    assert reject_res.status_code == 200

    trail = client.get(f"/cases/{case_id}/audit-trail")
    reviewed_event = next(e for e in trail.json()["events"] if e["event_type"] == "automation_reviewed")
    assert reviewed_event["after"] == "rejected"
    assert "ai_human_conflict" in reviewed_event["anomaly_flags"]
    assert reviewed_event["detail"] == "insufficient evidence"


def test_report_generated_event_labels_a_real_llm_report_as_llm(client, db_session):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 30))
    case_id = score_res.json()["case_id"]
    db_session.add(m.LlmReport(case_id=case_id, report_text="x", model_name="openai/gpt-oss-20b", source="groq"))
    db_session.commit()

    trail = client.get(f"/cases/{case_id}/audit-trail")
    report_events = [e for e in trail.json()["events"] if e["event_type"] == "report_generated"]
    assert any(e["detail"] == "Generated by LLM (Groq)" for e in report_events)


def test_report_generated_event_labels_a_fallback_report_as_deterministic(client, db_session):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 31))
    case_id = score_res.json()["case_id"]
    db_session.add(m.LlmReport(case_id=case_id, report_text="x", model_name="fallback", source="fallback"))
    db_session.commit()

    trail = client.get(f"/cases/{case_id}/audit-trail")
    report_events = [e for e in trail.json()["events"] if e["event_type"] == "report_generated"]
    assert len(report_events) == 2
    assert all(e["detail"] == "Deterministic fallback - no LLM call" for e in report_events)
    assert all(e["actor"] == "AI" for e in report_events)


def test_automation_proposed_event_is_always_labeled_rule_based(client, db_session):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 32))
    case_id, txn_id = score_res.json()["case_id"], score_res.json()["txn_id"]
    db_session.add(m.AutoBlockLog(
        transaction_id=txn_id, case_id=case_id,
        triggered_conditions={"eligible": True, "direction": "fraud", "reason": [], "failed_gates": []},
        review_status="proposed",
    ))
    db_session.commit()

    trail = client.get(f"/cases/{case_id}/audit-trail")
    proposed_event = next(e for e in trail.json()["events"] if e["event_type"] == "automation_proposed")
    assert proposed_event["detail"] == "Rule-based gate evaluation - no LLM involved"
    assert proposed_event["actor"] == "AI"


def test_bare_open_case_with_no_decisions_no_report_no_precedent(client):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 20))
    case_id = score_res.json()["case_id"]

    trail = client.get(f"/cases/{case_id}/audit-trail")
    assert trail.status_code == 200
    events = trail.json()["events"]

    event_types = [e["event_type"] for e in events]
    assert "scored" in event_types
    assert "case_opened" in event_types
    assert "decision" not in event_types
    assert "precedent_indexed" not in event_types
    assert all(e["anomaly_flags"] == [] for e in events)


def test_single_decision_case_has_no_rapid_redecision_flag(client):
    score_res = client.post("/score", json=_ghost_dest_payload(BASE_AMOUNT + 21))
    case_id = score_res.json()["case_id"]
    client.post(f"/cases/{case_id}/decision", json={
        "action_taken": "confirm_fraud", "analyst_reason_code": "test", "analyst_note": "single decision",
    })

    trail = client.get(f"/cases/{case_id}/audit-trail")
    decision_events = [e for e in trail.json()["events"] if e["event_type"] == "decision"]
    assert len(decision_events) == 1
    assert "rapid_redecision" not in decision_events[0]["anomaly_flags"]


def test_audit_trail_404s_for_unknown_case(client):
    res = client.get("/cases/999999999/audit-trail")
    assert res.status_code == 404
