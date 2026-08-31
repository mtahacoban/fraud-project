from __future__ import annotations

from types import SimpleNamespace

from backend import db_models as m
from backend.automation import evaluate_auto_decision, gate_bottleneck_stats, reject_rate_stats, shadow_agreement_stats

_DIRECTION_DETAIL = {
    "passed": "direction=fraud - automatable under current policy",
    "clean_blocked": "direction=clean - auto-clean is disabled (policy.auto_clean_enabled=false)",
    "escalate": "direction=escalate - escalation is never automated, always human-routed",
    "no_suggestion": "no Precedent Analysis suggestion (insufficient precedent) - nothing to automate",
}


def make_policy(**overrides):
    defaults = dict(
        fraud_similarity_threshold=0.95,
        min_precedent_count=10,
        min_consensus_ratio=0.90,
        min_calibrated_proba=0.95,
        hard_rule_required=False,
        auto_clean_enabled=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def eligible_precedent_summary(**overrides):
    defaults = dict(
        suggested_decision="confirm_fraud",
        avg_similarity=0.99,
        precedent_count=15,
        consensus_ratio=1.0,
    )
    defaults.update(overrides)
    return defaults


def test_evaluate_auto_decision_is_deterministic_across_repeated_calls():
    policy = make_policy()
    summary = eligible_precedent_summary()
    first = evaluate_auto_decision(summary, 1.0, [], policy, [])
    second = evaluate_auto_decision(summary, 1.0, [], policy, [])
    assert first == second


def test_all_six_gates_pass_for_a_clean_eligible_case():
    policy = make_policy()
    summary = eligible_precedent_summary()
    result = evaluate_auto_decision(summary, calibrated_proba=1.0, hard_rule_hits=[], policy=policy, clean_rule_hits=[])
    assert result["eligible"] is True
    assert result["direction"] == "fraud"
    assert result["failed_gates"] == []
    assert len(result["gates"]) == 6
    assert all(g["passed"] for g in result["gates"])


def test_gates_read_thresholds_from_the_policy_object_not_hardcoded():
    strict_policy = make_policy(fraud_similarity_threshold=0.999)
    summary = eligible_precedent_summary(avg_similarity=0.99)
    result = evaluate_auto_decision(summary, 1.0, [], strict_policy, [])
    assert result["eligible"] is False
    sim_gate = next(g for g in result["gates"] if g["gate"] == "similarity")
    assert sim_gate["passed"] is False
    assert sim_gate["threshold"] == 0.999


def test_direction_gate_fails_when_suggestion_is_none():
    policy = make_policy()
    summary = eligible_precedent_summary(suggested_decision=None)
    result = evaluate_auto_decision(summary, 1.0, [], policy, [])
    assert result["eligible"] is False
    assert result["direction"] is None
    gate = next(g for g in result["gates"] if g["gate"] == "direction_automatable")
    assert gate["passed"] is False
    assert gate["actual"] is None and gate["threshold"] is None


def test_direction_gate_fails_for_escalate_suggestion():
    policy = make_policy()
    summary = eligible_precedent_summary(suggested_decision="escalate")
    result = evaluate_auto_decision(summary, 1.0, [], policy, [])
    assert result["eligible"] is False
    assert result["direction"] == "escalate"


def test_direction_gate_fails_for_clean_when_auto_clean_disabled():
    policy = make_policy(auto_clean_enabled=False)
    summary = eligible_precedent_summary(suggested_decision="approve_clean")
    result = evaluate_auto_decision(summary, 1.0, [], policy, [])
    assert result["eligible"] is False


def test_similarity_gate_fails_below_threshold():
    policy = make_policy()
    summary = eligible_precedent_summary(avg_similarity=0.50)
    result = evaluate_auto_decision(summary, 1.0, [], policy, [])
    gate = next(g for g in result["gates"] if g["gate"] == "similarity")
    assert gate["passed"] is False
    assert gate["actual"] == 0.50
    assert result["eligible"] is False


def test_precedent_count_gate_fails_below_threshold():
    policy = make_policy()
    summary = eligible_precedent_summary(precedent_count=3)
    result = evaluate_auto_decision(summary, 1.0, [], policy, [])
    gate = next(g for g in result["gates"] if g["gate"] == "precedent_count")
    assert gate["passed"] is False
    assert gate["actual"] == 3
    assert result["eligible"] is False


def test_consensus_gate_fails_below_threshold():
    policy = make_policy()
    summary = eligible_precedent_summary(consensus_ratio=0.60)
    result = evaluate_auto_decision(summary, 1.0, [], policy, [])
    gate = next(g for g in result["gates"] if g["gate"] == "consensus")
    assert gate["passed"] is False
    assert result["eligible"] is False


def test_calibrated_proba_gate_fails_below_threshold():
    policy = make_policy()
    summary = eligible_precedent_summary()
    result = evaluate_auto_decision(summary, calibrated_proba=0.50, hard_rule_hits=[], policy=policy, clean_rule_hits=[])
    gate = next(g for g in result["gates"] if g["gate"] == "calibrated_proba")
    assert gate["passed"] is False
    assert gate["actual"] == 0.50
    assert result["eligible"] is False


def test_calibrated_proba_gate_boundary_is_inclusive():
    policy = make_policy(min_calibrated_proba=0.95)
    summary = eligible_precedent_summary()
    at_boundary = evaluate_auto_decision(summary, 0.95, [], policy, [])
    just_below = evaluate_auto_decision(summary, 0.9499999, [], policy, [])
    assert next(g for g in at_boundary["gates"] if g["gate"] == "calibrated_proba")["passed"] is True
    assert next(g for g in just_below["gates"] if g["gate"] == "calibrated_proba")["passed"] is False


def test_hard_rule_gate_fails_when_clean_confirming_rule_contradicts_fraud_direction():
    policy = make_policy(hard_rule_required=False)
    summary = eligible_precedent_summary()
    result = evaluate_auto_decision(summary, 1.0, [], policy, clean_rule_hits=["clean_confirmed"])
    gate = next(g for g in result["gates"] if g["gate"] == "hard_rule_conflict")
    assert gate["passed"] is False
    assert result["eligible"] is False


def test_hard_rule_gate_passes_by_default_with_no_conflict():
    policy = make_policy(hard_rule_required=False)
    summary = eligible_precedent_summary()
    result = evaluate_auto_decision(summary, 1.0, hard_rule_hits=[], policy=policy, clean_rule_hits=[])
    gate = next(g for g in result["gates"] if g["gate"] == "hard_rule_conflict")
    assert gate["passed"] is True


def test_hard_rule_gate_requires_a_hit_when_hard_rule_required_true():
    policy = make_policy(hard_rule_required=True)
    summary = eligible_precedent_summary()
    without_hit = evaluate_auto_decision(summary, 1.0, hard_rule_hits=[], policy=policy, clean_rule_hits=[])
    with_hit = evaluate_auto_decision(summary, 1.0, hard_rule_hits=["ghost_destination"], policy=policy, clean_rule_hits=[])
    assert next(g for g in without_hit["gates"] if g["gate"] == "hard_rule_conflict")["passed"] is False
    assert next(g for g in with_hit["gates"] if g["gate"] == "hard_rule_conflict")["passed"] is True


def test_reason_and_failed_gates_stay_string_lists_alongside_structured_gates():
    policy = make_policy()
    summary = eligible_precedent_summary(avg_similarity=0.10)
    result = evaluate_auto_decision(summary, 1.0, [], policy, [])
    assert isinstance(result["reason"], list) and all(isinstance(r, str) for r in result["reason"])
    assert isinstance(result["failed_gates"], list) and all(isinstance(r, str) for r in result["failed_gates"])
    assert len(result["reason"]) == 6
    assert isinstance(result["gates"], list) and all(isinstance(g, dict) for g in result["gates"])
    assert len(result["gates"]) == 6


def _score_ghost_dest_transfer(client, amount):
    payload = {
        "step": 3, "type": "TRANSFER", "amount": amount,
        "oldbalanceOrg": amount, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }
    res = client.post("/score", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["case_id"] is not None
    return body["txn_id"], body["case_id"]


def test_gate_bottleneck_stats_aggregates_both_triggered_conditions_shapes(client, db_session):
    before = {g["gate"]: g for g in gate_bottleneck_stats(db_session)["gates"]}
    before_n = gate_bottleneck_stats(db_session)["n_evaluations"]

    txn_a, case_a = _score_ghost_dest_transfer(client, 7_531_100.0)
    txn_b, case_b = _score_ghost_dest_transfer(client, 7_531_200.0)

    db_session.add(m.AutoBlockLog(
        transaction_id=txn_a, case_id=case_a, review_status="shadow",
        triggered_conditions={
            "eligible": False, "direction": "fraud",
            "gates": [
                {"gate": "direction_automatable", "passed": True, "actual": None, "threshold": None, "detail": "x"},
                {"gate": "similarity", "passed": True, "actual": 0.99, "threshold": 0.95, "detail": "x"},
                {"gate": "precedent_count", "passed": False, "actual": 2, "threshold": 10, "detail": "x"},
                {"gate": "consensus", "passed": True, "actual": 1.0, "threshold": 0.90, "detail": "x"},
                {"gate": "calibrated_proba", "passed": True, "actual": 1.0, "threshold": 0.95, "detail": "x"},
                {"gate": "hard_rule_conflict", "passed": True, "actual": None, "threshold": None, "detail": "x"},
            ],
        },
    ))

    reason = [
        "direction=fraud - automatable under current policy",
        "avg_similarity=0.10 < 0.95 required",
        "precedent_count=15 >= 10 required",
        "consensus_ratio=1.0000 >= 0.90 required",
        "calibrated_proba=1.0000 >= 0.95 required",
        "hard_rule_required=false, no clean-confirming rule conflict - passes by default",
    ]
    db_session.add(m.AutoBlockLog(
        transaction_id=txn_b, case_id=case_b, review_status="shadow",
        triggered_conditions={
            "eligible": False, "direction": "fraud",
            "reason": reason,
            "failed_gates": [f"similarity: {reason[1]}"],
        },
    ))
    db_session.commit()

    after = {g["gate"]: g for g in gate_bottleneck_stats(db_session)["gates"]}
    after_n = gate_bottleneck_stats(db_session)["n_evaluations"]

    assert after_n == before_n + 2
    assert after["direction_automatable"]["passed_count"] == before["direction_automatable"]["passed_count"] + 2
    assert after["similarity"]["passed_count"] == before["similarity"]["passed_count"] + 1
    assert after["similarity"]["failed_count"] == before["similarity"]["failed_count"] + 1
    assert after["precedent_count"]["passed_count"] == before["precedent_count"]["passed_count"] + 1
    assert after["precedent_count"]["failed_count"] == before["precedent_count"]["failed_count"] + 1
    assert after["hard_rule_conflict"]["passed_count"] == before["hard_rule_conflict"]["passed_count"] + 2
    assert all(g["total"] == g["passed_count"] + g["failed_count"] for g in after.values())


def test_gate_bottleneck_stats_skips_rows_with_an_unrecognized_reason_shape(client, db_session):
    before_n = gate_bottleneck_stats(db_session)["n_evaluations"]

    txn_id, case_id = _score_ghost_dest_transfer(client, 7_531_300.0)
    db_session.add(m.AutoBlockLog(
        transaction_id=txn_id, case_id=case_id, review_status="shadow",
        triggered_conditions={"eligible": False, "direction": None, "reason": ["only one entry"], "failed_gates": []},
    ))
    db_session.commit()

    after_n = gate_bottleneck_stats(db_session)["n_evaluations"]
    assert after_n == before_n + 1

    stats = gate_bottleneck_stats(db_session)
    total_gate_evaluations = sum(g["total"] for g in stats["gates"])
    assert total_gate_evaluations < after_n * 6


def test_direction_breakdown_classifies_all_four_templates_correctly(client, db_session):
    before = gate_bottleneck_stats(db_session)["direction_breakdown"]

    rows = [
        ("passed", True, "gates"),
        ("clean_blocked", False, "reason"),
        ("escalate", False, "gates"),
        ("no_suggestion", False, "reason"),
    ]
    for i, (bucket, direction_passed, shape) in enumerate(rows):
        payload = {
            "step": 3, "type": "TRANSFER", "amount": 6_642_000.0 + i,
            "oldbalanceOrg": 6_642_000.0 + i, "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        }
        res = client.post("/score", json=payload)
        assert res.status_code == 200
        body = res.json()
        detail = _DIRECTION_DETAIL[bucket]

        if shape == "gates":
            tc = {
                "eligible": False, "direction": None,
                "gates": [
                    {"gate": "direction_automatable", "passed": direction_passed, "actual": None, "threshold": None, "detail": detail},
                    {"gate": "similarity", "passed": True, "actual": 0.99, "threshold": 0.95, "detail": "x"},
                    {"gate": "precedent_count", "passed": True, "actual": 12, "threshold": 10, "detail": "x"},
                    {"gate": "consensus", "passed": True, "actual": 1.0, "threshold": 0.90, "detail": "x"},
                    {"gate": "calibrated_proba", "passed": True, "actual": 1.0, "threshold": 0.95, "detail": "x"},
                    {"gate": "hard_rule_conflict", "passed": True, "actual": None, "threshold": None, "detail": "x"},
                ],
            }
        else:
            reason = [
                detail,
                "avg_similarity=0.99 >= 0.95 required",
                "precedent_count=12 >= 10 required",
                "consensus_ratio=1.0000 >= 0.90 required",
                "calibrated_proba=1.0000 >= 0.95 required",
                "hard_rule_required=false, no clean-confirming rule conflict - passes by default",
            ]
            failed_gates = [] if direction_passed else [f"direction_automatable: {detail}"]
            tc = {"eligible": False, "direction": None, "reason": reason, "failed_gates": failed_gates}

        db_session.add(m.AutoBlockLog(
            transaction_id=body["txn_id"], case_id=body["case_id"], review_status="shadow",
            triggered_conditions=tc,
        ))
    db_session.commit()

    after = gate_bottleneck_stats(db_session)["direction_breakdown"]
    assert after["passed"] == before["passed"] + 1
    assert after["clean_blocked"] == before["clean_blocked"] + 1
    assert after["escalate"] == before["escalate"] + 1
    assert after["no_suggestion"] == before["no_suggestion"] + 1
    assert after["unrecognized"] == before["unrecognized"]


def test_direction_breakdown_sums_to_direction_automatable_gates_total(db_session):
    stats = gate_bottleneck_stats(db_session)
    direction_gate = next(g for g in stats["gates"] if g["gate"] == "direction_automatable")
    assert sum(stats["direction_breakdown"].values()) == direction_gate["total"]
    assert direction_gate["total"] <= stats["n_evaluations"]


def test_direction_breakdown_puts_an_unrecognized_detail_in_its_own_bucket(client, db_session):
    before = gate_bottleneck_stats(db_session)["direction_breakdown"]

    payload = {
        "step": 3, "type": "TRANSFER", "amount": 6_643_000.0,
        "oldbalanceOrg": 6_643_000.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }
    res = client.post("/score", json=payload)
    assert res.status_code == 200
    body = res.json()

    db_session.add(m.AutoBlockLog(
        transaction_id=body["txn_id"], case_id=body["case_id"], review_status="shadow",
        triggered_conditions={
            "eligible": False, "direction": None,
            "gates": [
                {"gate": "direction_automatable", "passed": False, "actual": None, "threshold": None, "detail": "some future template not in the 4 known ones"},
                {"gate": "similarity", "passed": True, "actual": 0.99, "threshold": 0.95, "detail": "x"},
                {"gate": "precedent_count", "passed": True, "actual": 12, "threshold": 10, "detail": "x"},
                {"gate": "consensus", "passed": True, "actual": 1.0, "threshold": 0.90, "detail": "x"},
                {"gate": "calibrated_proba", "passed": True, "actual": 1.0, "threshold": 0.95, "detail": "x"},
                {"gate": "hard_rule_conflict", "passed": True, "actual": None, "threshold": None, "detail": "x"},
            ],
        },
    ))
    db_session.commit()

    after = gate_bottleneck_stats(db_session)["direction_breakdown"]
    assert after["unrecognized"] == before["unrecognized"] + 1
    assert after["no_suggestion"] == before["no_suggestion"]


def _make_policy_version(db_session, **overrides):
    defaults = dict(version="test-reject-rate", mode="propose", circuit_breaker_min_confirmations=1)
    defaults.update(overrides)
    policy = m.AutomationPolicyVersion(**defaults)
    db_session.add(policy)
    db_session.commit()
    db_session.refresh(policy)
    return policy


def test_reject_rate_stats_returns_the_correct_case_ids_for_confirmed_rejected_pending(client, db_session):
    policy = _make_policy_version(db_session)

    txn_a, case_a = _score_ghost_dest_transfer(client, 7_544_100.0)
    txn_b, case_b = _score_ghost_dest_transfer(client, 7_544_200.0)
    txn_c, case_c = _score_ghost_dest_transfer(client, 7_544_300.0)

    db_session.add_all([
        m.AutoBlockLog(transaction_id=txn_a, case_id=case_a, policy_version_id=policy.id, review_status="confirmed", triggered_conditions={}),
        m.AutoBlockLog(transaction_id=txn_b, case_id=case_b, policy_version_id=policy.id, review_status="rejected", triggered_conditions={}),
        m.AutoBlockLog(transaction_id=txn_c, case_id=case_c, policy_version_id=policy.id, review_status="proposed", triggered_conditions={}),
    ])
    db_session.commit()

    stats = reject_rate_stats(db_session, policy_version_id=policy.id)
    assert stats["confirmed"] == 1
    assert stats["rejected"] == 1
    assert stats["pending"] == 1
    assert stats["n"] == 2
    assert stats["confirmed_case_ids"] == [case_a]
    assert stats["rejected_case_ids"] == [case_b]
    assert stats["pending_case_ids"] == [case_c]


def test_reject_rate_stats_case_id_lists_are_empty_when_no_rows_exist_for_the_policy(db_session):
    policy = _make_policy_version(db_session)
    stats = reject_rate_stats(db_session, policy_version_id=policy.id)
    assert stats["confirmed"] == 0 and stats["rejected"] == 0 and stats["pending"] == 0
    assert stats["confirmed_case_ids"] == []
    assert stats["rejected_case_ids"] == []
    assert stats["pending_case_ids"] == []


def test_reject_rate_stats_case_id_lists_stay_scoped_to_their_own_policy_version(client, db_session):
    policy_a = _make_policy_version(db_session, version="test-reject-rate-a")
    policy_b = _make_policy_version(db_session, version="test-reject-rate-b")

    txn, case = _score_ghost_dest_transfer(client, 7_544_400.0)
    db_session.add(m.AutoBlockLog(
        transaction_id=txn, case_id=case, policy_version_id=policy_a.id,
        review_status="confirmed", triggered_conditions={},
    ))
    db_session.commit()

    stats_a = reject_rate_stats(db_session, policy_version_id=policy_a.id)
    stats_b = reject_rate_stats(db_session, policy_version_id=policy_b.id)
    assert stats_a["confirmed_case_ids"] == [case]
    assert stats_b["confirmed_case_ids"] == []


def test_shadow_agreement_stats_returns_the_correct_case_ids_for_each_bucket(client, db_session):
    before = shadow_agreement_stats(db_session)

    txn_a, case_a = _score_ghost_dest_transfer(client, 7_545_100.0)
    res = client.post(f"/cases/{case_a}/decision", json={"action_taken": "confirm_fraud", "analyst_reason_code": "test"})
    assert res.status_code == 200

    txn_b, case_b = _score_ghost_dest_transfer(client, 7_545_200.0)
    res = client.post(f"/cases/{case_b}/decision", json={"action_taken": "approve_clean", "analyst_reason_code": "test"})
    assert res.status_code == 200

    txn_c, case_c = _score_ghost_dest_transfer(client, 7_545_300.0)

    eligible_tc = {"eligible": True, "direction": "fraud", "gates": []}
    db_session.add_all([
        m.AutoBlockLog(transaction_id=txn_a, case_id=case_a, review_status="shadow", triggered_conditions=eligible_tc),
        m.AutoBlockLog(transaction_id=txn_b, case_id=case_b, review_status="shadow", triggered_conditions=eligible_tc),
        m.AutoBlockLog(transaction_id=txn_c, case_id=case_c, review_status="shadow", triggered_conditions={"eligible": False, "direction": None}),
    ])
    db_session.commit()

    after = shadow_agreement_stats(db_session)
    assert after["n_shadow_evaluations"] >= before["n_shadow_evaluations"] + 3
    assert case_a in after["would_have_confirmed_correctly_case_ids"]
    assert case_b in after["would_have_been_wrong_case_ids"]
    assert case_c in after["not_eligible_case_ids"]
    assert case_a in after["eligible_case_ids"] and case_b in after["eligible_case_ids"]
    assert case_c not in after["eligible_case_ids"]
    assert {case_a, case_b, case_c} <= set(after["shadow_evaluation_case_ids"])


def test_shadow_agreement_stats_case_id_list_lengths_match_their_own_counts(db_session):
    stats = shadow_agreement_stats(db_session)
    assert len(stats["shadow_evaluation_case_ids"]) == stats["n_shadow_evaluations"]
    assert len(stats["not_eligible_case_ids"]) == stats["n_not_eligible"]
    assert len(stats["eligible_case_ids"]) == stats["n_eligible"]
    assert len(stats["would_have_confirmed_correctly_case_ids"]) == stats["would_have_confirmed_correctly"]
    assert len(stats["would_have_been_wrong_case_ids"]) == stats["would_have_been_wrong"]
