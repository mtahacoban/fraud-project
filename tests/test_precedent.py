from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from backend import db_models as m
from backend.precedent import KNOWN_AML_REASON_CODES, RISK_BANDS, find_precedents, summarize_precedents


@pytest.fixture(autouse=True)
def clean_precedent_index(db_session):
    db_session.query(m.PrecedentIndex).delete()
    db_session.commit()
    yield
    db_session.query(m.PrecedentIndex).delete()
    db_session.commit()


def _make_case(db_session, *, amount, step, risk_band, suffix):
    txn = m.Transaction(
        step=step, type="TRANSFER", amount=amount,
        oldbalance_org=amount, newbalance_orig=0.0,
        oldbalance_dest=0.0, newbalance_dest=0.0,
        error_balance_orig=0.0, error_balance_dest=amount,
        step_hour=step % 24, is_transfer=1, is_cashout=0,
        source="test", is_demo=True,
    )
    db_session.add(txn)
    db_session.flush()
    score = m.Score(
        transaction_id=txn.id, ml_score=87.0, rule_score=95.0, hybrid_score=87.0,
        risk_band=risk_band, calibrated_proba=0.98, model_version="test",
    )
    db_session.add(score)
    case = m.Case(transaction_id=txn.id, status="OPEN", priority="HIGH")
    db_session.add(case)
    db_session.flush()
    return case


def _raw_vector(amount, step_hour, err_orig, err_dest, is_transfer, is_cashout, risk_band):
    band_one_hot = [1.0 if risk_band == b else 0.0 for b in RISK_BANDS]
    return np.array([amount, step_hour, err_orig, err_dest, is_transfer, is_cashout] + band_one_hot, dtype=float)


def _fit_background_scaler():
    rng = np.random.default_rng(seed=42)
    background = []
    for _ in range(40):
        amount = float(rng.uniform(50, 2_000_000))
        step_hour = float(rng.integers(0, 24))
        is_transfer = float(rng.integers(0, 2))
        is_cashout = 1.0 - is_transfer
        err_orig = float(rng.uniform(-1000, 1000))
        err_dest = float(rng.uniform(-1000, 1_000_000))
        band = RISK_BANDS[int(rng.integers(0, 3))]
        background.append(_raw_vector(amount, step_hour, err_orig, err_dest, is_transfer, is_cashout, band))
    return StandardScaler().fit(np.array(background))


def _seed_pool(db_session, scaler, entries):
    for i, (raw, label) in enumerate(entries):
        scaled = scaler.transform(raw.reshape(1, -1))[0]
        db_session.add(m.PrecedentIndex(
            transaction_id=9000 + i, case_id=9000 + i,
            feature_vector=scaled.tolist(), label=label,
            created_at=datetime.now(timezone.utc),
        ))
    db_session.commit()



def test_find_precedents_is_deterministic_across_repeated_calls(db_session):
    query_case = _make_case(db_session, amount=500_000.0, step=3, risk_band="RED", suffix="q")
    raw_pool = [_raw_vector(500_000.0 + i * 1000, 3, 0.0, 500_000.0, 1.0, 0.0, "RED") for i in range(6)]
    scaler = _fit_background_scaler()
    _seed_pool(db_session, scaler, [(v, "confirm_fraud") for v in raw_pool])

    first = find_precedents(query_case, db_session, scaler, k=15)
    second = find_precedents(query_case, db_session, scaler, k=15)
    assert first == second



def test_find_precedents_returns_empty_list_on_cold_start(db_session):
    query_case = _make_case(db_session, amount=100.0, step=1, risk_band="GRAY", suffix="cold")
    scaler = _fit_background_scaler()
    result = find_precedents(query_case, db_session, scaler, k=15)
    assert result == []


def test_find_precedents_with_k_larger_than_the_pool_returns_all_of_it(db_session):
    query_case = _make_case(db_session, amount=500_000.0, step=3, risk_band="RED", suffix="onepool")
    scaler = _fit_background_scaler()
    only_neighbor = _raw_vector(500_000.0, 3, 0.0, 500_000.0, 1.0, 0.0, "RED")
    _seed_pool(db_session, scaler, [(only_neighbor, "confirm_fraud")])

    result = find_precedents(query_case, db_session, scaler, k=15)
    assert len(result) == 1
    assert result[0]["analyst_decision"] == "confirm_fraud"


def test_find_precedents_never_returns_the_query_case_itself(db_session):
    query_case = _make_case(db_session, amount=500_000.0, step=3, risk_band="RED", suffix="self")
    raw = _raw_vector(500_000.0, 3, 0.0, 500_000.0, 1.0, 0.0, "RED")
    scaler = _fit_background_scaler()
    db_session.add(m.PrecedentIndex(
        transaction_id=query_case.transaction_id, case_id=query_case.id,
        feature_vector=scaler.transform(raw.reshape(1, -1))[0].tolist(), label="confirm_fraud",
    ))
    db_session.commit()
    result = find_precedents(query_case, db_session, scaler, k=15)
    assert all(n["case_id"] != query_case.id for n in result)


def test_find_precedents_returns_similarity_and_label_per_neighbor(db_session):
    query_case = _make_case(db_session, amount=500_000.0, step=3, risk_band="RED", suffix="nb")
    raw_pool = [_raw_vector(500_000.0 + i * 1000, 3, 0.0, 500_000.0, 1.0, 0.0, "RED") for i in range(6)]
    scaler = _fit_background_scaler()
    _seed_pool(db_session, scaler, [(v, "confirm_fraud") for v in raw_pool])

    result = find_precedents(query_case, db_session, scaler, k=15)
    assert len(result) == 6
    for n in result:
        assert set(n.keys()) == {"case_id", "similarity", "analyst_decision"}
        assert n["analyst_decision"] == "confirm_fraud"
        assert 0.0 <= n["similarity"] <= 1.0001


def test_find_precedents_ranks_closer_vectors_first(db_session):
    query_case = _make_case(db_session, amount=500_000.0, step=3, risk_band="RED", suffix="rank")
    close = _raw_vector(500_000.0, 3, 0.0, 500_000.0, 1.0, 0.0, "RED")
    far = _raw_vector(5_000.0, 14, 0.0, 5_000.0, 0.0, 1.0, "GREEN")
    scaler = _fit_background_scaler()
    _seed_pool(db_session, scaler, [(far, "approve_clean"), (close, "confirm_fraud")])

    result = find_precedents(query_case, db_session, scaler, k=15)
    assert result[0]["analyst_decision"] == "confirm_fraud"
    assert result[0]["similarity"] > result[1]["similarity"]



def test_summary_suggests_when_all_three_gates_clear(db_session):
    query_case = _make_case(db_session, amount=500_000.0, step=3, risk_band="RED", suffix="sug")
    raw_pool = [_raw_vector(500_000.0 + i * 500, 3, 0.0, 500_000.0, 1.0, 0.0, "RED") for i in range(6)]
    scaler = _fit_background_scaler()
    _seed_pool(db_session, scaler, [(v, "confirm_fraud") for v in raw_pool])

    neighbors = find_precedents(query_case, db_session, scaler, k=15)
    summary = summarize_precedents(neighbors)
    assert summary["suggested_decision"] == "confirm_fraud"
    assert summary["precedent_count"] == 6
    assert summary["consensus_ratio"] == 1.0
    assert summary["avg_similarity"] >= 0.85
    assert summary["note"] is None


def test_summary_withholds_suggestion_below_min_precedent_count(db_session):
    query_case = _make_case(db_session, amount=500_000.0, step=3, risk_band="RED", suffix="few")
    raw_pool = [_raw_vector(500_000.0 + i * 500, 3, 0.0, 500_000.0, 1.0, 0.0, "RED") for i in range(3)]
    scaler = _fit_background_scaler()
    _seed_pool(db_session, scaler, [(v, "confirm_fraud") for v in raw_pool])

    neighbors = find_precedents(query_case, db_session, scaler, k=15)
    summary = summarize_precedents(neighbors)
    assert summary["suggested_decision"] is None
    assert summary["note"] is not None


def test_summary_withholds_suggestion_below_min_consensus(db_session):
    query_case = _make_case(db_session, amount=500_000.0, step=3, risk_band="RED", suffix="split")
    raw_pool = [_raw_vector(500_000.0 + i * 500, 3, 0.0, 500_000.0, 1.0, 0.0, "RED") for i in range(6)]
    scaler = _fit_background_scaler()
    labels = ["confirm_fraud", "approve_clean"] * 3
    _seed_pool(db_session, scaler, list(zip(raw_pool, labels)))

    neighbors = find_precedents(query_case, db_session, scaler, k=15)
    summary = summarize_precedents(neighbors)
    assert summary["consensus_ratio"] < 0.70
    assert summary["suggested_decision"] is None


def test_summary_withholds_suggestion_below_min_avg_similarity(db_session):
    query_case = _make_case(db_session, amount=500_000.0, step=3, risk_band="RED", suffix="dissim")
    dissimilar = [
        _raw_vector(5.0, 20, 0.0, 5.0, 0.0, 1.0, "GREEN"),
        _raw_vector(6.0, 21, 0.0, 6.0, 0.0, 1.0, "GREEN"),
        _raw_vector(7.0, 22, 0.0, 7.0, 0.0, 1.0, "GREEN"),
        _raw_vector(8.0, 23, 0.0, 8.0, 0.0, 1.0, "GREEN"),
        _raw_vector(9.0, 20, 0.0, 9.0, 0.0, 1.0, "GREEN"),
    ]
    scaler = _fit_background_scaler()
    _seed_pool(db_session, scaler, [(v, "confirm_fraud") for v in dissimilar])

    neighbors = find_precedents(query_case, db_session, scaler, k=15)
    summary = summarize_precedents(neighbors)
    if summary["precedent_count"] >= 5:
        assert summary["avg_similarity"] < 0.85 or summary["suggested_decision"] is None
    assert summary["suggested_decision"] is None



def _neighbor(similarity, decision, rule_hits, reason_code=None):
    return {
        "case_id": 0, "similarity": similarity, "analyst_decision": decision,
        "rule_hits": rule_hits, "analyst_reason_code": reason_code,
    }


def test_common_patterns_counts_rule_frequency_across_counted_precedents():
    neighbors = [
        _neighbor(0.95, "confirm_fraud", ["ghost_destination", "drain_account", "high_amount_transfer"]),
        _neighbor(0.94, "confirm_fraud", ["ghost_destination", "drain_account"]),
        _neighbor(0.90, "confirm_fraud", ["ghost_destination", "drain_account", "night_transaction"]),
        _neighbor(0.85, "confirm_fraud", ["drain_account"]),
        _neighbor(0.80, "confirm_fraud", ["ghost_destination"]),
    ]
    summary = summarize_precedents(neighbors)
    assert summary["precedent_count"] == 5
    assert summary["common_patterns"] == [
        {"rule": "drain_account", "count": 4, "total": 5},
        {"rule": "ghost_destination", "count": 4, "total": 5},
    ]


def test_common_patterns_only_top_two_even_with_more_distinct_rules():
    neighbors = [
        _neighbor(0.95, "confirm_fraud", ["ghost_destination"] * 1),
        _neighbor(0.94, "confirm_fraud", ["ghost_destination", "drain_account"]),
        _neighbor(0.90, "confirm_fraud", ["ghost_destination", "drain_account", "night_transaction"]),
        _neighbor(0.85, "confirm_fraud", ["ghost_destination", "high_amount_transfer"]),
        _neighbor(0.80, "confirm_fraud", ["ghost_destination"]),
    ]
    summary = summarize_precedents(neighbors)
    assert len(summary["common_patterns"]) == 2
    assert summary["common_patterns"][0] == {"rule": "ghost_destination", "count": 5, "total": 5}


def test_common_patterns_counts_a_repeated_rule_name_once_per_case():
    neighbors = [
        _neighbor(0.95, "confirm_fraud", ["ghost_destination", "ghost_destination"]),
        _neighbor(0.90, "confirm_fraud", ["drain_account"]),
        _neighbor(0.85, "confirm_fraud", ["drain_account"]),
        _neighbor(0.80, "confirm_fraud", ["drain_account"]),
        _neighbor(0.75, "confirm_fraud", ["drain_account"]),
    ]
    summary = summarize_precedents(neighbors)
    ghost_entry = next(p for p in summary["common_patterns"] if p["rule"] == "ghost_destination")
    assert ghost_entry["count"] == 1


def test_common_patterns_empty_when_neighbors_carry_no_rule_hits():
    neighbors = [
        _neighbor(0.95, "confirm_fraud", []),
        _neighbor(0.90, "confirm_fraud", []),
        _neighbor(0.85, "confirm_fraud", []),
        _neighbor(0.80, "confirm_fraud", []),
        _neighbor(0.75, "confirm_fraud", []),
    ]
    summary = summarize_precedents(neighbors)
    assert summary["precedent_count"] == 5
    assert summary["common_patterns"] == []


def test_common_patterns_empty_when_no_precedent_at_all():
    summary = summarize_precedents([])
    assert summary["precedent_count"] == 0
    assert summary["common_patterns"] == []


def test_common_patterns_ignores_neighbors_below_the_similarity_floor():
    neighbors = [
        _neighbor(0.95, "confirm_fraud", ["ghost_destination"]),
        _neighbor(0.90, "confirm_fraud", ["ghost_destination"]),
        _neighbor(0.85, "confirm_fraud", ["ghost_destination"]),
        _neighbor(0.80, "confirm_fraud", ["ghost_destination"]),
        _neighbor(0.75, "confirm_fraud", ["ghost_destination"]),
        _neighbor(0.10, "confirm_fraud", ["night_transaction"]),
    ]
    summary = summarize_precedents(neighbors)
    assert summary["precedent_count"] == 5
    assert summary["common_patterns"] == [{"rule": "ghost_destination", "count": 5, "total": 5}]



def test_known_aml_reason_codes_has_exactly_the_18_documented_codes():
    assert len(KNOWN_AML_REASON_CODES) == 18
    assert "account_takeover" in KNOWN_AML_REASON_CODES
    assert "money_laundering" in KNOWN_AML_REASON_CODES
    assert "false_positive" in KNOWN_AML_REASON_CODES


def test_common_reason_codes_excludes_seed_ground_truth_paysim():
    neighbors = [
        _neighbor(0.95, "confirm_fraud", ["ghost_destination"], reason_code="seed_ground_truth_paysim"),
        _neighbor(0.90, "confirm_fraud", ["ghost_destination"], reason_code="seed_ground_truth_paysim"),
        _neighbor(0.85, "confirm_fraud", ["ghost_destination"], reason_code="seed_ground_truth_paysim"),
        _neighbor(0.80, "confirm_fraud", ["ghost_destination"], reason_code="seed_ground_truth_paysim"),
        _neighbor(0.75, "confirm_fraud", ["ghost_destination"], reason_code="seed_ground_truth_paysim"),
    ]
    summary = summarize_precedents(neighbors)
    assert summary["common_reason_codes"] == []


def test_common_reason_codes_excludes_other_system_and_blank_tags():
    neighbors = [
        _neighbor(0.95, "confirm_fraud", [], reason_code="audit_trail_test"),
        _neighbor(0.90, "confirm_fraud", [], reason_code="ai2_proposal_confirmed"),
        _neighbor(0.85, "confirm_fraud", [], reason_code=None),
        _neighbor(0.80, "confirm_fraud", [], reason_code="some_future_tag_nobody_anticipated"),
        _neighbor(0.75, "confirm_fraud", [], reason_code=""),
    ]
    summary = summarize_precedents(neighbors)
    assert summary["common_reason_codes"] == []


def test_common_reason_codes_counts_real_aml_codes():
    neighbors = [
        _neighbor(0.95, "confirm_fraud", [], reason_code="account_takeover"),
        _neighbor(0.90, "confirm_fraud", [], reason_code="account_takeover"),
        _neighbor(0.85, "confirm_fraud", [], reason_code="account_takeover"),
        _neighbor(0.80, "confirm_fraud", [], reason_code="money_laundering"),
        _neighbor(0.75, "approve_clean", [], reason_code="seed_ground_truth_paysim"),
    ]
    summary = summarize_precedents(neighbors)
    assert summary["common_reason_codes"] == [
        {"rule": "account_takeover", "count": 3, "total": 5},
        {"rule": "money_laundering", "count": 1, "total": 5},
    ]

def test_common_reason_codes_empty_when_no_precedent_at_all():
    summary = summarize_precedents([])
    assert summary["common_reason_codes"] == []
