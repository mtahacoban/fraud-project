from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend import db_models as m
from backend.main import _top_rules_by_transaction_id

BASE_AMOUNT = 7_654_320.0


def _post_score(client, **overrides):
    payload = {
        "step": 3, "type": "TRANSFER", "amount": BASE_AMOUNT,
        "oldbalanceOrg": BASE_AMOUNT, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }
    payload.update(overrides)
    res = client.post("/score", json=payload)
    assert res.status_code == 200
    return res.json()


@pytest.fixture(scope="module")
def seeded(client):
    t1 = _post_score(client, amount=BASE_AMOUNT + 1, oldbalanceOrg=BASE_AMOUNT + 1, login_country="TR")
    t2 = _post_score(client, amount=BASE_AMOUNT + 2, oldbalanceOrg=BASE_AMOUNT + 2, login_country="DE")
    t3 = _post_score(client, amount=BASE_AMOUNT + 3, oldbalanceOrg=BASE_AMOUNT + 3, type="CASH_OUT", login_country="TR")
    t4 = _post_score(
        client, amount=BASE_AMOUNT + 4, type="PAYMENT",
        oldbalanceOrg=20_000.0, newbalanceOrig=10_000.0,
        oldbalanceDest=0.0, newbalanceDest=BASE_AMOUNT + 4, login_country="TR",
    )
    t5 = _post_score(
        client, amount=BASE_AMOUNT + 5, oldbalanceOrg=50_000.0, newbalanceOrig=49_500.0,
        oldbalanceDest=10_000.0, newbalanceDest=10_500.0, login_country="TR",
    )

    assert t1["risk_band"] == "RED" and t1["case_id"] is not None
    assert t2["risk_band"] == "RED" and t2["case_id"] is not None
    assert t3["risk_band"] == "RED" and t3["case_id"] is not None
    assert t4["case_id"] is None
    return {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5}


def _list(client, **params):
    params.setdefault("amount_min", BASE_AMOUNT)
    params.setdefault("amount_max", BASE_AMOUNT + 5)
    res = client.get("/cases", params=params)
    assert res.status_code == 200
    return res.json()


def test_status_all_merges_real_cases_and_auto_clean(client, seeded):
    body = _list(client)
    ids = {item["transaction_id"] for item in body["items"]}
    assert seeded["t1"]["txn_id"] in ids
    assert seeded["t2"]["txn_id"] in ids
    assert seeded["t3"]["txn_id"] in ids
    assert seeded["t4"]["txn_id"] in ids
    assert seeded["t5"]["txn_id"] in ids
    assert body["total"] == 5


def test_status_auto_clean_returns_only_fast_path_and_green_rows(client, seeded):
    body = _list(client, status="AUTO_CLEAN")
    ids = {item["transaction_id"] for item in body["items"]}
    assert seeded["t4"]["txn_id"] in ids
    assert seeded["t1"]["txn_id"] not in ids
    for item in body["items"]:
        assert item["case_id"] is None
        assert item["status"] == "AUTO_CLEAN"


def test_status_open_returns_only_real_open_cases_not_auto_clean(client, seeded):
    body = _list(client, status="OPEN")
    ids = {item["transaction_id"] for item in body["items"]}
    assert seeded["t1"]["txn_id"] in ids
    assert seeded["t4"]["txn_id"] not in ids
    for item in body["items"]:
        assert item["status"] == "OPEN"


def test_filter_by_risk_band_red(client, seeded):
    body = _list(client, risk_band="RED")
    ids = {item["transaction_id"] for item in body["items"]}
    assert {seeded["t1"]["txn_id"], seeded["t2"]["txn_id"], seeded["t3"]["txn_id"]} <= ids
    assert seeded["t4"]["txn_id"] not in ids


def test_filter_by_type_transfer_excludes_cash_out(client, seeded):
    body = _list(client, type="TRANSFER")
    ids = {item["transaction_id"] for item in body["items"]}
    assert seeded["t1"]["txn_id"] in ids
    assert seeded["t3"]["txn_id"] not in ids


def test_filter_by_type_cash_out_excludes_transfer(client, seeded):
    body = _list(client, type="CASH_OUT")
    ids = {item["transaction_id"] for item in body["items"]}
    assert seeded["t3"]["txn_id"] in ids
    assert seeded["t1"]["txn_id"] not in ids


def test_filter_by_country(client, seeded):
    body = _list(client, country="DE")
    ids = {item["transaction_id"] for item in body["items"]}
    assert seeded["t2"]["txn_id"] in ids
    assert seeded["t1"]["txn_id"] not in ids


def test_filter_by_amount_range_excludes_outside_values(client, seeded):
    res = client.get("/cases", params={"amount_min": BASE_AMOUNT + 1, "amount_max": BASE_AMOUNT + 1})
    body = res.json()
    ids = {item["transaction_id"] for item in body["items"]}
    assert seeded["t1"]["txn_id"] in ids
    assert seeded["t2"]["txn_id"] not in ids


def test_filter_by_date_range_includes_today(client, seeded):
    today = date.today().isoformat()
    body = _list(client, date_from=today, date_to=today)
    ids = {item["transaction_id"] for item in body["items"]}
    assert seeded["t1"]["txn_id"] in ids


def test_filter_by_date_range_excludes_before_today(client, seeded):
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    body = _list(client, date_to=yesterday)
    ids = {item["transaction_id"] for item in body["items"]}
    assert seeded["t1"]["txn_id"] not in ids


def test_filter_by_query_matches_case_id(client, seeded):
    case_id = seeded["t1"]["case_id"]
    res = client.get("/cases", params={"q": str(case_id)})
    body = res.json()
    ids = {item["case_id"] for item in body["items"]}
    assert case_id in ids


def test_combined_filters_type_and_risk_band_and_country(client, seeded):
    body = _list(client, type="TRANSFER", risk_band="RED", country="TR")
    ids = {item["transaction_id"] for item in body["items"]}
    assert seeded["t1"]["txn_id"] in ids
    assert seeded["t2"]["txn_id"] not in ids
    assert seeded["t3"]["txn_id"] not in ids


def test_total_reflects_the_filtered_count_not_the_page_size(client, seeded):
    body = _list(client, risk_band="RED", limit=1)
    assert len(body["items"]) == 1
    assert body["total"] == 3


def test_no_matching_filter_returns_empty_not_error(client, seeded):
    body = _list(client, country="ZZ")
    assert body["items"] == []
    assert body["total"] == 0


def test_reversed_date_range_returns_empty_not_error(client, seeded):
    today = date.today().isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    body = _list(client, date_from=tomorrow, date_to=today)
    assert body["items"] == []
    assert body["total"] == 0


def test_reversed_amount_range_returns_empty_not_error(client, seeded):
    body = _list(client, amount_min=BASE_AMOUNT + 5, amount_max=BASE_AMOUNT)
    assert body["items"] == []
    assert body["total"] == 0


def test_unknown_country_returns_empty_not_error(client, seeded):
    body = _list(client, country="XX")
    assert body["items"] == []
    assert body["total"] == 0


def test_offset_past_the_last_page_returns_empty_items_but_correct_total(client, seeded):
    body = _list(client, risk_band="RED", offset=1000)
    assert body["items"] == []
    assert body["total"] == 3


def test_limit_is_clamped_not_rejected_when_over_the_server_max(client, seeded):
    res = client.get("/cases", params={"amount_min": BASE_AMOUNT, "amount_max": BASE_AMOUNT + 5, "limit": 10_000})
    assert res.status_code == 200


def test_case_summary_amount_matches_the_real_transaction_across_all_three_branches(client, seeded):
    all_by_txn = {item["transaction_id"]: item for item in _list(client)["items"]}
    open_by_txn = {item["transaction_id"]: item for item in _list(client, status="OPEN")["items"]}
    auto_clean_by_txn = {item["transaction_id"]: item for item in _list(client, status="AUTO_CLEAN")["items"]}

    assert all_by_txn[seeded["t1"]["txn_id"]]["amount"] == BASE_AMOUNT + 1
    assert open_by_txn[seeded["t1"]["txn_id"]]["amount"] == BASE_AMOUNT + 1
    assert all_by_txn[seeded["t4"]["txn_id"]]["amount"] == BASE_AMOUNT + 4
    assert auto_clean_by_txn[seeded["t4"]["txn_id"]]["amount"] == BASE_AMOUNT + 4


def test_case_summary_top_rules_reflects_the_real_ghost_destination_hit(client, seeded):
    all_by_txn = {item["transaction_id"]: item for item in _list(client)["items"]}
    open_by_txn = {item["transaction_id"]: item for item in _list(client, status="OPEN")["items"]}
    assert "ghost_destination" in all_by_txn[seeded["t1"]["txn_id"]]["top_rules"]
    assert "ghost_destination" in open_by_txn[seeded["t1"]["txn_id"]]["top_rules"]
    assert "ghost_destination" in all_by_txn[seeded["t3"]["txn_id"]]["top_rules"]


def test_top_rules_by_transaction_id_excludes_clean_type_rule_hits(client, db_session, seeded):
    txn_id = seeded["t1"]["txn_id"]
    db_session.add(m.RuleHit(transaction_id=txn_id, rule_name="clean_confirmed", rule_type="clean", severity="info"))
    db_session.commit()

    result = _top_rules_by_transaction_id(db_session, [txn_id])
    assert "ghost_destination" in result[txn_id]
    assert "clean_confirmed" not in result[txn_id]


def test_top_rules_by_transaction_id_returns_empty_dict_for_empty_input(db_session):
    assert _top_rules_by_transaction_id(db_session, []) == {}


def test_case_summary_top_rules_is_empty_list_not_error_for_a_rule_free_transaction(client, seeded):
    rule_free = [
        t for t in seeded.values()
        if not t.get("hard_rule_hits") and not t.get("soft_rule_hits")
    ]
    assert rule_free, "expected at least one seeded transaction with zero rule hits"
    txn_id = rule_free[0]["txn_id"]

    all_by_txn = {item["transaction_id"]: item for item in _list(client)["items"]}
    assert all_by_txn[txn_id]["top_rules"] == []
