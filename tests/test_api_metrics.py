from __future__ import annotations


def _post_score(client, **overrides):
    payload = {
        "step": 3, "type": "TRANSFER", "amount": 500.0,
        "oldbalanceOrg": 500.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 500.0, "newbalanceDest": 1000.0,
    }
    payload.update(overrides)
    res = client.post("/score", json=payload)
    assert res.status_code == 200
    return res.json()


def _today_point(client):
    res = client.get("/metrics/trends", params={"days": 1})
    assert res.status_code == 200
    points = res.json()
    assert len(points) == 1
    return points[0]


def test_scored_count_is_always_at_least_case_count(client):
    _post_score(client)
    point = _today_point(client)
    assert point["scored_count"] >= point["case_count"]


def test_scored_count_increases_by_exactly_one_per_transaction_scored(client):
    before = _today_point(client)["scored_count"]
    result = _post_score(
        client, type="PAYMENT", amount=1.0, oldbalanceOrg=1.0, newbalanceOrig=0.0,
        oldbalanceDest=0.0, newbalanceDest=0.0,
    )
    assert result["case_id"] is None
    after = _today_point(client)["scored_count"]
    assert after == before + 1


def test_a_day_with_no_activity_has_scored_count_zero(client):
    res = client.get("/metrics/trends", params={"days": 90})
    assert res.status_code == 200
    points = res.json()
    earliest = points[0]
    assert earliest["case_count"] == 0
    assert earliest["scored_count"] == 0
    assert earliest["red_rate"] is None
    assert earliest["avg_score"] is None
