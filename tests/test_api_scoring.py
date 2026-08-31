from __future__ import annotations

CANONICAL_FRAUD_PAYLOAD = {
    "step": 3, "type": "TRANSFER", "amount": 320_000.0,
    "oldbalanceOrg": 320_000.0, "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}


def test_score_endpoint_returns_the_canonical_regression_values(client):
    res = client.post("/score", json=CANONICAL_FRAUD_PAYLOAD)
    assert res.status_code == 200
    body = res.json()
    assert body["risk_band"] == "RED"
    assert body["hybrid_score"] == 99.35
    assert body["calibrated_proba"] == 0.9852941036224365
    assert body["case_id"] is not None
    assert "ghost_destination" in body["hard_rule_hits"]
    assert len(body["shap_factors"]) == 6


def test_score_endpoint_response_matches_score_out_shape(client):
    res = client.post("/score", json=CANONICAL_FRAUD_PAYLOAD)
    body = res.json()
    assert set(body.keys()) == {
        "txn_id", "case_id", "ml_score", "soft_score", "hybrid_score", "risk_band",
        "calibrated_proba", "hard_rule_hits", "soft_rule_hits", "shap_factors",
        "model_version", "band_reason",
    }


def test_score_endpoint_fast_path_type_never_opens_a_case(client):
    payload = {
        "step": 10, "type": "PAYMENT", "amount": 9_999.0,
        "oldbalanceOrg": 20_000.0, "newbalanceOrig": 10_001.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 9_999.0,
    }
    res = client.post("/score", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["risk_band"] == "GREEN"
    assert body["case_id"] is None


def test_score_endpoint_rejects_invalid_payload(client):
    bad_payload = dict(CANONICAL_FRAUD_PAYLOAD)
    bad_payload["amount"] = -100.0
    res = client.post("/score", json=bad_payload)
    assert res.status_code == 422


def test_score_endpoint_rejects_unknown_transaction_type(client):
    bad_payload = dict(CANONICAL_FRAUD_PAYLOAD)
    bad_payload["type"] = "WIRE_TRANSFER"
    res = client.post("/score", json=bad_payload)
    assert res.status_code == 422


def test_score_endpoint_rejects_missing_required_field(client):
    bad_payload = dict(CANONICAL_FRAUD_PAYLOAD)
    del bad_payload["oldbalanceOrg"]
    res = client.post("/score", json=bad_payload)
    assert res.status_code == 422


def test_score_endpoint_rejects_non_numeric_amount(client):
    bad_payload = dict(CANONICAL_FRAUD_PAYLOAD)
    bad_payload["amount"] = "not a number"
    res = client.post("/score", json=bad_payload)
    assert res.status_code == 422


def test_score_endpoint_accepts_zero_amount(client):
    payload = dict(CANONICAL_FRAUD_PAYLOAD)
    payload["amount"] = 0.0
    payload["oldbalanceOrg"] = 0.0
    res = client.post("/score", json=payload)
    assert res.status_code == 200
    assert res.json()["risk_band"] in ("RED", "GRAY", "GREEN")


def test_simulation_run_single_matches_score_endpoint_on_the_same_payload(client):
    score_res = client.post("/score", json=CANONICAL_FRAUD_PAYLOAD)
    sim_res = client.post("/simulation/run", json={"transaction": CANONICAL_FRAUD_PAYLOAD, "count": 1})
    assert score_res.status_code == 200 and sim_res.status_code == 200

    score_body = score_res.json()
    sim_result = sim_res.json()["results"][0]
    assert sim_result["risk_band"] == score_body["risk_band"] == "RED"
    assert sim_result["hybrid_score"] == score_body["hybrid_score"]
    assert sim_result["calibrated_proba"] == score_body["calibrated_proba"]
    assert sim_result["hard_rule_hits"] == score_body["hard_rule_hits"] == ["ghost_destination"]
    assert sim_result["band_reason"] == score_body["band_reason"]


def test_simulation_run_opens_a_real_case_for_red_and_none_for_green(client):
    red_res = client.post("/simulation/run", json={"transaction": CANONICAL_FRAUD_PAYLOAD, "count": 1})
    red_result = red_res.json()["results"][0]
    assert red_result["case_id"] is not None

    green_payload = {
        "step": 14, "type": "TRANSFER", "amount": 500.0,
        "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0, "oldbalanceDest": 18340.0, "newbalanceDest": 18840.0,
    }
    green_res = client.post("/simulation/run", json={"transaction": green_payload, "count": 1})
    green_result = green_res.json()["results"][0]
    assert green_result["risk_band"] == "GREEN"
    assert green_result["case_id"] is None


def test_simulation_run_multi_count_still_returns_rule_hits_per_result(client):
    res = client.post("/simulation/run", json={"transaction": CANONICAL_FRAUD_PAYLOAD, "count": 3})
    assert res.status_code == 200
    body = res.json()
    assert body["scored"] == 3
    for result in body["results"]:
        assert result["shap_factors"] == []
        assert isinstance(result["hard_rule_hits"], list)
