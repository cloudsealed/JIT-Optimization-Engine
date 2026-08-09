import pytest
from fastapi.testclient import TestClient

from cloudsealed_jit.api import app

client = TestClient(app)


def test_health_reports_version_and_jit_status():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "jitEnabled" in body


def test_analyze_billing_matches_the_framework4d_contract(spiked_billing):
    response = client.post(
        "/v1/analyze-billing",
        json={
            "companyName": "Acme Ltda",
            "csvContent": spiked_billing,
            "analysisType": "waste-audit",
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {"anomalies", "metrics", "recommendations", "summary"}
    assert set(body["metrics"]) == {
        "averageDailyCost", "stdDeviation", "sharpeRatio", "wastePercentage"
    }
    assert set(body["anomalies"][0]) == {
        "date", "expectedCost", "actualCost", "deviation",
        "zScore", "severity", "description",
    }
    assert set(body["recommendations"][0]) == {
        "title", "description", "potentialSavings", "effort",
    }


def test_analysis_type_defaults_to_waste_audit(stable_billing):
    response = client.post(
        "/v1/analyze-billing",
        json={"companyName": "Acme", "csvContent": stable_billing},
    )
    assert response.status_code == 200


def test_budget_adds_forecast_without_breaking_default_shape(spiked_billing):
    # A budget request must surface a forecast object; the default (no budget)
    # response must stay free of the field to preserve the frozen contract.
    with_budget = client.post(
        "/v1/analyze-billing",
        json={"companyName": "Acme", "csvContent": spiked_billing, "budget": 100.0},
    ).json()
    assert "forecast" in with_budget
    assert with_budget["forecast"]["horizonDays"] == 30
    assert with_budget["forecast"]["budget"] == 100.0

    without = client.post(
        "/v1/analyze-billing",
        json={"companyName": "Acme", "csvContent": spiked_billing},
    ).json()
    assert "forecast" not in without


def test_unreadable_export_returns_422_not_500():
    response = client.post(
        "/v1/analyze-billing",
        json={"companyName": "Acme", "csvContent": "region,owner\nus-east-1,eng\n"},
    )
    assert response.status_code == 422
    assert "date and a cost column" in response.json()["detail"]


def test_empty_company_name_is_rejected(stable_billing):
    response = client.post(
        "/v1/analyze-billing",
        json={"companyName": "", "csvContent": stable_billing},
    )
    assert response.status_code == 422


def test_oversized_payload_is_rejected(monkeypatch):
    from cloudsealed_jit import api

    monkeypatch.setattr(api, "MAX_CSV_BYTES", 10)
    response = client.post(
        "/v1/analyze-billing",
        json={"companyName": "Acme", "csvContent": "date,cost\n2026-01-01,1\n"},
    )
    assert response.status_code == 413


def test_api_key_is_enforced_when_configured(monkeypatch, stable_billing):
    from cloudsealed_jit import api

    monkeypatch.setattr(api, "API_KEY", "secret")
    payload = {"companyName": "Acme", "csvContent": stable_billing}

    assert client.post("/v1/analyze-billing", json=payload).status_code == 401
    ok = client.post(
        "/v1/analyze-billing", json=payload, headers={"X-Api-Key": "secret"}
    )
    assert ok.status_code == 200
