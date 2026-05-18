"""Tests for the /api/v1/predict endpoint."""


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_lightgbm_predict(client, sample_request):
    r = client.post("/api/v1/predict/lightgbm", json=sample_request)
    assert r.status_code == 200
    data = r.json()
    assert "predicted_current_uA" in data
    assert len(data["predicted_current_uA"]) == len(sample_request["potential_V"])
