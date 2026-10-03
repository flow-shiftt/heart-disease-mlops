import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    from cardiorisk.api.app import app

    with TestClient(app) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert body["label"] in ("disease", "no_disease")
    assert body["model_name"] == "test_lr"
    assert r.headers["x-request-id"] == body["request_id"]


def test_predict_rejects_out_of_range(client, sample_patient):
    r = client.post("/predict", json={**sample_patient, "age": 250})
    assert r.status_code == 422


def test_predict_rejects_unknown_field(client, sample_patient):
    r = client.post("/predict", json={**sample_patient, "bmi": 30})
    assert r.status_code == 422


def test_batch_endpoint(client, sample_patient):
    r = client.post("/predict/batch", json={"patients": [sample_patient] * 3})
    assert r.status_code == 200
    assert r.json()["count"] == 3


def test_metrics_exposed_after_prediction(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "cardiorisk_predictions_total" in text
    assert "cardiorisk_http_request_duration_seconds" in text
    assert 'cardiorisk_model_info{name="test_lr",version="test"} 1.0' in text


def test_request_logged_as_json(client, sample_patient, capsys):
    client.post("/predict", json=sample_patient, headers={"x-request-id": "abc123"})
    out = capsys.readouterr().out
    assert '"request_id": "abc123"' in out
    assert '"msg": "prediction"' in out
