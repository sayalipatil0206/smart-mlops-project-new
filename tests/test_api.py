import src.api as api
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

client = TestClient(api.app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_input():
    features = load_breast_cancer().data[0].tolist()

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 200
    assert response.json()["prediction_code"] in [0, 1]
    assert response.json()["predicted_label"] in ["malignant", "benign"]


def test_predict_invalid_input():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0]}
    )

    assert response.status_code == 422


def test_predict_model_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api,
        "MODEL_PATH",
        tmp_path / "missing_model.joblib"
    )

    features = [1.0] * 30

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 503