from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app
from app.services.model_service import fraud_model_service


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_endpoint_returns_payload(monkeypatch) -> None:
    expected_payload = {
        "is_trained": True,
        "feature_count": 3,
        "target_column": "is_fraud",
        "class_distribution": {"non_fraud": 100, "fraud": 5},
        "validation_metrics": {
            "precision": 0.7,
            "recall": 0.8,
            "f1_score": 0.75,
            "roc_auc": 0.9,
        },
        "test_metrics": {
            "precision": 0.6,
            "recall": 0.85,
            "f1_score": 0.7,
            "roc_auc": 0.91,
        },
    }

    monkeypatch.setattr(fraud_model_service, "get_status", lambda: expected_payload)

    client = TestClient(app)
    response = client.get("/api/v1/fraud/status")

    assert response.status_code == 200
    assert response.json() == expected_payload


def test_train_endpoint_returns_training_result(monkeypatch) -> None:
    expected_payload = {
        "message": "Model training completed and frozen artifacts were saved successfully.",
        "target_column": "is_fraud",
        "class_distribution": {"non_fraud": 900, "fraud": 100},
        "validation_metrics": {
            "precision": 0.72,
            "recall": 0.81,
            "f1_score": 0.76,
            "roc_auc": 0.91,
        },
        "test_metrics": {
            "precision": 0.7,
            "recall": 0.79,
            "f1_score": 0.74,
            "roc_auc": 0.95,
        },
    }

    monkeypatch.setattr(fraud_model_service, "train_model", lambda: expected_payload)

    client = TestClient(app)
    response = client.post("/api/v1/fraud/train")

    assert response.status_code == 200
    assert response.json() == expected_payload


def test_predict_endpoint_returns_predictions(monkeypatch) -> None:
    expected_payload = {
        "total_rows": 1,
        "predictions": [
            {
                "row_index": 0,
                "fraud_probability_percent": 88.2,
                "recommendation": "High probability of fraud",
                "risk_level": "high",
            }
        ],
    }

    async def mock_predict_csv(_upload_file):
        return expected_payload

    monkeypatch.setattr(fraud_model_service, "predict_csv", mock_predict_csv)

    client = TestClient(app)
    files = {"file": ("transactions.csv", BytesIO(b"amount,merchant\n100,store\n"), "text/csv")}
    response = client.post("/api/v1/fraud/predict", files=files)

    assert response.status_code == 200
    assert response.json() == expected_payload
