from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "Random Forest"
    assert data["fraud_threshold"] == 0.395
    assert data["number_of_features"] == 30


def test_prediction():
    transaction = {
        "Time": 406,
        "V1": -2.3,
        "V2": 1.4,
        "V3": -1.2,
        "V4": 0.5,
        "V5": -0.8,
        "V6": 0.2,
        "V7": -1.1,
        "V8": 0.3,
        "V9": -0.5,
        "V10": -1.0,
        "V11": 0.8,
        "V12": -0.7,
        "V13": 0.1,
        "V14": -1.5,
        "V15": 0.4,
        "V16": -0.9,
        "V17": 0.2,
        "V18": -0.3,
        "V19": 0.6,
        "V20": -0.1,
        "V21": 0.2,
        "V22": -0.2,
        "V23": 0.1,
        "V24": -0.4,
        "V25": 0.3,
        "V26": -0.1,
        "V27": 0.05,
        "V28": 0.02,
        "Amount": 10.0
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data
    assert "risk_score" in data
    assert "risk_level" in data
    assert "is_fraud" in data

    assert 0 <= data["fraud_probability"] <= 1
    assert 0 <= data["risk_score"] <= 100
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(data["is_fraud"], bool)