from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_valid_text():
    response = client.post(
        "/predict",
        json={"text": "Ceci est un excellent film avec un super scénario."}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "predicted_category" in json_data
    assert "confidence" in json_data

def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={"text": "   "}
    )
    assert response.status_code == 400