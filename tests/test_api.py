from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health():
    result = client.get("/health")
    assert result.status_code == 200
    assert isinstance(result.json()["model_ready"], bool)


def test_invalid_input_negative():
    """Отрицательная кислотность должна дать 422."""
    payload = {
        "fixed_acidity": -1.0,
        "volatile_acidity": 0.27,
        "citric_acid": 0.3,
        "residual_sugar": 5.0,
        "chlorides": 0.04,
        "free_sulfur_dioxide": 30.0,
        "total_sulfur_dioxide": 120.0,
        "density": 0.995,
        "pH": 3.2,
        "sulphates": 0.5,
        "alcohol": 10.0,
    }
    result = client.post("/predict", json=payload)
    assert result.status_code == 422


def test_predict_success_if_model_exists():
    """Если модель уже обучена — успешный /predict."""
    health = client.get("/health").json()
    if not health["model_ready"]:
        return  # пропускаем, если модель ещё не обучена

    payload = {
        "fixed_acidity": 7.0,
        "volatile_acidity": 0.27,
        "citric_acid": 0.36,
        "residual_sugar": 20.7,
        "chlorides": 0.045,
        "free_sulfur_dioxide": 45.0,
        "total_sulfur_dioxide": 170.0,
        "density": 1.001,
        "pH": 3.0,
        "sulphates": 0.45,
        "alcohol": 8.8,
    }
    result = client.post("/predict", json=payload)
    assert result.status_code == 200
    body = result.json()
    assert "predicted_quality" in body
    assert isinstance(body["predicted_quality"], (int, float))
