from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "API работает"}

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict():
    response = client.post(
        "/predict",
        json={
            "text": "Студенты изучают машинное обучение в университете",
            "categories": [
                "спорт",
                "образование",
                "бизнес",
                "технологии",
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "label" in data
    assert "score" in data
    assert isinstance(data["label"], str)
    assert isinstance(data["score"], float)
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) > 0

    first_prediction = data["predictions"][0]

    assert "label" in first_prediction
    assert "score" in first_prediction
    assert isinstance(first_prediction["label"], str)
    assert isinstance(first_prediction["score"], float)

def test_empty_text():
    response = client.post(
        "/predict",
        json={
            "text": "",
            "categories": ["спорт", "образование"],
        },
    )

    assert response.status_code == 422

def test_empty_categories():
    response = client.post(
        "/predict",
        json={
            "text": "Тестовый текст",
            "categories": [],
        },
    )

    assert response.status_code == 422

def test_missing_text():
    response = client.post(
        "/predict",
        json={
            "categories": ["спорт", "образование"],
        },
    )

    assert response.status_code == 422

def test_missing_categories():
    response = client.post(
        "/predict",
        json={
            "text": "Какой-то текст",
        },
    )

    assert response.status_code == 422

def test_whitespace_text():
    response = client.post(
        "/predict",
        json={
            "text": "     ",
            "categories": ["спорт", "образование"],
        },
    )

    assert response.status_code == 422

def test_whitespace_categories():
    response = client.post(
        "/predict",
        json={
            "text": "Какой-то текст",
            "categories": ["", "   "],
        },
    )

    assert response.status_code == 422

def test_prediction_label_is_from_categories():
    categories = [
        "спорт",
        "образование",
        "бизнес",
        "технологии",
    ]

    response = client.post(
        "/predict",
        json={
            "text": "Футбольная команда выиграла матч",
            "categories": categories,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["label"] in categories

def test_score_range():
    response = client.post(
        "/predict",
        json={
            "text": "Компания выпустила новый процессор",
            "categories": [
                "спорт",
                "технологии",
                "образование",
            ],
        },
    )

    assert response.status_code == 200

    score = response.json()["score"]

    assert 0 <= score <= 1

def test_invalid_text_type():
    response = client.post(
        "/predict",
        json={
            "text": ["это", "не", "строка"],
            "categories": ["спорт", "образование"],
        },
    )

    assert response.status_code == 422

def test_prediction_label_matches_first_prediction():
    response = client.post(
        "/predict",
        json={
            "text": "Студенты изучают машинное обучение в университете",
            "categories": [
                "спорт",
                "образование",
                "бизнес",
                "технологии",
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["label"] == data["predictions"][0]["label"]
    assert data["score"] == data["predictions"][0]["score"]