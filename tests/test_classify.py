from unittest.mock import patch

from fastapi.testclient import TestClient

from app.llm_client import LLMBadResponseError, LLMUnavailableError
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.main.classify_text")
def test_classify_success(mock_classify):
    mock_classify.return_value = {
        "category": "поддержка",
        "priority": "high",
        "summary": "проблема с оплатой заказа",
        "suggested_reply": "Здравствуйте! Уточните, пожалуйста, номер заказа — разберёмся.",
    }

    response = client.post("/classify", json={"text": "Не могу оплатить заказ третий день"})

    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "поддержка"
    assert body["priority"] == "high"
    mock_classify.assert_called_once()


def test_classify_empty_text_returns_422():
    response = client.post("/classify", json={"text": "ab"})  # короче min_length
    assert response.status_code == 422


@patch("app.main.classify_text")
def test_classify_llm_unavailable_returns_503(mock_classify):
    mock_classify.side_effect = LLMUnavailableError("timeout")
    response = client.post("/classify", json={"text": "Здравствуйте, есть вопрос по заказу"})
    assert response.status_code == 503


@patch("app.main.classify_text")
def test_classify_bad_llm_response_returns_502(mock_classify):
    mock_classify.side_effect = LLMBadResponseError("not json")
    response = client.post("/classify", json={"text": "Здравствуйте, есть вопрос по заказу"})
    assert response.status_code == 502
