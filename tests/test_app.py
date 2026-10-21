import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from app import create_app


@pytest.fixture
def client(tmp_path):
    texts = ["WIN a FREE prize now, call to claim cash", "Free entry, txt WIN to claim",
             "see you at lunch tomorrow", "can you call me when you get home"]
    model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())]).fit(texts, [1, 1, 0, 0])
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "name": "test"}, path)
    return create_app(path).test_client()


def test_api_detects_spam(client):
    response = client.post("/api/predict", json={"text": "Claim your FREE prize, WIN cash now"})
    assert response.status_code == 200
    assert response.get_json()["label"] == "spam"


def test_api_detects_ham(client):
    assert client.post("/api/predict", json={"text": "see you at home"}).get_json()["is_spam"] is False


def test_api_rejects_empty_text(client):
    assert client.post("/api/predict", json={}).status_code == 400


def test_form_page(client):
    assert client.get("/").status_code == 200
    page = client.post("/", data={"message": "WIN a FREE prize"}).get_data(as_text=True)
    assert "spam" in page
