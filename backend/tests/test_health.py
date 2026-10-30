"""M1 smoke test: /api/health reports every pipeline component."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

EXPECTED_COMPONENTS = {"db", "bm25", "embedding", "reranker", "ragas", "llm"}


def test_root_ok():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Ask ITO"


def test_health_reports_all_components():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert set(body["components"].keys()) == EXPECTED_COMPONENTS
    # Every component must report a status string.
    for name, comp in body["components"].items():
        assert "status" in comp, f"{name} missing status"
    # Corpus summary present for the frontend footer.
    assert "corpus" in body


def test_health_llm_mock_mode():
    resp = client.get("/api/health")
    llm = resp.json()["components"]["llm"]
    assert llm["mode"] == "mock"
    assert llm["status"] == "ok"
