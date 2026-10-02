import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_hotel_search():
    response = client.post("/api/v1/chat", json={"message": "Hotels in Dubai under 15K", "trip_id": "test_1"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["success", "partial"]
    assert len(data["reply"]) > 0

def test_destination_research():
    response = client.post("/api/v1/chat", json={"message": "Destination research in Dubai", "trip_id": "test_2"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["success", "partial"]
    assert len(data["reply"]) > 0

def test_flight_search():
    response = client.post("/api/v1/chat", json={"message": "Find flights from TRV to LHR", "trip_id": "test_3"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["success", "partial"]
    assert len(data["reply"]) > 0

def test_tool_failure(monkeypatch):
    import app.tools.tavily_search
    async def mock_execute(*args, **kwargs):
        raise app.tools.tavily_search.TavilyConfigError("Mock tool failure")
    monkeypatch.setattr(app.tools.tavily_search.TavilySearchClient, "search", mock_execute)
    
    response = client.post("/api/v1/chat", json={"message": "Hotels in Dubai under 15K", "trip_id": "test_4"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["error", "partial"]
    assert any(err.get("code") == "INTERNAL_ERROR" for err in data.get("errors", []))

def test_empty_retrieval(monkeypatch):
    import app.tools.tavily_search
    import app.tools.tavily_search as ts
    async def mock_execute(*args, **kwargs):
        return ts.TavilySearchResponse(results=[])
    monkeypatch.setattr(app.tools.tavily_search.TavilySearchClient, "search", mock_execute)
    
    response = client.post("/api/v1/chat", json={"message": "Hotels in Dubai under 15K", "trip_id": "test_5"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["success", "partial"]
    assert len(data["reply"]) > 0

def test_missing_api_key(monkeypatch):
    import app.core.config
    monkeypatch.setattr(app.core.config.settings, "TAVILY_API_KEY", "")
    
    response = client.post("/api/v1/chat", json={"message": "Hotels in Dubai under 15K", "trip_id": "test_6"})
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "error"
    assert any(err.get("code") == "PROVIDER_UNAVAILABLE" for err in data.get("errors", []))

def test_llm_failure(monkeypatch):
    import app.core.config
    monkeypatch.setattr(app.core.config.settings, "OPENAI_API_KEY", "")
    
    response = client.post("/api/v1/chat", json={"message": "Hotels in Dubai under 15K", "trip_id": "test_7"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["reply"]) > 0
