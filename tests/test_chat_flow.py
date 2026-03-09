from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import MessageRequest
from app.services.orchestrator import orchestrator

client = TestClient(app)


def test_health_endpoint_returns_healthy() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"


def test_chat_endpoint_emergency_protocol() -> None:
    response = client.post(
        "/api/v1/chat/message",
        json={"session_id": "sess-1", "message": "estou com dor no peito"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["emergency_protocol"] is True
    assert payload["nlp_analysis"]["is_emergency"] is True


def test_orchestrator_scheduling_suggests_action() -> None:
    request = MessageRequest(session_id="sess-2", message="quero agendar uma consulta")
    response = __import__("asyncio").run(orchestrator.process_message(request))
    assert response.emergency_protocol is False
    assert "Agendar consulta" in response.suggested_actions
