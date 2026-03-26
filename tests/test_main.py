from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.models import LLMResponse, ProviderStatus

client = TestClient(app)


def test_health_returns_provider_statuses(monkeypatch):
    mock_service = AsyncMock()
    mock_service.refresh_health.return_value = [
        ProviderStatus(name="ollama", available=True, default_model="qwen2.5:0.5b"),
        ProviderStatus(
            name="openai", available=False, default_model="gpt-3.5-turbo", error="missing key"
        ),
    ]
    monkeypatch.setattr(main_module, "failover_service", mock_service)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == [
        {"name": "ollama", "available": True, "default_model": "qwen2.5:0.5b", "error": None},
        {
            "name": "openai",
            "available": False,
            "default_model": "gpt-3.5-turbo",
            "error": "missing key",
        },
    ]
    mock_service.refresh_health.assert_awaited_once()


def test_generate_returns_llm_response(monkeypatch):
    mock_service = AsyncMock()
    mock_service.generate.return_value = LLMResponse(
        content="4",
        model="gpt-4o-mini",
        provider="openai",
        input_tokens=8,
        output_tokens=1,
        finish_reason="stop",
    )
    monkeypatch.setattr(main_module, "failover_service", mock_service)

    response = client.post(
        "/generate",
        json={
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 10,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "content": "4",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "input_tokens": 8,
        "output_tokens": 1,
        "finish_reason": "stop",
    }
    mock_service.generate.assert_awaited_once()


def test_generate_returns_503_when_all_providers_fail(monkeypatch):
    mock_service = AsyncMock()
    mock_service.generate.side_effect = RuntimeError("All providers failed")
    monkeypatch.setattr(main_module, "failover_service", mock_service)

    response = client.post(
        "/generate",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        },
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "All providers failed"}
    mock_service.generate.assert_awaited_once()


def test_extract_returns_structured_json(monkeypatch):
    mock_service = AsyncMock()
    mock_service.generate.return_value = LLMResponse(
        content=(
            '{"entities":[{"name":"Alice","type":"person","context":"engineer"}],'
            '"summary":"Alice is mentioned."}'
        ),
        model="gpt-4o-mini",
        provider="openai",
        finish_reason="stop",
    )
    monkeypatch.setattr(main_module, "failover_service", mock_service)

    response = client.post(
        "/extract",
        json={"text": "Alice is an engineer."},
    )

    assert response.status_code == 200
    assert response.json() == {
        "entities": [{"name": "Alice", "type": "person", "context": "engineer"}],
        "summary": "Alice is mentioned.",
    }
    mock_service.generate.assert_awaited_once()


def test_extract_parses_markdown_wrapped_json(monkeypatch):
    mock_service = AsyncMock()
    mock_service.generate.return_value = LLMResponse(
        content=(
            "```json\n"
            '{"entities":[{"name":"Paris","type":"location","context":"city"}],'
            '"summary":"Paris is referenced."}\n'
            "```"
        ),
        model="gpt-4o-mini",
        provider="openai",
        finish_reason="stop",
    )
    monkeypatch.setattr(main_module, "failover_service", mock_service)

    response = client.post(
        "/extract",
        json={"text": "I visited Paris."},
    )

    assert response.status_code == 200
    assert response.json() == {
        "entities": [{"name": "Paris", "type": "location", "context": "city"}],
        "summary": "Paris is referenced.",
    }


def test_extract_returns_422_for_invalid_json(monkeypatch):
    mock_service = AsyncMock()
    mock_service.generate.return_value = LLMResponse(
        content="not valid json",
        model="gpt-4o-mini",
        provider="openai",
        finish_reason="stop",
    )
    monkeypatch.setattr(main_module, "failover_service", mock_service)

    response = client.post(
        "/extract",
        json={"text": "Alice is an engineer."},
    )

    assert response.status_code == 422
    assert "Failed to parse LLM response as JSON" in response.json()["detail"]


def test_extract_returns_503_when_generation_fails(monkeypatch):
    mock_service = AsyncMock()
    mock_service.generate.side_effect = RuntimeError("All providers failed")
    monkeypatch.setattr(main_module, "failover_service", mock_service)

    response = client.post(
        "/extract",
        json={"text": "Alice is an engineer."},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "All providers failed"}
