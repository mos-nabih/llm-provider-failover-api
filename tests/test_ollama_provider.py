from unittest.mock import AsyncMock, Mock

import pytest

from app.models import LLMRequest, Message, MessageRole
from app.ollama_provider import OllamaProvider


def build_request() -> LLMRequest:
    return LLMRequest(
        messages=[Message(role=MessageRole.USER, content="What is 2+2?")],
        temperature=0.2,
        max_tokens=12,
    )


@pytest.mark.asyncio
async def test_generate_sends_expected_payload():
    provider = OllamaProvider()
    mock_response = Mock()
    mock_response.json.return_value = {
        "message": {"content": "4"},
        "prompt_eval_count": 10,
        "eval_count": 2,
    }
    mock_response.raise_for_status = Mock()
    post_mock = AsyncMock(return_value=mock_response)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(provider.client, "post", post_mock)

    response = await provider.generate(build_request())

    assert response.content == "4"
    assert response.provider == "ollama"
    post_mock.assert_awaited_once_with(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen2.5:0.5b",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 12,
            },
        },
    )
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_health_check_reports_available_on_success():
    provider = OllamaProvider()
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    get_mock = AsyncMock(return_value=mock_response)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(provider.client, "get", get_mock)

    status = await provider.health_check()

    assert status.available is True
    assert status.name == "ollama"
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_health_check_reports_error_on_failure():
    provider = OllamaProvider()
    get_mock = AsyncMock(side_effect=RuntimeError("connection refused"))
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(provider.client, "get", get_mock)

    status = await provider.health_check()

    assert status.available is False
    assert status.error == "connection refused"
    monkeypatch.undo()
