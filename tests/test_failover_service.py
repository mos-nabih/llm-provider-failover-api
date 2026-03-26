from unittest.mock import AsyncMock

import pytest

from app.failover_service import FailoverService
from app.models import LLMRequest, LLMResponse, Message, MessageRole


def build_request() -> LLMRequest:
    return LLMRequest(
        messages=[Message(role=MessageRole.USER, content="Hello")],
        max_tokens=16,
    )


@pytest.mark.asyncio
async def test_generate_returns_first_successful_provider_response():
    first_provider = AsyncMock()
    first_provider.name = "openai"
    first_provider.generate.return_value = LLMResponse(
        content="hi",
        model="gpt-test",
        provider="openai",
        finish_reason="stop",
    )

    second_provider = AsyncMock()
    second_provider.name = "ollama"

    service = FailoverService([first_provider, second_provider])

    response = await service.generate(build_request())

    assert response.provider == "openai"
    first_provider.generate.assert_awaited_once()
    second_provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_generate_fails_over_to_next_provider_on_error():
    first_provider = AsyncMock()
    first_provider.name = "openai"
    first_provider.generate.side_effect = RuntimeError("boom")

    second_provider = AsyncMock()
    second_provider.name = "ollama"
    second_provider.generate.return_value = LLMResponse(
        content="fallback",
        model="qwen2.5:0.5b",
        provider="ollama",
        finish_reason="stop",
    )

    service = FailoverService([first_provider, second_provider])

    response = await service.generate(build_request())

    assert response.provider == "ollama"
    assert service._health_cache["openai"] is False
    first_provider.generate.assert_awaited_once()
    second_provider.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_raises_when_all_providers_fail():
    first_provider = AsyncMock()
    first_provider.name = "openai"
    first_provider.generate.side_effect = RuntimeError("missing key")

    second_provider = AsyncMock()
    second_provider.name = "ollama"
    second_provider.generate.side_effect = RuntimeError("service down")

    service = FailoverService([first_provider, second_provider])

    with pytest.raises(RuntimeError) as exc_info:
        await service.generate(build_request())

    message = str(exc_info.value)
    assert "All providers failed" in message
    assert "openai: missing key" in message
    assert "ollama: service down" in message
