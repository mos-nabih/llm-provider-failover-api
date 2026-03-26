import json
from collections.abc import AsyncIterator

import httpx

from .base_provider import BaseLLMProvider
from .models import LLMRequest, LLMResponse, ProviderStatus


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local LLM inference."""

    def __init__(self, host: str = "http://localhost:11434", default_model: str = "qwen2.5:0.5b"):
        super().__init__(name="ollama", default_model=default_model)
        self.host = host
        self.client = httpx.AsyncClient(timeout=120.0)

    def _format_messages(self, request: LLMRequest) -> list:
        """Format messages for Ollama API."""
        return [{"role": msg.role.value, "content": msg.content} for msg in request.messages]

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a complete response."""
        model = self.get_model(request)
        payload = {
            "model": model,
            "messages": self._format_messages(request),
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        response = await self.client.post(f"{self.host}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["message"]["content"],
            model=model,
            provider=self.name,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            finish_reason="stop",
        )

    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Generate a streaming response."""
        model = self.get_model(request)
        payload = {
            "model": model,
            "messages": self._format_messages(request),
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        async with self.client.stream("POST", f"{self.host}/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]

    async def health_check(self) -> ProviderStatus:
        """Check if Ollama is available."""
        try:
            response = await self.client.get(f"{self.host}/api/tags")
            response.raise_for_status()
            return ProviderStatus(name=self.name, available=True, default_model=self.default_model)
        except Exception as e:
            return ProviderStatus(
                name=self.name, available=False, default_model=self.default_model, error=str(e)
            )
