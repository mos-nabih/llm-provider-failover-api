import os
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .base_provider import BaseLLMProvider
from .models import LLMRequest, LLMResponse, ProviderStatus


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider for cloud LLM inference."""

    def __init__(self, api_key: str | None = None, default_model: str = "gpt-3.5-turbo"):
        super().__init__(name="openai", default_model=default_model)
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key and self.api_key != "your-key-here-if-available":
            self.client = AsyncOpenAI(api_key=self.api_key)

    def _format_messages(self, request: LLMRequest) -> list:
        """Format messages for OpenAI API."""
        return [{"role": msg.role.value, "content": msg.content} for msg in request.messages]

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a complete response."""
        if not self.client:
            raise RuntimeError("OpenAI API key not configured")

        model = self.get_model(request)
        response = await self.client.chat.completions.create(
            model=model,
            messages=self._format_messages(request),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
        )

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=model,
            provider=self.name,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            finish_reason=choice.finish_reason,
        )

    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Generate a streaming response."""
        if not self.client:
            raise RuntimeError("OpenAI API key not configured")

        model = self.get_model(request)
        stream = await self.client.chat.completions.create(
            model=model,
            messages=self._format_messages(request),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def health_check(self) -> ProviderStatus:
        """Check if OpenAI is available."""
        if not self.client:
            return ProviderStatus(
                name=self.name,
                available=False,
                default_model=self.default_model,
                error="API key not configured",
            )

        try:
            # List models to verify connectivity
            await self.client.models.list()
            return ProviderStatus(name=self.name, available=True, default_model=self.default_model)
        except Exception as e:
            return ProviderStatus(
                name=self.name, available=False, default_model=self.default_model, error=str(e)
            )
