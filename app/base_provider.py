from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .models import LLMRequest, LLMResponse, ProviderStatus


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str, default_model: str):
        self.name = name
        self.default_model = default_model

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a complete response from the LLM."""
        pass

    @abstractmethod
    def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM."""
        pass

    @abstractmethod
    async def health_check(self) -> ProviderStatus:
        """Check if the provider is available."""
        pass

    def get_model(self, request: LLMRequest) -> str:
        """Get the model to use, falling back to default."""
        return request.model or self.default_model
