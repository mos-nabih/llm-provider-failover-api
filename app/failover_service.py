import logging
from collections.abc import AsyncIterator

from .base_provider import BaseLLMProvider
from .models import LLMRequest, LLMResponse, ProviderStatus

logger = logging.getLogger(__name__)


class FailoverService:
    """Service that handles automatic failover between providers."""

    def __init__(self, providers: list[BaseLLMProvider]):
        self.providers = providers
        self._health_cache: dict[str, bool] = {}

    async def refresh_health(self) -> list[ProviderStatus]:
        """Refresh health status for all providers."""
        statuses = []
        for provider in self.providers:
            status = await provider.health_check()
            self._health_cache[provider.name] = status.available
            statuses.append(status)
        return statuses

    def get_available_providers(self) -> list[BaseLLMProvider]:
        """Get list of providers that were healthy at last check."""
        return [p for p in self.providers if self._health_cache.get(p.name, True)]

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response with automatic failover."""
        errors = []

        for provider in self.providers:
            try:
                logger.info(f"Attempting generation with {provider.name}")
                response = await provider.generate(request)
                logger.info(f"Successfully generated with {provider.name}")
                return response
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed: {e}")
                errors.append(f"{provider.name}: {str(e)}")
                self._health_cache[provider.name] = False
                continue

        raise RuntimeError(f"All providers failed: {'; '.join(errors)}")

    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Generate streaming response with automatic failover."""
        errors = []

        for provider in self.providers:
            try:
                logger.info(f"Attempting streaming with {provider.name}")
                async for chunk in provider.generate_stream(request):
                    yield chunk
                logger.info(f"Successfully completed streaming with {provider.name}")
                return
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed during streaming: {e}")
                errors.append(f"{provider.name}: {str(e)}")
                self._health_cache[provider.name] = False
                continue

        raise RuntimeError(f"All providers failed: {'; '.join(errors)}")
