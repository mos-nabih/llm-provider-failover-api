from .base_provider import BaseLLMProvider
from .failover_service import FailoverService
from .models import ProviderStatus


class ProviderRegistry:
    """Registry for managing LLM providers."""

    def __init__(self):
        self._providers: dict[str, BaseLLMProvider] = {}
        self._default_provider: str | None = None

    def register(self, provider: BaseLLMProvider, default: bool = False) -> None:
        """Register a provider."""
        self._providers[provider.name] = provider
        if default or self._default_provider is None:
            self._default_provider = provider.name

    def get(self, name: str) -> BaseLLMProvider | None:
        """Get a provider by name."""
        return self._providers.get(name)

    def get_default(self) -> BaseLLMProvider | None:
        """Get the default provider."""
        if self._default_provider:
            return self._providers.get(self._default_provider)
        return None

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    async def get_all_status(self) -> list[ProviderStatus]:
        """Get status of all providers."""
        statuses = []
        for provider in self._providers.values():
            status = await provider.health_check()
            statuses.append(status)
        return statuses


# Global registry instance
registry = ProviderRegistry()


def create_failover_service(registry: ProviderRegistry) -> FailoverService:
    """Create a failover service from registered providers."""
    providers: list[BaseLLMProvider] = [
        provider
        for name in registry.list_providers()
        if (provider := registry.get(name)) is not None
    ]
    return FailoverService(providers)
