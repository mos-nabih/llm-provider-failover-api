import app.openai_provider as openai_provider_module
from app.openai_provider import OpenAIProvider


def test_openai_provider_creates_client_when_api_key_exists(monkeypatch):
    monkeypatch.setattr(openai_provider_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    provider = OpenAIProvider()

    assert provider.api_key == "test-key"
    assert provider.client is not None


def test_openai_provider_has_no_client_without_api_key(monkeypatch):
    monkeypatch.setattr(openai_provider_module, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = OpenAIProvider()

    assert provider.api_key is None
    assert provider.client is None


def test_openai_provider_ignores_placeholder_key(monkeypatch):
    monkeypatch.setattr(openai_provider_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "your-key-here-if-available")

    provider = OpenAIProvider()

    assert provider.client is None
