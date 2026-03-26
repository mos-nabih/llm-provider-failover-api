# LLM Provider Failover API

A FastAPI service that routes requests across multiple LLM providers with failover support.

## Architecture

Architecture documentation is available in [`docs/architecture.md`](docs/architecture.md).

## Project Structure

```text
.
├── app/                    # FastAPI app, providers, models, and failover logic
├── docs/                   # Architecture and project documentation
├── tests/                  # Unit and API tests
├── .github/workflows/      # CI, security, and automation workflows
├── .pre-commit-config.yaml # Local git hooks
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependency versions for uv
├── pytest.ini              # Pytest configuration
└── README.md               # Project documentation
```

## Setup

Install `uv` and install dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --dev
```

Install git hooks:

```bash
uv run pre-commit install
```

If you want to use OpenAI, add your API key to `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
```

You can start from the provided example:

```bash
cp .env.example .env
```

This project targets Python 3.10+, with Python 3.14 recommended.

## Run The App

Start the API locally with:

```bash
uv run uvicorn app.main:app --reload
```

The app will be available at `http://localhost:8000`.

By default, the app tries `OpenAIProvider` first and falls back to `OllamaProvider` if OpenAI is unavailable.

## Testing

Run the full test suite locally with:

```bash
uv run pytest -q
```

Run formatting, linting, and type checking locally with:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy .
uv run pre-commit run --all-files
```

The tests include:
- unit tests for failover behavior
- unit tests for provider initialization and request handling
- API endpoint tests for `/health`, `/generate`, and `/extract`

## Manual Testing

Start the API:

```bash
uv run uvicorn app.main:app --reload
```

Then use the following commands in a separate terminal.

Check provider health:

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Generate a simple response:

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
    "max_tokens": 10
  }' | python -m json.tool
```

Test structured extraction:

```bash
curl -s -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "John Smith works at Acme Corp in New York City. He met Jane Doe at the conference."
  }' | python -m json.tool
```

Test failover directly in Python:

```bash
uv run python -c "
import asyncio
import logging
logging.basicConfig(level=logging.INFO)

from app.ollama_provider import OllamaProvider
from app.openai_provider import OpenAIProvider
from app.failover_service import FailoverService
from app.models import LLMRequest, Message, MessageRole

async def test():
    ollama = OllamaProvider()
    openai = OpenAIProvider()
    service = FailoverService([openai, ollama])

    request = LLMRequest(
        messages=[Message(role=MessageRole.USER, content='Say hello')],
        max_tokens=20
    )

    response = await service.generate(request)
    print(f'Response from {response.provider}: {response.content}')

asyncio.run(test())
"
```

Expected behavior:
- `/health` shows which providers are currently available
- `/generate` returns the provider name and model used
- `/extract` returns JSON with `entities` and `summary`
- the direct failover script tries OpenAI first and falls back to Ollama if OpenAI is unavailable

## CI

GitHub Actions runs:

- formatting checks with `ruff format --check`
- linting with `ruff check`
- type checking with `mypy`
- tests with `pytest`
- dependency review on pull requests
- CodeQL scanning for security analysis

Dependabot is also configured to keep `uv` dependencies and GitHub Actions up to date.

The main commands used in CI are:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest -q
```
