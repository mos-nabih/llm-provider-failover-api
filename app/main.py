import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .failover_service import FailoverService
from .models import LLMRequest, LLMResponse, Message, MessageRole, ProviderStatus
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Provider Failover API",
    description="LLM API with automatic failover and structured outputs",
    version="1.0.0",
)

# Initialize providers and failover service
ollama_provider = OllamaProvider()
openai_provider = OpenAIProvider()
failover_service = FailoverService([openai_provider, ollama_provider])


# Pydantic models for structured extraction
class ExtractedEntity(BaseModel):
    name: str = Field(description="Name of the entity")
    type: str = Field(description="Type: person, organization, or location")
    context: str | None = Field(None, description="Brief context about the entity")


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    summary: str = Field(description="Brief summary of the text")


class ExtractionRequest(BaseModel):
    text: str = Field(description="Text to extract entities from")


@app.get("/health")
async def health_check() -> list[ProviderStatus]:
    """Check health of all providers."""
    return await failover_service.refresh_health()


@app.post("/generate", response_model=LLMResponse)
async def generate(request: LLMRequest) -> LLMResponse:
    """Generate a response using available providers with failover."""
    try:
        return await failover_service.generate(request)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/generate/stream")
async def generate_stream(request: LLMRequest):
    """Generate a streaming response."""

    async def event_generator():
        try:
            async for chunk in failover_service.generate_stream(request):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except RuntimeError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/extract", response_model=ExtractionResult)
async def extract_entities(request: ExtractionRequest) -> ExtractionResult:
    """Extract structured entities from text using LLM."""
    system_prompt = """You are a data extraction assistant. Extract entities from the provided text.
Return ONLY valid JSON in this exact format:
{
  "entities": [
    {"name": "entity name", "type": "person|organization|location", "context": "brief context"}
  ],
  "summary": "brief summary of the text"
}"""

    llm_request = LLMRequest(
        messages=[
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=f"Extract entities from: {request.text}"),
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    try:
        response = await failover_service.generate(llm_request)
        # Parse the JSON response
        content = response.content.strip()
        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result_data = json.loads(content)
        return ExtractionResult(**result_data)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse LLM response as JSON: {e}",
        ) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
