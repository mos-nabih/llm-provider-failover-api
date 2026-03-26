from enum import Enum

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: MessageRole
    content: str


class LLMRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=1024, ge=1, le=4096)
    stream: bool = False


class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


class ProviderStatus(BaseModel):
    name: str
    available: bool
    default_model: str
    error: str | None = None
