from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("source_id", "text")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_id: str
    text: str
    chunk_index: int = Field(ge=0)
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievedChunk(DocumentChunk):
    distance: float = Field(ge=0)


class SourceCitation(BaseModel):
    chunk_id: str
    source_id: str
    chunk_index: int
    metadata: dict[str, str] = Field(default_factory=dict)


class ChatTurn(BaseModel):
    user: str
    assistant: str


class GeneratedAnswer(BaseModel):
    text: str = Field(min_length=1)
    used_chunk_ids: list[str] = Field(default_factory=list)


class RagAnswer(BaseModel):
    text: str
    sources: list[SourceCitation]


MessageRole = Literal["system", "user", "assistant"]


class PromptMessage(BaseModel):
    role: MessageRole
    content: str
