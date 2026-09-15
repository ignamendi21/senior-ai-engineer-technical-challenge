import os
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, model_validator

from improved.errors import ConfigurationError


class RagSettings(BaseModel):
    api_key: SecretStr | None = None
    chat_model: str
    embedding_model: str
    persistence_directory: Path = Path("chroma_data")
    collection_name: str = "documents"
    chunk_size: int = Field(default=1_200, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    embedding_batch_size: int = Field(default=64, ge=1, le=2_048)
    default_top_k: int = Field(default=5, ge=1, le=20)
    max_top_k: int = Field(default=10, ge=1, le=100)
    max_history_turns: int = Field(default=6, ge=1, le=50)
    max_context_characters: int = Field(default=6_000, ge=500)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)

    @model_validator(mode="after")
    def validate_limits(self) -> "RagSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be lower than chunk_size")
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k must not exceed max_top_k")
        return self

    @classmethod
    def from_environment(cls) -> "RagSettings":
        api_key = os.environ.get("OPENAI_API_KEY")
        chat_model = os.environ.get("RAG_CHAT_MODEL")
        embedding_model = os.environ.get("RAG_EMBEDDING_MODEL")
        missing = [
            name
            for name, value in (
                ("OPENAI_API_KEY", api_key),
                ("RAG_CHAT_MODEL", chat_model),
                ("RAG_EMBEDDING_MODEL", embedding_model),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(f"Missing required configuration: {', '.join(missing)}")
        return cls(
            api_key=SecretStr(api_key),
            chat_model=chat_model,
            embedding_model=embedding_model,
            persistence_directory=Path(os.environ.get("RAG_PERSIST_DIRECTORY", "chroma_data")),
        )
